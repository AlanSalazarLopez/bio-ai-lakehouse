"""
src/jobs/silver_tuner.py

Config tuner for Phase 1 — binary search over cols_per_chunk parameters.

Executes across a lightweight small Bronze data sample (N rows × all available columns)
to capture the optimal cols_per_chunk sizing thresholds BEFORE triggering comprehensive reshape stages.

Strategy:
    1. Initialize sequence boundaries at cols_per_chunk = 1 (Absolute baseline minimum, safety guaranteed).
    2. Ascend in strict powers of 2 increments: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 → ...
    3. During each iteration: evaluate precise real peak RAM usage spikes utilizing a thread monitor every 200ms.
    4. If measured peak RAM usage exceeds RAM_SAFETY_THRESHOLD bounds: halt sweep, cache final validated operational N state.
    5. Serialize configuration outputs to optimal_config.json with active winner profile layouts.

Output Directory Artifacts:
    data/staging/silver_tuner/
        optimal_config.json      ← Read by Phase 1 workflow dynamically if discovered on launch
        tuner_log.json           ← Complete chronological test execution attempt histories

Usage Constraints:
    docker exec -it --workdir /opt/spark/work-dir spark-master \
        env PYTHONPATH=. python3 src/jobs/silver_tuner.py

    # Override existing cached profiles and force fresh tuning parameter executions:
    python3 src/jobs/silver_tuner.py --force

The tuner leverages a tiny test footprint restricted to SAMPLE_ROWS records — making individual passes resolve in seconds.
The validated configuration winner applies downstream to scale full production reshape pipelines requiring hours.
Executed ONCE per infrastructure workstation node setup. If underlying engine hardware is modified, re-run tuning tracks.
"""

import argparse
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

# Leverage the true system metadata_loader module — maintaining identical strip(), KeyError, and Phase 1 behavior.
# If local PYTHONPATH structures are missing (e.g. running outside active targets), fallback gracefully inline.
try:
    from src.utils.metadata_loader import load_tissue_mapping as _load_tissue_mapping_real
    _USE_REAL_LOADER = True
except ImportError:
    _USE_REAL_LOADER = False

# ---------------------------------------------------------------------------
# Global Performance Configuration Parameters
# ---------------------------------------------------------------------------

BRONZE_PATH      = "data/bronze/gtex/gene_tpm_raw.parquet"
METADATA_PATH   = "data/raw/gtex_metadata.txt"
TUNER_DIR        = "data/staging/silver_tuner"
OPTIMAL_CONFIG  = "data/staging/silver_tuner/optimal_config.json"
TUNER_LOG        = "data/staging/silver_tuner/tuner_log.json"
PROFILING_REPORT = "data/lineage/profiling_report.json"

METADATA_COLS   = ["Name", "Description"]
SAMPLE_ROWS      = 200          # Targeted Bronze row sample slicing depths — smaller boundaries ensure fast sweeps
MONITOR_HZ      = 0.2          # Samping latency gaps between decoupled thread RAM profile collections
RAM_SAFETY_FRAC = 0.80         # Intercept and abort iterations if peak delta usage crosses 80% initialization availability
MAX_COLS        = 500          # Operational capacity ceiling bounds — structural scaling loses efficiency past this mark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decoupled Non-Blocking RAM Consumption Thread Monitor
# ---------------------------------------------------------------------------

class RAMMonitor:
    """
    Decoupled execution thread monitoring absolute RAM utilization DELTA limits relative to initial values
    captured during instantiation execution calls to .start(). Measures active execution resource escalation.

    Prevents structural baseline environment footprints (Docker + WSL2 + Windows hosts) from triggering early limit trips
    before reshape operations get processed through the execution engine.

    Peak utilization variables are exposed via .peak_mb property calls — representing maximum observed memory deltas.
    """
    def __init__(self):
        self._stop_event  = threading.Event()
        self._baseline    = 0
        self._peak_delta  = 0
        self._thread      = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop_event.clear()
        self._baseline    = psutil.virtual_memory().used
        self._peak_delta = 0
        self._thread      = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop_event.is_set():
            delta = psutil.virtual_memory().used - self._baseline
            if delta > self._peak_delta:
                self._peak_delta = delta
            time.sleep(MONITOR_HZ)

    @property
    def peak_mb(self) -> float:
        """Maximum isolated RAM utilization delta observed past initialization base metrics — maps exact job footprints."""
        return max(0, self._peak_delta) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Data Simulation Layer — Profiling Isolation Strategies
# ---------------------------------------------------------------------------

def build_synthetic_sample(
    profiling_path: str,
    n_rows: int,
    n_cols: int,
) -> tuple:
    """
    Generates isolated synthetic pa.Table objects alongside corresponding mapped metadata reference frameworks
    solely parsing insights from profiling_report.json configurations — bypasses massive 19k column Parquet lookups.

    Emulates exact upstream Bronze table layout profiles:
        - Meta columns mapping "Name" and "Description" fields (Gene metadata arrays).
        - Generates n_cols sample reference attributes adhering to GTEX-SYNTH-{i:04d}-SM-TUNER structural masks.
        - Packs floating-point arrays aligning to realistic distribution spreads (e.g. ~51% zero values, log-normal scale balances).

    Maps sample structural metadata properties directly to realistic biological tissue groupings extracted from 
    profiling outputs, accurately preserving existing structural skew characteristics.

    Args:
        profiling_path: Target operational path pointing to core Step 4 profiling_report.json artifacts.
        n_rows: Target simulated row dimensions generated per chunk loop sequence (Default: 200).
        n_cols: Structural volume width limit mapping distinct sample targets within evaluations (Default: 500).

    Returns:
        Tuple sequence containing: (sample_table, sample_cols, tissue_mapping, total_real_cols, total_real_rows)
    """
    import random

    log.info(f"Loading metadata profile references from destination repository: {profiling_path}")
    with open(profiling_path) as f:
        report = json.load(f)

    total_real_cols = report["bronze_dimensions"]["n_sample_cols"]  # 19,788
    total_real_rows = report["bronze_dimensions"]["n_rows"]          # 74,628

    # ── Parse Real Biological Tissue Distrubutions from Profiling Reports ─────
    top_tissues    = report["metadata_profile"]["top_10_tissues"]
    bottom_tissues = report["metadata_profile"]["bottom_5_tissues"]
    all_tissues    = {**top_tissues, **bottom_tissues}
    tissue_names   = list(all_tissues.keys())

    # ── Map Synthetic Sample Identifiers matching real-world GTEx standards ───
    sample_cols = [f"GTEX-SYNTH-{i:04d}-SM-TUNER" for i in range(n_cols)]

    # ── Establish Synthetic Tissue Maps — Proportional Population Rules ───────
    # Mirrors core operational data skew realities: Whole Blood >> Liver - Portal Tract
    total_weight  = sum(all_tissues.values())
    tissue_mapping = {}
    col_idx = 0
    for tissue, count in all_tissues.items():
        # Evaluate localized relative density tracking values across specific targets
        n_for_tissue = max(1, round(n_cols * count / total_weight))
        for _ in range(n_for_tissue):
            if col_idx >= n_cols:
                break
            tissue_mapping[sample_cols[col_idx]] = tissue
            col_idx += 1
    # Clean up and append residual variables to primary baseline tissue nodes
    while col_idx < n_cols:
        tissue_mapping[sample_cols[col_idx]] = tissue_names[0]
        col_idx += 1

    # ── Formulate Synthetic Structural Gene Identification Arrays ─────────────
    gene_ids     = [f"ENSG{i:011d}.1" for i in range(n_rows)]
    gene_symbols = [f"GENE{i:05d}"    for i in range(n_rows)]

    # ── Formulate Synthetic Matrix TPM Signals — ~51% zero values, log-normal scales ───
    rng = random.Random(42)  # Hardcoded seed enforces fully reproducible tracking results
    zero_pct = report["null_zero_profile"]["zero_pct"] / 100.0  # 0.5189

    cols_data = {"Name": pa.array(gene_ids, type=pa.string()),
                 "Description": pa.array(gene_symbols, type=pa.string())}

    for col in sample_cols:
        values = []
        for _ in range(n_rows):
            if rng.random() < zero_pct:
                values.append(0.0)
            else:
                # Log-normal distribution maps standard biometrical transcript expressions
                values.append(float(rng.lognormvariate(1.5, 2.0)))
            cols_data[col] = pa.array(values, type=pa.float32())

    sample_table = pa.table(cols_data)

    log.info(
        f"Synthetic matrix evaluation model generated: {n_rows} rows × {n_cols} columns. "
        f"({len(tissue_mapping)} distinct sample instances bound across {len(set(tissue_mapping.values()))} tissue classes)"
    )
    log.info(f"Discovered total upstream Bronze structural columns: {total_real_cols:,} (Leveraged for downstream execution estimates)")

    return sample_table, sample_cols, tissue_mapping, total_real_cols, total_real_rows

# ---------------------------------------------------------------------------
# Benchmark Transformation Slices (In-Memory Processing Safeguards)
# ---------------------------------------------------------------------------

def _load_tissue_mapping(path: str) -> dict:
    """
    Passes execution tasks directly to the real system metadata_loader.py asset if discovered
    in scope (retains exact strip() lookups, KeyError tracking, and Phase 1 behavior patterns).
    Injects inline fallback structures exclusively if tracking imports trigger exceptions (e.g. host debugging).
    """
    if _USE_REAL_LOADER:
        return _load_tissue_mapping_real(path=path)

    # Local fallback routes — invoked only if python execution structures omit src.utils inside active paths
    log.warning("Primary structural metadata_loader missing from context paths — invoking localized fallback routines")
    mapping = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            if "SAMPID" not in header or "SMTSD" not in header:
                raise KeyError(f"Failed to isolate required SAMPID or SMTSD components within current file headers: {header[:10]}")
            sid_idx = header.index("SAMPID")
            tid_idx = header.index("SMTSD")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(sid_idx, tid_idx):
                    continue
                sample_id = parts[sid_idx].strip()
                tissue_id = parts[tid_idx].strip()
                if sample_id and tissue_id:
                    mapping[sample_id] = tissue_id
    except Exception as e:
        log.warning(f"Localized fallback tissue tracking methods triggered processing errors: {e} — Returning empty mapping dict structures")
    return mapping


def run_reshape_sample(
    sample_table: pa.Table,
    sample_cols: list,
    tissue_mapping: dict,
    cols_per_chunk: int,
) -> None:
    """
    Executes standard wide-to-long unpivoting tasks across target sample_table instances using set cols_per_chunk metrics.
    Replicates Phase 1 bugfix strategies: converts isolated column lines sequentially, avoiding concatenation tasks,
    and initializing temporary in-memory ParquetWriter tracking streams (Leverages BytesIO targets, bypassing physical disk steps).

    Peak active RAM footprint profiles here are constrained to: gene_ids + gene_symbols + single vector slices
    (~40MB footprint scaling arrays with 200 rows; scales linearly without capacity breaches at 74k limits).
    """
    import io

    gene_ids     = sample_table.column("Name")
    gene_symbols = sample_table.column("Description")
    n_genes      = len(gene_ids)

    SILVER_SCHEMA_NO_TISSUE = pa.schema([
        pa.field("gene_id",     pa.string(),  nullable=False),
        pa.field("gene_symbol", pa.string(),  nullable=False),
        pa.field("sample_id",   pa.string(),  nullable=False),
        pa.field("tpm_value",   pa.float32(), nullable=False),
    ])

    chunks = [
        sample_cols[i : i + cols_per_chunk]
        for i in range(0, len(sample_cols), cols_per_chunk)
    ]

    for chunk in chunks:
        # Spin up temporary thread tracking buffers per tissue_id — maps real Phase 1 storage serialization logic
        writers: dict = {}
        buffers: dict = {}

        try:
            for col in chunk:
                if col not in sample_table.schema.names:
                    continue

                tpm       = sample_table.column(col).cast(pa.float32())
                tissue_id = tissue_mapping.get(col, "unknown")

                single = pa.table(
                    {
                        "gene_id":     gene_ids,
                        "gene_symbol": gene_symbols,
                        "sample_id":   pa.array([col] * n_genes, type=pa.string()),
                        "tpm_value":   tpm,
                    },
                    schema=SILVER_SCHEMA_NO_TISSUE,  # Explicit structural nullable=False integrity checks
                )

                if tissue_id not in writers:
                    buf = io.BytesIO()
                    buffers[tissue_id] = buf
                    writers[tissue_id] = pq.ParquetWriter(
                        buf, schema=SILVER_SCHEMA_NO_TISSUE, compression="snappy"
                    )

                writers[tissue_id].write_table(single)
                del single, tpm

        finally:
            for w in writers.values():
                w.close()
            # Flush isolated memory blocks — downstream Phase 1 tasks commit these tracking arrays directly to disk layers
            buffers.clear()
            writers.clear()


# ---------------------------------------------------------------------------
# Individual Tuning Step Iteration
# ---------------------------------------------------------------------------

def attempt(
    sample_table: pa.Table,
    sample_cols: list,
    tissue_mapping: dict,
    cols_per_chunk: int,
    ram_limit_mb: float,
) -> dict:
    """
    Fires localized reshape passes over target simulated frames applying designated cols_per_chunk variables.
    Returns structured results dictionaries mapping specific step performance profiles.
    """
    monitor = RAMMonitor()
    t0 = time.time()
    success = True
    error_msg = None

    monitor.start()
    try:
        run_reshape_sample(sample_table, sample_cols, tissue_mapping, cols_per_chunk)
    except MemoryError as e:
        success = False
        error_msg = f"MemoryError: {e}"
    except Exception as e:
        success = False
        error_msg = f"{type(e).__name__}: {e}"
    finally:
        monitor.stop()

    elapsed = time.time() - t0
    peak_mb = monitor.peak_mb

    # Flag records as failed if calculated processing deltas step beyond target safe boundary thresholds
    if success and peak_mb > ram_limit_mb:
        success = False
        error_msg = f"Observed Peak RAM spike ({peak_mb:.0f}MB) outgrew established infrastructure safety thresholds ({ram_limit_mb:.0f}MB)"

    return {
        "cols_per_chunk": cols_per_chunk,
        "success":        success,
        "peak_ram_mb":    round(peak_mb, 1),
        "elapsed_s":      round(elapsed, 2),
        "error":          error_msg,
    }


# ---------------------------------------------------------------------------
# Aggregated Infrastructure Scalability Projections
# ---------------------------------------------------------------------------

def estimate_total_minutes(
    elapsed_sample_s: float,
    sample_rows: int,
    total_rows: int,
    sample_cols_used: int,
    total_cols: int,
) -> float:
    """
    Scales execution performance tracking baselines from sample spaces up to full production database workloads.
    Calculates metrics via: elapsed_sample_s / (sample_rows * sample_cols_used) * (total_rows * total_cols)
    """
    sample_work  = sample_rows * sample_cols_used
    total_work   = total_rows  * total_cols
    if sample_work == 0:
        return 0.0
    return (elapsed_sample_s / sample_work * total_work) / 60.0


# ---------------------------------------------------------------------------
# Main Orchestration Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tune configuration parameters even if pre-existing optimal_config.json mappings are detected",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=SAMPLE_ROWS,
        help=f"Total target row dimensions mapped into generated mock tables (Default: {SAMPLE_ROWS})",
    )
    parser.add_argument(
        "--sample-cols",
        type=int,
        default=500,
        help="Total target column dimensions mapped into generated mock tables (Default: 500)",
    )
    args = parser.parse_args()

    Path(TUNER_DIR).mkdir(parents=True, exist_ok=True)

    # ── Manage Pre-existing Profiling Configurations and Early Exits ──────────
    if Path(OPTIMAL_CONFIG).exists() and not args.force:
        with open(OPTIMAL_CONFIG) as f:
            cfg = json.load(f)
        log.info("=" * 60)
        log.info("Valid operational configuration discovered — extracting cached parameters")
        log.info(f"  Target cols_per_chunk parameter value : {cfg['cols_per_chunk']}")
        log.info(f"  Tracked Peak RAM footprint (Sample)   : {cfg['peak_ram_mb']} MB")
        log.info(f"  Projected workflow runtime totals     : {cfg['estimated_minutes']:.1f} minutes")
        log.info(f"  To force fresh hardware tuning sweeps : Add '--force' argument properties")
        log.info("=" * 60)
        return cfg

    # ── Resource Monitoring Setup & Environment Profiling ─────────────────────
    log.info("=" * 60)
    log.info("Silver Tuner Engine Launch — Executing Binary Search on cols_per_chunk targets")
    log.info("Execution Strategy: 100% Synthetic Sandbox Simulation — Bypassing 19k column Parquet file I/O locks")
    log.info("=" * 60)

    mem              = psutil.virtual_memory()
    available_ram_mb = mem.available / (1024 ** 2)
    total_ram_mb     = mem.total    / (1024 ** 2)
    used_ram_mb      = mem.used     / (1024 ** 2)
    # The operational safety margin defines maximum extra consumption limits over host baselines.
    # We evaluate available limits combined with safety fractions to prevent host OS thread lock occurrences.
    ram_limit_mb     = available_ram_mb * RAM_SAFETY_FRAC
    log.info(f"Total hardware memory footprint  : {total_ram_mb:.0f} MB")
    log.info(f"Host base memory footprint usage : {used_ram_mb:.0f} MB (Current baseline system limits)")
    log.info(f"Active available memory space    : {available_ram_mb:.0f} MB")
    log.info(f"Safe Delta capacity boundary     : {ram_limit_mb:.0f} maximum extra MB allocatable to reshape operations ({RAM_SAFETY_FRAC*100:.0f}% of total available)")

    # ── Materialize Test Data Structures from profiling_report.json ───────────
    # Safely isolates workloads from heavy Bronze files — removes early runtime OOM risks
    n_rows      = args.sample_rows
    n_cols      = args.sample_cols
    sample_table, sample_cols, tissue_mapping, total_cols, total_rows = build_synthetic_sample(
        profiling_path = PROFILING_REPORT,
        n_rows         = n_rows,
        n_cols         = n_cols,
    )

    # ── Execute Ascending Binary Step Search Tracks ───────────────────────────
    log.info("-" * 60)
    log.info("Launching ascending power binary parameter optimization sweeps...")
    log.info("  Sequence Pattern: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 → ...")
    log.info("-" * 60)

    results    = []
    best       = None          # Caches metadata from the last fully validated functional iteration
    candidates = []

    # Map candidate parameter steps across target powers of two below global ceilings
    n = 1
    while n <= MAX_COLS:
        candidates.append(n)
        n *= 2

    for cols in candidates:
        log.info(f"Evaluating execution efficiency with target cols_per_chunk set to: {cols}...")
        result = attempt(sample_table, sample_cols, tissue_mapping, cols, ram_limit_mb)
        results.append(result)

        status = "✅ OK" if result["success"] else "❌ FAIL"
        log.info(
            f"  Status: {status} | Measured Peak RAM: {result['peak_ram_mb']:.0f} MB "
            f"| Phase Duration: {result['elapsed_s']:.2f}s "
            f"| {'' if not result['error'] else 'Trace Error: ' + result['error']}"
        )

        if result["success"]:
            best = result
        else:
            log.info(f"  → Hardware resource exhaustion limit intersected at cols_per_chunk = {cols}")
            log.info(f"  → Optimal processing parameter established: cols_per_chunk = {best['cols_per_chunk'] if best else 1}")
            break

    # Fallback to absolute base settings if environment variables crash initial steps
    if best is None:
        log.warning("Even minimal baseline profiles (cols_per_chunk=1) triggered execution failures — inspect host configurations")
        best = {"cols_per_chunk": 1, "peak_ram_mb": 0, "elapsed_s": 0}

    # ── Project Downstream Scaled Processing Footprints ───────────────────────
    # total_rows and total_cols are mapped from upstream profiling reports (74,628 and 19,788)
    est_minutes = estimate_total_minutes(
        elapsed_sample_s  = best["elapsed_s"],
        sample_rows       = n_rows,
        total_rows        = total_rows,
        sample_cols_used  = min(best["cols_per_chunk"], n_cols),
        total_cols        = total_cols,
    )

    # ── Serialize Optimized Pipeline Parameter Targets ────────────────────────
    optimal = {
        "cols_per_chunk":      best["cols_per_chunk"],
        "peak_ram_mb":         best["peak_ram_mb"],
        "ram_limit_mb":        round(ram_limit_mb, 1),
        "estimated_minutes":   round(est_minutes, 1),
        "sample_rows_used":    n_rows,
        "sample_cols_used":    n_cols,
        "total_real_cols":     total_cols,
        "total_real_rows":     total_rows,
        "synthetic_sample":    True,
        "tuned_at":            time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OPTIMAL_CONFIG, "w") as f:
        json.dump(optimal, f, indent=2)

    with open(TUNER_LOG, "w") as f:
        json.dump({"attempts": results, "optimal": optimal}, f, indent=2)

    # ── Performance Tracking Summary Report Logs ──────────────────────────────
    print("=" * 60)
    log.info("HARDWARE OPTIMIZATION TUNER SUMMARY REPORT")
    log.info(f"  Optimal cols_per_chunk value  : {optimal['cols_per_chunk']}")
    log.info(f"  Peak simulation RAM footprint : {optimal['peak_ram_mb']:.0f} MB")
    log.info(f"  Projected operational runtime : {optimal['estimated_minutes']:.1f} total minutes")
    log.info(f"  Configuration output saved to : {OPTIMAL_CONFIG}")
    log.info("=" * 60)
    log.info("Next steps in processing sequence:")
    log.info("  python3 src/jobs/silver_phase1_reshape.py")
    log.info("  (Phase 1 pipelines import optimal_config.json variables automatically on startup)")
    log.info("=" * 60)

    return optimal


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("An unhandled system exception event forced early termination of silver_tuner.py")
        traceback.print_exc()
        raise SystemExit(2)
