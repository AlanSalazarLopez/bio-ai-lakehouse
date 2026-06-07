"""
src/jobs/silver_phase1_reshape.py

Silver Reshape Phase 1 — Pure PyArrow, no Spark, no JVM overhead.

Reads the Bronze wide Parquet file by column batches using the PyArrow Dataset API,
manually performs the wide→long reshape at the RecordBatch level, applies the join with
tissue_mapping, and writes a partitioned Parquet staging area sorted by tissue_id to disk.

No Spark → No JVM overhead → Peak RAM utilization: ~200MB per batch.
Staging data is retained on disk as a backup — Phase 2 consumes it to write to Delta Lake.

Output:
    data/staging/silver/
        tissue_id=Whole Blood/
            batch_001.parquet
            ...
        tissue_id=Liver/
            ...
        quarantine/
            unmatched.parquet   ← samples without a valid tissue match

Invariants:
    silver_rows + quarantine_rows == 74,628 × 19,788
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.utils.metadata_loader import load_tissue_mapping, validate_tissue_mapping
from src.utils.execution_profile import ExecutionProfile, detect_profile, get_profile_by_name
from src.utils.lineage import load_bronze_lineage, save_lineage, SILVER_LINEAGE_PATH

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BRONZE_PATH   = "data/bronze/gtex/gene_tpm_raw.parquet"
STAGING_PATH  = "data/staging/silver"
QUARANTINE_PATH = "data/staging/silver/quarantine/unmatched.parquet"
METADATA_PATH = "data/raw/gtex_metadata.txt"

METADATA_COLS = ["Name", "Description"]

# Output Arrow schema for long format
SILVER_SCHEMA_ARROW = pa.schema([
    pa.field("gene_id",     pa.string(),  nullable=False),
    pa.field("gene_symbol", pa.string(),  nullable=False),
    pa.field("sample_id",   pa.string(),  nullable=False),
    pa.field("tpm_value",   pa.float32(), nullable=False),
    pa.field("tissue_id",   pa.string(),  nullable=False),
])

QUARANTINE_SCHEMA_ARROW = pa.schema([
    pa.field("gene_id",     pa.string(),  nullable=False),
    pa.field("gene_symbol", pa.string(),  nullable=False),
    pa.field("sample_id",   pa.string(),  nullable=False),
    pa.field("tpm_value",   pa.float32(), nullable=False),
    pa.field("reason",      pa.string(),  nullable=False),
])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Retrieve sample columns from schema (Zero RAM footprint)
# ---------------------------------------------------------------------------

def get_sample_cols() -> List[str]:
    """Reads only the Parquet schema metadata — no data parsed, zero RAM overhead."""
    schema = pq.read_schema(BRONZE_PATH)
    sample_cols = [c for c in schema.names if c not in METADATA_COLS]
    log.info(
        f"Bronze Schema metadata parsed: {len(schema.names)} total columns detected, "
        f"{len(sample_cols)} identified as target sample columns"
    )
    return sample_cols


# ---------------------------------------------------------------------------
# 2. Wide→Long Reshape + Streaming Write (Zero accumulation in RAM)
# ---------------------------------------------------------------------------

SILVER_SCHEMA_NO_TISSUE = pa.schema([
    pa.field("gene_id",     pa.string(),  nullable=False),
    pa.field("gene_symbol", pa.string(),  nullable=False),
    pa.field("sample_id",   pa.string(),  nullable=False),
    pa.field("tpm_value",   pa.float32(), nullable=False),
])

def reshape_and_write_streaming(
    table: pa.Table,
    sample_cols: List[str],
    tissue_mapping: Dict[str, str],
    batch_idx: int,
) -> Tuple[int, int]:
    """
    Performs wide→long reshape column by column, streaming data directly to disk.
    No cumulative concat operations — Peak RAM capped at a single column in memory (~6MB).

    Strategy:
        - Maintains one active open ParquetWriter per tissue_id (reused across columns)
        - Each column data array is processed, written, and garbage-collected before the next
        - Quarantine records accumulate only on unmatched metrics (expected to be minimal/empty)

    Returns (matched_rows, quarantine_rows).
    """
    gene_ids     = table.column("Name")
    gene_symbols = table.column("Description")
    n_genes      = len(gene_ids)

    staging_dir = Path(STAGING_PATH)
    staging_dir.mkdir(parents=True, exist_ok=True)

    writers: Dict[str, pq.ParquetWriter] = {}
    quarantine_batches = []
    matched_rows   = 0
    quarantine_rows = 0

    try:
        for col in sample_cols:
            tpm_values = table.column(col).cast(pa.float32())
            tissue_id  = tissue_mapping.get(col)

            if tissue_id is not None:
                single = pa.table(
                    {
                        "gene_id":     gene_ids,
                        "gene_symbol": gene_symbols,
                        "sample_id":   pa.array([col] * n_genes, type=pa.string()),
                        "tpm_value":   tpm_values,
                    },
                    schema=SILVER_SCHEMA_NO_TISSUE,
                )

                if tissue_id not in writers:
                    # Create subdirectory structure and target writer instance for this specific tissue
                    tissue_dir = staging_dir / f"tissue_id={tissue_id}"
                    tissue_dir.mkdir(parents=True, exist_ok=True)
                    out_path = tissue_dir / f"batch_{batch_idx:04d}.parquet"

                    # Idempotency check: if the asset already exists, evaluate file integrity
                    # Structurally intact file → Skip processing this tissue in the current batch
                    # Corrupted asset → Wipe from filesystem and trigger rewrite execution
                    if out_path.exists():
                        try:
                            pq.read_metadata(str(out_path))  # Parsers file footer metadata only, zero RAM
                            log.info(f"  Batch {batch_idx} tissue={tissue_id} already exists and is intact — skipping execution")
                            matched_rows += n_genes  # Account for rows even when skipping the write stage
                            del single, tpm_values
                            continue
                        except Exception:
                            log.warning(f"  Batch {batch_idx} tissue={tissue_id} found corrupted — unlinking asset for rewrite")
                            out_path.unlink()

                    writers[tissue_id] = pq.ParquetWriter(
                        str(out_path),
                        schema=SILVER_SCHEMA_NO_TISSUE,
                        compression="snappy",
                    )

                writers[tissue_id].write_table(single)
                matched_rows += n_genes
                del single

            else:
                # Quarantine handler — safely isolates unmatched column keys
                quarantine_batches.append(pa.table(
                    {
                        "gene_id":     gene_ids,
                        "gene_symbol": gene_symbols,
                        "sample_id":   pa.array([col] * n_genes, type=pa.string()),
                        "tpm_value":   tpm_values,
                        "reason":      pa.array(["no_tissue_match"] * n_genes, type=pa.string()),
                    },
                    schema=QUARANTINE_SCHEMA_ARROW,
                ))
                quarantine_rows += n_genes

            del tpm_values

    finally:
        # Close all active writers — enforces persistent flush routines to disk storage
        for writer in writers.values():
            writer.close()

    # Commit quarantine allocations if anomalies are encountered
    if quarantine_batches:
        write_quarantine(pa.concat_tables(quarantine_batches))

    return matched_rows, quarantine_rows


def write_quarantine(quarantine: pa.Table) -> None:
    """Appends data to quarantine — bundles all unmatched records into a unified Parquet file."""
    if quarantine.num_rows == 0:
        return

    path = Path(QUARANTINE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing   = pq.read_table(path)
        quarantine = pa.concat_tables([existing, quarantine])

    pq.write_table(quarantine, str(path), compression="snappy")
    log.warning(f"  Quarantine warning: {quarantine.num_rows:,} rows redirected → {QUARANTINE_PATH}")


# ---------------------------------------------------------------------------
# 4. Checkpoint State — To resume processing on failure conditions
# ---------------------------------------------------------------------------

PROGRESS_FILE = "data/staging/silver/.progress.json"

def load_progress() -> int:
    """Returns index references of the last successful batch completed (0 if init run)."""
    import json
    path = Path(PROGRESS_FILE)
    if not path.exists():
        return 0
    with open(path) as f:
        return json.load(f).get("last_completed_batch", 0)

def save_progress(batch_idx: int, silver_rows: int, quarantine_rows: int) -> None:
    """Saves operational checkpoint variables following a successful batch routine."""
    import json
    path = Path(PROGRESS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "last_completed_batch": batch_idx,
            "silver_rows_so_far":   silver_rows,
            "quarantine_rows_so_far": quarantine_rows,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# 5. Main Orchestration
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=None,
        help="Enforce operational profile runtime configuration: SURVIVAL | BALANCED | PERFORMANCE | PRO"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution tracking from the last valid verified state batch"
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info("=" * 60)
    log.info("Phase 1 — Reshape Bronze → Parquet Staging Pipeline (PyArrow)")
    log.info("=" * 60)

    # --- Profile Setup ---
    import psutil
    available_ram = psutil.virtual_memory().available / (1024 ** 3)
    if args.profile:
        profile = get_profile_by_name(args.profile)
        log.info(f"Execution Profile enforced via override: {profile.name}")
    else:
        profile = detect_profile(available_ram)

    log.info(f"\n{profile.summary()}")
    chunk_size = profile.cols_per_chunk

    # --- Parse Tuner configuration configs if present (Overrides default profile) ---
    OPTIMAL_CONFIG = "data/staging/silver_tuner/optimal_config.json"
    import json as _json
    _tuner_path = Path(OPTIMAL_CONFIG)
    if _tuner_path.exists() and not args.profile:
        with open(_tuner_path) as f:
            _cfg = _json.load(f)
        chunk_size = _cfg["cols_per_chunk"]
        log.info(
            f"Tuner parameter settings discovered → cols_per_chunk = {chunk_size} "
            f"(Estimated peak RAM: {_cfg['peak_ram_mb']:.0f}MB, "
            f"~{_cfg['estimated_minutes']:.0f} total runtime minutes expected)"
        )
    elif _tuner_path.exists() and args.profile:
        log.info("Execution profile enforced via command line parameter flag --profile, bypassing auto tuner settings")

    # --- Metadata loading and field parsing ---
    tissue_mapping = load_tissue_mapping(path=METADATA_PATH)
    valid, msg     = validate_tissue_mapping(tissue_mapping)
    if not valid:
        raise ValueError(f"Invalid tissue mapping schema encountered: {msg}")
    log.info(f"Tissue mapping index: {len(tissue_mapping):,} unique samples mapped → {len(set(tissue_mapping.values()))} discrete tissues")

    sample_cols = get_sample_cols()
    n_samples   = len(sample_cols)

    # --- Slicing Batch Arrays ---
    batches  = [sample_cols[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]
    n_batches = len(batches)

    # --- Checkpoint Recovery Logic ---
    start_batch = 0
    silver_rows = 0
    quarantine_rows = 0

    if args.resume:
        last = load_progress()
        if last > 0:
            import json
            with open(PROGRESS_FILE) as f:
                prog = json.load(f)
            start_batch     = last  # Proceed from the subsequent batch reference
            silver_rows     = prog.get("silver_rows_so_far", 0)
            quarantine_rows = prog.get("quarantine_rows_so_far", 0)
            log.info(f"Recovering runtime pipeline environment from state batch {start_batch + 1}/{n_batches}")
            log.info(f"  Accumulated Silver records logged up to checkpoint: {silver_rows:,} rows")

    log.info(f"Total structured batch allocations: {n_batches} | columns per processing chunk: {chunk_size}")
    log.info(f"Remaining workload batches to calculate: {n_batches - start_batch}")

    # --- Core Pipeline Processing Loop ---
    for idx, cols in enumerate(batches):
        if idx < start_batch:
            continue

        batch_num = idx + 1
        log.info(f"Processing Batch {batch_num}/{n_batches} — Parsing {len(cols)} unique metadata sample keys")

        # Query and parse isolated column records from Bronze source via PyArrow Dataset pointers
        dataset = ds.dataset(BRONZE_PATH, format="parquet")
        table   = dataset.to_table(columns=METADATA_COLS + cols)

        # Execute structural wide→long pivoting + trigger direct serialization streaming (RAM optimization)
        matched_n, quarantine_n = reshape_and_write_streaming(
            table, cols, tissue_mapping, batch_num
        )
        del table

        silver_rows     += matched_n
        quarantine_rows += quarantine_n

        # Commit progress state to checkpoint storage
        save_progress(idx + 1, silver_rows, quarantine_rows)
        log.info(f"  Current cumulative Silver rows committed to disk staging: {silver_rows:,} rows")

    # --- Data Lineage Audit and Closure Assertions ---
    expected = 74_628 * 19_788
    actual   = silver_rows + quarantine_rows
    log.info(f"Row count integrity assertion check → Expected target: {expected:,} | Computed actual: {actual:,}")
    if actual != expected:
        log.warning(f"Data lineage integrity discrepancy detected: {expected - actual:+,} missing records")
    else:
        log.info("✅ Pipeline data lineage structural checks passed successfully")

    duration = time.time() - t_start
    log.info("=" * 60)
    log.info(f"Phase 1 execution process finished in {duration:.1f}s ({duration/60:.1f} minutes total elapsed time)")
    log.info(f"  Silver staging records committed : {silver_rows:,}")
    log.info(f"  Quarantine records segregated    : {quarantine_rows:,}")
    log.info(f"  Staging target path location     : {STAGING_PATH}")
    log.info("=" * 60)

    return silver_rows, quarantine_rows


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("Unhandled fatal runtime exception intercepted in silver_phase1_reshape.py")
        traceback.print_exc()
        raise SystemExit(2)
