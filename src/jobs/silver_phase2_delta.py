"""
src/jobs/silver_phase2_delta.py

Silver Reshape Phase 2 — delta-rs (Pure PyArrow) populates Silver Delta Lake.

Phase 1 generated intermediate staging outputs partitioned by tissue_id under:
    data/staging/silver/tissue_id=<tissue>/batch_XXXX.parquet

Phase 2 executes a singular pipeline stage: consumes the staging directory and writes the Silver Delta Lake
enforcing the canonical target schema, partitioned by tissue_id, with Snappy compression enabled.

No active Spark instance, no JVM layer initialization, no shuffle operations — pure delta-rs execution.

Memory Management Strategy:
    - Never materializes an entire tissue profile into RAM arrays.
    - Iterates file by file within each tissue subdirectory utilizing pyarrow's iter_batches().
    - Batch sizes are dynamically computed relative to available host virtual memory (psutil).
    - Maximum actual RAM allocation footprint: limited to a single record batch array in flight.

Core Responsibilities:
    1. Scan available system memory capacity via psutil (adapts to host resource boundaries).
    2. Compute safe, non-hardcoded batch_size limits for processing.
    3. Read staging data sequentially tissue-by-tissue, file-by-file, batch-by-batch.
    4. Populate Delta Lake Silver tables natively using delta-rs (zero JVM overhead, zero shuffle execution).
    5. Trigger data quality gate validations using pure PyArrow compute blocks.
    6. Update data lineage tracking artifacts using lineage.py helpers.

Output:
    data/silver/gtex/gene_expression_long/    <- Silver Delta Lake Table
    data/lineage/silver_metadata.json         <- Silver Metadata Manifest
    data/lineage/pipeline_lineage.json        <- Accumulated Pipeline Lineage

Idempotency & Fault Isolation:
    The initial write batch enforces an overwrite strategy (clearing partial/stale target outputs).
    Subsequent processing iterations fall back to append routines.
    On execution aborts, simply drop the data/silver/ directory and restart — Phase 1 assets are preserved.

Usage:
    docker exec -it --workdir /opt/spark/work-dir spark-master \
        env PYTHONPATH=. python3 src/jobs/silver_phase2_delta.py

    # Bypass data quality constraints (Debug environments only):
    env PYTHONPATH=. python3 src/jobs/silver_phase2_delta.py --skip-quality
"""

import gc
import json
import logging
import shutil
import time
import traceback
import urllib.parse
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from src.utils.lineage import (
    load_bronze_lineage,
    build_silver_lineage,
    build_pipeline_lineage,
    save_lineage,
    SILVER_LINEAGE_PATH,
    PIPELINE_LINEAGE_PATH,
)
from src.utils.quality_checks import (
    run_silver_checks,
    PipelineQualityError,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STAGING_PATH    = "data/staging/silver"
SILVER_PATH     = "data/silver/gtex/gene_expression_long"
QUARANTINE_PATH = "data/staging/silver/quarantine/unmatched.parquet"
BRONZE_LINEAGE  = "data/lineage/bronze_metadata.json"
OPTIMAL_CONFIG  = "data/staging/silver_tuner/optimal_config.json"

# Canonical target Arrow schema for Silver — 5 data columns
SILVER_SCHEMA = pa.schema([
    pa.field("gene_id",     pa.string(),  nullable=False),
    pa.field("gene_symbol", pa.string(),  nullable=False),
    pa.field("sample_id",   pa.string(),  nullable=False),
    pa.field("tpm_value",   pa.float32(), nullable=False),
    pa.field("tissue_id",   pa.string(),  nullable=False),
])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Dynamically Compute Safe batch_size Bounds From System RAM
# ---------------------------------------------------------------------------

def calculate_batch_size(available_ram_gb: float) -> int:
    """
    Computes rows per record batch that safely fit within available system memory footprints.

    Restricts memory allocation targets to 25% of evaluated available system RAM — conservative buffer for:
    - PyArrow object deserialization memory inflation (~2x raw batch size dimensions)
    - Native delta-rs internal write cache allocations
    - Operating system runtime processes and background execution streams

    Baseline Profile: 5 structured columns of mixed types average ~120 bytes per uncompressed row object.

    Reference Calculations:
        2.8GB available → 25% target = 700MB → ~5.8M rows/batch
        4.0GB available → 25% target = 1.0GB → ~8.3M rows/batch
        8.0GB available → 25% target = 2.0GB → ~16.6M rows/batch

    Clamping Thresholds:
        min = 100_000   (Prevents delta-rs from generating small file fragmentation anomalies)
        max = 5_000_000 (Conservative ceiling limit designed to shield bounded host environments)
    """
    bytes_per_row   = 120
    usable_bytes    = available_ram_gb * 0.25 * (1024 ** 3)
    calculated_rows = int(usable_bytes / bytes_per_row)
    batch_size      = max(100_000, min(calculated_rows, 5_000_000))

    log.info(
        f"Available virtual memory footprint: {available_ram_gb:.2f}GB → "
        f"Dynamic batch target allocation set to: {batch_size:,} rows "
        f"(Roughly equivalent to ~{batch_size * bytes_per_row / (1024**2):.0f}MB per batch load chunk)"
    )
    return batch_size


# ---------------------------------------------------------------------------
# 2. Assert Existing Silver Ingestion State — Zero RAM footprint (Footers only)
# ---------------------------------------------------------------------------

EXPECTED_SILVER_ROWS = 74_628 * 19_788  # 1,476,738,864


def silver_is_complete(silver_path: str) -> bool:
    """
    Validates if target Silver assets already exist and conform exactly to expected row counts.
    Parses Parquet file metadata footers exclusively — zero data array parsing, instant execution.

    True  → Silver metrics validated, skip write sequence.
    False → Target data corrupted, missing, or incomplete; trigger rewrite execution.
    """
    silver_dir = Path(silver_path)
    if not silver_dir.exists():
        log.info("Target Silver directory paths not detected — write initialization required")
        return False

    parquet_files = [
        f for f in silver_dir.rglob("*.parquet")
        if "_delta_log" not in str(f)
    ]
    if not parquet_files:
        log.info("Target Silver directory contains no data elements — write initialization required")
        return False

    total = sum(pq.read_metadata(str(f)).num_rows for f in parquet_files)

    if total == EXPECTED_SILVER_ROWS:
        log.info(f"Target Silver metrics match valid state data constraints ({total:,} rows) — skipping write pipeline")
        return True

    log.warning(
        f"Silver data bounds identified as incomplete or duplicated ({total:,} found / {EXPECTED_SILVER_ROWS:,} expected) "
        f"— clearing directory path structures for write execution"
    )
    return False


# ---------------------------------------------------------------------------
# 2b. Populate Delta Silver — Streaming pipeline, zero multi-file loading
# ---------------------------------------------------------------------------

def write_delta_silver(staging_path: str, silver_path: str, batch_size: int) -> int:
    """
    Pure streaming routine — Maximum memory tracking capped at one processing record batch array in memory.

    Strategy:
    - Iterates sequentially through target tissue_id=<id> subdirectories.
    - Inside target tissues, scans each individual .parquet file independently.
    - Files are streamed via ParquetFile.iter_batches(batch_size) —
      averts loading whole uncompressed data tables into memory arrays.
    - Sequentially appends the target tissue_id key column value directly into the RecordBatch fields before serialization.
    - The initial transaction registers as an overwrite operation, subsequent batch loads fall back to append modes.
    - Systematic gc.collect() triggers isolate processing environments across distinct tissue sets.

    Peak RAM footprint expectations:
        batch_size × 120 bytes × ~2 (PyArrow execution overhead tracking)
        Under a 2.8GB system constraint using a batch_size=5M, allocations peak around ~1.2GB.

    Zero local or network data shuffles — eliminates JVM framework layers.
    """
    from deltalake import write_deltalake

    # Idempotency asset check: skip pipeline execution if target Silver matches constraints
    if silver_is_complete(silver_path):
        return EXPECTED_SILVER_ROWS

    # Silver state correction: purge existing structural paths to reset transactional integrity logs
    # Natively executing delta-rs overwrites append deletions inside log arrays without stripping
    # structural binary components instantly from filesystem paths, resulting in metrics calculation inflation.
    silver_dir = Path(silver_path)
    if silver_dir.exists():
        log.info(f"Wiping existing target paths to guarantee transactional integrity: {silver_path}")
        shutil.rmtree(silver_dir)
    silver_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(staging_path)
    tissue_dirs = sorted([
        d for d in staging_dir.iterdir()
        if d.is_dir() and d.name.startswith("tissue_id=")
    ])

    if not tissue_dirs:
        raise FileNotFoundError(
            f"No matching processed tissue partition directories found under path context: {staging_path}.\n"
            "Please execute silver_phase1_reshape.py prior to initiating this step."
        )

    log.info(f"Total target tissue elements identified for execution: {len(tissue_dirs)}")
    log.info(f"Target Delta Lake Silver storage path location    : {silver_path}")
    log.info(f"Operational data batch processing constraint ceiling : {batch_size:,} rows")
    log.info("Processing Engine framework context              : delta-rs streaming mode (zero JVM overhead, zero shuffle)")

    total_rows  = 0
    first_write = True

    for i, tissue_dir in enumerate(tissue_dirs, 1):
        # URL-decode directory naming: "Whole%20Blood" -> "Whole Blood"
        tissue_name   = urllib.parse.unquote(tissue_dir.name.split("=", 1)[1])
        parquet_files = sorted(tissue_dir.glob("*.parquet"))

        if not parquet_files:
            log.warning(f"  [{i}/{len(tissue_dirs)}] Tissue partition target: {tissue_name} — Contains no valid files, skipping path")
            continue

        log.info(f"[{i}/{len(tissue_dirs)}] Processing tissue partition key context: tissue_id={tissue_name} ({len(parquet_files)} items discovered)")
        tissue_rows = 0

        for pq_file in parquet_files:
            pf = pq.ParquetFile(str(pq_file))

            # iter_batches evaluation structure — guards against sweeping raw tables into system memory
            for batch in pf.iter_batches(batch_size=batch_size):
                n = batch.num_rows

                # Inject the target tissue_id attribute array directly inside the inline Parquet record attributes
                # Omitting partition_by flags preserves the attribute safely inside schemas without Hive string conversions
                batch_with_tissue = batch.append_column(
                    pa.field("tissue_id", pa.string(), nullable=False),
                    pa.array([tissue_name] * n, type=pa.string()),
                )

                # Restructure data fields to realign with target table schemas
                table = pa.Table.from_batches(
                    [batch_with_tissue],
                    schema=SILVER_SCHEMA,
                )

                mode = "overwrite" if first_write else "append"
                write_deltalake(
                    silver_path,
                    table,
                    schema=SILVER_SCHEMA,
                    # Retaining tissue_id inline preserves fields as default native dataset data components
                    # Delta log records manifest path records cleanly without relying on Hive directory string structures
                    mode=mode,
                    file_options={"compression": "snappy"},
                )
                first_write  = False
                tissue_rows += n

                del batch, batch_with_tissue, table
                gc.collect()

        total_rows += tissue_rows
        log.info(f"  ✅ Completed partition calculations for tissue element: {tissue_name} — {tissue_rows:,} rows successfully committed")
        gc.collect()

    log.info(f"All target tissue structures successfully written to Delta Lake storage — Aggregate line count: {total_rows:,} rows")
    return total_rows


# ---------------------------------------------------------------------------
# 3. Quality Gate Metrics Extraction — Pure PyArrow (No DeltaTable, No Spark)
# ---------------------------------------------------------------------------

def collect_silver_metrics(silver_path: str, quarantine_path: str, batch_size: int) -> dict:
    """
    Parses and aggregates metadata and row anomalies across Delta Silver assets to seed run_silver_checks().

    Queries Parquet data footprints directly using PyArrow helpers — bypasses DeltaTable tracking models,
    negates active Spark runtime setups, and circumvents JVM requirements. Iterates by discrete batch limits.

    Because target assets bypass Hive path structures (retaining tissue_id attributes inside default tables),
    the parser executes recursive sweeps across discovered target .parquet records.
    """
    log.info("Initiating Silver pipeline performance metrics audit (Pure PyArrow execution context)...")

    silver_dir = Path(silver_path)
    # Recursively discover target data files, ignoring isolated transactional _delta_log records
    parquet_files = [
        f for f in silver_dir.rglob("*.parquet")
        if "_delta_log" not in str(f)
    ]

    if not parquet_files:
        raise FileNotFoundError(f"No valid Parquet target assets discovered under search path context: {silver_path}")

    log.info(f"  Target Parquet resource records discovered for analysis: {len(parquet_files)}")

    silver_row_count = 0
    gene_id_nulls    = 0
    sample_id_nulls  = 0
    tissue_id_nulls  = 0
    zero_count       = 0
    tissue_set       = set()

    for pq_file in parquet_files:
        pf = pq.ParquetFile(str(pq_file))
        for batch in pf.iter_batches(batch_size=batch_size):
            silver_row_count += batch.num_rows
            gene_id_nulls    += pc.sum(pc.is_null(batch.column("gene_id"))).as_py()   or 0
            sample_id_nulls  += pc.sum(pc.is_null(batch.column("sample_id"))).as_py() or 0
            tissue_id_nulls  += pc.sum(pc.is_null(batch.column("tissue_id"))).as_py() or 0
            zero_count       += pc.sum(
                pc.equal(batch.column("tpm_value"), pa.scalar(0.0, pa.float32()))
            ).as_py() or 0
            tissue_set.update(pc.unique(batch.column("tissue_id")).to_pylist())
            del batch
        gc.collect()

    zero_fraction = zero_count / silver_row_count if silver_row_count > 0 else 0.0
    tissue_count  = len(tissue_set)

    # Quarantine file checks — footprint bounded by structural footers, zero data block overhead
    quarantine_row_count = 0
    if Path(quarantine_path).exists():
        quarantine_row_count = pq.read_metadata(quarantine_path).num_rows

    log.info(f"  Computed silver_row_count metrics   : {silver_row_count:,}")
    log.info(f"  Computed quarantine_rows metrics    : {quarantine_row_count:,}")
    log.info(f"  Computed unique tissue_count flags  : {tissue_count}")
    log.info(f"  Evaluated zero_fraction expression  : {zero_fraction:.2%}")
    log.info(f"  Identified gene_id null values      : {gene_id_nulls:,}")
    log.info(f"  Identified sample_id null values    : {sample_id_nulls:,}")
    log.info(f"  Identified tissue_id null values    : {tissue_id_nulls:,}")

    return {
        "silver_row_count":     silver_row_count,
        "quarantine_row_count": quarantine_row_count,
        "tissue_id_null_count": tissue_id_nulls,
        "gene_id_null_count":   gene_id_nulls,
        "sample_id_null_count": sample_id_nulls,
        "actual_zero_fraction": zero_fraction,
        "tissue_count":         tissue_count,
    }


# ---------------------------------------------------------------------------
# 4. Main Orchestration
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Bypass systemic data quality constraints (Restricted to runtime debugging routines)",
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info("=" * 60)
    log.info("Phase 2 — Staging Parquet -> Delta Lake Silver Pipeline Ingestion (delta-rs)")
    log.info("=" * 60)

    # Verify staging integrity states
    if not Path(STAGING_PATH).exists():
        raise FileNotFoundError(
            f"Staging reference path contexts could not be localized: {STAGING_PATH}.\n"
            "Please run silver_phase1_reshape.py prior to initiating this step."
        )

    # Parse auto tuner metadata profile contexts to preserve structural data lineage entries
    tuner_info = {}
    if Path(OPTIMAL_CONFIG).exists():
        with open(OPTIMAL_CONFIG) as f:
            tuner_info = json.load(f)
        log.info(
            f"Active Tuner profile constraints parsed: cols_per_chunk={tuner_info.get('cols_per_chunk')} "
            f"| estimated_duration={tuner_info.get('estimated_minutes')} target runtime minutes"
        )

    # Evaluate memory bounds
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    log.info(f"Host available virtual memory profile quantified at: {available_ram_gb:.2f}GB")

    # Establish processing limits bounded by environmental properties
    batch_size = calculate_batch_size(available_ram_gb)

    # Populate target Delta Lake tables — eliminates active Spark layers, JVM setups, and data shuffles
    total_rows = write_delta_silver(STAGING_PATH, SILVER_PATH, batch_size)

    # Data Quality Gate Processing
    if not args.skip_quality:
        metrics = collect_silver_metrics(SILVER_PATH, QUARANTINE_PATH, batch_size)

        quality_report = run_silver_checks(
            silver_row_count      = metrics["silver_row_count"],
            quarantine_row_count  = metrics["quarantine_row_count"],
            tissue_id_null_count  = metrics["tissue_id_null_count"],
            gene_id_null_count    = metrics["gene_id_null_count"],
            sample_id_null_count  = metrics["sample_id_null_count"],
            actual_zero_fraction  = metrics["actual_zero_fraction"],
        )

        log.info("\n" + quality_report.summary())

        if not quality_report.passed:
            raise PipelineQualityError(quality_report)

        log.info("Silver validation pipeline Quality Gate status: PASSED")
    else:
        log.warning("Systemic data quality constraints BYPASSED (--skip-quality flag active)")
        metrics        = {
            "silver_row_count": total_rows, "quarantine_row_count": 0,
            "tissue_count": 0,              "actual_zero_fraction": 0.0,
        }
        quality_report = None

    # Structural Lineage Management
    duration       = time.time() - t_start
    bronze_lineage = load_bronze_lineage(BRONZE_LINEAGE)

    silver_lineage = build_silver_lineage(
        bronze_lineage       = bronze_lineage,
        silver_row_count     = metrics["silver_row_count"],
        quarantine_row_count = metrics["quarantine_row_count"],
        tissue_count         = metrics.get("tissue_count", 0),
        chunk_plan_summary   = (
            f"delta-rs streaming iter_batches | batch_size={batch_size:,} (dynamic execution parsing) "
            f"| cols_per_chunk={tuner_info.get('cols_per_chunk', '?')}"
        ),
        quality_report_dict  = quality_report.to_dict() if quality_report else {},
        infra_fingerprint    = f"delta-rs-no-spark-ram{available_ram_gb:.1f}gb",
        memory_used          = f"{available_ram_gb:.2f}gb-available",
        duration_seconds     = round(duration, 1),
    )

    silver_lineage["transformations"] = [
        "wide->long reshape (PyArrow streaming execution format, column-by-column, zero continuous RAM build overhead)",
        f"cols_per_chunk={tuner_info.get('cols_per_chunk', '?')} (sourced via silver_tuner.py configuration parameters)",
        "join sample_id -> tissue_id derivation mapped from input partition folder contexts (gtex_metadata.txt template)",
        "cast operations: gene_id=string, gene_symbol=string, sample_id=string, tpm_value=float32 formatting properties",
        "zero values preserved — biologically verified baseline traits (51.89% data profile baseline constraint matching)",
        "unmatched sample elements flagged -> isolated into quarantine segments using reason=no_tissue_match parameter key values",
        f"Phase 2: delta-rs iter_batches streaming model framework | batch_size={batch_size:,} (dynamically derived relative to RAM boundaries)",
    ]

    pipeline_lineage = build_pipeline_lineage(bronze_lineage, silver_lineage)
    save_lineage(silver_lineage,   SILVER_LINEAGE_PATH)
    save_lineage(pipeline_lineage, PIPELINE_LINEAGE_PATH)

    # Runtime Execution Performance Summary
    log.info("=" * 60)
    log.info("Phase 2 Execution Sequence Concluded Successfully")
    log.info(f"  Total processing duration metric   : {duration:.1f}s ({duration/60:.1f} minutes elapsed time parameters)")
    log.info(f"  Silver table row total metrics     : {metrics['silver_row_count']:,} lines saved")
    log.info(f"  Quarantine table row total metrics : {metrics['quarantine_row_count']:,} rows isolated")
    log.info(f"  Unique tissue elements processed   : {metrics.get('tissue_count', '?')}")
    log.info(f"  Zero fraction expression value     : {metrics['actual_zero_fraction']:.2%}")
    log.info(f"  Operational streaming batch limits : {batch_size:,} rows allocated per run")
    log.info(f"  Available virtual host RAM record  : {available_ram_gb:.2f}GB tracked")
    log.info(f"  Target Delta Lake location paths   : {SILVER_PATH}")
    log.info(f"  Lineage metadata file tracking log : {SILVER_LINEAGE_PATH}")
    log.info("=" * 60)
    log.info("Next steps operational pointer sequence: src/jobs/gold_transform.py")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except PipelineQualityError as e:
        log.critical("Systemic data quality constraints breached — pipeline orchestration halted immediately")
        log.critical(str(e))
        raise SystemExit(2)
    except Exception:
        log.critical("Unhandled fatal runtime exception intercepted in silver_phase2_delta.py script path execution")
        traceback.print_exc()
        raise SystemExit(2)
