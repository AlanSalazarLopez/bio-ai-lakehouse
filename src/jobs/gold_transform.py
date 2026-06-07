"""
src/jobs/gold_transform.py

Gold Orchestrator — Final layer of the Bio-AI Lakehouse pipeline.

What it does:
    1. Loads Silver lineage to retrieve the expected tissue_count
    2. Iterates over Silver in batches via SilverBatchReader (avoiding massive RAM overhead)
    3. Accumulates statistics per group (gene_id, gene_symbol, tissue_id)
       using online Welford (mean/std) and reservoir sampling (median)
       via numpy memmap — disk-backed arrays keeping RAM constant at ~200MB
    4. Serializes accumulators into a pa.Table
    5. Writes Gold Delta Lake with ZSTD compression via write_deltalake
    6. Runs Gold quality gate checks (run_gold_checks)
    7. Updates data lineage (gold_metadata.json + pipeline_lineage.json)

Estimated peak RAM:
    ~200MB active memmap + ~100MB Silver batch = ~300MB constant footprint
    No OOM exceptions regardless of the number of unique groups.

Usage:
    docker exec -it --workdir /opt/spark/work-dir spark-master \
    env PYTHONPATH=. python3 src/jobs/gold_transform.py

    # Skip quality gate (debug only):
    env PYTHONPATH=. python3 src/jobs/gold_transform.py --skip-quality

    # Resume from an existing cache (if the process was interrupted):
    env PYTHONPATH=. python3 src/jobs/gold_transform.py --resume

Idempotency:
    write_deltalake using mode='overwrite' — if a partial Gold state exists, it is wiped clean.
    Re-running without --resume is safe.
"""

import argparse
import logging
import sys
import time
from typing import Any, Dict

import psutil
import pyarrow as pa
import pyarrow.compute as pc
from deltalake import write_deltalake

from src.utils.gold_accumulators import GoldAccumulatorMap
from src.utils.gold_batch_reader import SilverBatchReader
from src.utils.lineage import (
    build_gold_lineage,
    build_pipeline_lineage,
    load_lineage,
    save_lineage,
    PIPELINE_LINEAGE_PATH,
)
from src.utils.quality_checks import (
    PipelineQualityError,
    run_gold_checks,
)

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

SILVER_ROOT          = "data/silver/gtex/gene_expression_long"
GOLD_PATH            = "data/gold/gtex/gene_tissue_summary"
GOLD_CACHE_DIR       = "data/gold_cache"
GOLD_LINEAGE_PATH    = "data/lineage/gold_metadata.json"
SILVER_LINEAGE_PATH  = "data/lineage/silver_metadata.json"

EXPECTED_GENES   = 74_628
EXPECTED_TISSUES = 68

# Flush cache to disk every N rows — protects against unexpected crashes
FLUSH_EVERY_ROWS = 50_000_000

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Step 1 — Load Silver Lineage
# ─────────────────────────────────────────────

def load_silver_lineage() -> Dict[str, Any]:
    lineage = load_lineage(SILVER_LINEAGE_PATH)
    if not lineage:
        logger.warning(
            "Silver lineage not found at %s — falling back to defaults",
            SILVER_LINEAGE_PATH,
        )
        return {
            "output": {
                "tissue_count": EXPECTED_TISSUES,
                "row_count":    1_476_738_864,
                "path":         SILVER_ROOT,
            },
            "generated_at_utc": "unknown",
        }
    logger.info(
        "Silver lineage loaded — tissue_count=%s, row_count=%s",
        lineage.get("output", {}).get("tissue_count", "?"),
        f"{lineage.get('output', {}).get('row_count', 0):,}",
    )
    return lineage


# ─────────────────────────────────────────────
#  Step 2-3 — Iterate Silver and Accumulate
# ─────────────────────────────────────────────

def accumulate_silver(acc_map: GoldAccumulatorMap, resume: bool = False) -> SilverBatchReader:
    """
    Reads the entire Silver layer in batches and updates the memmap accumulators.
    Flushes cache to disk every FLUSH_EVERY_ROWS to protect against crashes.
    """
    reader     = SilverBatchReader(SILVER_ROOT)
    total_rows = 0
    rows_since_flush = 0
    start_ts   = time.time()

    logger.info("Starting Silver → Gold accumulation (%s assets)...",
                f"{reader.total_files:,}")

    for batch in reader.iter_batches():
        acc_map.update_from_batch(batch)
        total_rows       += batch.num_rows
        rows_since_flush += batch.num_rows

        # Periodic disk flush
        if rows_since_flush >= FLUSH_EVERY_ROWS:
            acc_map.flush()
            rows_since_flush = 0
            logger.info("Cache successfully flushed to disk — %s unique groups tracked",
                        f"{acc_map.group_count:,}")

        # Progress tracking log every 50M rows
        if total_rows % 50_000_000 < batch.num_rows:
            elapsed  = time.time() - start_ts
            pct      = total_rows / 1_476_738_864 * 100
            ram_gb   = psutil.virtual_memory().available / 1024 ** 3
            logger.info(
                "Progress: %s rows transformed (%.1f%%) | unique_groups=%s | Free RAM=%.1fGB | Time: %.0fs",
                f"{total_rows:,}", pct,
                f"{acc_map.group_count:,}",
                ram_gb, elapsed,
            )

    # Final persistent flush
    acc_map.flush()

    elapsed_total = time.time() - start_ts
    logger.info(
        "Accumulation completed successfully: %s rows handled in %.0fs | %s total unique groups",
        f"{reader.rows_yielded:,}", elapsed_total,
        f"{acc_map.group_count:,}",
    )
    return reader


# ─────────────────────────────────────────────
#  Step 4-5 — Serialize and Commit Gold Delta
# ─────────────────────────────────────────────

def write_gold(gold_table: pa.Table) -> None:
    logger.info(
        "Committing Gold Delta Lake table: %s rows → %s",
        f"{gold_table.num_rows:,}", GOLD_PATH,
    )

    write_deltalake(
        GOLD_PATH,
        gold_table,
        mode              = "overwrite",
        partition_by      = ["tissue_id"],
        storage_options   = {"allow_http": "true"},
        configuration     = {
            "delta.minWriterVersion": "2",
            "delta.minReaderVersion": "1",
        },
    )

    logger.info("Gold layer successfully committed to storage at %s", GOLD_PATH)


# ─────────────────────────────────────────────
#  Step 6 — Gold Quality Gate Checks
# ─────────────────────────────────────────────

def run_quality_gate(gold_table: pa.Table, skip: bool = False) -> Dict[str, Any]:
    if skip:
        logger.warning("Gold quality gate checks SKIPPED (--skip-quality flag active)")
        return {"layer": "gold", "passed": True, "skipped": True, "checks": []}

    rows = gold_table.to_pydict()
    n    = gold_table.num_rows

    gene_id_nulls     = gold_table.column("gene_id").null_count
    gene_symbol_nulls = gold_table.column("gene_symbol").null_count
    tissue_id_nulls   = gold_table.column("tissue_id").null_count
    tissue_count      = len(set(rows["tissue_id"]))
    min_sample_count  = int(pc.min(gold_table.column("sample_count")).as_py())
    avg_zero_fraction = float(pc.mean(gold_table.column("zero_fraction")).as_py())

    report = run_gold_checks(
        gold_row_count         = n,
        tissue_count           = tissue_count,
        gene_id_null_count     = gene_id_nulls,
        gene_symbol_null_count = gene_symbol_nulls,
        tissue_id_null_count   = tissue_id_nulls,
        min_sample_count       = min_sample_count,
        avg_zero_fraction      = avg_zero_fraction,
        expected_genes         = EXPECTED_GENES,
        expected_tissues       = EXPECTED_TISSUES,
    )

    logger.info("\n%s", report.summary())

    if not report.passed:
        raise PipelineQualityError(report)

    return report.to_dict()


# ─────────────────────────────────────────────
#  Step 7 — Metadata Lineage Tracking
# ─────────────────────────────────────────────

def update_lineage(
    gold_table:          pa.Table,
    quality_report_dict: Dict[str, Any],
    duration_seconds:    float,
) -> None:
    tissue_count = len(set(gold_table.column("tissue_id").to_pylist()))
    ram_used_gb  = round(
        (psutil.virtual_memory().total - psutil.virtual_memory().available)
        / 1024 ** 3, 2
    )

    import hashlib, json as _json, platform
    snap = {
        "total_ram": psutil.virtual_memory().total,
        "platform":  platform.system(),
    }
    fingerprint = hashlib.sha256(
        _json.dumps(snap, sort_keys=True).encode()
    ).hexdigest()[:16]

    silver_lineage = load_lineage(SILVER_LINEAGE_PATH)
    bronze_lineage = load_lineage("data/lineage/bronze_metadata.json")

    gold_lineage = build_gold_lineage(
        silver_lineage      = silver_lineage,
        gold_row_count      = gold_table.num_rows,
        tissue_count        = tissue_count,
        quality_report_dict = quality_report_dict,
        infra_fingerprint   = fingerprint,
        memory_used         = f"{ram_used_gb}g",
        duration_seconds    = duration_seconds,
    )

    pipeline_lineage = build_pipeline_lineage(
        bronze_lineage = bronze_lineage,
        silver_lineage = silver_lineage,
        gold_lineage   = gold_lineage,
    )

    save_lineage(gold_lineage,     GOLD_LINEAGE_PATH)
    save_lineage(pipeline_lineage, PIPELINE_LINEAGE_PATH)

    logger.info("Data lineage records successfully updated:")
    logger.info("  %s", GOLD_LINEAGE_PATH)
    logger.info("  %s", PIPELINE_LINEAGE_PATH)
    logger.info("  Pipeline operational status: %s", pipeline_lineage.get("status"))


# ─────────────────────────────────────────────
#  Main Orchestrator Entrypoint
# ─────────────────────────────────────────────

def main(skip_quality: bool = False, resume: bool = False) -> None:
    start_total = time.time()

    logger.info("══════════════════════════════════════════")
    logger.info("  Bio-AI Lakehouse — Gold Transform v3")
    logger.info("  Execution Mode: %s", "RESUME" if resume else "FRESH")
    logger.info("══════════════════════════════════════════")

    # Step 1 — Load context from upstream Silver lineage
    silver_lineage = load_silver_lineage()

    # Step 2-3 — Initialize memmap architecture and stream compute data
    acc_map = GoldAccumulatorMap(
        cache_dir = GOLD_CACHE_DIR,
        resume    = resume,
    )
    reader = accumulate_silver(acc_map, resume=resume)

    # Step 4 — Serialize structured computational arrays to pa.Table
    logger.info("Serializing analytical accumulators into a pa.Table structure...")
    gold_table = acc_map.to_arrow_table()
    logger.info("pa.Table processing ready: %s rows × %s columns mapped",
                f"{gold_table.num_rows:,}", gold_table.num_columns)

    # Step 5 — Commit state to Gold Delta Lake target
    write_gold(gold_table)

    # Step 6 — Run schema validation and evaluation quality gate
    quality_dict = run_quality_gate(gold_table, skip=skip_quality)

    # Step 7 — Package metrics and audit lineage
    duration = time.time() - start_total
    update_lineage(gold_table, quality_dict, duration)

    logger.info("══════════════════════════════════════════")
    logger.info("  Gold Transform SUCCESSFUL in %.0fs (%.1f min)",
                duration, duration / 60)
    logger.info("  Target location: %s", GOLD_PATH)
    logger.info("  Total rows committed: %s", f"{gold_table.num_rows:,}")
    logger.info("══════════════════════════════════════════")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Transform — Bio-AI Lakehouse")
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip operational Gold quality gate checks (debug environment only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from existing memmap analytical cache state",
    )
    args = parser.parse_args()

    try:
        main(skip_quality=args.skip_quality, resume=args.resume)
    except PipelineQualityError as e:
        logger.error("Pipeline run terminated by quality gate assertion: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected runtime error intercepted: %s", e, exc_info=True)
        sys.exit(1)
