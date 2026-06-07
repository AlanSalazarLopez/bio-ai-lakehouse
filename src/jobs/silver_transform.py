"""
silver_transform.py — Step 7: Silver Execution
Bio-AI Lakehouse · GTEx gene expression

Strategy:
  - Read Bronze schema to extract sample metadata column references (zero memory footprint).
  - Iterate across structured data in chunks of 200 columns: parse only targeted column metrics from Parquet.
  - Execute wide→long unpivoting shapes utilizing Pandas melt.
  - Perform structured lookups with tissue_mapping metadata dict records.
  - Route unmatched sample rows to isolated quarantine storage (chunk-level appends, preventing memory footprint accumulation).
  - Commit Silver tables into Delta Lake storage structures partitioned dynamically by tissue_id keys.
  - Evaluate pipeline quality gate metrics and append updated data lineage snapshots at execution exit.

Silver Target Table Schema:
  gene_id      STRING  NOT NULL  ← Name (Parquet index properties, reset as native column array)
  gene_symbol  STRING  NOT NULL  ← Description
  sample_id    STRING  NOT NULL  ← Original Bronze source column naming conventions
  tpm_value    FLOAT   NOT NULL  ← Target matrix expression record values
  tissue_id    STRING  NOT NULL  ← Derived via direct join lookups against gtex_metadata.txt definitions
"""

import math
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType

from src.utils.resources import get_spark_memory_settings, apply_to_spark_session
from src.utils.execution_profile import (
    ExecutionProfile, detect_profile, get_profile_by_name
)
from src.utils.quality_checks import run_silver_checks, PipelineQualityError
from src.utils.metadata_loader import load_tissue_mapping, validate_tissue_mapping
from src.utils.lineage import (
    load_bronze_lineage,
    build_silver_lineage,
    build_pipeline_lineage,
    save_lineage,
    SILVER_LINEAGE_PATH,
    PIPELINE_LINEAGE_PATH,
)

# ---------------------------------------------------------------------------
# Configuration Constraints & Target References
# ---------------------------------------------------------------------------

BRONZE_PATH     = "data/bronze/gtex/gene_tpm_raw.parquet"
SILVER_PATH     = "data/silver/gtex/gene_expression_long"
QUARANTINE_PATH = "data/quarantine/silver_unmatched_samples.parquet"
METADATA_PATH   = "data/raw/gtex_metadata.txt"

METADATA_COLS = ["Name", "Description"]  # Non-sample column boundaries inside Bronze layers

SILVER_SCHEMA = StructType([
    StructField("gene_id",     StringType(), nullable=False),
    StructField("gene_symbol", StringType(), nullable=False),
    StructField("sample_id",   StringType(), nullable=False),
    StructField("tpm_value",   FloatType(),  nullable=False),
    StructField("tissue_id",   StringType(), nullable=False),
])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Spark Session Factory Initialization
# ---------------------------------------------------------------------------

def build_spark(profile=None) -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("silver_transform")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    builder = apply_to_spark_session(builder, mode="local")

    # Static optimization parameters from active profile must be bound prior to triggering getOrCreate()
    if profile:
        for key, value in profile.static_configs().items():
            builder = builder.config(key, value)

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log.info("Target SparkSession instance initialized and ready.")
    return spark


# ---------------------------------------------------------------------------
# 2. Lazy Column Metadata Extraction
# ---------------------------------------------------------------------------

def get_bronze_sample_cols() -> List[str]:
    """
    Scans internal Parquet structural schemas exclusively — skips record data parsing to preserve memory limits.
    Returns array list containing sample column references (all discovered elements excluding METADATA_COLS boundaries).
    """
    schema = pq.read_schema(BRONZE_PATH)
    all_cols = schema.names
    sample_cols = [c for c in all_cols if c not in METADATA_COLS]
    log.info(
        f"Bronze structural schema read verified: {len(all_cols)} aggregate columns registered, "
        f"{len(sample_cols)} sample data target columns isolated."
    )
    return sample_cols


def load_bronze_chunk(cols: List[str]) -> pd.DataFrame:
    """
    Selectively targets and loads only METADATA_COLS + specific batch sequence data arrays from Bronze sources.
    Guarantees active RAM allocations never exceed chunk_size cols × 74,628 gene row lengths simultaneously.
    """
    df = pd.read_parquet(BRONZE_PATH, columns=METADATA_COLS + cols)

    # Re-normalize data structures if 'Name' fields fallback into index states
    if df.index.name == "Name":
        df = df.reset_index()

    return df


# ---------------------------------------------------------------------------
# 3. Reference Metadata Core Integration
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    mapping = load_tissue_mapping(path=METADATA_PATH)
    valid, msg = validate_tissue_mapping(mapping)
    if not valid:
        raise ValueError(f"Encountered non-conforming tissue mapping layout criteria: {msg}")
    log.info(
        f"Tissue mapping reference dict loaded: {len(mapping):,} mapped sample configurations "
        f"→ resolved into {len(set(mapping.values()))} unique biological tissue targets."
    )
    return mapping


# ---------------------------------------------------------------------------
# 4. In-Memory Wide-to-Long Matrix Transformations (Pandas Melt)
# ---------------------------------------------------------------------------

def reshape_chunk(
    df_chunk: pd.DataFrame,
    sample_cols: List[str],
) -> pd.DataFrame:
    """
    Consumes isolated batch matrix inputs (METADATA_COLS + sample_cols) and restructures values
    into localized tall transactional row records:
        gene_id | gene_symbol | sample_id | tpm_value

    Omits tissue_id resolution definitions here — attributes are appended inside join_tissue operations.
    Expression metrics indicating absolute zeroes denote valid biological states — DO NOT strip or filter records.
    """
    long = df_chunk.melt(
        id_vars=METADATA_COLS,
        value_vars=sample_cols,
        var_name="sample_id",
        value_name="tpm_value",
    )
    long = long.rename(columns={"Name": "gene_id", "Description": "gene_symbol"})
    long["tpm_value"] = long["tpm_value"].astype("float32")
    return long[["gene_id", "gene_symbol", "sample_id", "tpm_value"]]


# ---------------------------------------------------------------------------
# 5. Metadata Mapping Lookups & Quality Quarantine Isolation
# ---------------------------------------------------------------------------

def join_tissue(
    df_long: pd.DataFrame,
    tissue_mapping: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Injects matching tissue_id properties into long data tables via direct key lookup maps.
    Returns structurally split dataframe references containing: (matched_df, quarantine_df).
    """
    df_long["tissue_id"] = df_long["sample_id"].map(tissue_mapping)

    matched   = df_long[df_long["tissue_id"].notna()].copy()
    unmatched = df_long[df_long["tissue_id"].isna()].copy()

    if not unmatched.empty:
        unmatched["reason"] = "no_tissue_match"

    return matched, unmatched


# ---------------------------------------------------------------------------
# 6. Delta Lake Core Writing Pipeline Execution
# ---------------------------------------------------------------------------

def write_silver(
    spark: SparkSession,
    df_chunk: pd.DataFrame,
    first_write: bool,
    n_partitions: int = 10,
    max_records: int = 500_000,
) -> None:
    """
    Converts local Pandas batch matrices → PySpark DataFrame structures → Delta Lake target tables.
    Initial transactional batch commit: forces overwrite mode → preserves full pipeline idempotency guarantees.
    Succeeding processing sequence frames: toggle append mode → accumulative streaming without reloading history.

    Applying repartition(n_partitions) ensures bulk data batches break down into highly manageable worker steps.
    Enforcing maxRecordsPerFile constrains sizing bounds of target Parquet objects stored in deep disk layers —
    larger file scales = prevents small file fragmentation = reduces transactional log tracking load overhead.
    """
    sdf = spark.createDataFrame(df_chunk, schema=SILVER_SCHEMA)
    sdf = sdf.repartition(n_partitions)
    mode = "overwrite" if first_write else "append"

    (
        sdf.write
        .format("delta")
        .mode(mode)
        .partitionBy("tissue_id")
        .option("compression", "snappy")
        .option("maxRecordsPerFile", str(max_records))
        .save(SILVER_PATH)
    )


# ---------------------------------------------------------------------------
# 7. Low-Footprint Quarantine Streaming Append Strategies
# ---------------------------------------------------------------------------

def write_quarantine_chunk(df_unmatched: pd.DataFrame) -> int:
    """
    Commits non-conforming sample arrays immediately to disk — limits local RAM accumulation bounds.
    Inspects and evaluates existing disk targets, merges matching schemas, and triggers overwrite commits.
    Returns total volume count of mismatched records handled inside current data slice.
    """
    if df_unmatched.empty:
        return 0

    path = Path(QUARANTINE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df_unmatched], ignore_index=True)
    else:
        combined = df_unmatched

    combined.to_parquet(path, index=False, compression="snappy")
    log.warning(
        f"  Quarantine Routing: Appended +{len(df_unmatched):,} unmatched record lines "
        f"(Aggregated quarantine footprint total: {len(combined):,}) → Destination: {QUARANTINE_PATH}"
    )
    return len(df_unmatched)


# ---------------------------------------------------------------------------
# 8. Post-Ingestion Quality Gate Audit Routing
# ---------------------------------------------------------------------------

def run_quality_gate(
    silver_row_count: int,
    quarantine_row_count: int,
    spark: SparkSession,
):
    """
    Connects to target Silver tables via Delta engine contexts, computes truth metrics, and evaluates run_silver_checks.
    Halts pipeline orchestration and throws a PipelineQualityError on critical threshold breaches.
    """
    log.info("Initiating post-ingestion verification sweep across Silver target data spaces…")

    sdf   = spark.read.format("delta").load(SILVER_PATH)
    total = sdf.count()

    tissue_id_nulls = sdf.filter(F.col("tissue_id").isNull()).count()
    gene_id_nulls   = sdf.filter(F.col("gene_id").isNull()).count()
    sample_id_nulls = sdf.filter(F.col("sample_id").isNull()).count()

    zero_count       = sdf.filter(F.col("tpm_value") == 0.0).count()
    actual_zero_frac = zero_count / total if total > 0 else 0.0

    report = run_silver_checks(
        silver_row_count     = silver_row_count,
        quarantine_row_count = quarantine_row_count,
        tissue_id_null_count = tissue_id_nulls,
        gene_id_null_count   = gene_id_nulls,
        sample_id_null_count = sample_id_nulls,
        actual_zero_fraction = actual_zero_frac,
    )

    log.info(report.summary())

    if not report.passed:
        raise PipelineQualityError(report)

    log.info("Silver validation pipeline Quality Gate status: PASSED ✓")
    return report


# ---------------------------------------------------------------------------
# 9. Meta-Lineage Trace Synchronization
# ---------------------------------------------------------------------------

def update_lineage(
    bronze_lineage: dict,
    silver_row_count: int,
    quarantine_row_count: int,
    tissue_count: int,
    chunk_plan_summary: str,
    quality_report,
    duration_seconds: float,
) -> None:
    import psutil
    import platform

    infra = {
        "os":     platform.system(),
        "python": platform.python_version(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }
    mem_used = round(psutil.virtual_memory().used / 1e9, 2)

    silver_meta = build_silver_lineage(
        bronze_lineage       = bronze_lineage,
        silver_row_count     = silver_row_count,
        quarantine_row_count = quarantine_row_count,
        tissue_count         = tissue_count,
        chunk_plan_summary   = chunk_plan_summary,
        quality_report_dict  = quality_report.to_dict(),
        infra_fingerprint    = infra,
        memory_used          = mem_used,
        duration_seconds     = round(duration_seconds, 1),
    )

    save_lineage(silver_meta, SILVER_LINEAGE_PATH)

    pipeline_meta = build_pipeline_lineage(bronze_lineage, silver_meta)
    save_lineage(pipeline_meta, PIPELINE_LINEAGE_PATH)

    log.info(f"Pipeline lineage trace successfully serialized and persisted → {SILVER_LINEAGE_PATH}")


# ---------------------------------------------------------------------------
# 10. Core Application Orchestration Wrapper
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=None,
        help="Enforce manual resource profiles: SURVIVAL | BALANCED | PERFORMANCE | PRO"
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info("=" * 60)
    log.info("Initializing execution routine: silver_transform.py")
    log.info("=" * 60)

    # --- Identify deployment profiling state BEFORE spinning up active Spark engines (Intersperse configs to builder) ---
    mem_settings  = get_spark_memory_settings(mode="local")
    available_ram = mem_settings["meta"]["available_ram_gb"]

    if args.profile:
        profile = get_profile_by_name(args.profile)
        log.info(f"Target runtime optimization profile explicitly overridden via CLI context: {profile.name}")
    else:
        profile = detect_profile(available_ram)

    log.info(f"\n{profile.summary()}")

    # --- Initialize Spark Engine Core (Inject static structural profile parameters directly to the context builder) ---
    spark = build_spark(profile=profile)

    # --- Extract metadata references from schema structures (Zero resource array loading costs) ---
    tissue_mapping = load_metadata()
    bronze_lineage = load_bronze_lineage()
    sample_cols    = get_bronze_sample_cols()
    n_samples      = len(sample_cols)
    log.info(f"Aggregate volume matrix sample profiles isolated for execution: {n_samples:,}")

    # Bind post-initialization dynamic config settings across runtime Spark boundaries
    for key, value in profile.dynamic_configs().items():
        spark.conf.set(key, value)
    log.info("Dynamic execution performance variables successfully mapped across the active Spark landscape.")

    # Sizing metrics for column loops are extracted directly from system profiles
    chunk_size = profile.cols_per_chunk

    # --- Column Processing Chunk Loop & Buffered Write Infrastructure Stage ---
    # Instead of firing unmitigated individual table writes (396 concurrent calls degrade Delta storage layers),
    # data files are cached using chunks_per_write blocks to commit combined datasets within single transactional windows.
    # Yields: ~80 transactions instead of 396 operations → decelerates log footprint inflation roughly 5x.
    silver_row_count     = 0
    quarantine_row_count = 0
    first_write          = True
    batch_buffer         = []   # Collects valid chunk dataframes up to structural limits defined by chunks_per_write
    batch_row_count      = 0

    chunks   = [sample_cols[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]
    n_chunks = len(chunks)
    log.info(f"Constructed chunk layout: {n_chunks} iterations mapped | limits: {profile.chunks_per_write} chunks per transactional block.")
    log.info(f"Total projected discrete storage write sequences: ~{math.ceil(n_chunks / profile.chunks_per_write)}")

    for idx, cols in enumerate(chunks, start=1):
        log.info(f"Processing chunk slice sequence {idx}/{n_chunks} — Evaluating {len(cols)} current sample features")

        # Selectively parse and materialize target sample subsets from underlying Parquet files
        df_chunk = load_bronze_chunk(cols)

        # Reshape metrics from wide profiles to extended long row patterns
        df_long = reshape_chunk(df_chunk, cols)
        del df_chunk

        # Map metadata constraints to tissue keys
        matched, unmatched = join_tissue(df_long, tissue_mapping)
        del df_long

        # Process unmatched items: flush directly to disk endpoints to control memory boundaries
        quarantine_row_count += write_quarantine_chunk(unmatched)
        del unmatched

        # Empty match responses indicate missing upstream metadata associations
        if matched.empty:
            log.error(
                f"Processing anomaly discovered at chunk iteration {idx}/{n_chunks}: Returned ZERO valid data correlations. "
                f"Leading sample tags within block: {cols[:3]}… "
                "Target identifier strings lack valid lookups inside gtex_metadata.txt — verify metadata tables."
            )
            raise RuntimeError(
                f"Data slice block iteration {idx} generated exactly 0 matched layout records. "
                "Halt process tracking immediately to avoid downstream table corruption or incomplete states."
            )

        # Append clean structures directly to batch queues
        batch_buffer.append(matched)
        batch_row_count += len(matched)
        del matched

        # Trigger transaction writes once buffers reach capacity boundaries OR final records are parsed
        is_last_chunk    = (idx == n_chunks)
        buffer_is_full   = (len(batch_buffer) >= profile.chunks_per_write)

        if buffer_is_full or is_last_chunk:
            log.info(
                f"  Flushing transaction buffer blocks: serializing {len(batch_buffer)} combined slice structures, "
                f"aggregating {batch_row_count:,} records → Committing to Delta storage layers"
            )
            df_batch = pd.concat(batch_buffer, ignore_index=True)
            batch_buffer.clear()

            write_silver(
                spark,
                df_batch,
                first_write    = first_write,
                n_partitions   = profile.n_partitions,
                max_records    = profile.max_records_per_file,
            )
            first_write        = False
            silver_row_count += batch_row_count
            batch_row_count   = 0
            del df_batch

            log.info(f"  Aggregated tracking footprint currently committed to Silver table spaces: {silver_row_count:,} data records")

    # --- Systemic Data Completeness Validation Checks ---
    expected = n_samples * 74_628  # Target rows established during early data profiling stages
    actual   = silver_row_count + quarantine_row_count
    log.info(f"Completeness assertion check → Expected: {expected:,} rows | Calculated: {actual:,} rows")
    if actual != expected:
        log.warning(f"Discovered processing line margin delta of: {expected - actual:,} missing data fields.")

    # --- Post-Processing Quality Gate Evaluation Stage ---
    quality_report = run_quality_gate(silver_row_count, quarantine_row_count, spark)

    # --- Update Storage Meta Trace Histories ---
    tissue_count = len(set(tissue_mapping.values()))
    duration     = time.time() - t_start

    update_lineage(
        bronze_lineage       = bronze_lineage,
        silver_row_count     = silver_row_count,
        quarantine_row_count = quarantine_row_count,
        tissue_count         = tissue_count,
        chunk_plan_summary   = profile.summary(),
        quality_report       = quality_report,
        duration_seconds     = duration,
    )

    log.info("=" * 60)
    log.info(f"Execution run for silver_transform.py concluded successfully within {duration:.1f}s")
    log.info(f"  Silver destination data volume counts : {silver_row_count:,} rows")
    log.info(f"  Quarantine repository tracking bounds : {quarantine_row_count:,} rows")
    log.info(f"  Unique biological tissues resolved   : {tissue_count}")
    log.info("=" * 60)

    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except PipelineQualityError as e:
        log.critical(f"PIPELINE ORCHESTRATION TERMINATED — Quality gate check failed to meet validation criteria:\n{e}")
        raise SystemExit(1)
    except Exception:
        log.critical("An unhandled exception event managed to escape out of the silver_transform.py execution stack")
        traceback.print_exc()
        raise SystemExit(2)
