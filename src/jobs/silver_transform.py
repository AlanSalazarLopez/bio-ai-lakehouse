"""
silver_transform.py — Paso 7: Ejecución Silver
Bio-AI Lakehouse · GTEx gene expression

Estrategia:
  - Lee schema de Bronze para obtener columnas de muestras (cero RAM)
  - Itera en chunks de 200 columnas: lee solo esas cols del parquet
  - wide→long con Pandas melt
  - Join con tissue_mapping de metadata
  - Separa unmatched samples a quarantine (append por chunk, sin acumular)
  - Escribe Silver en Delta Lake particionado por tissue_id
  - Quality gate + lineage al final

Silver schema:
  gene_id     STRING  NOT NULL  ← Name (índice parquet, reset como columna)
  gene_symbol STRING  NOT NULL  ← Description
  sample_id   STRING  NOT NULL  ← nombre de columna Bronze
  tpm_value   FLOAT   NOT NULL  ← valor de celda
  tissue_id   STRING  NOT NULL  ← join con gtex_metadata.txt
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
# Config
# ---------------------------------------------------------------------------

BRONZE_PATH     = "data/bronze/gtex/gene_tpm_raw.parquet"
SILVER_PATH     = "data/silver/gtex/gene_expression_long"
QUARANTINE_PATH = "data/quarantine/silver_unmatched_samples.parquet"
METADATA_PATH   = "data/raw/gtex_metadata.txt"

METADATA_COLS = ["Name", "Description"]  # columnas no-muestra en Bronze

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
# 1. Spark session
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

    # Configs estáticas del perfil — deben ir antes de getOrCreate()
    if profile:
        for key, value in profile.static_configs().items():
            builder = builder.config(key, value)

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log.info("SparkSession iniciada.")
    return spark


# ---------------------------------------------------------------------------
# 2. Bronze — lectura lazy por chunk de columnas
# ---------------------------------------------------------------------------

def get_bronze_sample_cols() -> List[str]:
    """
    Lee solo el schema del parquet — sin datos, costo mínimo de RAM.
    Retorna la lista de columnas de muestras (todo excepto METADATA_COLS).
    """
    schema = pq.read_schema(BRONZE_PATH)
    all_cols = schema.names
    sample_cols = [c for c in all_cols if c not in METADATA_COLS]
    log.info(
        f"Schema Bronze leído: {len(all_cols)} columnas totales, "
        f"{len(sample_cols)} columnas de muestras"
    )
    return sample_cols


def load_bronze_chunk(cols: List[str]) -> pd.DataFrame:
    """
    Lee del parquet Bronze SOLO METADATA_COLS + cols del chunk actual.
    Nunca carga más de chunk_size cols × 74,628 genes en RAM simultáneamente.
    """
    df = pd.read_parquet(BRONZE_PATH, columns=METADATA_COLS + cols)

    # Si 'Name' llegó como índice lo reseteamos
    if df.index.name == "Name":
        df = df.reset_index()

    return df


# ---------------------------------------------------------------------------
# 3. Carga de metadata
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    mapping = load_tissue_mapping(path=METADATA_PATH)
    valid, msg = validate_tissue_mapping(mapping)
    if not valid:
        raise ValueError(f"Tissue mapping inválido: {msg}")
    log.info(
        f"Tissue mapping cargado: {len(mapping):,} muestras "
        f"→ {len(set(mapping.values()))} tejidos"
    )
    return mapping


# ---------------------------------------------------------------------------
# 4. Reshape de un chunk (wide → long, Pandas)
# ---------------------------------------------------------------------------

def reshape_chunk(
    df_chunk: pd.DataFrame,
    sample_cols: List[str],
) -> pd.DataFrame:
    """
    Recibe el chunk ya leído (METADATA_COLS + sample_cols) y lo convierte
    a formato long:
        gene_id | gene_symbol | sample_id | tpm_value

    Sin tissue_id aún — se añade en join_tissue().
    Zeros son válidos biológicamente — NO se filtran.
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
# 5. Join con tissue mapping + separación de quarantine
# ---------------------------------------------------------------------------

def join_tissue(
    df_long: pd.DataFrame,
    tissue_mapping: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Añade tissue_id via lookup directo en el dict.
    Retorna (matched_df, quarantine_df).
    """
    df_long["tissue_id"] = df_long["sample_id"].map(tissue_mapping)

    matched   = df_long[df_long["tissue_id"].notna()].copy()
    unmatched = df_long[df_long["tissue_id"].isna()].copy()

    if not unmatched.empty:
        unmatched["reason"] = "no_tissue_match"

    return matched, unmatched


# ---------------------------------------------------------------------------
# 6. Escritura Silver (Delta, append por chunk)
# ---------------------------------------------------------------------------

def write_silver(
    spark: SparkSession,
    df_chunk: pd.DataFrame,
    first_write: bool,
    n_partitions: int = 10,
    max_records: int = 500_000,
) -> None:
    """
    Pandas batch → Spark DataFrame → Delta.
    Primer write: overwrite → idempotencia completa del pipeline.
    Resto:        append    → acumulación sin recargar lo ya escrito.

    repartition(n_partitions) divide el batch en tareas manejables.
    maxRecordsPerFile controla el tamaño de archivos Delta en disco —
    archivos grandes = menos small files = transaction log más liviano.
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
# 7. Escritura quarantine (append por chunk, sin acumulación en RAM)
# ---------------------------------------------------------------------------

def write_quarantine_chunk(df_unmatched: pd.DataFrame) -> int:
    """
    Escribe inmediatamente al parquet de cuarentena — sin acumular en RAM.
    Lee el existente si ya hay algo, concatena y sobreescribe.
    Retorna el número de filas escritas en este chunk.
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
        f"  Cuarentena: +{len(df_unmatched):,} filas unmatched "
        f"(total acumulado: {len(combined):,}) → {QUARANTINE_PATH}"
    )
    return len(df_unmatched)


# ---------------------------------------------------------------------------
# 8. Quality gate Silver
# ---------------------------------------------------------------------------

def run_quality_gate(
    silver_row_count: int,
    quarantine_row_count: int,
    spark: SparkSession,
):
    """
    Lee Silver desde Delta, calcula métricas reales y ejecuta run_silver_checks.
    Lanza PipelineQualityError si algún check crítico falla.
    """
    log.info("Ejecutando quality gate Silver…")

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

    log.info("Quality gate Silver: PASSED ✓")
    return report


# ---------------------------------------------------------------------------
# 9. Lineage
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

    log.info(f"Lineage guardado → {SILVER_LINEAGE_PATH}")


# ---------------------------------------------------------------------------
# 10. main — orquestación
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=None,
        help="Forzar perfil: SURVIVAL | BALANCED | PERFORMANCE | PRO"
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info("=" * 60)
    log.info("Iniciando silver_transform.py")
    log.info("=" * 60)

    # --- Detectar perfil ANTES de arrancar Spark (static configs van al builder) ---
    mem_settings  = get_spark_memory_settings(mode="local")
    available_ram = mem_settings["meta"]["available_ram_gb"]

    if args.profile:
        profile = get_profile_by_name(args.profile)
        log.info(f"Perfil forzado por CLI: {profile.name}")
    else:
        profile = detect_profile(available_ram)

    log.info(f"\n{profile.summary()}")

    # --- Spark (con static configs del perfil en el builder) ---
    spark = build_spark(profile=profile)

    # --- Columnas de muestras (solo schema, cero RAM de datos) ---
    tissue_mapping = load_metadata()
    bronze_lineage = load_bronze_lineage()
    sample_cols    = get_bronze_sample_cols()
    n_samples      = len(sample_cols)
    log.info(f"Muestras a procesar: {n_samples:,}")

    # Aplicar configs dinámicas del perfil (modificables post-inicio)
    for key, value in profile.dynamic_configs().items():
        spark.conf.set(key, value)
    log.info("Spark config dinámica del perfil aplicada.")

    # chunk_size viene del perfil
    chunk_size = profile.cols_per_chunk

    # --- Loop de chunks con escritura por lotes ---
    # En lugar de escribir cada chunk individualmente (396 writes → Delta se degrada),
    # acumulamos chunks_per_write chunks y hacemos un solo write por lote.
    # Resultado: ~80 writes en lugar de 396 → transaction log crece 5x más lento.
    silver_row_count     = 0
    quarantine_row_count = 0
    first_write          = True
    batch_buffer         = []   # acumula DataFrames matched hasta chunks_per_write
    batch_row_count      = 0

    chunks   = [sample_cols[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]
    n_chunks = len(chunks)
    log.info(f"Total chunks: {n_chunks} | chunks/write: {profile.chunks_per_write}")
    log.info(f"Total writes esperados: ~{math.ceil(n_chunks / profile.chunks_per_write)}")

    for idx, cols in enumerate(chunks, start=1):
        log.info(f"Chunk {idx}/{n_chunks} — {len(cols)} muestras")

        # Leer solo las columnas de este chunk del parquet
        df_chunk = load_bronze_chunk(cols)

        # wide → long
        df_long = reshape_chunk(df_chunk, cols)
        del df_chunk

        # join tissue
        matched, unmatched = join_tissue(df_long, tissue_mapping)
        del df_long

        # cuarentena: escribir inmediatamente, sin acumular en RAM
        quarantine_row_count += write_quarantine_chunk(unmatched)
        del unmatched

        # matched vacío es un bug
        if matched.empty:
            log.error(
                f"Chunk {idx}/{n_chunks}: CERO filas matched. "
                f"Primeras cols del chunk: {cols[:3]}… "
                "Estos sample IDs no están en el tissue mapping — revisar gtex_metadata.txt."
            )
            raise RuntimeError(
                f"Chunk {idx} produjo 0 filas matched. "
                "Pipeline detenido para evitar Silver incompleto."
            )

        # Acumular en el buffer del lote
        batch_buffer.append(matched)
        batch_row_count += len(matched)
        del matched

        # Escribir cuando el buffer llega a chunks_per_write O es el último chunk
        is_last_chunk    = (idx == n_chunks)
        buffer_is_full   = (len(batch_buffer) >= profile.chunks_per_write)

        if buffer_is_full or is_last_chunk:
            log.info(
                f"  Escribiendo lote: {len(batch_buffer)} chunks, "
                f"{batch_row_count:,} filas → Delta"
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
            first_write       = False
            silver_row_count += batch_row_count
            batch_row_count   = 0
            del df_batch

            log.info(f"  Silver acumulado: {silver_row_count:,} filas")

    # --- Validación de conteo total ---
    expected = n_samples * 74_628  # genes confirmados en profiling
    actual   = silver_row_count + quarantine_row_count
    log.info(f"Row count check → esperado: {expected:,} | real: {actual:,}")
    if actual != expected:
        log.warning(f"Discrepancia de {expected - actual:,} filas.")

    # --- Quality gate ---
    quality_report = run_quality_gate(silver_row_count, quarantine_row_count, spark)

    # --- Lineage ---
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
    log.info(f"silver_transform.py completado en {duration:.1f}s")
    log.info(f"  Silver rows     : {silver_row_count:,}")
    log.info(f"  Quarantine rows : {quarantine_row_count:,}")
    log.info(f"  Tejidos únicos  : {tissue_count}")
    log.info("=" * 60)

    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except PipelineQualityError as e:
        log.critical(f"PIPELINE DETENIDO — Quality gate falló:\n{e}")
        raise SystemExit(1)
    except Exception:
        log.critical("Error no manejado en silver_transform.py")
        traceback.print_exc()
        raise SystemExit(2)