"""
src/jobs/silver_phase2_delta.py

Phase 2 del reshape Silver — delta-rs (PyArrow puro) escribe Delta Lake Silver.

Phase 1 dejó el staging particionado por tissue_id en:
    data/staging/silver/tissue_id=<tejido>/batch_XXXX.parquet

Phase 2 hace UNA sola cosa: leer ese staging y escribir Delta Lake Silver
con el schema correcto, particionado por tissue_id, compresión Snappy.

Sin Spark, sin JVM, sin shuffle — delta-rs puro.

Estrategia de memoria:
    - Nunca se carga un tejido completo en RAM
    - Se itera archivo por archivo dentro de cada tejido con iter_batches()
    - batch_size calculado dinámicamente según RAM disponible (psutil)
    - RAM máxima real: un solo batch en memoria a la vez

Responsabilidades:
    1. Detectar RAM disponible via psutil (elástico por entorno)
    2. Calcular batch_size seguro sin valores hardcodeados
    3. Leer staging tejido por tejido, archivo por archivo, batch por batch
    4. Escribir Delta Lake Silver via delta-rs (cero JVM, cero shuffle)
    5. Correr quality checks (PyArrow puro, sin Spark)
    6. Actualizar data lineage via lineage.py

Output:
    data/silver/gtex/gene_expression_long/   <- Delta Lake Silver
    data/lineage/silver_metadata.json        <- lineage Silver
    data/lineage/pipeline_lineage.json       <- lineage acumulado

Idempotencia:
    Primer batch escrito = overwrite (limpia Silver parcial previo).
    Resto = append.
    Si falla a mitad, borrar data/silver/ y reintentar — Phase 1 no se repite.

Uso:
    docker exec -it --workdir /opt/spark/work-dir spark-master \\
        env PYTHONPATH=. python3 src/jobs/silver_phase2_delta.py

    # Saltar quality gate (solo debug):
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
# Config
# ---------------------------------------------------------------------------

STAGING_PATH    = "data/staging/silver"
SILVER_PATH     = "data/silver/gtex/gene_expression_long"
QUARANTINE_PATH = "data/staging/silver/quarantine/unmatched.parquet"
BRONZE_LINEAGE  = "data/lineage/bronze_metadata.json"
OPTIMAL_CONFIG  = "data/staging/silver_tuner/optimal_config.json"

# Schema Arrow canónico del Silver — 5 columnas
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
# 1. Calcular batch_size seguro según RAM disponible real
# ---------------------------------------------------------------------------

def calculate_batch_size(available_ram_gb: float) -> int:
    """
    Calcula filas por batch que caben en RAM disponible.

    Usa solo el 25% de la RAM disponible — margen conservador para:
    - PyArrow overhead de deserialización (~2x el tamaño del batch)
    - delta-rs write buffer
    - OS y otros procesos corriendo

    Baseline: 5 columnas mixed types ~ 120 bytes por fila descomprimida.

    Ejemplos:
        2.8GB disponibles → 25% = 700MB → ~5.8M filas/batch
        4.0GB disponibles → 25% = 1.0GB → ~8.3M filas/batch
        8.0GB disponibles → 25% = 2.0GB → ~16.6M filas/batch

    Clampeo:
        min = 100_000  (mínimo para que delta-rs no genere archivos basura)
        max = 5_000_000 (máximo conservador para RAM limitada)
    """
    bytes_per_row   = 120
    usable_bytes    = available_ram_gb * 0.25 * (1024 ** 3)
    calculated_rows = int(usable_bytes / bytes_per_row)
    batch_size      = max(100_000, min(calculated_rows, 5_000_000))

    log.info(
        f"RAM disponible: {available_ram_gb:.2f}GB → "
        f"batch_size: {batch_size:,} filas "
        f"(~{batch_size * bytes_per_row / (1024**2):.0f}MB por batch)"
    )
    return batch_size


# ---------------------------------------------------------------------------
# 2. Verificar si Silver ya está completo — cero RAM, solo footers
# ---------------------------------------------------------------------------

EXPECTED_SILVER_ROWS = 74_628 * 19_788  # 1,476,738,864


def silver_is_complete(silver_path: str) -> bool:
    """
    Verifica si el Silver ya existe y tiene el row count correcto.
    Lee solo footers de los Parquet — cero RAM, instantáneo.

    True  → Silver completo, saltar write
    False → Silver ausente o incompleto, reescribir
    """
    silver_dir = Path(silver_path)
    if not silver_dir.exists():
        log.info("Silver no existe — write necesario")
        return False

    parquet_files = [
        f for f in silver_dir.rglob("*.parquet")
        if "_delta_log" not in str(f)
    ]
    if not parquet_files:
        log.info("Silver vacío — write necesario")
        return False

    total = sum(pq.read_metadata(str(f)).num_rows for f in parquet_files)

    if total == EXPECTED_SILVER_ROWS:
        log.info(f"Silver ya completo ({total:,} filas) — saltando write")
        return True

    log.warning(
        f"Silver incompleto o duplicado ({total:,} / {EXPECTED_SILVER_ROWS:,}) "
        f"— borrando y reescribiendo"
    )
    return False


# ---------------------------------------------------------------------------
# 2b. Escribir Delta Silver — streaming puro, nunca carga tejido completo
# ---------------------------------------------------------------------------

def write_delta_silver(staging_path: str, silver_path: str, batch_size: int) -> int:
    """
    Streaming puro — RAM máxima = un batch en memoria a la vez.

    Estrategia:
    - Itera directorios tissue_id=<nombre> uno a la vez
    - Dentro de cada tejido, itera cada archivo .parquet por separado
    - Cada archivo se lee con ParquetFile.iter_batches(batch_size) —
      nunca materializa el archivo completo en RAM
    - Agrega columna tissue_id al RecordBatch antes de escribir
    - Primer batch global = overwrite, resto = append
    - gc.collect() entre tejidos

    RAM máxima real:
        batch_size × 120 bytes × ~2 (PyArrow overhead)
        Con 2.8GB y batch_size=5M → ~1.2GB máximo

    Shuffle files: cero — sin JVM
    """
    from deltalake import write_deltalake

    # Idempotencia: si Silver ya está completo y correcto, no reescribir
    if silver_is_complete(silver_path):
        return EXPECTED_SILVER_ROWS

    # Silver incompleto o duplicado — borrar físicamente y reescribir limpio
    # delta-rs overwrite solo marca archivos como eliminados en el log,
    # los físicos quedan en disco y rglob los cuenta doble en las métricas
    silver_dir = Path(silver_path)
    if silver_dir.exists():
        log.info(f"Borrando Silver para write limpio: {silver_path}")
        shutil.rmtree(silver_dir)
    silver_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(staging_path)
    tissue_dirs = sorted([
        d for d in staging_dir.iterdir()
        if d.is_dir() and d.name.startswith("tissue_id=")
    ])

    if not tissue_dirs:
        raise FileNotFoundError(
            f"No se encontraron directorios tissue_id= en {staging_path}.\n"
            "Corre silver_phase1_reshape.py primero."
        )

    log.info(f"Tejidos a escribir : {len(tissue_dirs)}")
    log.info(f"Delta Silver path  : {silver_path}")
    log.info(f"Batch size         : {batch_size:,} filas")
    log.info("Motor              : delta-rs streaming (cero JVM, cero shuffle)")

    total_rows  = 0
    first_write = True

    for i, tissue_dir in enumerate(tissue_dirs, 1):
        # URL-decode: "Whole%20Blood" -> "Whole Blood"
        tissue_name   = urllib.parse.unquote(tissue_dir.name.split("=", 1)[1])
        parquet_files = sorted(tissue_dir.glob("*.parquet"))

        if not parquet_files:
            log.warning(f"  [{i}/{len(tissue_dirs)}] {tissue_name} — sin archivos, saltando")
            continue

        log.info(f"[{i}/{len(tissue_dirs)}] tissue_id={tissue_name} ({len(parquet_files)} archivos)")
        tissue_rows = 0

        for pq_file in parquet_files:
            pf = pq.ParquetFile(str(pq_file))

            # iter_batches — nunca carga el archivo completo en RAM
            for batch in pf.iter_batches(batch_size=batch_size):
                n = batch.num_rows

                # Agregar tissue_id como columna dentro del Parquet
                # SIN partition_by — tissue_id queda en el schema, sin URL encoding
                batch_with_tissue = batch.append_column(
                    pa.field("tissue_id", pa.string(), nullable=False),
                    pa.array([tissue_name] * n, type=pa.string()),
                )

                # Reordenar al schema canónico
                table = pa.Table.from_batches(
                    [batch_with_tissue],
                    schema=SILVER_SCHEMA,
                )

                mode = "overwrite" if first_write else "append"
                write_deltalake(
                    silver_path,
                    table,
                    schema=SILVER_SCHEMA,
                    # Sin partition_by — tissue_id queda como columna normal
                    # Delta log registra los archivos sin Hive encoding
                    mode=mode,
                    file_options={"compression": "snappy"},
                )
                first_write  = False
                tissue_rows += n

                del batch, batch_with_tissue, table
                gc.collect()

        total_rows += tissue_rows
        log.info(f"  ✅ {tissue_name} — {tissue_rows:,} filas escritas")
        gc.collect()

    log.info(f"Todos los tejidos escritos — total: {total_rows:,} filas")
    return total_rows


# ---------------------------------------------------------------------------
# 3. Métricas para quality gate — PyArrow puro, sin DeltaTable, sin Spark
# ---------------------------------------------------------------------------

def collect_silver_metrics(silver_path: str, quarantine_path: str, batch_size: int) -> dict:
    """
    Recolecta métricas del Delta Silver para run_silver_checks().

    Lee los Parquet del Silver directamente con PyArrow — sin DeltaTable,
    sin Spark, sin JVM. Itera archivo por archivo en batch_size filas.

    El Silver no está particionado por Hive (tissue_id es columna normal),
    así que simplemente escaneamos todos los .parquet recursivamente.
    """
    log.info("Recolectando métricas Silver (PyArrow puro, sin DeltaTable)...")

    silver_dir = Path(silver_path)
    # Buscar todos los parquet excepto los del _delta_log
    parquet_files = [
        f for f in silver_dir.rglob("*.parquet")
        if "_delta_log" not in str(f)
    ]

    if not parquet_files:
        raise FileNotFoundError(f"No se encontraron archivos Parquet en {silver_path}")

    log.info(f"  Archivos Parquet encontrados: {len(parquet_files)}")

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

    # Quarantine — solo footer, cero RAM
    quarantine_row_count = 0
    if Path(quarantine_path).exists():
        quarantine_row_count = pq.read_metadata(quarantine_path).num_rows

    log.info(f"  silver_row_count   : {silver_row_count:,}")
    log.info(f"  quarantine_rows    : {quarantine_row_count:,}")
    log.info(f"  tissue_count       : {tissue_count}")
    log.info(f"  zero_fraction      : {zero_fraction:.2%}")
    log.info(f"  gene_id_nulls      : {gene_id_nulls:,}")
    log.info(f"  sample_id_nulls    : {sample_id_nulls:,}")
    log.info(f"  tissue_id_nulls    : {tissue_id_nulls:,}")

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
# 4. Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Saltar quality gate (solo para debug)",
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info("=" * 60)
    log.info("Phase 2 — Staging Parquet -> Delta Lake Silver (delta-rs)")
    log.info("=" * 60)

    # Verificar staging
    if not Path(STAGING_PATH).exists():
        raise FileNotFoundError(
            f"Staging no encontrado en {STAGING_PATH}.\n"
            "Corre silver_phase1_reshape.py primero."
        )

    # Info del tuner para el lineage
    tuner_info = {}
    if Path(OPTIMAL_CONFIG).exists():
        with open(OPTIMAL_CONFIG) as f:
            tuner_info = json.load(f)
        log.info(
            f"Tuner config: cols_per_chunk={tuner_info.get('cols_per_chunk')} "
            f"| estimated={tuner_info.get('estimated_minutes')} min"
        )

    # Detectar RAM disponible
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    log.info(f"RAM disponible detectada: {available_ram_gb:.2f}GB")

    # Calcular batch_size seguro para este entorno
    batch_size = calculate_batch_size(available_ram_gb)

    # Escribir Delta Silver — cero Spark, cero JVM, cero shuffle
    total_rows = write_delta_silver(STAGING_PATH, SILVER_PATH, batch_size)

    # Quality gate
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

        log.info("Quality gate Silver APROBADO")
    else:
        log.warning("Quality gate SALTADO (--skip-quality activo)")
        metrics        = {
            "silver_row_count": total_rows, "quarantine_row_count": 0,
            "tissue_count": 0,              "actual_zero_fraction": 0.0,
        }
        quality_report = None

    # Lineage
    duration       = time.time() - t_start
    bronze_lineage = load_bronze_lineage(BRONZE_LINEAGE)

    silver_lineage = build_silver_lineage(
        bronze_lineage       = bronze_lineage,
        silver_row_count     = metrics["silver_row_count"],
        quarantine_row_count = metrics["quarantine_row_count"],
        tissue_count         = metrics.get("tissue_count", 0),
        chunk_plan_summary   = (
            f"delta-rs streaming iter_batches | batch_size={batch_size:,} (dinámico) "
            f"| cols_per_chunk={tuner_info.get('cols_per_chunk', '?')}"
        ),
        quality_report_dict  = quality_report.to_dict() if quality_report else {},
        infra_fingerprint    = f"delta-rs-no-spark-ram{available_ram_gb:.1f}gb",
        memory_used          = f"{available_ram_gb:.2f}gb-available",
        duration_seconds     = round(duration, 1),
    )

    silver_lineage["transformations"] = [
        "wide->long reshape (PyArrow streaming, col-by-col, sin acumulacion en RAM)",
        f"cols_per_chunk={tuner_info.get('cols_per_chunk', '?')} (silver_tuner.py config)",
        "join sample_id -> tissue_id inferido del path Hive (gtex_metadata.txt)",
        "cast: gene_id=string, gene_symbol=string, sample_id=string, tpm_value=float32",
        "zeros preservados — biológicamente válidos (51.89% baseline profiling)",
        "unmatched samples -> cuarentena con reason=no_tissue_match",
        f"Phase 2: delta-rs iter_batches streaming | batch_size={batch_size:,} (dinámico por RAM)",
    ]

    pipeline_lineage = build_pipeline_lineage(bronze_lineage, silver_lineage)
    save_lineage(silver_lineage,   SILVER_LINEAGE_PATH)
    save_lineage(pipeline_lineage, PIPELINE_LINEAGE_PATH)

    # Resumen final
    log.info("=" * 60)
    log.info("Phase 2 completada")
    log.info(f"  Tiempo              : {duration:.1f}s ({duration/60:.1f} min)")
    log.info(f"  Silver rows         : {metrics['silver_row_count']:,}")
    log.info(f"  Quarantine rows     : {metrics['quarantine_row_count']:,}")
    log.info(f"  Tejidos únicos      : {metrics.get('tissue_count', '?')}")
    log.info(f"  Zero fraction       : {metrics['actual_zero_fraction']:.2%}")
    log.info(f"  Batch size usado    : {batch_size:,} filas")
    log.info(f"  RAM disponible      : {available_ram_gb:.2f}GB")
    log.info(f"  Delta Silver path   : {SILVER_PATH}")
    log.info(f"  Lineage             : {SILVER_LINEAGE_PATH}")
    log.info("=" * 60)
    log.info("Siguiente paso: src/jobs/gold_transform.py")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except PipelineQualityError as e:
        log.critical("Quality gate fallo — pipeline detenido")
        log.critical(str(e))
        raise SystemExit(2)
    except Exception:
        log.critical("Error no manejado en silver_phase2_delta.py")
        traceback.print_exc()
        raise SystemExit(2)