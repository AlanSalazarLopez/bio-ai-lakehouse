"""
src/jobs/gold_transform.py

Orquestador Gold — capa final del pipeline Bio-AI Lakehouse.

Qué hace:
    1. Carga Silver lineage para obtener tissue_count esperado
    2. Itera Silver en batches via SilverBatchReader (sin cargar todo en RAM)
    3. Acumula estadísticos por grupo (gene_id, gene_symbol, tissue_id)
       con Welford online (mean/std) y reservoir sampling (median)
       via numpy memmap — arrays en disco, RAM constante ~200MB
    4. Serializa acumuladores a pa.Table
    5. Escribe Delta Lake Gold con ZSTD via write_deltalake
    6. Corre quality gate Gold (run_gold_checks)
    7. Actualiza data lineage (gold_metadata.json + pipeline_lineage.json)

RAM máxima estimada:
    ~200MB memmap activo + ~100MB batch Silver = ~300MB constante
    Sin OOM sin importar cuántos grupos haya.

Uso:
    docker exec -it --workdir /opt/spark/work-dir spark-master \
    env PYTHONPATH=. python3 src/jobs/gold_transform.py

    # Saltar quality gate (solo debug):
    env PYTHONPATH=. python3 src/jobs/gold_transform.py --skip-quality

    # Resumir desde cache existente (si el proceso se interrumpió):
    env PYTHONPATH=. python3 src/jobs/gold_transform.py --resume

Idempotencia:
    write_deltalake con mode='overwrite' — si Gold existe parcial, se limpia.
    Re-ejecutar sin --resume es seguro.
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
#  Configuración
# ─────────────────────────────────────────────

SILVER_ROOT          = "data/silver/gtex/gene_expression_long"
GOLD_PATH            = "data/gold/gtex/gene_tissue_summary"
GOLD_CACHE_DIR       = "data/gold_cache"
GOLD_LINEAGE_PATH    = "data/lineage/gold_metadata.json"
SILVER_LINEAGE_PATH  = "data/lineage/silver_metadata.json"

EXPECTED_GENES   = 74_628
EXPECTED_TISSUES = 68

# Flush del cache a disco cada N filas — protege contra crashes
FLUSH_EVERY_ROWS = 50_000_000

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Paso 1 — Cargar Silver lineage
# ─────────────────────────────────────────────

def load_silver_lineage() -> Dict[str, Any]:
    lineage = load_lineage(SILVER_LINEAGE_PATH)
    if not lineage:
        logger.warning(
            "Silver lineage no encontrado en %s — usando defaults",
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
        "Silver lineage cargado — tissue_count=%s, row_count=%s",
        lineage.get("output", {}).get("tissue_count", "?"),
        f"{lineage.get('output', {}).get('row_count', 0):,}",
    )
    return lineage


# ─────────────────────────────────────────────
#  Paso 2-3 — Iterar Silver y acumular
# ─────────────────────────────────────────────

def accumulate_silver(acc_map: GoldAccumulatorMap, resume: bool = False) -> SilverBatchReader:
    """
    Lee Silver completo en batches y actualiza los acumuladores memmap.
    Flushea a disco cada FLUSH_EVERY_ROWS filas — protege contra crashes.
    """
    reader     = SilverBatchReader(SILVER_ROOT)
    total_rows = 0
    rows_since_flush = 0
    start_ts   = time.time()

    logger.info("Iniciando acumulación Silver → Gold (%s archivos)...",
                f"{reader.total_files:,}")

    for batch in reader.iter_batches():
        acc_map.update_from_batch(batch)
        total_rows       += batch.num_rows
        rows_since_flush += batch.num_rows

        # Flush periódico a disco
        if rows_since_flush >= FLUSH_EVERY_ROWS:
            acc_map.flush()
            rows_since_flush = 0
            logger.info("Cache flusheado a disco — %s grupos únicos",
                        f"{acc_map.group_count:,}")

        # Log de progreso cada 50M filas
        if total_rows % 50_000_000 < batch.num_rows:
            elapsed  = time.time() - start_ts
            pct      = total_rows / 1_476_738_864 * 100
            ram_gb   = psutil.virtual_memory().available / 1024 ** 3
            logger.info(
                "Progreso: %s filas (%.1f%%) | grupos=%s | RAM libre=%.1fGB | %.0fs",
                f"{total_rows:,}", pct,
                f"{acc_map.group_count:,}",
                ram_gb, elapsed,
            )

    # Flush final
    acc_map.flush()

    elapsed_total = time.time() - start_ts
    logger.info(
        "Acumulación completa: %s filas en %.0fs | %s grupos únicos",
        f"{reader.rows_yielded:,}", elapsed_total,
        f"{acc_map.group_count:,}",
    )
    return reader


# ─────────────────────────────────────────────
#  Paso 4-5 — Serializar y escribir Delta Gold
# ─────────────────────────────────────────────

def write_gold(gold_table: pa.Table) -> None:
    logger.info(
        "Escribiendo Gold Delta Lake: %s filas → %s",
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

    logger.info("Gold escrito exitosamente en %s", GOLD_PATH)


# ─────────────────────────────────────────────
#  Paso 6 — Quality gate Gold
# ─────────────────────────────────────────────

def run_quality_gate(gold_table: pa.Table, skip: bool = False) -> Dict[str, Any]:
    if skip:
        logger.warning("Quality gate Gold OMITIDO (--skip-quality activo)")
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
#  Paso 7 — Lineage
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

    logger.info("Lineage actualizado:")
    logger.info("  %s", GOLD_LINEAGE_PATH)
    logger.info("  %s", PIPELINE_LINEAGE_PATH)
    logger.info("  Pipeline status: %s", pipeline_lineage.get("status"))


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main(skip_quality: bool = False, resume: bool = False) -> None:
    start_total = time.time()

    logger.info("══════════════════════════════════════════")
    logger.info("  Bio-AI Lakehouse — Gold Transform v3")
    logger.info("  Modo: %s", "RESUME" if resume else "FRESH")
    logger.info("══════════════════════════════════════════")

    # Paso 1 — Silver lineage
    silver_lineage = load_silver_lineage()

    # Paso 2-3 — Acumular Silver via memmap
    acc_map = GoldAccumulatorMap(
        cache_dir = GOLD_CACHE_DIR,
        resume    = resume,
    )
    reader = accumulate_silver(acc_map, resume=resume)

    # Paso 4 — Serializar a pa.Table
    logger.info("Serializando acumuladores a pa.Table...")
    gold_table = acc_map.to_arrow_table()
    logger.info("pa.Table lista: %s filas × %s columnas",
                f"{gold_table.num_rows:,}", gold_table.num_columns)

    # Paso 5 — Escribir Delta Gold
    write_gold(gold_table)

    # Paso 6 — Quality gate
    quality_dict = run_quality_gate(gold_table, skip=skip_quality)

    # Paso 7 — Lineage
    duration = time.time() - start_total
    update_lineage(gold_table, quality_dict, duration)

    logger.info("══════════════════════════════════════════")
    logger.info("  Gold Transform COMPLETO en %.0fs (%.1f min)",
                duration, duration / 60)
    logger.info("  Output: %s", GOLD_PATH)
    logger.info("  Filas : %s", f"{gold_table.num_rows:,}")
    logger.info("══════════════════════════════════════════")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Transform — Bio-AI Lakehouse")
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Omitir quality gate Gold (solo para debug)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resumir desde cache memmap existente (si el proceso se interrumpió)",
    )
    args = parser.parse_args()

    try:
        main(skip_quality=args.skip_quality, resume=args.resume)
    except PipelineQualityError as e:
        logger.error("Pipeline detenido por quality gate: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Error inesperado: %s", e, exc_info=True)
        sys.exit(1)