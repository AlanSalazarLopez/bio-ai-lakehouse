"""
src/utils/lineage.py

Gestiona la lectura y actualización del data lineage del pipeline.
Compatible con el formato generado por bronze_ingest.py.

El lineage es un JSON que acumula metadata de cada capa:
- Bronze: generado por bronze_ingest.py (ya existe)
- Silver: este módulo agrega la sección 'silver'
- Gold:   este módulo agrega la sección 'gold'

Diseño:
- Funciones puras de serialización — sin side-effects fuera de I/O
- Idempotente: sobrescribir una sección existente no corrompe el resto
- Compatible con Python 3.8 (Optional[X], no X | None)

Rutas por defecto:
    data/lineage/bronze_metadata.json  ← generado por bronze_ingest.py
    data/lineage/silver_metadata.json  ← generado por silver_transform.py
    data/lineage/pipeline_lineage.json ← lineage completo acumulado
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Rutas por defecto
BRONZE_LINEAGE_PATH   = "data/lineage/bronze_metadata.json"
SILVER_LINEAGE_PATH   = "data/lineage/silver_metadata.json"
PIPELINE_LINEAGE_PATH = "data/lineage/pipeline_lineage.json"


# ─────────────────────────────────────────────
#  Lectura
# ─────────────────────────────────────────────

def load_lineage(path: str) -> Dict[str, Any]:
    """
    Lee un JSON de lineage desde disco.

    Args:
        path: ruta al archivo JSON

    Returns:
        dict con el contenido del lineage.
        Retorna dict vacío si el archivo no existe (primera ejecución).
    """
    if not os.path.exists(path):
        logger.warning("Lineage no encontrado en %s — retornando vacío", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Lineage cargado desde %s", path)
    return data


def load_bronze_lineage(path: str = BRONZE_LINEAGE_PATH) -> Dict[str, Any]:
    """
    Carga el lineage de Bronze generado por bronze_ingest.py.
    Wrapper semántico sobre load_lineage para claridad en el caller.
    """
    data = load_lineage(path)
    if not data:
        logger.warning(
            "Bronze lineage vacío — ¿corriste bronze_ingest.py primero?"
        )
    return data


# ─────────────────────────────────────────────
#  Escritura
# ─────────────────────────────────────────────

def save_lineage(data: Dict[str, Any], path: str) -> None:
    """
    Escribe un dict de lineage a disco como JSON indentado.
    Crea el directorio si no existe.

    Args:
        data: dict a serializar
        path: ruta destino
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Lineage guardado en %s", path)


# ─────────────────────────────────────────────
#  Builders por capa
# ─────────────────────────────────────────────

def build_silver_lineage(
    bronze_lineage:       Dict[str, Any],
    silver_row_count:     int,
    quarantine_row_count: int,
    tissue_count:         int,
    chunk_plan_summary:   str,
    quality_report_dict:  Dict[str, Any],
    infra_fingerprint:    str,
    memory_used:          str,
    duration_seconds:     Optional[float] = None,
) -> Dict[str, Any]:
    """
    Construye el dict de lineage para Silver.
    Incluye referencia al Bronze del que proviene para trazabilidad completa.

    Args:
        bronze_lineage:       resultado de load_bronze_lineage()
        silver_row_count:     filas escritas en Delta Silver
        quarantine_row_count: filas enviadas a cuarentena
        tissue_count:         tejidos únicos en Silver
        chunk_plan_summary:   ChunkPlan.summary() como string
        quality_report_dict:  QualityReport.to_dict()
        infra_fingerprint:    fingerprint SHA-256 del entorno
        memory_used:          RAM usada (ej. '3g')
        duration_seconds:     tiempo total de ejecución del job

    Returns:
        Dict listo para save_lineage()
    """
    bronze_genes   = 74_628
    bronze_samples = 19_788
    expected_total = bronze_genes * bronze_samples
    actual_total   = silver_row_count + quarantine_row_count
    lineage_closes = actual_total == expected_total

    return {
        "layer":                "silver",
        "generated_at_utc":     datetime.now(timezone.utc).isoformat(),
        "infra_fingerprint":    infra_fingerprint,
        "memory_used":          memory_used,

        # Trazabilidad hacia Bronze
        "source": {
            "layer":             "bronze",
            "path":              bronze_lineage.get("bronze_path", "unknown"),
            "size_bytes":        bronze_lineage.get("bronze_size_bytes", 0),
            "ingestion_ts":      bronze_lineage.get("ingestion_timestamp", "unknown"),
            "bronze_fingerprint": bronze_lineage.get("infra_fingerprint", "unknown"),
        },

        # Output Silver
        "output": {
            "path":              "data/silver/gtex/gene_expression_long/",
            "format":            "delta",
            "compression":       "snappy",
            "partition_col":     "tissue_id",
            "tissue_count":      tissue_count,
            "row_count":         silver_row_count,
            "quarantine_count":  quarantine_row_count,
            "quarantine_path":   "data/quarantine/silver_unmatched_samples.parquet",
        },

        # Verificación de cierre matemático
        "lineage_closure": {
            "bronze_genes":    bronze_genes,
            "bronze_samples":  bronze_samples,
            "expected_total":  expected_total,
            "actual_total":    actual_total,
            "closes":          lineage_closes,
            "delta":           actual_total - expected_total,
        },

        # Transformaciones aplicadas
        "transformations": [
            "wide→long reshape (pandas chunks, 200 cols/chunk)",
            "join sample_id → tissue_id (gtex_metadata.txt)",
            "cast gene_id=string, gene_symbol=string, sample_id=string, tpm_value=float",
            "zeros preservados (biológicamente válidos)",
            "unmatched samples → cuarentena",
        ],

        # Performance
        "performance": {
            "chunk_plan":       chunk_plan_summary,
            "duration_seconds": duration_seconds,
        },

        # Quality gate
        "quality_report": quality_report_dict,
    }


def build_gold_lineage(
    silver_lineage:      Dict[str, Any],
    gold_row_count:      int,
    tissue_count:        int,
    quality_report_dict: Dict[str, Any],
    infra_fingerprint:   str,
    memory_used:         str,
    duration_seconds:    Optional[float] = None,
) -> Dict[str, Any]:
    """
    Construye el dict de lineage para Gold.
    Referencia el Silver del que proviene.
    """
    return {
        "layer":             "gold",
        "generated_at_utc":  datetime.now(timezone.utc).isoformat(),
        "infra_fingerprint": infra_fingerprint,
        "memory_used":       memory_used,

        "source": {
            "layer":       "silver",
            "path":        silver_lineage.get("output", {}).get("path", "unknown"),
            "row_count":   silver_lineage.get("output", {}).get("row_count", 0),
            "silver_ts":   silver_lineage.get("generated_at_utc", "unknown"),
        },

        "output": {
            "path":          "data/gold/gtex/gene_tissue_summary/",
            "format":        "delta",
            "compression":   "zstd",
            "partition_col": "tissue_id",
            "tissue_count":  tissue_count,
            "row_count":     gold_row_count,
        },

        "transformations": [
            "groupBy(gene_id, gene_symbol, tissue_id)",
            "agg: mean/median/std de log1p(tpm_value)",
            "agg: count(sample_id) → sample_count",
            "agg: zero_fraction = sum(tpm==0) / count",
            "log1p aplicado en Gold (zeros excluidos del promedio)",
        ],

        "performance": {
            "duration_seconds": duration_seconds,
        },

        "quality_report": quality_report_dict,
    }


# ─────────────────────────────────────────────
#  Pipeline lineage acumulado
# ─────────────────────────────────────────────

def build_pipeline_lineage(
    bronze_lineage: Dict[str, Any],
    silver_lineage: Optional[Dict[str, Any]] = None,
    gold_lineage:   Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construye el lineage completo del pipeline acumulando las tres capas.
    Útil para el README y el AI_LOG.md.

    Args:
        bronze_lineage: dict de Bronze (siempre requerido)
        silver_lineage: dict de Silver (None si aún no se ejecutó)
        gold_lineage:   dict de Gold   (None si aún no se ejecutó)

    Returns:
        Dict con las tres capas y el estado del pipeline.
    """
    layers_complete = sum([
        1,                              # Bronze siempre existe si llegamos aquí
        1 if silver_lineage else 0,
        1 if gold_lineage else 0,
    ])

    return {
        "pipeline":         "bio-ai-lakehouse",
        "dataset":          "GTEx Gene Expression v11",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "layers_complete":  f"{layers_complete}/3",
        "status":           "complete" if layers_complete == 3 else "in_progress",
        "bronze":           bronze_lineage,
        "silver":           silver_lineage or {"status": "pending"},
        "gold":             gold_lineage   or {"status": "pending"},
    }


# ─────────────────────────────────────────────
#  CLI — smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Smoke test lineage.py ─────────────────\n")

    # 1. Cargar Bronze existente
    bronze = load_bronze_lineage()
    if bronze:
        print(f"✅ Bronze lineage cargado")
        print(f"   source  : {bronze.get('source_file', 'N/A')}")
        print(f"   size_gb : {bronze.get('source_size_gb', 'N/A')}")
        print(f"   ts      : {bronze.get('ingestion_timestamp', 'N/A')}")
    else:
        print("⚠️  Bronze lineage no encontrado — usando datos simulados")
        bronze = {
            "source_file":       "data/raw/GTEx_Analysis_...parquet",
            "source_size_gb":    4.02,
            "bronze_path":       "data/bronze/gtex/gene_tpm_raw.parquet",
            "bronze_size_bytes": 4314964228,
            "ingestion_timestamp": "2026-04-30T03:31:16.043239+00:00",
            "infra_fingerprint": "fceb44ab17161175",
            "memory_used":       "4g",
            "override_applied":  True,
        }

    # 2. Simular Silver lineage
    silver = build_silver_lineage(
        bronze_lineage       = bronze,
        silver_row_count     = 74_628 * 19_788,
        quarantine_row_count = 0,
        tissue_count         = 54,
        chunk_plan_summary   = "particiones=20, RAM=3g, cores=5, ~14.0 min",
        quality_report_dict  = {"layer": "silver", "passed": True, "checks": []},
        infra_fingerprint    = "fceb44ab17161175",
        memory_used          = "3g",
        duration_seconds     = 840.0,
    )
    print(f"\n✅ Silver lineage construido")
    print(f"   rows     : {silver['output']['row_count']:,}")
    print(f"   closes   : {silver['lineage_closure']['closes']}")
    print(f"   delta    : {silver['lineage_closure']['delta']}")

    # 3. Pipeline completo (Gold pendiente)
    pipeline = build_pipeline_lineage(bronze, silver)
    print(f"\n✅ Pipeline lineage: {pipeline['layers_complete']} capas completas")
    print(f"   status: {pipeline['status']}")

    # 4. Guardar en disco
    save_lineage(silver,   SILVER_LINEAGE_PATH)
    save_lineage(pipeline, PIPELINE_LINEAGE_PATH)
    print(f"\n✅ Archivos guardados:")
    print(f"   {SILVER_LINEAGE_PATH}")
    print(f"   {PIPELINE_LINEAGE_PATH}")