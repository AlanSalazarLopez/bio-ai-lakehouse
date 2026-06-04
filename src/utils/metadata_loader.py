"""
src/utils/metadata_loader.py

Carga el archivo de metadata de muestras GTEx y construye el mapping
SAMPID → SMTSD (sample_id → tissue_id) para el join en Silver.

El archivo gtex_metadata.txt es un TSV con ~100 columnas. Solo nos
interesan dos: SAMPID (key) y SMTSD (tejido destino).

Función principal: load_tissue_mapping
    → retorna dict {sample_id: tissue_id} listo para el join en Silver

Función secundaria: validate_tissue_mapping
    → verifica que el mapping no esté vacío y loguea stats básicas

Compatible con Python 3.8 — usa Optional[X], no X | None.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Columnas que nos importan del TSV — el resto se ignora
SAMPID_COL = "SAMPID"
SMTSD_COL  = "SMTSD"

# Ruta por defecto dentro del contenedor
DEFAULT_METADATA_PATH = "data/raw/gtex_metadata.txt"


# ─────────────────────────────────────────────
#  Función principal
# ─────────────────────────────────────────────

def load_tissue_mapping(
    path: str = DEFAULT_METADATA_PATH,
) -> Dict[str, str]:
    """
    Lee gtex_metadata.txt y retorna un dict {sample_id → tissue_id}.

    Solo carga las columnas SAMPID y SMTSD — ignora las ~100 restantes.
    Filas con SAMPID o SMTSD vacíos se descartan y se loguean como warning.

    Args:
        path: ruta al archivo TSV de metadata GTEx.

    Returns:
        Dict[str, str] — {sample_id: tissue_id}
        Ejemplo: {"GTEX-1117F-0005-SM-HL9SH": "Thyroid"}

    Raises:
        FileNotFoundError: si el archivo no existe en la ruta indicada.
        KeyError: si el archivo no tiene las columnas SAMPID o SMTSD.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metadata GTEx no encontrada en: {path}\n"
            f"Asegúrate de que el archivo esté en data/raw/"
        )

    mapping: Dict[str, str] = {}
    skipped_empty = 0
    total_rows    = 0

    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        columns     = header_line.split("\t")

        # Validar que las columnas clave existen
        if SAMPID_COL not in columns:
            raise KeyError(
                f"Columna '{SAMPID_COL}' no encontrada en {path}. "
                f"Columnas disponibles: {columns[:10]}..."
            )
        if SMTSD_COL not in columns:
            raise KeyError(
                f"Columna '{SMTSD_COL}' no encontrada en {path}. "
                f"Columnas disponibles: {columns[:10]}..."
            )

        sampid_idx = columns.index(SAMPID_COL)
        smtsd_idx  = columns.index(SMTSD_COL)

        for line in f:
            total_rows += 1
            parts = line.rstrip("\n").split("\t")

            # Protección contra líneas cortas malformadas
            if len(parts) <= max(sampid_idx, smtsd_idx):
                skipped_empty += 1
                continue

            sample_id = parts[sampid_idx].strip()
            tissue_id = parts[smtsd_idx].strip()

            if not sample_id or not tissue_id:
                skipped_empty += 1
                logger.debug(
                    "Fila descartada — SAMPID='%s' SMTSD='%s'",
                    sample_id, tissue_id
                )
                continue

            mapping[sample_id] = tissue_id

    if skipped_empty > 0:
        logger.warning(
            "metadata_loader: %d/%d filas descartadas por SAMPID o SMTSD vacío",
            skipped_empty, total_rows
        )

    logger.info(
        "metadata_loader: %d muestras cargadas desde %s",
        len(mapping), path
    )

    return mapping


# ─────────────────────────────────────────────
#  Validación del mapping
# ─────────────────────────────────────────────

def validate_tissue_mapping(
    mapping: Dict[str, str],
    min_samples: int = 100,
) -> Tuple[bool, str]:
    """
    Verifica que el mapping tenga contenido suficiente para el join.

    Args:
        mapping:     resultado de load_tissue_mapping()
        min_samples: mínimo de muestras esperadas (default 100)

    Returns:
        Tuple[bool, str] — (passed, mensaje)
    """
    if not mapping:
        return False, "mapping vacío — gtex_metadata.txt no cargó ninguna muestra"

    if len(mapping) < min_samples:
        return False, (
            f"mapping demasiado pequeño: {len(mapping)} muestras < {min_samples} mínimo"
        )

    tissues       = set(mapping.values())
    tissue_counts = {}
    for tissue in mapping.values():
        tissue_counts[tissue] = tissue_counts.get(tissue, 0) + 1

    top_tissue    = max(tissue_counts, key=tissue_counts.__getitem__)
    bottom_tissue = min(tissue_counts, key=tissue_counts.__getitem__)

    msg = (
        f"mapping válido: {len(mapping):,} muestras × {len(tissues)} tejidos | "
        f"mayor={top_tissue} ({tissue_counts[top_tissue]:,}) | "
        f"menor={bottom_tissue} ({tissue_counts[bottom_tissue]:,})"
    )

    logger.info("validate_tissue_mapping: %s", msg)
    return True, msg


# ─────────────────────────────────────────────
#  Helper para el join en Silver
# ─────────────────────────────────────────────

def get_tissue_or_unknown(
    mapping: Dict[str, str],
    sample_id: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """
    Lookup seguro del tissue_id dado un sample_id.
    Retorna None (o fallback) si no hay match — el caller decide si va a cuarentena.

    Uso en silver_transform.py:
        tissue = get_tissue_or_unknown(mapping, sample_id)
        if tissue is None:
            → quarantine
    """
    return mapping.get(sample_id, fallback)


# ─────────────────────────────────────────────
#  CLI — smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_METADATA_PATH

    print(f"\n── Cargando metadata desde: {path} ──────────")
    mapping = load_tissue_mapping(path)

    passed, msg = validate_tissue_mapping(mapping)
    status = "✅" if passed else "❌"
    print(f"{status} {msg}")

    # Mostrar 3 ejemplos del mapping
    print("\n── Ejemplos del mapping ──────────────────")
    for i, (sample_id, tissue_id) in enumerate(mapping.items()):
        print(f"  {sample_id} → {tissue_id}")
        if i >= 2:
            break

    # Test de lookup
    print("\n── Test get_tissue_or_unknown ────────────")
    test_id = "GTEX-FAKE-0000-SM-XXXXX"
    result  = get_tissue_or_unknown(mapping, test_id)
    print(f"  sample no existente → {result}")

    first_sample = next(iter(mapping))
    result2 = get_tissue_or_unknown(mapping, first_sample)
    print(f"  primera muestra real → {result2}")