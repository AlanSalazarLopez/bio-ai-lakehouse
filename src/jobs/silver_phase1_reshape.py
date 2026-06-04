"""
src/jobs/silver_phase1_reshape.py

Phase 1 del reshape Silver — PyArrow puro, sin Spark, sin JVM.

Lee el parquet Bronze wide por batches de columnas usando PyArrow Dataset API,
hace el reshape wide→long manualmente a nivel de RecordBatch, hace el join con
tissue_mapping, y escribe Parquet staging particionado por tissue_id en disco.

Sin Spark → sin JVM → RAM máxima usada: ~200MB por batch.
El staging se conserva en disco como backup — Phase 2 lo consume para Delta.

Output:
    data/staging/silver/
        tissue_id=Whole Blood/
            batch_001.parquet
            ...
        tissue_id=Liver/
            ...
        quarantine/
            unmatched.parquet   ← samples sin tissue match

Invariantes:
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
# Config
# ---------------------------------------------------------------------------

BRONZE_PATH   = "data/bronze/gtex/gene_tpm_raw.parquet"
STAGING_PATH  = "data/staging/silver"
QUARANTINE_PATH = "data/staging/silver/quarantine/unmatched.parquet"
METADATA_PATH = "data/raw/gtex_metadata.txt"

METADATA_COLS = ["Name", "Description"]

# Schema Arrow del output long
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
# 1. Obtener columnas de muestras del schema (cero RAM)
# ---------------------------------------------------------------------------

def get_sample_cols() -> List[str]:
    """Lee solo el schema del parquet — sin datos, cero RAM."""
    schema = pq.read_schema(BRONZE_PATH)
    sample_cols = [c for c in schema.names if c not in METADATA_COLS]
    log.info(
        f"Schema Bronze: {len(schema.names)} columnas totales, "
        f"{len(sample_cols)} columnas de muestras"
    )
    return sample_cols


# ---------------------------------------------------------------------------
# 2. Reshape wide→long + escritura streaming (sin acumulación en RAM)
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
    Reshape wide→long columna por columna, escribiendo a disco inmediatamente.
    Sin concat acumulativo — RAM máxima: una sola columna en memoria (~6MB).

    Estrategia:
        - Un ParquetWriter abierto por tissue_id (reutilizado entre columnas)
        - Cada columna se procesa, escribe y libera antes de la siguiente
        - Quarantine se acumula solo si hay unmatched (debería ser vacío o mínimo)

    Retorna (matched_rows, quarantine_rows).
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
                    # Crear subdirectorio y writer para este tejido
                    tissue_dir = staging_dir / f"tissue_id={tissue_id}"
                    tissue_dir.mkdir(parents=True, exist_ok=True)
                    out_path = tissue_dir / f"batch_{batch_idx:04d}.parquet"

                    # Idempotencia: si el archivo ya existe, verificar integridad
                    # Archivo íntegro → saltar este tissue en este batch
                    # Archivo corrupto → borrar y reescribir
                    if out_path.exists():
                        try:
                            pq.read_metadata(str(out_path))  # solo lee footer, cero RAM
                            log.info(f"  Batch {batch_idx} tissue={tissue_id} ya existe y es íntegro — saltando")
                            matched_rows += n_genes  # contar las filas aunque no se reescriban
                            del single, tpm_values
                            continue
                        except Exception:
                            log.warning(f"  Batch {batch_idx} tissue={tissue_id} corrupto — reescribiendo")
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
                # Quarantine — acumular solo unmatched (mínimos o cero)
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
        # Cerrar todos los writers — garantiza flush a disco
        for writer in writers.values():
            writer.close()

    # Escribir quarantine si hubo unmatched
    if quarantine_batches:
        write_quarantine(pa.concat_tables(quarantine_batches))

    return matched_rows, quarantine_rows


def write_quarantine(quarantine: pa.Table) -> None:
    """Append a cuarentena — acumula todos los unmatched en un solo parquet."""
    if quarantine.num_rows == 0:
        return

    path = Path(QUARANTINE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing  = pq.read_table(path)
        quarantine = pa.concat_tables([existing, quarantine])

    pq.write_table(quarantine, str(path), compression="snappy")
    log.warning(f"  Cuarentena: {quarantine.num_rows:,} filas → {QUARANTINE_PATH}")


# ---------------------------------------------------------------------------
# 4. Progreso — para retomar si falla
# ---------------------------------------------------------------------------

PROGRESS_FILE = "data/staging/silver/.progress.json"

def load_progress() -> int:
    """Retorna el último batch completado (0 si nunca corrió)."""
    import json
    path = Path(PROGRESS_FILE)
    if not path.exists():
        return 0
    with open(path) as f:
        return json.load(f).get("last_completed_batch", 0)

def save_progress(batch_idx: int, silver_rows: int, quarantine_rows: int) -> None:
    """Guarda el progreso después de cada batch exitoso."""
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
# 5. main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=None,
        help="Forzar perfil: SURVIVAL | BALANCED | PERFORMANCE | PRO"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retomar desde el último batch completado"
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info("=" * 60)
    log.info("Phase 1 — Reshape Bronze → Parquet Staging (PyArrow)")
    log.info("=" * 60)

    # --- Perfil ---
    import psutil
    available_ram = psutil.virtual_memory().available / (1024 ** 3)
    if args.profile:
        profile = get_profile_by_name(args.profile)
        log.info(f"Perfil forzado: {profile.name}")
    else:
        profile = detect_profile(available_ram)

    log.info(f"\n{profile.summary()}")
    chunk_size = profile.cols_per_chunk

    # --- Leer config del tuner si existe (override del perfil) ---
    OPTIMAL_CONFIG = "data/staging/silver_tuner/optimal_config.json"
    import json as _json
    _tuner_path = Path(OPTIMAL_CONFIG)
    if _tuner_path.exists() and not args.profile:
        with open(_tuner_path) as f:
            _cfg = _json.load(f)
        chunk_size = _cfg["cols_per_chunk"]
        log.info(
            f"Tuner config encontrada → cols_per_chunk = {chunk_size} "
            f"(RAM pico estimado: {_cfg['peak_ram_mb']:.0f}MB, "
            f"~{_cfg['estimated_minutes']:.0f} min total)"
        )
    elif _tuner_path.exists() and args.profile:
        log.info("Perfil forzado por --profile, ignorando tuner config")

    # --- Metadata y columnas ---
    tissue_mapping = load_tissue_mapping(path=METADATA_PATH)
    valid, msg     = validate_tissue_mapping(tissue_mapping)
    if not valid:
        raise ValueError(f"Tissue mapping inválido: {msg}")
    log.info(f"Tissue mapping: {len(tissue_mapping):,} muestras → {len(set(tissue_mapping.values()))} tejidos")

    sample_cols = get_sample_cols()
    n_samples   = len(sample_cols)

    # --- Batches ---
    batches  = [sample_cols[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]
    n_batches = len(batches)

    # --- Resume ---
    start_batch = 0
    silver_rows = 0
    quarantine_rows = 0

    if args.resume:
        last = load_progress()
        if last > 0:
            import json
            with open(PROGRESS_FILE) as f:
                prog = json.load(f)
            start_batch     = last  # continuar desde el siguiente
            silver_rows     = prog.get("silver_rows_so_far", 0)
            quarantine_rows = prog.get("quarantine_rows_so_far", 0)
            log.info(f"Resumiendo desde batch {start_batch + 1}/{n_batches}")
            log.info(f"  Silver acumulado hasta ahora: {silver_rows:,} filas")

    log.info(f"Total batches: {n_batches} | cols/batch: {chunk_size}")
    log.info(f"Batches a procesar: {n_batches - start_batch}")

    # --- Loop principal ---
    for idx, cols in enumerate(batches):
        if idx < start_batch:
            continue

        batch_num = idx + 1
        log.info(f"Batch {batch_num}/{n_batches} — {len(cols)} muestras")

        # Leer solo estas columnas del parquet Bronze — PyArrow Dataset
        dataset = ds.dataset(BRONZE_PATH, format="parquet")
        table   = dataset.to_table(columns=METADATA_COLS + cols)

        # Reshape wide→long + write streaming (sin acumulación en RAM)
        matched_n, quarantine_n = reshape_and_write_streaming(
            table, cols, tissue_mapping, batch_num
        )
        del table

        silver_rows     += matched_n
        quarantine_rows += quarantine_n

        # Guardar progreso
        save_progress(idx + 1, silver_rows, quarantine_rows)
        log.info(f"  Silver acumulado: {silver_rows:,} filas")

    # --- Validación de cierre ---
    expected = 74_628 * 19_788
    actual   = silver_rows + quarantine_rows
    log.info(f"Row count check → esperado: {expected:,} | real: {actual:,}")
    if actual != expected:
        log.warning(f"Discrepancia de {expected - actual:+,} filas")
    else:
        log.info("✅ Lineage cierra perfectamente")

    duration = time.time() - t_start
    log.info("=" * 60)
    log.info(f"Phase 1 completada en {duration:.1f}s ({duration/60:.1f} min)")
    log.info(f"  Silver staging rows : {silver_rows:,}")
    log.info(f"  Quarantine rows     : {quarantine_rows:,}")
    log.info(f"  Staging path        : {STAGING_PATH}")
    log.info("=" * 60)

    return silver_rows, quarantine_rows


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("Error no manejado en silver_phase1_reshape.py")
        traceback.print_exc()
        raise SystemExit(2)