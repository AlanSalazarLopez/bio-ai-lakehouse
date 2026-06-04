"""
src/utils/gold_batch_reader.py

Lectura de Silver en batches para Gold — sin DeltaTable, sin Spark.

Problema: Silver tiene 3,800 archivos Parquet y 1.5B filas. Leer todo
en RAM es imposible con 2-3GB disponibles.

Solución: generador que yield RecordBatches uno a la vez. El caller
(gold_transform.py) procesa cada batch y lo descarta — RAM máxima
usada es un solo batch en memoria a la vez.

Estrategia de batch_size dinámico:
    - psutil mide RAM disponible antes de cada archivo
    - batch_size se recalcula si la RAM cambió significativamente
    - Nunca hardcodea filas — se adapta al entorno real

Compatibilidad:
    - Lee con pq.ParquetFile().iter_batches() — igual que Silver Phase 2
    - Excluye _delta_log automáticamente
    - Compatible con Python 3.8 (Optional[X], no X | None)

Uso típico en gold_transform.py:
    reader = SilverBatchReader(silver_root="data/silver/gtex/gene_expression_long")
    for batch in reader.iter_batches():
        acc_map.update_from_batch(batch)
"""

import logging
import pathlib
from typing import Generator, List, Optional

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Columnas que Gold necesita — no leer sample_id (ahorra ~20% RAM por batch)
GOLD_COLUMNS = ["gene_id", "gene_symbol", "tissue_id", "tpm_value"]

# Límites de seguridad para batch_size
MIN_BATCH_ROWS = 10_000
MAX_BATCH_ROWS = 500_000

# Fracción de RAM disponible que puede usar un batch
RAM_BATCH_FRACTION = 0.15  # 15% — conservador, deja margen para acumuladores

# Bytes estimados por fila de Silver en RAM (4 strings + 1 float32 ≈ ~200 bytes)
BYTES_PER_ROW_ESTIMATE = 200


# ─────────────────────────────────────────────
#  Cálculo dinámico de batch_size — función pura
# ─────────────────────────────────────────────

def calculate_batch_size(available_ram_bytes: Optional[int] = None) -> int:
    """
    Calcula el número de filas por batch basado en RAM disponible.

    Args:
        available_ram_bytes: RAM disponible en bytes.
                             Si None, se mide con psutil en el momento.

    Returns:
        Número de filas por batch, clampado entre MIN y MAX.

    Ejemplo con 2GB disponibles:
        budget = 2GB × 0.15 = 307MB
        rows   = 307MB / 200 bytes = ~1,535,000 → clamp → 500,000

    Ejemplo con 512MB disponibles:
        budget = 512MB × 0.15 = 76MB
        rows   = 76MB / 200 bytes = ~380,000 → clamp → 380,000
    """
    if available_ram_bytes is None:
        available_ram_bytes = psutil.virtual_memory().available

    budget_bytes = available_ram_bytes * RAM_BATCH_FRACTION
    raw_rows     = int(budget_bytes / BYTES_PER_ROW_ESTIMATE)

    batch_size = max(MIN_BATCH_ROWS, min(raw_rows, MAX_BATCH_ROWS))

    logger.debug(
        "batch_size calculado: %s filas (ram_disponible=%.1fGB, budget=%.0fMB)",
        f"{batch_size:,}",
        available_ram_bytes / (1024 ** 3),
        budget_bytes / (1024 ** 2),
    )

    return batch_size


# ─────────────────────────────────────────────
#  Descubrimiento de archivos Silver
# ─────────────────────────────────────────────

def discover_silver_files(silver_root: str) -> List[pathlib.Path]:
    """
    Encuentra todos los Parquet de Silver excluyendo _delta_log.

    Args:
        silver_root: ruta raíz de Silver Delta Lake

    Returns:
        Lista de paths ordenada — orden determinista para reproducibilidad.

    Raises:
        FileNotFoundError: si silver_root no existe
        RuntimeError: si no encuentra ningún Parquet
    """
    root = pathlib.Path(silver_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Silver root no encontrado: {silver_root}\n"
            "¿Corriste silver_phase2_delta.py primero?"
        )

    files = sorted([
        p for p in root.rglob("*.parquet")
        if "_delta_log" not in str(p)
    ])

    if not files:
        raise RuntimeError(
            f"No se encontraron archivos Parquet en {silver_root}\n"
            "Silver puede estar vacío o corrupto."
        )

    logger.info("Silver: %s archivos Parquet encontrados en %s", f"{len(files):,}", silver_root)
    return files


# ─────────────────────────────────────────────
#  Reader principal
# ─────────────────────────────────────────────

class SilverBatchReader:
    """
    Generador de RecordBatches de Silver para Gold.

    Lee archivo por archivo, batch por batch.
    RAM máxima: un solo batch en memoria a la vez.

    Atributos públicos post-iteración:
        files_processed  : archivos leídos
        batches_yielded  : batches emitidos al caller
        rows_yielded     : filas totales emitidas
    """

    def __init__(
        self,
        silver_root: str = "data/silver/gtex/gene_expression_long",
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            silver_root: ruta raíz de Silver Delta Lake
            columns:     columnas a leer. Default: GOLD_COLUMNS
                         (excluye sample_id que Gold no necesita)
        """
        self.silver_root = silver_root
        self.columns     = columns or GOLD_COLUMNS

        # Métricas — disponibles después de iter_batches()
        self.files_processed: int = 0
        self.batches_yielded: int = 0
        self.rows_yielded:    int = 0

        # Descubrir archivos en construcción — falla rápido si Silver no existe
        self._files = discover_silver_files(silver_root)

    @property
    def total_files(self) -> int:
        return len(self._files)

    def iter_batches(self) -> Generator[pa.RecordBatch, None, None]:
        """
        Generador principal. Yield RecordBatch uno a la vez.

        Recalcula batch_size cada 100 archivos para adaptarse a cambios
        de RAM (otros procesos, GC del acumulador).

        Uso:
            for batch in reader.iter_batches():
                acc_map.update_from_batch(batch)
        """
        # Reset métricas al inicio de cada iteración
        self.files_processed = 0
        self.batches_yielded = 0
        self.rows_yielded    = 0

        batch_size   = calculate_batch_size()
        recalc_every = 100  # recalcular RAM cada N archivos

        logger.info(
            "Iniciando lectura Silver: %s archivos, batch_size=%s filas, cols=%s",
            f"{self.total_files:,}", f"{batch_size:,}", self.columns,
        )

        for file_idx, parquet_path in enumerate(self._files):

            # Recalcular batch_size periódicamente
            if file_idx > 0 and file_idx % recalc_every == 0:
                new_batch_size = calculate_batch_size()
                if new_batch_size != batch_size:
                    logger.info(
                        "batch_size ajustado: %s → %s (archivo %s/%s)",
                        f"{batch_size:,}", f"{new_batch_size:,}",
                        f"{file_idx:,}", f"{self.total_files:,}",
                    )
                    batch_size = new_batch_size

            # Log de progreso cada 500 archivos
            if file_idx % 500 == 0 and file_idx > 0:
                pct = file_idx / self.total_files * 100
                logger.info(
                    "Progreso: %s/%s archivos (%.1f%%) — %s filas emitidas",
                    f"{file_idx:,}", f"{self.total_files:,}",
                    pct, f"{self.rows_yielded:,}",
                )

            try:
                pf = pq.ParquetFile(parquet_path)
                for batch in pf.iter_batches(
                    batch_size=batch_size,
                    columns=self.columns,
                ):
                    self.batches_yielded += 1
                    self.rows_yielded    += batch.num_rows
                    yield batch

            except Exception as e:
                # Log y continuar — un archivo corrupto no detiene Gold
                logger.warning(
                    "Archivo saltado por error: %s — %s", parquet_path, e
                )
                continue

            self.files_processed += 1

        logger.info(
            "Lectura completa: %s archivos, %s batches, %s filas",
            f"{self.files_processed:,}",
            f"{self.batches_yielded:,}",
            f"{self.rows_yielded:,}",
        )

    def summary(self) -> str:
        """Resumen legible post-iteración para el lineage."""
        return (
            f"files={self.files_processed:,} "
            f"batches={self.batches_yielded:,} "
            f"rows={self.rows_yielded:,} "
            f"batch_size_initial={calculate_batch_size():,}"
        )


# ─────────────────────────────────────────────
#  CLI — smoke test sin Silver real
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n── Test 1: calculate_batch_size ──────────────────────────────\n")

    # Simular diferentes RAMs disponibles
    casos = [
        512  * 1024 ** 2,   # 512 MB
        1    * 1024 ** 3,   # 1 GB
        2    * 1024 ** 3,   # 2 GB  ← entorno real
        8    * 1024 ** 3,   # 8 GB
    ]
    for ram in casos:
        bs = calculate_batch_size(ram)
        print(f"  RAM={ram / 1024**3:.1f}GB → batch_size={bs:,}")

    print("\n── Test 2: batch_size con RAM real ───────────────────────────\n")
    bs_real = calculate_batch_size()
    ram_gb  = psutil.virtual_memory().available / 1024 ** 3
    print(f"  RAM disponible ahora : {ram_gb:.2f} GB")
    print(f"  batch_size calculado : {bs_real:,} filas")
    print(f"  dentro de límites    : {MIN_BATCH_ROWS <= bs_real <= MAX_BATCH_ROWS}")

    print("\n── Test 3: discover_silver_files (Silver real) ───────────────\n")
    silver_path = "data/silver/gtex/gene_expression_long"
    try:
        files = discover_silver_files(silver_path)
        print(f"  Archivos encontrados : {len(files):,}")
        print(f"  Primero              : {files[0].name}")
        print(f"  Último               : {files[-1].name}")
        print(f"  _delta_log excluido  : {not any('_delta_log' in str(f) for f in files)}")
    except FileNotFoundError as e:
        print(f"  Silver no disponible en este entorno — OK para CI")
        print(f"  ({e})")
        sys.exit(0)

    print("\n── Test 4: iter_batches (primeros 3 archivos) ────────────────\n")
    reader  = SilverBatchReader(silver_path)
    batches = 0
    rows    = 0

    # Parchear _files para leer solo 3
    reader._files = reader._files[:3]

    for batch in reader.iter_batches():
        batches += 1
        rows    += batch.num_rows
        # Verificar schema en el primer batch
        if batches == 1:
            cols = batch.schema.names
            print(f"  Columnas del batch   : {cols}")
            assert "gene_id"    in cols, "❌ gene_id ausente"
            assert "tpm_value"  in cols, "❌ tpm_value ausente"
            assert "sample_id" not in cols, "❌ sample_id debería estar excluido"
            print(f"  sample_id excluido   : ✅")

    print(f"  Batches emitidos     : {batches}")
    print(f"  Filas totales        : {rows:,}")
    print(f"  files_processed      : {reader.files_processed}")
    print(f"  summary()            : {reader.summary()}")
    print(f"\n✅ gold_batch_reader.py listo\n")