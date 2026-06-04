"""
src/utils/chunk_calculator.py

Calcula el número óptimo de particiones de Spark dado:
- El tamaño del archivo a procesar
- La RAM disponible (inferida o del override del .env)

Árbitro de seguridad: si el override del .env pide más RAM de la que
realmente hay disponible, usa la inferencia para evitar OOM.

Función pura — no depende de Spark, testeable sin cluster.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────

# Factor de expansión en memoria vs disco:
# Parquet comprimido ocupa ~3-4x más en RAM al deserializarse
PARQUET_EXPANSION_FACTOR = 3.5

# Overhead de Spark por partición (shuffle buffers, metadata, etc.)
SPARK_OVERHEAD_FACTOR = 0.75  # usamos solo 75% de la RAM asignada para datos

# Mínimo absoluto de RAM para operar (Paso 3 Framework)
MIN_RAM_GB = 2.0

# Mínimo de cores para operar
MIN_CORES = 1

# Tamaño objetivo por partición en GB — chunks más pequeños = más seguro
TARGET_PARTITION_GB = 0.25  # 256MB por partición


# ─────────────────────────────────────────────
#  Resultado del cálculo
# ─────────────────────────────────────────────

@dataclass
class ChunkPlan:
    """
    Plan de ejecución calculado para el job de Spark.
    Todos los campos son deterministas dado el mismo input.

    cols_per_chunk es relevante solo cuando se pasa total_cols a
    calculate_chunk_plan — en caso contrario vale -1 (no aplicable).
    """
    partitions:          int    # número de particiones para repartition()
    safe_memory_gb:      float  # RAM que realmente vamos a usar
    memory_source:       str    # 'override' | 'inferred' | 'conservative'
    cores:               int    # cores disponibles para el job
    estimated_minutes:   float  # estimación de tiempo de ejecución
    override_rejected:   bool   # True si el override fue ignorado por seguridad
    rejection_reason:    str    # por qué se rechazó el override (si aplica)
    cols_per_chunk:      int    # columnas de muestras por iteración Pandas (-1 si N/A)

    def summary(self) -> str:
        lines = [
            "── Chunk Plan ────────────────────────────",
            f"  particiones      : {self.partitions}",
            f"  RAM segura       : {self.safe_memory_gb}g",
            f"  fuente de RAM    : {self.memory_source}",
            f"  cores            : {self.cores}",
            f"  tiempo estimado  : ~{self.estimated_minutes:.1f} min",
        ]
        if self.cols_per_chunk > 0:
            lines.append(f"  cols/chunk       : {self.cols_per_chunk}")
        if self.override_rejected:
            lines.append(f"  ⚠️  override rechazado: {self.rejection_reason}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Función principal — pura, sin side-effects
# ─────────────────────────────────────────────

def calculate_chunk_plan(
    file_size_bytes:     int,
    available_ram_gb:    float,
    total_ram_gb:        float,
    override_memory_str: Optional[str],
    inferred_memory_gb:  float,
    cpu_physical:        int,
    total_cols:          Optional[int] = None,
) -> ChunkPlan:
    """
    Calcula el plan óptimo de particiones para procesar un archivo con Spark.

    Árbitro de seguridad:
    - Si el override pide más RAM de la disponible → se ignora y se usa la inferencia
    - Si la inferencia también es mayor a la disponible → se usa lo disponible * 0.85
    - Siempre garantiza mínimo MIN_RAM_GB y MIN_CORES

    Args:
        file_size_bytes:     tamaño del archivo fuente en bytes
        available_ram_gb:    RAM disponible en el sistema ahora mismo
        total_ram_gb:        RAM total del sistema
        override_memory_str: valor del SPARK_EXECUTOR_MEMORY_OVERRIDE ('4g', '512m', None)
        inferred_memory_gb:  valor calculado por resources.py (60% del total)
        cpu_physical:        cores físicos disponibles
        total_cols:          número total de columnas de muestras en el dataset wide
                             (opcional — si se pasa, calcula cols_per_chunk)

    Returns:
        ChunkPlan con particiones, RAM segura, estimación de tiempo
        y cols_per_chunk si total_cols fue provisto.
    """
    file_size_gb     = file_size_bytes / (1024 ** 3)
    expanded_size_gb = file_size_gb * PARQUET_EXPANSION_FACTOR

    # ── 1. Resolver RAM segura ─────────────────────────────────────────────
    override_rejected = False
    rejection_reason  = ""

    if override_memory_str:
        override_gb = _parse_memory_to_gb(override_memory_str)

        if override_gb > available_ram_gb:
            # Override pide más de lo disponible → peligro de OOM
            override_rejected = True
            rejection_reason  = (
                f"Override={override_gb}g > disponible={available_ram_gb:.1f}g. "
                "Usando inferencia para evitar OOM."
            )
            logger.warning(rejection_reason)

            # Intentar con la inferencia
            if inferred_memory_gb <= available_ram_gb:
                safe_memory_gb = inferred_memory_gb
                memory_source  = "inferred"
            else:
                # Inferencia también es mayor → usar 85% de lo disponible
                safe_memory_gb = round(available_ram_gb * 0.85, 1)
                memory_source  = "conservative"
                logger.warning(
                    "Inferencia (%.1fg) también supera disponible (%.1fg). "
                    "Usando conservador: %.1fg",
                    inferred_memory_gb, available_ram_gb, safe_memory_gb
                )
        else:
            # Override es seguro
            safe_memory_gb = override_gb
            memory_source  = "override"

    elif inferred_memory_gb <= available_ram_gb:
        safe_memory_gb = inferred_memory_gb
        memory_source  = "inferred"
    else:
        # Sin override y la inferencia supera lo disponible → conservador
        safe_memory_gb = round(available_ram_gb * 0.85, 1)
        memory_source  = "conservative"
        logger.warning(
            "RAM inferida (%.1fg) supera disponible (%.1fg). Usando conservador: %.1fg",
            inferred_memory_gb, available_ram_gb, safe_memory_gb
        )

    # Garantizar mínimo absoluto
    safe_memory_gb = max(safe_memory_gb, MIN_RAM_GB)
    safe_memory_gb = math.floor(safe_memory_gb)

    # ── 2. Calcular particiones ────────────────────────────────────────────
    # RAM utilizable para datos después del overhead de Spark
    usable_ram_gb = safe_memory_gb * SPARK_OVERHEAD_FACTOR

    # Particiones por RAM: cuántos chunks caben en la RAM usable
    partitions_by_ram = math.ceil(expanded_size_gb / usable_ram_gb)

    # Particiones por tamaño objetivo: chunks de TARGET_PARTITION_GB
    partitions_by_size = math.ceil(file_size_gb / TARGET_PARTITION_GB)

    # Usar el mayor — más particiones = chunks más pequeños = más seguro
    partitions = max(partitions_by_ram, partitions_by_size, 1)

    # Redondear al siguiente múltiplo de cores para paralelismo óptimo
    usable_cores = max(cpu_physical - 1, MIN_CORES)
    partitions   = math.ceil(partitions / usable_cores) * usable_cores

    # ── 3. Estimar tiempo ──────────────────────────────────────────────────
    # Throughput estimado: ~0.5 GB/min por core con parquet + snappy en local
    throughput_gb_per_min = 0.5 * usable_cores
    estimated_minutes     = round(expanded_size_gb / throughput_gb_per_min, 1)

    # ── 4. Calcular cols_per_chunk (solo si total_cols fue provisto) ────────
    cols_per_chunk = _calculate_cols_per_chunk(
        file_size_bytes = file_size_bytes,
        total_cols      = total_cols,
        safe_memory_gb  = safe_memory_gb,
    )

    logger.info(
        "ChunkPlan: partitions=%d, safe_memory=%.1fg, source=%s, ~%.1f min%s",
        partitions, safe_memory_gb, memory_source, estimated_minutes,
        f", cols_per_chunk={cols_per_chunk}" if cols_per_chunk > 0 else "",
    )

    return ChunkPlan(
        partitions        = partitions,
        safe_memory_gb    = safe_memory_gb,
        memory_source     = memory_source,
        cores             = usable_cores,
        estimated_minutes = estimated_minutes,
        override_rejected = override_rejected,
        rejection_reason  = rejection_reason,
        cols_per_chunk    = cols_per_chunk,
    )


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _calculate_cols_per_chunk(
    file_size_bytes: int,
    total_cols:      Optional[int],
    safe_memory_gb:  float,
) -> int:
    """
    Calcula cuántas columnas de muestras caben en la RAM segura por iteración
    Pandas (wide→long). Retorna -1 si total_cols no fue provisto.

    Lógica:
      bytes_por_col  = file_size_bytes / total_cols
      ram_para_pandas = safe_memory_gb * SPARK_OVERHEAD_FACTOR * 0.5
          ↑ Spark y Python comparten RAM — Pandas toma la mitad de lo usable
      cols_en_ram    = ram_para_pandas (bytes) / (bytes_por_col * PARQUET_EXPANSION_FACTOR)

    Se aplica un cap de 500 y un mínimo de 50 para evitar extremos.
    """
    if not total_cols or total_cols <= 0:
        return -1

    bytes_por_col    = file_size_bytes / total_cols
    # Pandas comparte RAM con Spark — usa máximo la mitad de la RAM usable
    pandas_ram_bytes = safe_memory_gb * SPARK_OVERHEAD_FACTOR * 0.5 * (1024 ** 3)
    cols_in_ram      = pandas_ram_bytes / (bytes_por_col * PARQUET_EXPANSION_FACTOR)

    cols_per_chunk = math.floor(cols_in_ram)
    cols_per_chunk = max(cols_per_chunk, 50)   # mínimo: 50 cols/chunk
    cols_per_chunk = min(cols_per_chunk, 100)  # cap: 100 cols/chunk

    return cols_per_chunk


def _parse_memory_to_gb(memory_str: str) -> float:
    """Convierte '4g', '512m' → float en GB."""
    memory_str = memory_str.strip().lower()
    if memory_str.endswith("g"):
        return float(memory_str[:-1])
    if memory_str.endswith("m"):
        return float(memory_str[:-1]) / 1024
    raise ValueError(f"Formato de memoria no reconocido: '{memory_str}'")


# ─────────────────────────────────────────────
#  CLI — prueba sin Spark
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.utils.resources import get_spark_memory_settings
    import psutil

    # Simular con el archivo real del proyecto
    file_path = "data/raw/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet"

    if not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        print("Simulando con 4GB de archivo de prueba...")
        file_size_bytes = int(4.02 * (1024 ** 3))
    else:
        file_size_bytes = os.path.getsize(file_path)

    infra = get_spark_memory_settings(mode="local")
    mem   = psutil.virtual_memory()

    plan = calculate_chunk_plan(
        file_size_bytes      = file_size_bytes,
        available_ram_gb     = mem.available / (1024 ** 3),
        total_ram_gb         = mem.total     / (1024 ** 3),
        override_memory_str  = infra["env_snapshot"]["override_env"],
        inferred_memory_gb   = float(infra["meta"]["memory_used"].replace("g", "")),
        cpu_physical         = infra["meta"]["cpu_physical"],
        total_cols           = 19_788,  # columnas de muestras GTEx
    )

    print(plan.summary())