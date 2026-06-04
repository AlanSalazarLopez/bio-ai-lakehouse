"""
src/utils/execution_profile.py

Define los perfiles de ejecución del pipeline Bio-AI Lakehouse.
Cada perfil encapsula todos los parámetros que cambian con la RAM disponible:
cols_per_chunk, n_partitions, spill_to_disk, AQE config, etc.

Perfiles (de menor a mayor agresividad):
    SURVIVAL    < 4GB   → que termine, sin importar el tiempo
    BALANCED    4-16GB  → balance razonable
    PERFORMANCE 16-32GB → agresivo pero seguro
    PRO         > 32GB  → máximo aprovechamiento

Uso típico:
    profile = detect_profile(available_ram_gb)
    # o forzar desde CLI:
    profile = get_profile_by_name("BALANCED")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass del perfil
# ---------------------------------------------------------------------------

@dataclass
class ExecutionProfile:
    """
    Todos los parámetros que varían con el hardware en un solo objeto.
    silver_transform.py consume esto — no sabe nada de RAM directamente.
    """
    name:               str    # "SURVIVAL" | "BALANCED" | "PERFORMANCE" | "PRO"
    min_ram_gb:         float  # umbral mínimo de RAM para este perfil
    cols_per_chunk:     int    # columnas de muestras por iteración Pandas
    n_partitions:       int    # particiones para repartition() antes del write
    spill_to_disk:      bool   # activar off-heap + spill para presión de RAM
    aqe_aggressive:     bool   # AQE con coalesce agresivo para RAM limitada
    chunks_per_write:   int    # chunks a acumular antes de cada write a Delta
    max_records_per_file: int  # filas máximas por archivo Delta (controla tamaño)
    spark_extra_config: Dict[str, str] = field(default_factory=dict)

    # Configs que deben ir en el SparkSession builder (no modificables post-inicio)
    STATIC_CONFIGS = {
        "spark.memory.fraction",
        "spark.memory.storageFraction",
        "spark.memory.offHeap.enabled",
        "spark.memory.offHeap.size",
    }

    def static_configs(self) -> dict:
        """Configs que van en el builder — no modificables después de getOrCreate()."""
        return {k: v for k, v in self.spark_extra_config.items()
                if k in self.STATIC_CONFIGS}

    def dynamic_configs(self) -> dict:
        """Configs aplicables post-inicio via spark.conf.set()."""
        return {k: v for k, v in self.spark_extra_config.items()
                if k not in self.STATIC_CONFIGS}

    def summary(self) -> str:
        lines = [
            f"── Execution Profile: {self.name} ────────────────",
            f"  cols/chunk       : {self.cols_per_chunk}",
            f"  n_partitions     : {self.n_partitions}",
            f"  spill_to_disk    : {self.spill_to_disk}",
            f"  aqe_aggressive   : {self.aqe_aggressive}",
        ]
        lines.append(f"  chunks/write     : {self.chunks_per_write}")
        lines.append(f"  max_records/file : {self.max_records_per_file:,}")
        if self.spark_extra_config:
            for k, v in self.spark_extra_config.items():
                lines.append(f"  {k} = {v}")
        return "\n".join(lines)

    def downgrade(self) -> Optional["ExecutionProfile"]:
        """
        Retorna el perfil inmediatamente inferior.
        Retorna None si ya estamos en SURVIVAL — no se puede bajar más.
        """
        order = PROFILE_ORDER
        idx   = order.index(self.name)
        if idx == 0:
            return None  # ya en SURVIVAL
        return PROFILES[order[idx - 1]]


# ---------------------------------------------------------------------------
# Definición de los cuatro perfiles
# ---------------------------------------------------------------------------

PROFILES: Dict[str, ExecutionProfile] = {

    "SURVIVAL": ExecutionProfile(
        name                 = "SURVIVAL",
        min_ram_gb           = 0.0,
        cols_per_chunk       = 50,
        n_partitions         = 5,
        spill_to_disk        = True,
        aqe_aggressive       = True,
        chunks_per_write     = 5,
        max_records_per_file = 500_000,
        spark_extra_config   = {
            # AQE — coalesce agresivo para reducir shuffle en RAM limitada
            "spark.sql.adaptive.enabled":                          "true",
            "spark.sql.adaptive.coalescePartitions.enabled":       "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes":     "32mb",
            "spark.sql.adaptive.skewJoin.enabled":                 "true",
            # Forzar sort-merge join — nunca broadcast (OOM en RAM baja)
            "spark.sql.autoBroadcastJoinThreshold":                "-1",
            # Spill a disco cuando la RAM se agota
            "spark.memory.storageFraction":                        "0.2",
            "spark.memory.fraction":                               "0.6",
            # Off-heap para reducir presión en el heap de JVM
            "spark.memory.offHeap.enabled":                        "true",
            "spark.memory.offHeap.size":                           "512m",
            # Delta — reducir small files y espaciar checkpoints
            "spark.databricks.delta.checkpointInterval":           "10",
            "spark.sql.files.maxRecordsPerFile":                   "500000",
        },
    ),

    "BALANCED": ExecutionProfile(
        name                 = "BALANCED",
        min_ram_gb           = 4.0,
        cols_per_chunk       = 100,
        n_partitions         = 10,
        spill_to_disk        = True,
        aqe_aggressive       = True,
        chunks_per_write     = 10,
        max_records_per_file = 1_000_000,
        spark_extra_config   = {
            "spark.sql.adaptive.enabled":                          "true",
            "spark.sql.adaptive.coalescePartitions.enabled":       "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes":     "64mb",
            "spark.sql.adaptive.skewJoin.enabled":                 "true",
            "spark.sql.autoBroadcastJoinThreshold":                "-1",
            "spark.memory.storageFraction":                        "0.3",
            "spark.memory.fraction":                               "0.6",
            "spark.memory.offHeap.enabled":                        "true",
            "spark.memory.offHeap.size":                           "1g",
            # Delta — reducir small files
            "spark.databricks.delta.checkpointInterval":           "10",
            "spark.sql.files.maxRecordsPerFile":                   "1000000",
        },
    ),

    "PERFORMANCE": ExecutionProfile(
        name                 = "PERFORMANCE",
        min_ram_gb           = 16.0,
        cols_per_chunk       = 200,
        n_partitions         = 20,
        spill_to_disk        = False,
        aqe_aggressive       = False,
        chunks_per_write     = 20,
        max_records_per_file = 2_000_000,
        spark_extra_config   = {
            "spark.sql.adaptive.enabled":                          "true",
            "spark.sql.adaptive.coalescePartitions.enabled":       "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes":     "128mb",
            "spark.sql.adaptive.skewJoin.enabled":                 "true",
            # Broadcast habilitado para tablas pequeñas
            "spark.sql.autoBroadcastJoinThreshold":                "10mb",
            "spark.memory.storageFraction":                        "0.5",
            "spark.memory.fraction":                               "0.75",
        },
    ),

    "PRO": ExecutionProfile(
        name                 = "PRO",
        min_ram_gb           = 32.0,
        cols_per_chunk       = 400,
        n_partitions         = 40,
        spill_to_disk        = False,
        aqe_aggressive       = False,
        chunks_per_write     = 40,
        max_records_per_file = 5_000_000,
        spark_extra_config   = {
            "spark.sql.adaptive.enabled":                          "true",
            "spark.sql.adaptive.coalescePartitions.enabled":       "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes":     "256mb",
            "spark.sql.adaptive.skewJoin.enabled":                 "true",
            "spark.sql.autoBroadcastJoinThreshold":                "50mb",
            "spark.memory.storageFraction":                        "0.6",
            "spark.memory.fraction":                               "0.8",
        },
    ),
}

# Orden de menor a mayor — usado por downgrade()
PROFILE_ORDER: List[str] = ["SURVIVAL", "BALANCED", "PERFORMANCE", "PRO"]


# ---------------------------------------------------------------------------
# Detección automática
# ---------------------------------------------------------------------------

def detect_profile(available_ram_gb: float) -> ExecutionProfile:
    """
    Detecta el perfil más agresivo que puede soportar la RAM disponible.
    Siempre intenta el perfil más alto posible — el retry wrapper baja si falla.

    Args:
        available_ram_gb: RAM disponible ahora mismo (de psutil)

    Returns:
        ExecutionProfile correspondiente
    """
    # Recorrer de mayor a menor — primer perfil que cabe en la RAM disponible
    for name in reversed(PROFILE_ORDER):
        profile = PROFILES[name]
        if available_ram_gb >= profile.min_ram_gb:
            logger.info(
                "RAM disponible: %.1fGB → perfil detectado: %s "
                "(cols/chunk=%d, partitions=%d)",
                available_ram_gb, profile.name,
                profile.cols_per_chunk, profile.n_partitions,
            )
            return profile

    # Fallback absoluto — nunca debería llegar aquí
    logger.warning(
        "RAM disponible (%.1fGB) por debajo de todos los umbrales. "
        "Usando SURVIVAL como fallback.",
        available_ram_gb,
    )
    return PROFILES["SURVIVAL"]


def get_profile_by_name(name: str) -> ExecutionProfile:
    """
    Retorna un perfil por nombre — usado para override desde CLI.

    Args:
        name: "SURVIVAL" | "BALANCED" | "PERFORMANCE" | "PRO"

    Raises:
        ValueError si el nombre no existe
    """
    name = name.upper().strip()
    if name not in PROFILES:
        raise ValueError(
            f"Perfil '{name}' no existe. Opciones: {PROFILE_ORDER}"
        )
    return PROFILES[name]


# ---------------------------------------------------------------------------
# CLI — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import psutil

    available = psutil.virtual_memory().available / (1024 ** 3)
    total     = psutil.virtual_memory().total     / (1024 ** 3)

    print(f"\nRAM total     : {total:.1f}GB")
    print(f"RAM disponible: {available:.1f}GB")

    profile = detect_profile(available)
    print(f"\n{profile.summary()}")

    # Simular downgrade chain
    print("\n── Downgrade chain ───────────────────────")
    current = profile
    while current:
        print(f"  {current.name}")
        current = current.downgrade()