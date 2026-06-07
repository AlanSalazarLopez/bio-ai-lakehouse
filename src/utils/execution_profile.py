"""
src/utils/execution_profile.py

Defines execution profile configurations for the Bio-AI Lakehouse pipeline.
Each profile encapsulates every parameter shifting alongside available hardware RAM:
cols_per_chunk, n_partitions, spill_to_disk, AQE configurations, etc.

Profiles (Sorted from least to most aggressive):
    SURVIVAL    < 4GB   → Force job completion, discarding time constraints
    BALANCED    4-16GB  → Balanced stability and throughput compromise
    PERFORMANCE 16-32GB → Aggressive but secure performance optimizations
    PRO         > 32GB  → Maximum hardware infrastructure utilization

Typical Usage:
    profile = detect_profile(available_ram_gb)
    # Or force manually via CLI overrides:
    profile = get_profile_by_name("BALANCED")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile Dataclass Model
# ---------------------------------------------------------------------------

@dataclass
class ExecutionProfile:
    """
    Encapsulates all hardware-dependent execution parameters within a single object.
    Consumed by silver_transform.py — abstracts raw RAM evaluations entirely.
    """
    name:                 str    # "SURVIVAL" | "BALANCED" | "PERFORMANCE" | "PRO"
    min_ram_gb:           float  # Minimum memory threshold boundary to qualify for this profile
    cols_per_chunk:       int    # Total sample column steps parsed per Pandas iteration loop
    n_partitions:         int    # Target partition counts applied to repartition() before writing
    spill_to_disk:        bool   # Controls off-heap + physical disk spill under heavy memory pressure
    aqe_aggressive:       bool   # Activates aggressive AQE partition coalescing constraints
    chunks_per_write:     int    # In-memory chunks accumulated before committing write operations to Delta
    max_records_per_file: int    # Target row ceiling threshold per individual Delta file (controls sizing)
    spark_extra_config: Dict[str, str] = field(default_factory=dict)

    # Static engine parameters requiring binding during SparkSession builder initialization
    STATIC_CONFIGS = {
        "spark.memory.fraction",
        "spark.memory.storageFraction",
        "spark.memory.offHeap.enabled",
        "spark.memory.offHeap.size",
    }

    def static_configs(self) -> dict:
        """Extracts configuration settings bound directly to the SparkSession builder."""
        return {k: v for k, v in self.spark_extra_config.items()
                if k in self.STATIC_CONFIGS}

    def dynamic_configs(self) -> dict:
        """Extracts configuration settings applicable mid-lifecycle via spark.conf.set()."""
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
        Safely identifies and returns the next fallback execution profile.
        Returns None if active profile matches SURVIVAL limits (lowest boundary block).
        """
        order = PROFILE_ORDER
        idx   = order.index(self.name)
        if idx == 0:
            return None  # Already operating at SURVIVAL baseline levels
        return PROFILES[order[idx - 1]]


# ---------------------------------------------------------------------------
# Definition of the Four Execution Profiles
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
            # AQE — Aggressive partition coalescence to reduce shuffles under constrained memory bounds
            "spark.sql.adaptive.enabled":                          "true",
            "spark.sql.adaptive.coalescePartitions.enabled":       "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes":     "32mb",
            "spark.sql.adaptive.skewJoin.enabled":                 "true",
            # Enforce sort-merge joins exclusively — suppress broadcasts to prevent low-memory Driver OOM errors
            "spark.sql.autoBroadcastJoinThreshold":                "-1",
            # Storage memory fraction allocations, optimizing persistent disk spills when limits break
            "spark.memory.storageFraction":                        "0.2",
            "spark.memory.fraction":                               "0.6",
            # Off-heap configuration parameters to minimize tracking footprint pressures inside JVM heap spaces
            "spark.memory.offHeap.enabled":                        "true",
            "spark.memory.offHeap.size":                           "512m",
            # Delta Lake — Suppress small file generation footprints and space execution checkpoint cycles
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
            # Delta Lake — Standard operational small file optimizations
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
            # Broadcast strategies activated for metadata mapping sets and small structural joins safely
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

# Sequential execution order tracking array — utilized directly by downgrade() logic
PROFILE_ORDER: List[str] = ["SURVIVAL", "BALANCED", "PERFORMANCE", "PRO"]


# ---------------------------------------------------------------------------
# Automated Profile Detection Routing
# ---------------------------------------------------------------------------

def detect_profile(available_ram_gb: float) -> ExecutionProfile:
    """
    Evaluates system telemetry metrics to flag the most efficient operational profile.
    Always initializes matching calculations against upper thresholds — fallback retry loops downgrade if required.

    Args:
        available_ram_gb: Active unallocated physical system RAM tracked right now via psutil metrics.

    Returns:
        ExecutionProfile mapping matched structural processing configurations.
    """
    # Evaluate configurations from top to bottom — returning the highest boundary tier supported by host RAM
    for name in reversed(PROFILE_ORDER):
        profile = PROFILES[name]
        if available_ram_gb >= profile.min_ram_gb:
            logger.info(
                "Available capacity verified: %.1fGB → Resolved execution profile target: %s "
                "(cols/chunk=%d, partitions=%d)",
                available_ram_gb, profile.name,
                profile.cols_per_chunk, profile.n_partitions,
            )
            return profile

    # Absolute fallback barrier logic — theoretically unreachable due to SURVIVAL tracking configurations at 0.0
    logger.warning(
        "Available telemetry values (%.1fGB) registered below standard baselines. "
        "Routing task loops to SURVIVAL fallback settings.",
        available_ram_gb,
    )
    return PROFILES["SURVIVAL"]


def get_profile_by_name(name: str) -> ExecutionProfile:
    """
    Fetches an explicit execution profile directly matching target configuration name tags (CLI runtime routing).

    Args:
        name: Profile tag string: "SURVIVAL" | "BALANCED" | "PERFORMANCE" | "PRO"

    Raises:
        ValueError if string parameters break known configuration schemas.
    """
    name = name.upper().strip()
    if name not in PROFILES:
        raise ValueError(
            f"Requested configuration signature block '{name}' is not recognized. Options: {PROFILE_ORDER}"
        )
    return PROFILES[name]


# ---------------------------------------------------------------------------
# CLI Sandbox Smoke Testing Enclosures
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import psutil

    available = psutil.virtual_memory().available / (1024 ** 3)
    total     = psutil.virtual_memory().total     / (1024 ** 3)

    print(f"\nTotal physical memory footprint detected : {total:.1f}GB")
    print(f"Available unallocated capacity tracking : {available:.1f}GB")

    profile = detect_profile(available)
    print(f"\n{profile.summary()}")

    # Simulated local pipeline downgrade validation sequence loops
    print("\n── Validating Downgrade Chain Intersections ───────────────────────")
    current = profile
    while current:
        print(f"  {current.name}")
        current = current.downgrade()
