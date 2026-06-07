"""
src/utils/chunk_calculator.py

Calculates the optimal number of Spark partitions based on:
- The size of the file to be processed
- The available RAM (inferred or from the .env override)

Safety arbiter: if the .env override requests more RAM than what is
actually available, it utilizes the inference to prevent OOM.

Pure function — independent of Spark, fully testable without a cluster.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

# In-memory vs on-disk expansion factor:
# Compressed Parquet occupies ~3-4x more space in RAM when deserialized
PARQUET_EXPANSION_FACTOR = 3.5

# Spark overhead factor per partition (shuffle buffers, metadata, etc.)
SPARK_OVERHEAD_FACTOR = 0.75  # We reserve exactly 75% of assigned RAM for data

# Absolute minimum RAM required to operate (Phase 3 Framework)
MIN_RAM_GB = 2.0

# Minimum core baseline required to operate
MIN_CORES = 1

# Target processing allocation size per partition in GB — smaller chunks = safer processing
TARGET_PARTITION_GB = 0.25  # 256MB allocated per partition


# ─────────────────────────────────────────────
#  Calculation Artifact Outputs
# ─────────────────────────────────────────────

@dataclass
class ChunkPlan:
    """
    Calculated execution plan mapped for the Spark job targets.
    All attributes are strictly deterministic given identical input fields.

    cols_per_chunk applies exclusively when total_cols is supplied to
    calculate_chunk_plan — otherwise defaults to -1 (Not Applicable).
    """
    partitions:          int    # Target partition counts passed to repartition()
    safe_memory_gb:      float  # Safe system RAM thresholds allocated to the job
    memory_source:       str    # Origin source profile: 'override' | 'inferred' | 'conservative'
    cores:               int    # Active hardware core execution slices bound to operations
    estimated_minutes:   float  # Projected end-to-end execution runtime calculations
    override_rejected:   bool   # Flag set to True if safety rules drop env overrides
    rejection_reason:    str    # Micro-logs capturing override drop triggers if applicable
    cols_per_chunk:      int    # Target matrix samples chunked per Pandas iteration (-1 if N/A)

    def summary(self) -> str:
        lines = [
            "── Chunk Plan Summary ────────────────────",
            f"  partitions       : {self.partitions}",
            f"  safe memory      : {self.safe_memory_gb}g",
            f"  memory source    : {self.memory_source}",
            f"  cores allocated  : {self.cores}",
            f"  estimated time   : ~{self.estimated_minutes:.1f} min",
        ]
        if self.cols_per_chunk > 0:
            lines.append(f"  cols/chunk       : {self.cols_per_chunk}")
        if self.override_rejected:
            lines.append(f"  ⚠️  override rejected: {self.rejection_reason}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Core Processing Logic — Pure Function
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
    Calculates the optimal partition execution strategy to transform dataset structures via Spark.

    Safety Arbiter Rules:
    - If the environment override exceeds actual active capacity limits → Dropped, applying inference targets.
    - If calculated inference profiles step past active availability bounds → Falls back to safety marks (available * 0.85).
    - Enforces physical engineering minimums via MIN_RAM_GB and MIN_CORES variables automatically.

    Args:
        file_size_bytes:     Total source file size footprint measured in bytes.
        available_ram_gb:    Active unallocated hardware RAM capacity on the host node right now.
        total_ram_gb:        Total overall physical memory capacity bound to the system.
        override_memory_str: Value captured from SPARK_EXECUTOR_MEMORY_OVERRIDE settings ('4g', '512m', None).
        inferred_memory_gb:  Calculated safety memory target mapped by resources.py (60% baseline limits).
        cpu_physical:        Active physical CPU computing cores available inside the environment.
        total_cols:          Total column dimensions inside wide target datasets
                             (Optional — triggers cols_per_chunk logic if supplied).

    Returns:
        ChunkPlan instance populated with safe execution boundaries, partitions, and timing scales.
    """
    file_size_gb     = file_size_bytes / (1024 ** 3)
    expanded_size_gb = file_size_gb * PARQUET_EXPANSION_FACTOR

    # ── 1. Resolve Safe Memory Allocation Schemes ─────────────────────────────
    override_rejected = False
    rejection_reason  = ""

    if override_memory_str:
    override_gb = _parse_memory_to_gb(override_memory_str)

    if override_gb > available_ram_gb:
        # Override metrics step beyond system capacities → OOM Risk
        override_rejected = True
        rejection_reason  = (
            f"Override={override_gb}g > available={available_ram_gb:.1f}g. "
            "Switching to inferred configurations to mitigate OOM threats."
        )
        logger.warning(rejection_reason)

        # Fall back and evaluate system inference vectors safely
        if inferred_memory_gb <= available_ram_gb:
            safe_memory_gb = inferred_memory_gb
            memory_source  = "inferred"
        else:
            # System inferences breach limits too → Restrict allocation safely to 85% of active headroom
            safe_memory_gb = round(available_ram_gb * 0.85, 1)
            memory_source  = "conservative"
            logger.warning(
                "Inferred memory targets (%.1fg) exceed active unallocated headroom (%.1fg). "
                "Applying conservative constraints: %.1fg",
                inferred_memory_gb, available_ram_gb, safe_memory_gb
            )
    else:
        # Override configurations fall within safe bounds
        safe_memory_gb = override_gb
        memory_source  = "override"

    elif inferred_memory_gb <= available_ram_gb:
        safe_memory_gb = inferred_memory_gb
        memory_source  = "inferred"
    else:
        # Absence of custom configurations combined with boundary breaches triggers strict conservative mode
        safe_memory_gb = round(available_ram_gb * 0.85, 1)
        memory_source  = "conservative"
        logger.warning(
            "Inferred RAM configuration targets (%.1fg) exceed active unallocated headroom (%.1fg). "
            "Applying conservative constraints: %.1fg",
            inferred_memory_gb, available_ram_gb, safe_memory_gb
        )

    # Protect operational baselines against sub-minimum limits
    safe_memory_gb = max(safe_memory_gb, MIN_RAM_GB)
    safe_memory_gb = math.floor(safe_memory_gb)

    # ── 2. Determine Optimal Spark Partition Splits ───────────────────────────
    # Calculate usable capacity blocks remaining for data matrices post Spark system overheads
    usable_ram_gb = safe_memory_gb * SPARK_OVERHEAD_FACTOR

    # Memory allocation scales: track how many data matrix steps fit across usable limits
    partitions_by_ram = math.ceil(expanded_size_gb / usable_ram_gb)

    # Volume layout allocation scales: partition target sets into chunks of TARGET_PARTITION_GB size
    partitions_by_size = math.ceil(file_size_gb / TARGET_PARTITION_GB)

    # Pick the absolute maximum — higher partition steps equal smaller execution footprints, lowering OOM threats
    partitions = max(partitions_by_ram, partitions_by_size, 1)

    # Factor layout splits to the nearest CPU thread coefficient to guarantee maximized compute utilization
    usable_cores = max(cpu_physical - 1, MIN_CORES)
    partitions   = math.ceil(partitions / usable_cores) * usable_cores

    # ── 3. Calculate Projected System Runtime Scales ──────────────────────────
    # Expected local processing speeds: roughly ~0.5 GB/min computed per execution core (Parquet + Snappy)
    throughput_gb_per_min = 0.5 * usable_cores
    estimated_minutes     = round(expanded_size_gb / throughput_gb_per_min, 1)

    # ── 4. Evaluate Matrix Column Chunks (Executed exclusively if total_cols is present) ──
    cols_per_chunk = _calculate_cols_per_chunk(
        file_size_bytes = file_size_bytes,
        total_cols      = total_cols,
        safe_memory_gb  = safe_memory_gb,
    )

    logger.info(
        "ChunkPlan calculated: partitions=%d, safe_memory=%.1fg, source=%s, ~%.1f min%s",
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
#  Local Utility Workers
# ─────────────────────────────────────────────

def _calculate_cols_per_chunk(
    file_size_bytes: int,
    total_cols:      Optional[int],
    safe_memory_gb:  float,
) -> int:
    """
    Calculates total sample column structures that fit safely across RAM limits per local Pandas loop execution pass
    (wide→long matrix operations). Yields -1 if total_cols parameter maps are missing.

    Formulaic approach:
       bytes_per_col   = file_size_bytes / total_cols
       ram_for_pandas  = safe_memory_gb * SPARK_OVERHEAD_FACTOR * 0.5
           ↑ Python engine runtimes share resources with JVM limits — Pandas claims half the usable limits
       cols_in_ram     = ram_for_pandas (bytes) / (bytes_per_col * PARQUET_EXPANSION_FACTOR)

    Applies hardcoded tracking cap ceilings between 50 and 100 to mitigate pipeline variance anomalies.
    """
    if not total_cols or total_cols <= 0:
        return -1

    bytes_per_col    = file_size_bytes / total_cols
    # Pandas partitions share active memory structures with underlying Spark jobs — locks to half of available space
    pandas_ram_bytes = safe_memory_gb * SPARK_OVERHEAD_FACTOR * 0.5 * (1024 ** 3)
    cols_in_ram      = pandas_ram_bytes / (bytes_per_col * PARQUET_EXPANSION_FACTOR)

    cols_per_chunk = math.floor(cols_in_ram)
    cols_per_chunk = max(cols_per_chunk, 50)   # Floor threshold limit: 50 cols/chunk minimum
    cols_per_chunk = min(cols_per_chunk, 100)  # Ceiling tracking cap: 100 cols/chunk maximum

    return cols_per_chunk


def _parse_memory_to_gb(memory_str: str) -> float:
    """Converts standard operational environment allocation flags (e.g., '4g', '512m') into float representations of GB."""
    memory_str = memory_str.strip().lower()
    if memory_str.endswith("g"):
        return float(memory_str[:-1])
    if memory_str.endswith("m"):
        return float(memory_str[:-1]) / 1024
    raise ValueError(f"Unrecognized structural memory assignment layout syntax: '{memory_str}'")


# ─────────────────────────────────────────────
#  Isolated Command Line Interface Sandboxing (Run without Spark dependencies)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.utils.resources import get_spark_memory_settings
    import psutil

    # Run validations using actual operational path settings if discovered on host architecture
    file_path = "data/raw/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet"

    if not os.path.exists(file_path):
        print(f"Target target production file entity not resolved at endpoint: {file_path}")
        print("Simulating calculations against a default 4.02 GB mock dataset profile...")
        file_size_bytes = int(4.02 * (1024 ** 3))
    else:
        file_size_bytes = os.path.getsize(file_path)

    infra = get_spark_memory_settings(mode="local")
    mem   = psutil.virtual_memory()

    plan = calculate_chunk_plan(
        file_size_bytes      = file_size_bytes,
        available_ram_gb     = mem.available / (1024 ** 3),
        total_ram_gb         = mem.total      / (1024 ** 3),
        override_memory_str  = infra["env_snapshot"]["override_env"],
        inferred_memory_gb   = float(infra["meta"]["memory_used"].replace("g", "")),
        cpu_physical         = infra["meta"]["cpu_physical"],
        total_cols           = 19_788,  # Evaluated GTEx core tissue column size constants
    )

    print(plan.summary())
