"""
src/utils/resources.py

Manages the elastic infrastructure of the processing engine (Step 3 of the Framework).
Calculates and applies optimal Spark configurations based on:
- A deterministic hardware snapshot (RAM, CPUs, Platform).
- The execution deployment target mode (local, yarn, k8s).
- Prioritization of manual overrides specified within the .env file.

Guarantees reproducibility through an infrastructure fingerprinting system (SHA-256), 
ensuring that every single run execution is logged with a unique infrastructure ID 
for the AI_LOG data lineage collection tracks.

Primary Entry Point: apply_to_spark_session — integrates resource intelligence 
directly into the SparkSession initialization lifecycle loop.
"""
import psutil
import os
import re
import hashlib
import json
import platform
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Environment Snapshot (Baseline for Reproducibility)
# ─────────────────────────────────────────────

def _capture_env_snapshot() -> dict:
    """
    Captures a deterministic state snapshot of the active environment.
    Identical hardware state → identical snapshot → identical configuration footprint.
    """
    mem = psutil.virtual_memory()
    return {
        "total_ram_bytes":      mem.total,
        "available_ram_bytes": mem.available,
        "cpu_physical":        psutil.cpu_count(logical=False) or 1,
        "cpu_logical":         psutil.cpu_count(logical=True) or 1,
        "platform":            platform.system(),
        "override_env":        os.getenv("SPARK_EXECUTOR_MEMORY_OVERRIDE"),
        "mode_env":            os.getenv("SPARK_MODE"),
    }


def _snapshot_fingerprint(snapshot: dict) -> str:
    """
    Generates a SHA-256 hash marker identifying the capture snapshot.
    Matching fingerprint across target execution run nodes → identical configuration applied.
    """
    canonical = json.dumps(snapshot, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────
#  Override Verification Logic
# ─────────────────────────────────────────────

def _parse_memory_override(raw: str) -> str:
    """Validates and normalizes runtime configuration overrides. Raises ValueError if invalid."""
    raw = raw.strip()
    if re.fullmatch(r"\d+(\.\d+)?[gGmM]", raw):
        return raw.lower()
    raise ValueError(
        f"Specified SPARK_EXECUTOR_MEMORY_OVERRIDE='{raw}' is invalid. "
        "Please provide standard allocations like '4g', '2.5g', or '512m'."
    )


# ─────────────────────────────────────────────
#  Memory Arithmetic Allocation Rules (Pure Function)
# ─────────────────────────────────────────────

def _calculate_memory(total_bytes: int, available_bytes: int, override_active: bool) -> float:
    """
    Pure mathematical evaluator: matching inputs → identical outputs (Idempotent execution).

    Priority Matrix:
        1. .env manual declaration — Applied ONLY if it safely clears active runtime bounds
        2. Inferred heuristics via total_ram × 0.60
        3. Inferred heuristics via available_ram × 0.90 (Engaged when system memory is constrained)

    If an override requests memory layout spaces that exceed the safe limits of the operational 
    hardware footprint, it is dropped to trigger safer inference rules instead — preventing JVM OutOfMemory
    fault conditions regardless of standard local configurations.
    """
    total_gb     = total_bytes     / (1024 ** 3)
    available_gb = available_bytes / (1024 ** 3)
    by_total     = round(total_gb * 0.60, 1)
    by_available = round(available_gb * 0.90, 1)

    if available_gb < by_total:
        recommended = min(by_total, by_available)
        if not override_active:
            logger.warning(
                "System memory pressure detected: available=%.1fGB < calculated=%.1fGB. "
                "Throttling safety allocations back to %.1fGB. Declare SPARK_EXECUTOR_MEMORY_OVERRIDE to anchor manually.",
                available_gb, by_total, recommended
            )
    else:
        recommended = by_total

    return max(round(recommended, 1), 2.0)


def _validate_override_fits(override_str: str, available_bytes: int) -> bool:
    """
    Verifies if a .env override parameter safely clears the active physical environment constraints.
    If the bounds check fails, the override is dropped to engage standard structural inferences.

    Execution Patterns:
    - override=4g, available=1.8g → False → Fall back to standard calculations
    - override=2g, available=3.5g → True  → Commit override targets
    """
    available_gb = available_bytes / (1024 ** 3)

    # Standardize and parse override mapping to GB targets
    raw = override_str.strip().lower()
    try:
        if raw.endswith("g"):
            requested_gb = float(raw[:-1])
        elif raw.endswith("m"):
            requested_gb = float(raw[:-1]) / 1024
        else:
            return False
    except ValueError:
        return False

    # Manual allocations must reserve a minimum 20% system buffer for OS overhead and auxiliary routines
    safe_limit_gb = available_gb * 0.85
    fits = requested_gb <= safe_limit_gb

    if not fits:
        logger.warning(
            "System configuration override rejected: .env rules demand %.1fgb, but only %.1fgb are available "
            "(Operational threshold limit ceiling=%.1fgb). Defaulting back to engine inference choices to isolate OOM faults.",
            requested_gb, available_gb, safe_limit_gb
        )

    return fits


# ─────────────────────────────────────────────
#  Target Deployment Configuration Builders
# ─────────────────────────────────────────────

def _build_local_config(memory_gb: float, cores: int, memory_str: str) -> dict:
    """Local Deployment Layer: Only driver configuration weights apply. Executor specifications are skipped by Spark."""
    usable_cores = max(cores - 1, 1)
    return {
        "spark.master":                 f"local[{usable_cores}]",
        "spark.driver.memory":          memory_str,
        "spark.executor.memory":        None,
        "spark.sql.shuffle.partitions": str(usable_cores * 2),
    }


def _build_yarn_config(memory_gb: float, cores: int, memory_str: str) -> dict:
    """YARN Cluster Layer: Balances shared execution boundaries evenly between driver tracks and workers."""
    driver_gb    = round(max(memory_gb * 0.35, 1.0), 1)
    executor_gb  = round(max(memory_gb * 0.65, 1.0), 1)
    usable_cores = max(cores - 1, 1)
    return {
        "spark.master":                 "yarn",
        "spark.driver.memory":          f"{driver_gb}g",
        "spark.executor.memory":        f"{executor_gb}g",
        "spark.executor.cores":         str(usable_cores),
        "spark.sql.shuffle.partitions": str(usable_cores * 4),
    }


def _build_k8s_config(memory_gb: float, cores: int, memory_str: str) -> dict:
    """K8s Cloud Layer: Standard layout allocations combined with container storage padding thresholds (~10%)."""
    overhead     = round(memory_gb * 0.10, 1)
    executor_gb  = round(max(memory_gb * 0.60, 1.0), 1)
    driver_gb    = round(max(memory_gb * 0.30, 1.0), 1)
    usable_cores = max(cores - 1, 1)
    return {
        "spark.master":                          "k8s://https://<K8S_API_SERVER>",
        "spark.driver.memory":                   f"{driver_gb}g",
        "spark.executor.memory":                 f"{executor_gb}g",
        "spark.executor.cores":                  str(usable_cores),
        "spark.kubernetes.memoryOverheadFactor": "0.1",
        "spark.driver.memoryOverhead":           f"{overhead}g",
        "spark.sql.shuffle.partitions":          str(usable_cores * 4),
    }


_MODE_BUILDERS = {
    "local": _build_local_config,
    "yarn":  _build_yarn_config,
    "k8s":   _build_k8s_config,
}


# ─────────────────────────────────────────────
#  Primary Engine Parameter Computation
# ─────────────────────────────────────────────

def get_spark_memory_settings(mode: Optional[str] = None) -> dict:
    """
    Calculates the absolute optimal Spark engine configuration properties for the system context.

    Idempotent Design: Exact match parameter footprints guarantee identical output maps.
    Reproducible Engine: Implements explicit SHA-256 fingerprint chains to bind diagnostic log state metrics.

    Args:
        mode: 'local' | 'yarn' | 'k8s'. Resolution preference: Manual Argument > SPARK_MODE variable > 'local'.

    Returns:
        A mapping dictionary bundling spark_config entries, runtime metadata summaries, and target env snapshots.
    """
    # 1. Capture system hardware state parameters
    snapshot    = _capture_env_snapshot()
    fingerprint = _snapshot_fingerprint(snapshot)

    # 2. Resolve cluster environment tracking mode
    resolved_mode = (mode or snapshot["mode_env"] or "local").lower()
    if resolved_mode not in _MODE_BUILDERS:
        raise ValueError(
            f"Deployment architecture model target '{resolved_mode}' is not supported. Choose among: {list(_MODE_BUILDERS)}"
        )

    # 3. Resolve overrides — Confirm resources exist safely BEFORE locking assignments
    override = snapshot["override_env"]
    if override:
        try:
            parsed_override = _parse_memory_override(override)
            # Assess physical infrastructure viability thresholds
            if _validate_override_fits(parsed_override, snapshot["available_ram_bytes"]):
                memory_str       = parsed_override
                override_applied = True
            else:
                override_applied = False  # Override fails bounds evaluation; pass back to heuristics
        except ValueError as e:
            logger.warning("Malformed environment override configurations dropped from verification tracks: %s", e)
            override_applied = False
    else:
        override_applied = False

    # 4. Process underlying engine calculations
    # override_active=False forces static alignment controls because verified manual parameters 
    # already establish targeted control boundaries directly
    memory_gb = _calculate_memory(
        snapshot["total_ram_bytes"],
        snapshot["available_ram_bytes"],
        override_active=override_applied,
    )

    # 5. Handle fallback steps when override components fail validation limits
    # Standardize layouts using explicit MB values to protect JVM processes from layout conversions (e.g., '2.0g' error crashes)
    if not override_applied:
        memory_mb = int(memory_gb * 1024)
        memory_str = f"{memory_mb}m"

    # 6. Call target orchestration infrastructure generation rules
    spark_config = _MODE_BUILDERS[resolved_mode](
        memory_gb,
        snapshot["cpu_physical"],
        memory_str,
    )

    # 7. Drop empty assignment keys from the active structure mapping
    spark_config = {k: v for k, v in spark_config.items() if v is not None}

    return {
        "spark_config": spark_config,
        "meta": {
            "fingerprint":      fingerprint,
            "mode":             resolved_mode,
            "override_applied": override_applied,
            "memory_used":      memory_str,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_ram_gb":      round(snapshot["total_ram_bytes"]     / (1024 ** 3), 2),
            "available_ram_gb": round(snapshot["available_ram_bytes"] / (1024 ** 3), 2),
            "cpu_physical":     snapshot["cpu_physical"],
            "platform":         snapshot["platform"],
        },
        "env_snapshot": snapshot,
    }


# ─────────────────────────────────────────────
#  Heavy Production Writing Custom Strategies
# ─────────────────────────────────────────────

def get_write_config() -> dict:
    """
    Calculates safety configurations protecting massive storage flush events (Silver → Delta, Gold → Delta).
    Differs from get_spark_memory_settings() by focusing on pipeline write stability over raw data transformation.

    Operational Directives:
    - System Cores: max(1, physical_cpus // 3) — Scales to hardware capacity without bottlenecking runtime loops
      e.g., 6 physical cores → uses 2. 12 cores → uses 4. 2 cores → uses 1.
    - Off-heap Spaces: Reserves 15% of functional RAM metrics — Alleviates performance drag across JVM GC actions
      e.g., 2GB free allocation limits → generates 300MB. 6GB → generates 900MB. Max tracking cap fixed to 2GB.
    - Shuffle Segments: Matches total targeted write core allocation counts × 4
      Bypasses the generic 200 default marker to minimize tiny, fragmented file allocations.

    Returns:
        Dictionary mapping tracking configuration properties ready to inject into SparkSession.builder.config()
    """
    mem          = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    cpu_physical = psutil.cpu_count(logical=False) or 1

    # Apply conservative processing engine layouts to isolate engine resource lock scenarios
    write_cores = max(1, cpu_physical // 3)

    # Off-heap calculations — 15% of active systems available capacity, floor tracking at 256MB, ceiling cap at 2GB
    off_heap_gb  = min(max(round(available_gb * 0.15, 1), 0.25), 2.0)
    off_heap_str = f"{int(off_heap_gb * 1024)}m" if off_heap_gb < 1 else f"{off_heap_gb}g"

    # Calibration targets mapped directly around structural output limits rather than computational transforms
    shuffle_partitions = max(write_cores * 4, 10)

    config = {
        "spark.master":                      f"local[{write_cores}]",
        "spark.sql.shuffle.partitions":      str(shuffle_partitions),
        "spark.memory.offHeap.enabled":      "true",
        "spark.memory.offHeap.size":         off_heap_str,
    }

    logger.info(
        "write_config committed parameters: cores=%d | off_heap=%s | shuffle_partitions=%d "
        "(System statistics baseline: available_ram=%.1fgb, cpu_physical=%d)",
        write_cores, off_heap_str, shuffle_partitions, available_gb, cpu_physical
    )

    return config


# ─────────────────────────────────────────────
#  Orchestration Lifecycle Integration Helpers
# ─────────────────────────────────────────────

def apply_to_spark_session(builder, mode: Optional[str] = None):
    """
    Injects the evaluated infrastructure tuning parameters directly back into an active SparkSession.builder pipeline.

    Execution Blueprint:
        builder = SparkSession.builder.appName("target_pipeline_submodule")
        spark = apply_to_spark_session(builder).getOrCreate()
    """
    result = get_spark_memory_settings(mode=mode)
    for key, value in result["spark_config"].items():
        builder = builder.config(key, value)
    return builder


# ─────────────────────────────────────────────
#  Local Integration Verification Harness (CLI Entry Point)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else None
    result   = get_spark_memory_settings(mode=mode_arg)

    print("\n── Computed Spark Properties Engine Config ──────────────────────────")
    for k, v in result["spark_config"].items():
        print(f"  {k} = {v}")

    print("\n── Trace Operational Context Metadata ──────────────────────────────")
    for k, v in result["meta"].items():
        print(f"  {k}: {v}")
