"""
src/utils/resources.py

Gestiona la infraestructura elástica del motor de procesamiento (Paso 3 del Framework).
Calcula y aplica la configuración óptima de Spark basándose en:
- Un snapshot determinista del hardware (RAM, CPUs, Plataforma).
- El modo de ejecución (local, yarn, k8s).
- Priorización de overrides manuales definidos en el archivo .env.

Garantiza la reproducibilidad mediante un sistema de 'fingerprinting' (SHA-256), 
asegurando que cada ejecución quede registrada con un ID único de infraestructura 
para el data lineage del AI_LOG.

Función principal: apply_to_spark_session — integra la inteligencia de recursos 
directamente en el ciclo de vida de la SparkSession.
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
#  Snapshot del entorno (base para reproducibilidad)
# ─────────────────────────────────────────────

def _capture_env_snapshot() -> dict:
    """
    Captura el estado determinista del entorno.
    Mismo entorno → mismo snapshot → misma config.
    """
    mem = psutil.virtual_memory()
    return {
        "total_ram_bytes":     mem.total,
        "available_ram_bytes": mem.available,
        "cpu_physical":        psutil.cpu_count(logical=False) or 1,
        "cpu_logical":         psutil.cpu_count(logical=True) or 1,
        "platform":            platform.system(),
        "override_env":        os.getenv("SPARK_EXECUTOR_MEMORY_OVERRIDE"),
        "mode_env":            os.getenv("SPARK_MODE"),
    }


def _snapshot_fingerprint(snapshot: dict) -> str:
    """
    Hash SHA-256 del snapshot.
    Fingerprint igual en dos máquinas → config idéntica.
    """
    canonical = json.dumps(snapshot, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────
#  Validación del override
# ─────────────────────────────────────────────

def _parse_memory_override(raw: str) -> str:
    """Valida y normaliza el override. Lanza ValueError si es inválido."""
    raw = raw.strip()
    if re.fullmatch(r"\d+(\.\d+)?[gGmM]", raw):
        return raw.lower()
    raise ValueError(
        f"SPARK_EXECUTOR_MEMORY_OVERRIDE='{raw}' inválido. "
        "Usa formato como '4g', '2.5g' o '512m'."
    )


# ─────────────────────────────────────────────
#  Cálculo de memoria — función pura
# ─────────────────────────────────────────────

def _calculate_memory(total_bytes: int, available_bytes: int, override_active: bool) -> float:
    """
    Función pura: misma entrada → misma salida (idempotente).

    Prioridad:
        1. override del .env — SOLO si cabe en la RAM disponible real
        2. inferencia por total_ram × 0.60
        3. inferencia por available_ram × 0.90 (si hay presión de memoria)

    Si el override pide más RAM de la que hay disponible, se ignora y se
    usa la inferencia — evita OOM de JVM aunque el .env diga otra cosa.
    """
    total_gb     = total_bytes     / (1024 ** 3)
    available_gb = available_bytes / (1024 ** 3)
    by_total     = round(total_gb * 0.60, 1)
    by_available = round(available_gb * 0.90, 1)

    if available_gb < by_total:
        recommended = min(by_total, by_available)
        if not override_active:
            logger.warning(
                "Presión de memoria: disponible=%.1fGB < calculado=%.1fGB. "
                "Ajustando a %.1fGB. Usa SPARK_EXECUTOR_MEMORY_OVERRIDE para fijar manualmente.",
                available_gb, by_total, recommended
            )
    else:
        recommended = by_total

    return max(round(recommended, 1), 2.0)


def _validate_override_fits(override_str: str, available_bytes: int) -> bool:
    """
    Verifica que el override del .env cabe en la RAM disponible real.
    Si no cabe, el override se ignora — la inferencia es más conservadora.

    Ejemplo: override=4g, available=1.8g → False → usar inferencia
    Ejemplo: override=2g, available=3.5g → True  → usar override
    """
    available_gb = available_bytes / (1024 ** 3)

    # Parsear el override a GB
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

    # El override debe dejar al menos 20% de RAM libre para el OS y otros procesos
    safe_limit_gb = available_gb * 0.85
    fits = requested_gb <= safe_limit_gb

    if not fits:
        logger.warning(
            "Override rechazado: .env pide %.1fgb pero solo hay %.1fgb disponibles "
            "(limite seguro=%.1fgb). Usando inferencia para evitar OOM.",
            requested_gb, available_gb, safe_limit_gb
        )

    return fits


# ─────────────────────────────────────────────
#  Builders por modo
# ─────────────────────────────────────────────

def _build_local_config(memory_gb: float, cores: int, memory_str: str) -> dict:
    """En local, solo driver.memory importa. executor.memory es ignorado por Spark."""
    usable_cores = max(cores - 1, 1)
    return {
        "spark.master":                 f"local[{usable_cores}]",
        "spark.driver.memory":          memory_str,
        "spark.executor.memory":        None,
        "spark.sql.shuffle.partitions": str(usable_cores * 2),
    }


def _build_yarn_config(memory_gb: float, cores: int, memory_str: str) -> dict:
    """YARN: divide memoria entre driver y executors."""
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
    """K8s: igual que YARN + overhead de contenedor (~10%)."""
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
#  Función principal
# ─────────────────────────────────────────────

def get_spark_memory_settings(mode: Optional[str] = None) -> dict:
    """
    Calcula la configuración óptima de Spark para el entorno actual.

    Idempotente:  misma entrada → mismo resultado siempre.
    Reproducible: fingerprint SHA-256 para trazabilidad en AI_LOG y lineage.

    Args:
        mode: 'local' | 'yarn' | 'k8s'. Prioridad: argumento > SPARK_MODE env > 'local'.

    Returns:
        dict con spark_config, meta y env_snapshot.
    """
    # 1. Snapshot determinista
    snapshot    = _capture_env_snapshot()
    fingerprint = _snapshot_fingerprint(snapshot)

    # 2. Resolver modo
    resolved_mode = (mode or snapshot["mode_env"] or "local").lower()
    if resolved_mode not in _MODE_BUILDERS:
        raise ValueError(
            f"Modo '{resolved_mode}' no soportado. Usa: {list(_MODE_BUILDERS)}"
        )

    # 3. Resolver override — validar que cabe en RAM disponible ANTES de aplicar
    override = snapshot["override_env"]
    if override:
        try:
            parsed_override = _parse_memory_override(override)
            # Verificar que el override realmente cabe — si no, usar inferencia
            if _validate_override_fits(parsed_override, snapshot["available_ram_bytes"]):
                memory_str       = parsed_override
                override_applied = True
            else:
                override_applied = False  # override rechazado, usar inferencia
        except ValueError as e:
            logger.warning("Override inválido ignorado: %s", e)
            override_applied = False
    else:
        override_applied = False

    # 4. Calcular memoria base por inferencia
    # override_active=False aquí siempre porque si el override fue aceptado
    # ya tenemos memory_str y no necesitamos el warning de presión
    memory_gb = _calculate_memory(
        snapshot["total_ram_bytes"],
        snapshot["available_ram_bytes"],
        override_active=override_applied,
    )

    # 5. Si override fue rechazado o no existe, usar la inferencia
    # Convertir a MB para evitar decimales invalidos en la JVM (2.0g falla, 2048m funciona)
    if not override_applied:
        memory_mb = int(memory_gb * 1024)
        memory_str = f"{memory_mb}m"

    # 6. Construir config del modo
    spark_config = _MODE_BUILDERS[resolved_mode](
        memory_gb,
        snapshot["cpu_physical"],
        memory_str,
    )

    # 7. Limpiar claves nulas
    spark_config = {k: v for k, v in spark_config.items() if v is not None}

    return {
        "spark_config": spark_config,
        "meta": {
            "fingerprint":      fingerprint,
            "mode":             resolved_mode,
            "override_applied": override_applied,
            "memory_used":      memory_str,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_ram_gb":     round(snapshot["total_ram_bytes"]     / (1024 ** 3), 2),
            "available_ram_gb": round(snapshot["available_ram_bytes"] / (1024 ** 3), 2),
            "cpu_physical":     snapshot["cpu_physical"],
            "platform":         snapshot["platform"],
        },
        "env_snapshot": snapshot,
    }


# ─────────────────────────────────────────────
#  Config especifica para writes pesados
# ─────────────────────────────────────────────

def get_write_config() -> dict:
    """
    Calcula config conservadora para writes pesados (Silver → Delta, Gold → Delta).
    Distinta de get_spark_memory_settings() que optimiza para procesamiento general.

    Estrategia:
    - Cores: max(1, cpu_fisicos // 3) — escala con el hardware sin saturar
      Con 6 cores → 2. Con 12 → 4. Con 2 → 1.
    - Off-heap: 15% de la RAM disponible — alivia presión del GC de JVM
      Con 2GB disponibles → 300MB. Con 6GB → 900MB. Max 2GB.
    - Shuffle partitions: igual al número de cores de write × 4
      Evita el default de 200 que genera demasiados archivos pequeños.

    Returns:
        dict con las config keys listas para SparkSession.builder.config()
    """
    mem          = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    cpu_physical = psutil.cpu_count(logical=False) or 1

    # Cores conservadores para write — no saturar la JVM
    write_cores = max(1, cpu_physical // 3)

    # Off-heap — 15% del disponible, mínimo 256MB, máximo 2GB
    off_heap_gb  = min(max(round(available_gb * 0.15, 1), 0.25), 2.0)
    off_heap_str = f"{int(off_heap_gb * 1024)}m" if off_heap_gb < 1 else f"{off_heap_gb}g"

    # Shuffle partitions calibradas para el write, no el procesamiento general
    shuffle_partitions = max(write_cores * 4, 10)

    config = {
        "spark.master":                      f"local[{write_cores}]",
        "spark.sql.shuffle.partitions":      str(shuffle_partitions),
        "spark.memory.offHeap.enabled":      "true",
        "spark.memory.offHeap.size":         off_heap_str,
    }

    logger.info(
        "write_config: cores=%d | off_heap=%s | shuffle_partitions=%d "
        "(available_ram=%.1fgb, cpu_physical=%d)",
        write_cores, off_heap_str, shuffle_partitions, available_gb, cpu_physical
    )

    return config


# ─────────────────────────────────────────────
#  Helper de integración con SparkSession
# ─────────────────────────────────────────────

def apply_to_spark_session(builder, mode: Optional[str] = None):
    """
    Aplica la config calculada directamente a un SparkSession.builder.

    Uso:
        builder = SparkSession.builder.appName("mi_app")
        spark = apply_to_spark_session(builder).getOrCreate()
    """
    result = get_spark_memory_settings(mode=mode)
    for key, value in result["spark_config"].items():
        builder = builder.config(key, value)
    return builder


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else None
    result   = get_spark_memory_settings(mode=mode_arg)

    print("\n── Spark Config ──────────────────────────")
    for k, v in result["spark_config"].items():
        print(f"  {k} = {v}")

    print("\n── Metadata ──────────────────────────────")
    for k, v in result["meta"].items():
        print(f"  {k}: {v}")