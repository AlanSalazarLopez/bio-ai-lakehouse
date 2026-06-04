"""
src/jobs/silver_tuner.py

Config tuner para Phase 1 — binary search sobre cols_per_chunk.

Corre sobre un sample pequeño del Bronze (N filas × todas las columnas)
para encontrar el cols_per_chunk óptimo ANTES de correr el reshape completo.

Estrategia:
    1. Arranca en cols_per_chunk = 1  (mínimo absoluto, garantizado)
    2. Sube en potencias de 2: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 → ...
    3. En cada intento: mide RAM pico real con thread monitor cada 200ms
    4. Si RAM pico > RAM_SAFETY_THRESHOLD: para, guarda el último N que funcionó
    5. Escribe optimal_config.json con la config ganadora

Output:
    data/staging/silver_tuner/
        optimal_config.json     ← Phase 1 lee esto si existe
        tuner_log.json          ← historial completo de intentos

Uso:
    docker exec -it --workdir /opt/spark/work-dir spark-master \\
        env PYTHONPATH=. python3 src/jobs/silver_tuner.py

    # Forzar re-tune aunque ya exista optimal_config.json:
    python3 src/jobs/silver_tuner.py --force

El tuner usa un sample de SAMPLE_ROWS filas — cada intento tarda segundos.
La config ganadora se usa en el reshape completo que tarda horas.
Se corre UNA vez por equipo. Si cambias de máquina, corre de nuevo.
"""

import argparse
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

# Usar el metadata_loader real — mismo strip(), mismo KeyError, idéntico al Phase 1.
# Si PYTHONPATH no está seteado (ej. correr fuera del contenedor), cae al fallback inline.
try:
    from src.utils.metadata_loader import load_tissue_mapping as _load_tissue_mapping_real
    _USE_REAL_LOADER = True
except ImportError:
    _USE_REAL_LOADER = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BRONZE_PATH     = "data/bronze/gtex/gene_tpm_raw.parquet"
METADATA_PATH   = "data/raw/gtex_metadata.txt"
TUNER_DIR       = "data/staging/silver_tuner"
OPTIMAL_CONFIG  = "data/staging/silver_tuner/optimal_config.json"
TUNER_LOG        = "data/staging/silver_tuner/tuner_log.json"
PROFILING_REPORT = "data/lineage/profiling_report.json"

METADATA_COLS   = ["Name", "Description"]
SAMPLE_ROWS     = 200          # filas del Bronze a usar — pequeño = intentos rápidos
MONITOR_HZ      = 0.2          # segundos entre lecturas del monitor de RAM
RAM_SAFETY_FRAC = 0.80         # parar si RAM pico > 80% de RAM disponible al inicio
MAX_COLS        = 500          # techo — no tiene sentido ir más allá

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monitor de RAM en thread separado
# ---------------------------------------------------------------------------

class RAMMonitor:
    """
    Thread que mide el DELTA de RAM usado respecto al baseline al momento
    de .start(). Mide cuánto sube la RAM, no cuánto está usada en total.

    Esto evita que el baseline del sistema (Docker + WSL2 + Windows) haga
    fallar el límite antes de que el reshape haga cualquier cosa.

    El pico se lee con .peak_delta_mb — incremento máximo sobre el baseline.
    """
    def __init__(self):
        self._stop_event  = threading.Event()
        self._baseline    = 0
        self._peak_delta  = 0
        self._thread      = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop_event.clear()
        self._baseline   = psutil.virtual_memory().used
        self._peak_delta = 0
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop_event.is_set():
            delta = psutil.virtual_memory().used - self._baseline
            if delta > self._peak_delta:
                self._peak_delta = delta
            time.sleep(MONITOR_HZ)

    @property
    def peak_mb(self) -> float:
        """Delta máximo de RAM sobre el baseline — cuánto subió el proceso."""
        return max(0, self._peak_delta) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Datos sintéticos desde profiling_report.json — cero contacto con Bronze
# ---------------------------------------------------------------------------

def build_synthetic_sample(
    profiling_path: str,
    n_rows: int,
    n_cols: int,
) -> tuple:
    """
    Genera una pa.Table sintética y un tissue_mapping sintético
    100% desde profiling_report.json — sin tocar el parquet de 19k columnas.

    Estructura idéntica al Bronze real:
        - Columnas "Name" y "Description" (gene metadata)
        - n_cols columnas de samples con formato GTEX-SYNTH-{i:04d}-SM-TUNER
        - float32 con distribución realista (51% zeros, resto positivos)

    El tissue_mapping asigna los sample IDs sintéticos a tejidos reales
    extraídos del profiling_report, con la misma distribución de skew.

    Args:
        profiling_path: ruta al profiling_report.json del Paso 4
        n_rows:         filas a generar (default 200)
        n_cols:         columnas de samples a generar (default 500)

    Returns:
        (sample_table, sample_cols, tissue_mapping, total_real_cols)
    """
    import random

    log.info(f"Cargando profiling report desde: {profiling_path}")
    with open(profiling_path) as f:
        report = json.load(f)

    total_real_cols = report["bronze_dimensions"]["n_sample_cols"]  # 19,788
    total_real_rows = report["bronze_dimensions"]["n_rows"]          # 74,628

    # ── Tejidos reales del profiling ──────────────────────────────────────────
    top_tissues    = report["metadata_profile"]["top_10_tissues"]
    bottom_tissues = report["metadata_profile"]["bottom_5_tissues"]
    all_tissues    = {**top_tissues, **bottom_tissues}
    tissue_names   = list(all_tissues.keys())

    # ── Sample IDs sintéticos con formato GTEx real ───────────────────────────
    sample_cols = [f"GTEX-SYNTH-{i:04d}-SM-TUNER" for i in range(n_cols)]

    # ── Tissue mapping sintético — distribuir por tejido proporcional al skew ─
    # Replica el skew real: Whole Blood >> Liver - Portal Tract
    total_weight  = sum(all_tissues.values())
    tissue_mapping = {}
    col_idx = 0
    for tissue, count in all_tissues.items():
        # cuántas cols le tocan proporcionalmente a este tejido
        n_for_tissue = max(1, round(n_cols * count / total_weight))
        for _ in range(n_for_tissue):
            if col_idx >= n_cols:
                break
            tissue_mapping[sample_cols[col_idx]] = tissue
            col_idx += 1
    # el resto va al tejido más grande si sobran
    while col_idx < n_cols:
        tissue_mapping[sample_cols[col_idx]] = tissue_names[0]
        col_idx += 1

    # ── Gene IDs y symbols sintéticos ────────────────────────────────────────
    gene_ids     = [f"ENSG{i:011d}.1" for i in range(n_rows)]
    gene_symbols = [f"GENE{i:05d}"    for i in range(n_rows)]

    # ── TPM values sintéticos — 51% zeros, resto float32 positivos ───────────
    rng = random.Random(42)  # seed fijo para reproducibilidad
    zero_pct = report["null_zero_profile"]["zero_pct"] / 100.0  # 0.5189

    cols_data = {"Name": pa.array(gene_ids, type=pa.string()),
                 "Description": pa.array(gene_symbols, type=pa.string())}

    for col in sample_cols:
        values = []
        for _ in range(n_rows):
            if rng.random() < zero_pct:
                values.append(0.0)
            else:
                # distribución log-normal típica de TPM
                values.append(float(rng.lognormvariate(1.5, 2.0)))
        cols_data[col] = pa.array(values, type=pa.float32())

    sample_table = pa.table(cols_data)

    log.info(
        f"Sample sintético generado: {n_rows} filas × {n_cols} cols "
        f"({len(tissue_mapping)} muestras mapeadas a {len(set(tissue_mapping.values()))} tejidos)"
    )
    log.info(f"Total cols reales del Bronze: {total_real_cols:,} (usado para estimación de tiempo)")

    return sample_table, sample_cols, tissue_mapping, total_real_cols, total_real_rows

# ---------------------------------------------------------------------------
# Reshape mínimo para benchmark (sin escribir a disco)
# ---------------------------------------------------------------------------

def _load_tissue_mapping(path: str) -> dict:
    """
    Delega al metadata_loader.py real si está disponible (mismo strip(),
    mismo KeyError, comportamiento idéntico al Phase 1).
    Fallback inline solo si el import falló — ej. correr fuera del contenedor.
    """
    if _USE_REAL_LOADER:
        return _load_tissue_mapping_real(path=path)

    # Fallback inline — solo si src.utils no está en PYTHONPATH
    log.warning("metadata_loader no disponible — usando fallback inline (solo para debug)")
    mapping = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            if "SAMPID" not in header or "SMTSD" not in header:
                raise KeyError(f"SAMPID o SMTSD no encontrados en header: {header[:10]}")
            sid_idx = header.index("SAMPID")
            tid_idx = header.index("SMTSD")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(sid_idx, tid_idx):
                    continue
                sample_id = parts[sid_idx].strip()
                tissue_id = parts[tid_idx].strip()
                if sample_id and tissue_id:
                    mapping[sample_id] = tissue_id
    except Exception as e:
        log.warning(f"Fallback tissue mapping falló: {e} — mapping vacío")
    return mapping


def run_reshape_sample(
    sample_table: pa.Table,
    sample_cols: list,
    tissue_mapping: dict,
    cols_per_chunk: int,
) -> None:
    """
    Hace el reshape wide→long sobre sample_table con el cols_per_chunk dado.
    Replica la lógica del fix de Phase 1: una columna a la vez, sin concat,
    con un ParquetWriter en memoria por tissue_id (usando BytesIO, sin tocar disco).

    El pico de RAM aquí es: gene_ids + gene_symbols + una sola tabla de col
    (~40MB con 200 filas, escalado a 74k filas sigue siendo bajo).
    """
    import io

    gene_ids     = sample_table.column("Name")
    gene_symbols = sample_table.column("Description")
    n_genes      = len(gene_ids)

    SILVER_SCHEMA_NO_TISSUE = pa.schema([
        pa.field("gene_id",     pa.string(),  nullable=False),
        pa.field("gene_symbol", pa.string(),  nullable=False),
        pa.field("sample_id",   pa.string(),  nullable=False),
        pa.field("tpm_value",   pa.float32(), nullable=False),
    ])

    chunks = [
        sample_cols[i : i + cols_per_chunk]
        for i in range(0, len(sample_cols), cols_per_chunk)
    ]

    for chunk in chunks:
        # Un writer en memoria por tissue_id — replica lo que hace Phase 1 en disco
        writers: dict = {}
        buffers: dict = {}

        try:
            for col in chunk:
                if col not in sample_table.schema.names:
                    continue

                tpm       = sample_table.column(col).cast(pa.float32())
                tissue_id = tissue_mapping.get(col, "unknown")

                single = pa.table(
                    {
                        "gene_id":     gene_ids,
                        "gene_symbol": gene_symbols,
                        "sample_id":   pa.array([col] * n_genes, type=pa.string()),
                        "tpm_value":   tpm,
                    },
                    schema=SILVER_SCHEMA_NO_TISSUE,  # nullable=False explícito
                )

                if tissue_id not in writers:
                    buf = io.BytesIO()
                    buffers[tissue_id] = buf
                    writers[tissue_id] = pq.ParquetWriter(
                        buf, schema=SILVER_SCHEMA_NO_TISSUE, compression="snappy"
                    )

                writers[tissue_id].write_table(single)
                del single, tpm

        finally:
            for w in writers.values():
                w.close()
            # Liberar buffers en memoria — en Phase 1 real estos van a disco
            buffers.clear()
            writers.clear()


# ---------------------------------------------------------------------------
# Un intento del tuner
# ---------------------------------------------------------------------------

def attempt(
    sample_table: pa.Table,
    sample_cols: list,
    tissue_mapping: dict,
    cols_per_chunk: int,
    ram_limit_mb: float,
) -> dict:
    """
    Corre un reshape sobre el sample con cols_per_chunk dado.
    Retorna dict con resultados del intento.
    """
    monitor = RAMMonitor()
    t0 = time.time()
    success = True
    error_msg = None

    monitor.start()
    try:
        run_reshape_sample(sample_table, sample_cols, tissue_mapping, cols_per_chunk)
    except MemoryError as e:
        success = False
        error_msg = f"MemoryError: {e}"
    except Exception as e:
        success = False
        error_msg = f"{type(e).__name__}: {e}"
    finally:
        monitor.stop()

    elapsed = time.time() - t0
    peak_mb = monitor.peak_mb

    # También falla si se pasó del límite aunque no tronó
    if success and peak_mb > ram_limit_mb:
        success = False
        error_msg = f"RAM pico ({peak_mb:.0f}MB) > límite seguro ({ram_limit_mb:.0f}MB)"

    return {
        "cols_per_chunk": cols_per_chunk,
        "success":        success,
        "peak_ram_mb":    round(peak_mb, 1),
        "elapsed_s":      round(elapsed, 2),
        "error":          error_msg,
    }


# ---------------------------------------------------------------------------
# Estimar tiempo total del reshape completo
# ---------------------------------------------------------------------------

def estimate_total_minutes(
    elapsed_sample_s: float,
    sample_rows: int,
    total_rows: int,
    sample_cols_used: int,
    total_cols: int,
) -> float:
    """
    Escala el tiempo del sample al reshape completo.
    elapsed_sample_s / (sample_rows * sample_cols_used) * (total_rows * total_cols)
    """
    sample_work  = sample_rows * sample_cols_used
    total_work   = total_rows  * total_cols
    if sample_work == 0:
        return 0.0
    return (elapsed_sample_s / sample_work * total_work) / 60.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tunear aunque ya exista optimal_config.json",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=SAMPLE_ROWS,
        help=f"Filas del sample sintético (default: {SAMPLE_ROWS})",
    )
    parser.add_argument(
        "--sample-cols",
        type=int,
        default=500,
        help="Columnas del sample sintético (default: 500)",
    )
    args = parser.parse_args()

    Path(TUNER_DIR).mkdir(parents=True, exist_ok=True)

    # ── Ya existe config y no se forzó re-tune ──────────────────────────────
    if Path(OPTIMAL_CONFIG).exists() and not args.force:
        with open(OPTIMAL_CONFIG) as f:
            cfg = json.load(f)
        log.info("=" * 60)
        log.info("Config óptima ya existe — usando la guardada")
        log.info(f"  cols_per_chunk      : {cfg['cols_per_chunk']}")
        log.info(f"  RAM pico (sample)   : {cfg['peak_ram_mb']} MB")
        log.info(f"  Tiempo estimado     : {cfg['estimated_minutes']:.1f} min")
        log.info(f"  Para re-tunear      : --force")
        log.info("=" * 60)
        return cfg

    # ── Setup ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Silver Tuner — binary search sobre cols_per_chunk")
    log.info("Modo: 100%% sintético — cero contacto con el Bronze de 19k cols")
    log.info("=" * 60)

    mem              = psutil.virtual_memory()
    available_ram_mb = mem.available / (1024 ** 2)
    total_ram_mb     = mem.total    / (1024 ** 2)
    used_ram_mb      = mem.used     / (1024 ** 2)
    # Límite = cuánto RAM EXTRA puede consumir el reshape sobre el baseline
    # Usamos el disponible × safety para no crashear el sistema
    ram_limit_mb     = available_ram_mb * RAM_SAFETY_FRAC
    log.info(f"RAM total       : {total_ram_mb:.0f} MB")
    log.info(f"RAM usada ahora : {used_ram_mb:.0f} MB (baseline del sistema)")
    log.info(f"RAM disponible  : {available_ram_mb:.0f} MB")
    log.info(f"Límite delta    : {ram_limit_mb:.0f} MB extra que puede usar el reshape ({RAM_SAFETY_FRAC*100:.0f}% del disponible)")

    # ── Generar sample sintético desde profiling_report.json ─────────────────
    # Sin tocar el parquet Bronze — cero riesgo de crash en el setup
    n_rows      = args.sample_rows
    n_cols      = args.sample_cols
    sample_table, sample_cols, tissue_mapping, total_cols, total_rows = build_synthetic_sample(
        profiling_path = PROFILING_REPORT,
        n_rows         = n_rows,
        n_cols         = n_cols,
    )

    # ── Binary search ascendente ─────────────────────────────────────────────
    log.info("-" * 60)
    log.info("Iniciando binary search...")
    log.info("  Secuencia: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 → ...")
    log.info("-" * 60)

    results    = []
    best       = None          # último intento exitoso
    candidates = []

    # Generar secuencia de potencias de 2 hasta MAX_COLS
    n = 1
    while n <= MAX_COLS:
        candidates.append(n)
        n *= 2

    for cols in candidates:
        log.info(f"Intentando cols_per_chunk = {cols}...")
        result = attempt(sample_table, sample_cols, tissue_mapping, cols, ram_limit_mb)
        results.append(result)

        status = "✅ OK" if result["success"] else "❌ FAIL"
        log.info(
            f"  {status} | RAM pico: {result['peak_ram_mb']:.0f}MB "
            f"| tiempo: {result['elapsed_s']:.2f}s "
            f"| {'error: ' + result['error'] if result['error'] else ''}"
        )

        if result["success"]:
            best = result
        else:
            log.info(f"  → Límite encontrado en cols_per_chunk = {cols}")
            log.info(f"  → Config óptima: cols_per_chunk = {best['cols_per_chunk'] if best else 1}")
            break

    # Si todos pasaron, el ganador es el más grande probado
    if best is None:
        log.warning("Incluso cols_per_chunk=1 falló — hay un problema de entorno")
        best = {"cols_per_chunk": 1, "peak_ram_mb": 0, "elapsed_s": 0}

    # ── Estimar tiempo total ──────────────────────────────────────────────────
    # total_rows y total_cols vienen del profiling_report (74,628 y 19,788)
    est_minutes = estimate_total_minutes(
        elapsed_sample_s  = best["elapsed_s"],
        sample_rows       = n_rows,
        total_rows        = total_rows,
        sample_cols_used  = min(best["cols_per_chunk"], n_cols),
        total_cols        = total_cols,
    )

    # ── Guardar resultados ────────────────────────────────────────────────────
    optimal = {
        "cols_per_chunk":      best["cols_per_chunk"],
        "peak_ram_mb":         best["peak_ram_mb"],
        "ram_limit_mb":        round(ram_limit_mb, 1),
        "estimated_minutes":   round(est_minutes, 1),
        "sample_rows_used":    n_rows,
        "sample_cols_used":    n_cols,
        "total_real_cols":     total_cols,
        "total_real_rows":     total_rows,
        "synthetic_sample":    True,
        "tuned_at":            time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OPTIMAL_CONFIG, "w") as f:
        json.dump(optimal, f, indent=2)

    with open(TUNER_LOG, "w") as f:
        json.dump({"attempts": results, "optimal": optimal}, f, indent=2)

    # ── Resumen ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RESULTADO DEL TUNER")
    log.info(f"  cols_per_chunk óptimo : {optimal['cols_per_chunk']}")
    log.info(f"  RAM pico (sample)     : {optimal['peak_ram_mb']:.0f} MB")
    log.info(f"  Tiempo estimado total : {optimal['estimated_minutes']:.1f} min")
    log.info(f"  Config guardada en    : {OPTIMAL_CONFIG}")
    log.info("=" * 60)
    log.info("Siguiente paso:")
    log.info("  python3 src/jobs/silver_phase1_reshape.py")
    log.info("  (Phase 1 leerá optimal_config.json automáticamente)")
    log.info("=" * 60)

    return optimal


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("Error no manejado en silver_tuner.py")
        traceback.print_exc()
        raise SystemExit(2)