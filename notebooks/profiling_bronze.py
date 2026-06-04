"""
notebooks/profiling_bronze.py — Paso 4 del Data Engineering Decision Framework
GTEx v11 · Bronze profiling + metadata analysis

Responde estas preguntas del framework:
  - Factor de multiplicación real del reshape (wide → long)
  - Distribución de nulls vs zeros en columnas de expresión
  - Skew por tejido (columna de partición candidata)
  - Naturaleza de los zeros en TPM
  - Sample representativo generado

Output: data/lineage/profiling_report.json

Corre DENTRO del contenedor spark-master:
    docker exec -it --workdir /opt/spark/work-dir spark-master \
        env PYTHONPATH=. python3 notebooks/profiling_bronze.py
"""

import os
import sys
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

# ── paths ──────────────────────────────────────────────────────────────────
BRONZE_PATH   = "data/bronze/gtex/gene_tpm_raw.parquet"
METADATA_PATH = "data/raw/gtex_metadata.txt"
LINEAGE_PATH  = "data/lineage/profiling_report.json"
BRONZE_META   = "data/lineage/bronze_metadata.json"

# ── constantes de profiling ────────────────────────────────────────────────
SAMPLE_COLS          = 200   # columnas de expresión a muestrear para null/zero analysis
PARQUET_EXPANSION    = 3.5   # factor de expansión en RAM vs disco
SILVER_COLS_FIXED    = 3     # gene_id, sample_id, tpm_value
MIN_RAM_GB           = 2.0


# ══════════════════════════════════════════════════════════════════════════
#  SECCIÓN 1 — Metadata de muestras (sin Spark, pandas puro)
# ══════════════════════════════════════════════════════════════════════════

def profile_metadata(metadata_path: str) -> Dict:
    """
    Lee el archivo SampleAttributesDS y calcula:
    - Conteo de muestras por tejido (SMTSD)
    - Skew: ratio max/min entre tejidos
    - Número de tejidos únicos
    - Recomendación de partición
    """
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas no disponible. Instalar: pip install pandas")
        sys.exit(1)

    if not os.path.exists(metadata_path):
        print(f"❌ Metadata no encontrada: {metadata_path}")
        print("   Descarga con: wget -O data/raw/gtex_metadata.txt <url>")
        sys.exit(1)

    print("\n── [1/4] Analizando metadatos de muestras ────────────────────")

    df = pd.read_csv(metadata_path, sep="\t", usecols=["SAMPID", "SMTSD"], low_memory=False)
    df = df.dropna(subset=["SMTSD"])

    tissue_counts = df["SMTSD"].value_counts()
    total_samples = len(df)
    n_tissues     = df["SMTSD"].nunique()

    max_count = int(tissue_counts.iloc[0])
    min_count = int(tissue_counts.iloc[-1])
    skew_ratio = round(max_count / min_count, 1) if min_count > 0 else 999

    print(f"  Total muestras   : {total_samples:,}")
    print(f"  Tejidos únicos   : {n_tissues}")
    print(f"  Tejido mayor     : {tissue_counts.index[0]} ({max_count:,} muestras)")
    print(f"  Tejido menor     : {tissue_counts.index[-1]} ({min_count:,} muestras)")
    print(f"  Skew ratio       : {skew_ratio}x (max/min)")

    if skew_ratio > 10:
        skew_level = "SEVERO"
        skew_note  = "Considerar salting o partición por chromosome como alternativa"
    elif skew_ratio > 5:
        skew_level = "MODERADO"
        skew_note  = "tissue_id como partición es viable con repartition() explícito"
    else:
        skew_level = "BAJO"
        skew_note  = "tissue_id como partición es óptimo"

    print(f"  Nivel de skew    : {skew_level} → {skew_note}")

    # Top 10 y bottom 5 para el reporte
    top_tissues = {k: int(v) for k, v in tissue_counts.head(10).items()}
    bot_tissues = {k: int(v) for k, v in tissue_counts.tail(5).items()}

    return {
        "total_samples":        total_samples,
        "n_tissues":            n_tissues,
        "skew_ratio":           skew_ratio,
        "skew_level":           skew_level,
        "skew_note":            skew_note,
        "max_tissue":           tissue_counts.index[0],
        "max_tissue_count":     max_count,
        "min_tissue":           tissue_counts.index[-1],
        "min_tissue_count":     min_count,
        "top_10_tissues":       top_tissues,
        "bottom_5_tissues":     bot_tissues,
        "partition_recommendation": "tissue_id (SMTSD)",
    }


# ══════════════════════════════════════════════════════════════════════════
#  SECCIÓN 2 — Dimensiones exactas del Bronze (Spark liviano)
# ══════════════════════════════════════════════════════════════════════════

def profile_bronze_dimensions(bronze_path: str, infra: Dict) -> Dict:
    """
    Lee el parquet de Bronze con Spark para obtener:
    - Row count exacto
    - Column count exacto
    - Lista de columnas de expresión (sample IDs)
    - Columnas de metadata (gene_id, Description, etc.)

    Solo hace count() y schema — no lee datos, no hace OOM.
    """
    print("\n── [2/4] Dimensiones exactas del Bronze (Spark) ──────────────")

    try:
        from pyspark.sql import SparkSession
        from src.utils.resources import apply_to_spark_session
    except ImportError as e:
        print(f"❌ PySpark no disponible: {e}")
        sys.exit(1)

    builder = SparkSession.builder.appName("gtex-profiling-dimensions")
    spark   = apply_to_spark_session(builder, mode="local").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("  Spark session iniciada.")
    print("  Leyendo schema del parquet (sin cargar datos)...")

    df        = spark.read.parquet(bronze_path)
    all_cols  = df.columns
    n_cols    = len(all_cols)

    # Columnas de metadata: todo lo que NO es un sample ID
    # En GTEx los sample IDs tienen formato GTEX-XXXXX o similar
    meta_cols   = [c for c in all_cols if not c.startswith("GTEX") and not c.startswith("K-")]
    sample_cols = [c for c in all_cols if c not in meta_cols]

    print(f"  Columnas totales    : {n_cols:,}")
    print(f"  Columnas metadata   : {len(meta_cols)} → {meta_cols}")
    print(f"  Columnas de samples : {len(sample_cols):,}")
    print("  Contando filas (esto puede tomar 1-2 min)...")

    n_rows = df.count()
    print(f"  Filas totales       : {n_rows:,}")

    # Healthcheck Spark
    spark.sql("SELECT 1").collect()
    print("  ✅ Healthcheck Spark : OK")

    spark.stop()

    return {
        "n_rows":        n_rows,
        "n_cols":        n_cols,
        "n_sample_cols": len(sample_cols),
        "n_meta_cols":   len(meta_cols),
        "meta_cols":     meta_cols,
        "sample_cols_preview": sample_cols[:5],
    }


# ══════════════════════════════════════════════════════════════════════════
#  SECCIÓN 3 — Null/Zero analysis sobre sample de columnas (Spark)
# ══════════════════════════════════════════════════════════════════════════

def get_all_sample_cols(bronze_path: str) -> List[str]:
    """Lee el schema del parquet y devuelve solo las columnas de samples."""
    try:
        from pyspark.sql import SparkSession
        from src.utils.resources import apply_to_spark_session
    except ImportError as e:
        print(f"❌ PySpark no disponible: {e}")
        sys.exit(1)

    builder = SparkSession.builder.appName("gtex-schema-only")
    spark   = apply_to_spark_session(builder, mode="local").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    all_cols    = spark.read.parquet(bronze_path).columns
    spark.stop()
    meta_cols   = [c for c in all_cols if not c.startswith("GTEX") and not c.startswith("K-")]
    return [c for c in all_cols if c not in meta_cols]


def profile_null_zero(
    bronze_path: str,
    sample_col_names: List[str],
    n_rows: int,
    infra: Dict,
) -> Dict:
    """
    Analiza nulls y zeros sobre SAMPLE_COLS columnas aleatorias.
    No hace full scan — solo las columnas muestreadas.

    En GTEx:
    - null  = dato faltante (no debería existir en cols de expresión)
    - 0.0   = gen no expresado en esa muestra (biológicamente válido)
    """
    print(f"\n── [3/4] Null/Zero analysis ({SAMPLE_COLS} columnas muestreadas) ─")

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from src.utils.resources import apply_to_spark_session
    except ImportError as e:
        print(f"❌ PySpark no disponible: {e}")
        sys.exit(1)

    # Muestra aleatoria de columnas de expresión
    random.seed(42)
    sampled_cols = random.sample(
        sample_col_names,
        min(SAMPLE_COLS, len(sample_col_names))
    )

    builder = SparkSession.builder.appName("gtex-profiling-nullzero")
    spark   = apply_to_spark_session(builder, mode="local").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Leer solo las columnas muestreadas + gene_id
    read_cols = ["Name"] + sampled_cols if "Name" in sample_col_names else sampled_cols
    # gene_id puede llamarse "Name" o "gene_id" en GTEx
    try:
        df = spark.read.parquet(bronze_path).select(sampled_cols)
    except Exception:
        df = spark.read.parquet(bronze_path).select(
            [c for c in sampled_cols if c in spark.read.parquet(bronze_path).columns]
        )

    total_cells = n_rows * len(sampled_cols)

    # Contar nulls por columna
    null_exprs = [F.sum(F.col(c).isNull().cast("long")).alias(c) for c in df.columns]
    null_row   = df.select(null_exprs).collect()[0]
    null_counts = {c: null_row[c] for c in df.columns}

    total_nulls  = sum(null_counts.values())
    null_pct     = round((total_nulls / total_cells) * 100, 4)

    # Contar zeros por columna (solo en columnas numéricas)
    zero_exprs = [F.sum((F.col(c) == 0.0).cast("long")).alias(c) for c in df.columns]
    zero_row   = df.select(zero_exprs).collect()[0]
    zero_counts = {c: zero_row[c] for c in df.columns}

    total_zeros  = sum(v for v in zero_counts.values() if v is not None)
    zero_pct     = round((total_zeros / total_cells) * 100, 2)

    # Columnas con nulls (para detectar corrupción)
    cols_with_nulls = {c: v for c, v in null_counts.items() if v and v > 0}

    print(f"  Columnas analizadas : {len(sampled_cols)}")
    print(f"  Celdas totales      : {total_cells:,}")
    print(f"  Nulls totales       : {total_nulls:,} ({null_pct}%)")
    print(f"  Zeros totales       : {total_zeros:,} ({zero_pct}%)")
    print(f"  Cols con nulls      : {len(cols_with_nulls)}")

    if null_pct < 0.1:
        null_verdict = "LIMPIO — nulls prácticamente inexistentes en expresión"
        null_strategy = "preservar zeros como válidos, cuarentena solo si null_pct > 1%"
    elif null_pct < 1.0:
        null_verdict = "ACEPTABLE — nulls menores al 1%"
        null_strategy = "cuarentena de filas con null en cualquier col de expresión"
    else:
        null_verdict = "REVISAR — nulls superiores al 1%, posible corrupción"
        null_strategy = "investigar columnas con mayor concentración de nulls"

    zero_verdict = (
        "zeros son BIOLÓGICAMENTE VÁLIDOS en TPM — "
        f"{zero_pct}% indica genes no expresados en muestras específicas. "
        "Preservar en Silver. Aplicar log1p(TPM) en Gold."
    )

    print(f"  Veredicto nulls     : {null_verdict}")
    print(f"  Veredicto zeros     : {zero_pct}% — biológicamente válidos (TPM)")

    spark.stop()

    return {
        "sampled_cols":     len(sampled_cols),
        "total_cells":      total_cells,
        "total_nulls":      total_nulls,
        "null_pct":         null_pct,
        "total_zeros":      total_zeros,
        "zero_pct":         zero_pct,
        "cols_with_nulls":  len(cols_with_nulls),
        "null_verdict":     null_verdict,
        "null_strategy":    null_strategy,
        "zero_verdict":     zero_verdict,
    }


# ══════════════════════════════════════════════════════════════════════════
#  SECCIÓN 4 — Factor de multiplicación y estimación Silver
# ══════════════════════════════════════════════════════════════════════════

def calculate_reshape_factor(
    n_rows: int,
    n_sample_cols: int,
    bronze_size_bytes: int,
) -> Dict:
    """
    Calcula el factor de multiplicación real del reshape wide→long
    y estima el tamaño de Silver en disco y en RAM.

    Silver long format:
        n_rows_silver = n_rows_bronze × n_sample_cols
        n_cols_silver = 3 (gene_id, sample_id, tpm_value)
    """
    print("\n── [4/4] Factor de multiplicación y estimación Silver ────────")

    # Filas Silver
    n_rows_silver  = n_rows * n_sample_cols
    reshape_factor = n_sample_cols  # cada fila bronze → n_sample_cols filas silver

    # Tamaño estimado Silver en disco
    # Long format con 3 cols comprime bien con Snappy/ZSTD
    # Estimación: cada fila Silver pesa ~40 bytes comprimida (gene_id string + sample_id string + float)
    bytes_per_row_silver = 40
    silver_size_bytes    = n_rows_silver * bytes_per_row_silver
    silver_size_gb       = silver_size_bytes / (1024 ** 3)

    # RAM necesaria para materializar Silver completo (NO lo haremos — es referencia)
    silver_ram_gb = silver_size_gb * PARQUET_EXPANSION

    # RAM necesaria por chunk si hacemos reshape de N columnas a la vez
    # Estrategia segura: chunks de 500 columnas
    chunk_sizes = [100, 200, 500, 1000]
    chunk_plans = {}
    for chunk in chunk_sizes:
        rows_in_chunk  = n_rows * chunk
        ram_for_chunk  = (rows_in_chunk * bytes_per_row_silver * PARQUET_EXPANSION) / (1024 ** 3)
        chunk_plans[f"chunk_{chunk}_cols"] = {
            "rows_per_chunk":    rows_in_chunk,
            "ram_needed_gb":     round(ram_for_chunk, 2),
            "n_chunks":          math.ceil(n_sample_cols / chunk),
        }

    # Gold estimación (agregado por tejido)
    n_tissues_approx  = 54
    n_rows_gold       = n_rows * n_tissues_approx  # genes × tejidos
    gold_size_gb      = round((n_rows_gold * 100) / (1024 ** 3), 3)

    bronze_size_gb = bronze_size_bytes / (1024 ** 3)

    print(f"  Bronze filas        : {n_rows:,}")
    print(f"  Bronze cols sample  : {n_sample_cols:,}")
    print(f"  Factor reshape      : ×{reshape_factor:,}")
    print(f"  Silver filas est.   : {n_rows_silver:,.0f}")
    print(f"  Silver disco est.   : ~{silver_size_gb:.1f} GB (Snappy)")
    print(f"  Silver RAM completo : ~{silver_ram_gb:.1f} GB ← NO hacer full scan")
    print(f"  Gold filas est.     : ~{n_rows_gold:,} ({n_tissues_approx} tejidos × {n_rows:,} genes)")
    print()
    print("  Estrategia de chunks recomendada:")
    for k, v in chunk_plans.items():
        flag = " ← recomendado" if "500" in k else ""
        print(f"    {k}: {v['n_chunks']} chunks, ~{v['ram_needed_gb']} GB RAM/chunk{flag}")

    # Determinar chunk seguro para 6GB WSL (4GB libres con Airflow apagado)
    available_for_spark = 4.0  # GB aproximados con Airflow parado
    safe_chunk = None
    for chunk in chunk_sizes:
        ram_needed = chunk_plans[f"chunk_{chunk}_cols"]["ram_needed_gb"]
        if ram_needed <= available_for_spark * 0.75:
            safe_chunk = chunk

    print(f"\n  ✅ Chunk seguro para 4GB disponibles : {safe_chunk} columnas/chunk")

    return {
        "n_rows_bronze":     n_rows,
        "n_sample_cols":     n_sample_cols,
        "reshape_factor":    reshape_factor,
        "n_rows_silver_est": n_rows_silver,
        "silver_size_gb_est": round(silver_size_gb, 2),
        "silver_ram_full_gb": round(silver_ram_gb, 2),
        "gold_rows_est":     n_rows_gold,
        "gold_size_gb_est":  gold_size_gb,
        "chunk_plans":       chunk_plans,
        "safe_chunk_cols":   safe_chunk,
        "reshape_strategy":  f"chunks de {safe_chunk} columnas, {math.ceil(n_sample_cols / safe_chunk)} iteraciones",
    }


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  GTEx Bronze Profiling — Paso 4 del Framework")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("═" * 60)

    # Verificar paths
    for path, name in [(BRONZE_PATH, "Bronze"), (METADATA_PATH, "Metadata")]:
        if not os.path.exists(path):
            print(f"❌ {name} no encontrado: {path}")
            sys.exit(1)

    # Cargar infra
    sys.path.insert(0, os.path.abspath("."))
    from src.utils.resources import get_spark_memory_settings
    from src.utils.chunk_calculator import calculate_chunk_plan
    import psutil

    infra = get_spark_memory_settings(mode="local")
    mem   = psutil.virtual_memory()

    print(f"\n  RAM total     : {mem.total / (1024**3):.1f} GB")
    print(f"  RAM disponible: {mem.available / (1024**3):.1f} GB")
    print(f"  Override activo: {infra['meta']['override_applied']}")
    print(f"  Fingerprint   : {infra['meta']['fingerprint']}")

    # Leer bronze_metadata si existe (para el size)
    bronze_size_bytes = os.path.getsize(BRONZE_PATH)
    if os.path.exists(BRONZE_META):
        with open(BRONZE_META) as f:
            bm = json.load(f)
            bronze_size_bytes = bm.get("bronze_size_bytes", bronze_size_bytes)

    # ── Ejecutar secciones ─────────────────────────────────────────────────
    metadata_profile  = profile_metadata(METADATA_PATH)
    bronze_dimensions = profile_bronze_dimensions(BRONZE_PATH, infra)
    print("\n  Obteniendo lista completa de columnas de samples...")
    all_sample_cols   = get_all_sample_cols(BRONZE_PATH)
    null_zero_profile = profile_null_zero(
        BRONZE_PATH,
        all_sample_cols,
        bronze_dimensions["n_rows"],
        infra,
    )
    reshape_plan      = calculate_reshape_factor(
        bronze_dimensions["n_rows"],
        bronze_dimensions["n_sample_cols"],
        bronze_size_bytes,
    )

    # ── Chunk plan final con chunk_calculator ──────────────────────────────
    chunk_plan = calculate_chunk_plan(
        file_size_bytes      = bronze_size_bytes,
        available_ram_gb     = mem.available / (1024 ** 3),
        total_ram_gb         = mem.total     / (1024 ** 3),
        override_memory_str  = infra["env_snapshot"]["override_env"],
        inferred_memory_gb   = float(infra["meta"]["memory_used"].replace("g", "")),
        cpu_physical         = infra["meta"]["cpu_physical"],
    )
    print(f"\n{chunk_plan.summary()}")

    # ── Decisiones del Paso 4 ──────────────────────────────────────────────
    decisions = {
        "zeros_en_tpm":       "VÁLIDOS — preservar en Silver, log1p en Gold",
        "nulls_en_expresion": f"{null_zero_profile['null_pct']}% — {null_zero_profile['null_verdict']}",
        "particion_silver":   f"tissue_id (SMTSD) — skew {metadata_profile['skew_ratio']}x ({metadata_profile['skew_level']})",
        "reshape_strategy":   reshape_plan["reshape_strategy"],
        "silver_format":      "long flat — gene_id, sample_id, tpm_value",
        "silver_compression": "Snappy (velocidad sobre espacio en Silver)",
        "gold_compression":   "ZSTD (espacio sobre velocidad en Gold)",
        "safe_chunk_cols":    reshape_plan["safe_chunk_cols"],
    }

    print("\n── Decisiones confirmadas del Paso 4 ────────────────────────")
    for k, v in decisions.items():
        print(f"  {k}: {v}")

    # ── Guardar reporte ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(LINEAGE_PATH), exist_ok=True)

    report = {
        "profiling_timestamp":  datetime.now(timezone.utc).isoformat(),
        "infra_fingerprint":    infra["meta"]["fingerprint"],
        "framework_step":       "Paso 4 — Profiling pre-reshape",
        "metadata_profile":     metadata_profile,
        "bronze_dimensions":    bronze_dimensions,
        "null_zero_profile":    null_zero_profile,
        "reshape_plan":         reshape_plan,
        "chunk_plan": {
            "partitions":      chunk_plan.partitions,
            "safe_memory_gb":  chunk_plan.safe_memory_gb,
            "memory_source":   chunk_plan.memory_source,
            "cores":           chunk_plan.cores,
            "estimated_min":   chunk_plan.estimated_minutes,
        },
        "decisions":            decisions,
        "next_step":            "Paso 5 — quality_gate_bronze.py",
    }

    with open(LINEAGE_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Reporte guardado: {LINEAGE_PATH}")
    print("\n  → Siguiente paso: quality_gate_bronze.py")
    print("═" * 60)


if __name__ == "__main__":
    main()