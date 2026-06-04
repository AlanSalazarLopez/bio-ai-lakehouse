from pyspark.sql import SparkSession
from utils.resources import get_spark_memory_settings
import os
import time

# Paso 3: El entorno sabe dónde vive antes de ejecutar [cite: 20]
ram_config = get_spark_memory_settings()

spark = SparkSession.builder \
    .appName("GTEx_Profiling_Paso1") \
    .config("spark.executor.memory", ram_config) \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Ruta al archivo original
path = "data/raw/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet"

print(f"\n{'='*40}")
print(f"INICIANDO PASO 1: PERFIL DEL DATO CRUDO")
print(f"{'='*40}")

if not os.path.exists(path):
    print(f"ERROR: No se encuentra el archivo en {path}")
else:
    start_time = time.time()
    
    # Leer el dataset
    df = spark.read.parquet(path)
    
    # 1. Dimensiones [cite: 5]
    row_count = df.count()
    col_count = len(df.columns)
    
    # 2. Peso en disco [cite: 9]
    size_gb = os.path.getsize(path) / (1024**3)
    
    # 3. Formato y Esquema [cite: 8]
    # Revisamos si es Wide (muchas columnas) o Long
    format_type = "Wide" if col_count > 100 else "Long"
    
    print(f"• RAM Asignada (60%): {ram_config} [cite: 23]")
    print(f"• Total Filas: {row_count:,} [cite: 5]")
    print(f"• Total Columnas: {col_count:,} [cite: 5]")
    print(f"• Formato Detectado: {format_type} [cite: 8]")
    print(f"• Peso Comprimido: {size_gb:.2f} GB [cite: 9]")
    
    print("\n--- Vista Previa de Columnas Clave ---")
    df.select(df.columns[:5]).show(5, truncate=False)
    
    # 4. Análisis de ceros en una muestra (Paso 1/4) [cite: 10, 49]
    print("--- Análisis de Ceros (Muestra de 3 columnas de expresión) ---")
    from pyspark.sql.functions import col, count, when
    sample_cols = df.columns[2:5] # Típicamente gene_id, gene_name y primera muestra
    df.select([count(when(col(c) == 0, c)).alias(f"zeros_{c}") for c in sample_cols]).show()

    end_time = time.time()
    print(f"\nTiempo de ejecución: {end_time - start_time:.2f} segundos")

print(f"{'='*40}")
spark.stop()