from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
from utils.resources import apply_to_spark_session
import os
import time

# ✅ Use the helper function to apply optimized config
spark = apply_to_spark_session(
    SparkSession.builder.appName("GTEx_Profiling_Phase1")
).config("spark.driver.memory", "4g").getOrCreate()

# ✅ Use parameterized path or environment variable
path = os.getenv("BRONZE_PATH", "data/bronze/gtex/gene_tpm_raw.parquet")

print(f"\n{'='*40}")
print("INITIALIZING PHASE 1: RAW BRONZE PROFILING")
print(f"{'='*40}")

if not os.path.exists(path):
    print(f"ERROR: Target file artifact not found at location: {path}")
else:
    start_time = time.time()
    
    df = spark.read.parquet(path)
    
    col_count = len(df.columns)
    row_count = df.count()
    size_gb = os.path.getsize(path) / (1024**3)
    format_type = "Wide" if col_count > 100 else "Long"
    
    print(f"• Total Dataset Rows                 : {row_count:,}")
    print(f"• Total Dataset Columns              : {col_count:,}")
    print(f"• Inferred Structural Topology       : {format_type}")
    print(f"• On-Disk Compressed Footprint       : {size_gb:.2f} GB")
    
    print("\n--- Key Metadata Columns Preview ---")
    df.select(df.columns[:5]).show(5, truncate=False)
    
    print("--- Expression Matrix Zero-Count Analysis (3 Sample Column Vectors) ---")
    sample_cols = df.columns[2:5]
    df.select([count(when(col(c) == 0, c)).alias(f"zeros_{c}") for c in sample_cols]).show()

    end_time = time.time()
    print(f"\nPhase execution completed in: {end_time - start_time:.2f} seconds")

print(f"{'='*40}")
spark.stop()
