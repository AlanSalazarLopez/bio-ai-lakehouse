from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
from utils.resources import get_spark_memory_settings
import os
import time

# Step 3: Extract calculated memory parameters to respect container bounds
ram_config = get_spark_memory_settings()

# Optimization: Added arrow validation and adjusted driver memory if needed.
# Since you are handling wide Parquet files (>19k columns), driver memory 
# requires sufficient headroom to store the heavy Schema metadata.
spark = SparkSession.builder \
    .appName("GTEx_Profiling_Phase1") \
    .config("spark.executor.memory", ram_config) \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# Path to target raw Bronze storage
path = "data/raw/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet"

print(f"\n{'='*40}")
print("INITIALIZING PHASE 1: RAW BRONZE PROFILING")
print(f"{'='*40}")

if not os.path.exists(path):
    print(f"ERROR: Target file artifact not found at location: {path}")
else:
    start_time = time.time()
    
    # Read raw Parquet file structure
    df = spark.read.parquet(path)
    
    # 1. Structural Dimensions
    # CRITICAL WARNING: len(df.columns) forces a driver-side schema metadata parsing scan.
    # For 19k+ columns, this is acceptable ONCE, but avoid repeating it inside loops.
    col_count = len(df.columns)
    row_count = df.count()
    
    # 2. Filesystem Compressed Size Allocation
    size_gb = os.path.getsize(path) / (1024**3)
    
    # 3. Structural Topology Evaluation
    format_type = "Wide" if col_count > 100 else "Long"
    
    print(f"• Dynamic Allocated RAM (60% Bounds): {ram_config}")
    print(f"• Total Dataset Rows                 : {row_count:,}")
    print(f"• Total Dataset Columns              : {col_count:,}")
    print(f"• Inferred Structural Topology       : {format_type}")
    print(f"• On-Disk Compressed Footprint        : {size_gb:.2f} GB")
    
    print("\n--- Key Metadata Columns Preview ---")
    # Fetching the first 5 columns safely. 
    # Assumes df.columns[0] is 'Name' (gene_id) and df.columns[1] is 'Description' (gene_name)
    df.select(df.columns[:5]).show(5, truncate=False)
    
    # 4. Controlled Expression Zero-Value Skew Analysis
    print("--- Expression Matrix Zero-Count Analysis (3 Sample Column Vectors) ---")
    
    # FIX: Shifted the slice window to df.columns[2:5] to skip 'Name' and 'Description',
    # targeting actual numerical expression columns (Samples) where zero-values reflect biological dropouts.
    sample_cols = df.columns[2:5]
    
    # Efficient execution: Computes zero matrix profile via a single localized aggregation pass
    df.select([count(when(col(c) == 0, c)).alias(f"zeros_{c}") for c in sample_cols]).show()

    end_time = time.time()
    print(f"\nPhase execution completed in: {end_time - start_time:.2f} seconds")

print(f"{'='*40}")
spark.stop()
