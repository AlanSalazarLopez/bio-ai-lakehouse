from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("connectivity-test") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "admin") \
    .config("spark.hadoop.fs.s3a.secret.key", "dev_password123") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

# Crear un DataFrame mínimo y escribirlo en bronze
data = [("test", 1), ("connectivity", 2)]
df = spark.createDataFrame(data, ["word", "value"])
df.write.mode("overwrite").parquet("s3a://bronze/test/")

print("✓ Spark puede escribir en MinIO bronze")

# Leerlo de vuelta
df_read = spark.read.parquet("s3a://bronze/test/")
df_read.show()

print("✓ Spark puede leer de MinIO bronze")

spark.stop()