FROM apache/spark:3.5.0

USER root

# Dependencias de Python
RUN pip install --no-cache-dir \
    psutil \
    python-dotenv \
    pyspark \
    pandas \
    pyarrow \
    delta-spark==3.0.0 \
    deltalake \
    tdigest

# JARs para S3A (MinIO/S3 compatible)
RUN curl -o /opt/spark/jars/hadoop-aws-3.3.4.jar \
    https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -o /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar \
    https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

WORKDIR /opt/spark/work-dir