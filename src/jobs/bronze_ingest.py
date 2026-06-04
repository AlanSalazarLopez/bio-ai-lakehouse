"""
src/jobs/bronze_ingest.py

Bronze = copia exacta e inmutable del dato crudo.
Sin transformaciones. Sin columnas extra. Sin row count.
El quality gate y el lineage van en scripts separados.
"""

import os
import json
import shutil
from datetime import datetime, timezone
from src.utils.resources import get_spark_memory_settings


def run_bronze_ingest():
    # 1. Recursos e inferencia — sin Spark todavía
    infra_info    = get_spark_memory_settings(mode="local")
    archivo_local = "data/raw/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet"
    ruta_bronze   = "data/bronze/gtex/gene_tpm_raw.parquet"
    ruta_metadata = "data/lineage/bronze_metadata.json"

    print("\n── Recursos Inferidos ────────────────────")
    print(f"  memoria    : {infra_info['meta']['memory_used']}")
    print(f"  override   : {infra_info['meta']['override_applied']}")
    print(f"  fingerprint: {infra_info['meta']['fingerprint']}")

    # 2. Validaciones pre-copia — sin memoria de Spark
    if not os.path.exists(archivo_local):
        print(f"❌ Archivo no encontrado: {archivo_local}")
        return

    size_gb = os.path.getsize(archivo_local) / (1024 ** 3)
    print(f"\n✅ Archivo encontrado : {size_gb:.2f} GB")
    print(f"✅ Destino Bronze     : {ruta_bronze}")

    # 3. Copia inmutable — Bronze es el archivo original sin tocar
    os.makedirs(os.path.dirname(ruta_bronze), exist_ok=True)
    print("\nCopiando a Bronze...")
    shutil.copy2(archivo_local, ruta_bronze)

    # Verificar integridad por tamaño
    size_bronze = os.path.getsize(ruta_bronze)
    if size_bronze != os.path.getsize(archivo_local):
        print("❌ Error: tamaño del archivo Bronze no coincide con el original.")
        return

    # 4. Metadata de lineage — sin Spark
    os.makedirs(os.path.dirname(ruta_metadata), exist_ok=True)
    metadata = {
        "source_file":         archivo_local,
        "source_size_bytes":   os.path.getsize(archivo_local),
        "source_size_gb":      round(size_gb, 2),
        "bronze_path":         ruta_bronze,
        "bronze_size_bytes":   size_bronze,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "infra_fingerprint":   infra_info["meta"]["fingerprint"],
        "memory_used":         infra_info["meta"]["memory_used"],
        "override_applied":    infra_info["meta"]["override_applied"],
        "transformation":      "none — Bronze es copia exacta del original",
    }
    with open(ruta_metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n── Ingesta Bronze Completada ─────────────")
    print(f"  ✅ Bronze            : {ruta_bronze}")
    print(f"  ✅ Tamaño verificado : {size_gb:.2f} GB")
    print(f"  ✅ Metadata          : {ruta_metadata}")
    print(f"  🆔 Fingerprint       : {infra_info['meta']['fingerprint']}")
    print("\n  → Corre quality_gate_bronze.py para validar schema e integridad.")


if __name__ == "__main__":
    run_bronze_ingest()