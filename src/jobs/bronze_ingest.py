"""
src/jobs/bronze_ingest.py

Bronze Layer = Exact, immutable copy of the raw data.
No transformations. No extra columns. No row counts.
Quality gate and data lineage are handled in separate scripts.
"""

import os
import json
import shutil
from datetime import datetime, timezone
from src.utils.resources import get_spark_memory_settings


def run_bronze_ingest():
    # 1. Resource and infrastructure inference — without Spark overhead yet
    infra_info    = get_spark_memory_settings(mode="local")
    archivo_local = "data/raw/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet"
    ruta_bronze   = "data/bronze/gtex/gene_tpm_raw.parquet"
    ruta_metadata = "data/lineage/bronze_metadata.json"

    print("\n── Inferred Resources ────────────────────")
    print(f"  memory used : {infra_info['meta']['memory_used']}")
    print(f"  override    : {infra_info['meta']['override_applied']}")
    print(f"  fingerprint : {infra_info['meta']['fingerprint']}")

    # 2. Pre-copy validations — lightweight OS checks before computing
    if not os.path.exists(archivo_local):
        print(f"❌ File not found: {archivo_local}")
        return

    size_gb = os.path.getsize(archivo_local) / (1024 ** 3)
    print(f"\n✅ File found : {size_gb:.2f} GB")
    print(f"✅ Bronze destination : {ruta_bronze}")

    # 3. Immutable copy — Bronze retains the untouched original file
    os.makedirs(os.path.dirname(ruta_bronze), exist_ok=True)
    print("\nCopying data to Bronze layer...")
    shutil.copy2(archivo_local, ruta_bronze)

    # Integrity verification via file size check
    size_bronze = os.path.getsize(ruta_bronze)
    if size_bronze != os.path.getsize(archivo_local):
        print("❌ Error: Bronze file size does not match the original file.")
        return

    # 4. Data lineage metadata architecture
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

    print("\n── Bronze Ingestion Completed ─────────────")
    print(f"  ✅ Bronze asset       : {ruta_bronze}")
    print(f"  ✅ Verified size      : {size_gb:.2f} GB")
    print(f"  ✅ Lineage metadata   : {ruta_metadata}")
    print(f"  🆔 Infra fingerprint  : {infra_info['meta']['fingerprint']}")
    print("\n  → Next step: Run quality_gate_bronze.py to validate schema and integrity.")


if __name__ == "__main__":
    run_bronze_ingest()
