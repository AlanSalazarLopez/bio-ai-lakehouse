# Ubicación: src/tests/test_resources.py
import sys
import os

# Esto permite que el test encuentre la carpeta 'src' aunque estés en la raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.resources import get_spark_memory_settings

def test_my_env():
    print("\n--- TEST DE INFRAESTRUCTURA ELÁSTICA ---")
    try:
        # Probamos el modo local que es el de tu WSL
        settings = get_spark_memory_settings(mode="local")
        
        conf = settings['spark_config']
        meta = settings['meta']
        
        print(f"✅ Entorno detectado: {meta['platform']}")
        print(f"✅ RAM Detectada: {meta['total_ram_gb']} GB")
        print(f"🚀 Configuración calculada para Spark:")
        print(f"   - Master: {conf['spark.master']}")
        print(f"   - Driver Memory: {conf['spark.driver.memory']} (Debería ser ~60% de tu RAM)")
        print(f"   - Particiones de Shuffle: {conf['spark.sql.shuffle.partitions']}")
        print(f"🆔 Fingerprint del sistema: {meta['fingerprint']}")
        
    except Exception as e:
        print(f"❌ Error al calcular recursos: {e}")

if __name__ == "__main__":
    test_my_env()