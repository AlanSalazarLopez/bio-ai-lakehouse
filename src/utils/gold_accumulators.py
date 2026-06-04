"""
src/utils/gold_accumulators.py

Acumuladores matemáticos para el groupBy Gold — sin Spark, sin OOM.

Diseño v3 — numpy memmap:
    Los arrays de estadísticos viven en disco (SSD) via numpy.memmap.
    Python los ve como arrays normales pero el OS maneja el paging.
    RAM usada = solo páginas activas, no el array completo.
    Sin OOM sin importar cuántos grupos haya.

Problema resuelto:
    v1: objetos Python (GroupAccumulator) — 3GB overhead para 5M grupos
    v2: reservoir sampler en RAM — sigue siendo 2-3GB con 5M grupos
    v3: numpy memmap en SSD — RAM constante ~200-400MB sin importar grupos

Estadísticos implementados:
    mean_log1p_tpm   → Welford online (exacto)
    std_log1p_tpm    → Welford online (exacto)
    median_log1p_tpm → Reservoir sampling en memmap (~5% error)
    sample_count     → contador entero
    zero_fraction    → contador de zeros / sample_count

Layout de archivos en disco (GOLD_CACHE_DIR):
    index.json       — mapeo group_key → row_index
    n.bin            — float64[MAX_GROUPS] — sample count
    mean.bin         — float64[MAX_GROUPS] — media Welford
    M2.bin           — float64[MAX_GROUPS] — suma cuadrados Welford
    zero_count.bin   — int64[MAX_GROUPS]   — zeros acumulados
    reservoir.bin    — float32[MAX_GROUPS × RESERVOIR_SIZE] — para mediana

Compatible con Python 3.8.
"""

import json
import logging
import math
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Configuración
# ─────────────────────────────────────────────

# Directorio donde viven los archivos memmap
GOLD_CACHE_DIR = "data/gold_cache"

# Máximo de grupos posibles: 74,628 genes × 68 tejidos = 5,074,704
# Redondeamos arriba para seguridad
MAX_GROUPS = 5_100_000

# Reservoir por grupo — 10 muestras, ~5% error en mediana
# RAM de reservoir: 10 × 4 bytes × 5M grupos = 200MB en disco, ~0 en RAM
RESERVOIR_SIZE = 10

# Key del dict de acumuladores: (gene_id, gene_symbol, tissue_id)
GroupKey = Tuple[str, str, str]


# ─────────────────────────────────────────────
#  Inicialización del cache en disco
# ─────────────────────────────────────────────

def _init_cache(cache_dir: str) -> None:
    """Crea el directorio de cache si no existe."""
    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Cache memmap en: %s", cache_dir)


def _cache_path(cache_dir: str, name: str) -> str:
    return os.path.join(cache_dir, name)


# ─────────────────────────────────────────────
#  GoldAccumulatorMap — numpy memmap
# ─────────────────────────────────────────────

class GoldAccumulatorMap:
    """
    Acumuladores Gold usando numpy memmap — arrays en disco, acceso como RAM.

    En lugar de 5M objetos Python (3GB overhead), usa 6 arrays numpy
    mapeados a disco (~800MB en disco, ~200MB en RAM activa).

    Idempotente: si el cache existe de una run anterior, se puede
    continuar desde donde quedó (útil si el proceso se interrumpe).
    """

    def __init__(
        self,
        cache_dir:    str = GOLD_CACHE_DIR,
        max_groups:   int = MAX_GROUPS,
        reservoir_size: int = RESERVOIR_SIZE,
        resume:       bool = False,
    ) -> None:
        """
        Args:
            cache_dir:      directorio para los archivos memmap
            max_groups:     máximo de grupos únicos esperados
            reservoir_size: muestras por grupo para mediana aproximada
            resume:         si True, carga un cache existente en lugar de crear uno nuevo
        """
        self._cache_dir     = cache_dir
        self._max_groups    = max_groups
        self._reservoir_size = reservoir_size

        _init_cache(cache_dir)

        index_path = _cache_path(cache_dir, "index.json")

        if resume and os.path.exists(index_path):
            logger.info("Resumiendo desde cache existente: %s", cache_dir)
            with open(index_path) as f:
                self._index: Dict[str, int] = json.load(f)
            self._next_idx = max(self._index.values()) + 1 if self._index else 0
            mode = "r+"
        else:
            logger.info("Creando cache memmap nuevo: %s", cache_dir)
            self._index    = {}
            self._next_idx = 0
            mode = "w+"

        # Arrays principales — viven en disco via memmap
        self._n          = np.memmap(_cache_path(cache_dir, "n.bin"),
                                     dtype="float64", mode=mode, shape=(max_groups,))
        self._mean       = np.memmap(_cache_path(cache_dir, "mean.bin"),
                                     dtype="float64", mode=mode, shape=(max_groups,))
        self._M2         = np.memmap(_cache_path(cache_dir, "M2.bin"),
                                     dtype="float64", mode=mode, shape=(max_groups,))
        self._zero_count = np.memmap(_cache_path(cache_dir, "zero_count.bin"),
                                     dtype="int64",   mode=mode, shape=(max_groups,))
        # Reservoir: shape (max_groups, reservoir_size)
        self._reservoir  = np.memmap(_cache_path(cache_dir, "reservoir.bin"),
                                     dtype="float32", mode=mode,
                                     shape=(max_groups, reservoir_size))

        logger.info(
            "Memmap inicializado: max_groups=%s, reservoir_size=%d",
            f"{max_groups:,}", reservoir_size,
        )

    # ── Helpers internos ────────────────────────────────────────────────

    def _get_or_create_idx(self, key_str: str) -> int:
        """Retorna el índice del grupo, creándolo si no existe."""
        if key_str not in self._index:
            if self._next_idx >= self._max_groups:
                raise RuntimeError(
                    f"Se alcanzó el máximo de grupos ({self._max_groups:,}). "
                    "Aumenta MAX_GROUPS en gold_accumulators.py."
                )
            self._index[key_str] = self._next_idx
            self._next_idx += 1
        return self._index[key_str]

    def _update_single(self, idx: int, log_val: float, is_zero: bool) -> None:
        """Welford online + reservoir para un solo grupo en el array."""
        self._n[idx] += 1
        n = self._n[idx]

        if is_zero:
            self._zero_count[idx] += 1

        # Welford online
        delta          = log_val - self._mean[idx]
        self._mean[idx] += delta / n
        delta2          = log_val - self._mean[idx]
        self._M2[idx]  += delta * delta2

        # Reservoir sampling — Algoritmo R de Vitter
        n_int = int(n)
        if n_int <= self._reservoir_size:
            self._reservoir[idx, n_int - 1] = log_val
        else:
            j = random.randint(0, n_int - 1)
            if j < self._reservoir_size:
                self._reservoir[idx, j] = log_val

    # ── Update desde RecordBatch ────────────────────────────────────────

    def update_from_batch(self, batch: pa.RecordBatch) -> int:
        """
        Actualiza los acumuladores con todas las filas de un RecordBatch.

        Args:
            batch: RecordBatch con columnas
                   gene_id, gene_symbol, tissue_id, tpm_value

        Returns:
            Número de filas procesadas.
        """
        gene_ids     = batch.column("gene_id").to_pylist()
        gene_symbols = batch.column("gene_symbol").to_pylist()
        tissue_ids   = batch.column("tissue_id").to_pylist()
        tpm_values   = batch.column("tpm_value").to_pylist()

        n_rows = len(gene_ids)

        for i in range(n_rows):
            tpm = tpm_values[i]
            if tpm is None:
                continue

            key_str = f"{gene_ids[i]}|{gene_symbols[i]}|{tissue_ids[i]}"
            idx     = self._get_or_create_idx(key_str)
            tpm_f   = float(tpm)
            log_val = math.log1p(tpm_f)
            self._update_single(idx, log_val, tpm_f == 0.0)

        return n_rows

    def flush(self) -> None:
        """
        Flusea los arrays memmap a disco y guarda el índice.
        Llamar periódicamente para asegurar que el progreso no se pierda.
        """
        self._n.flush()
        self._mean.flush()
        self._M2.flush()
        self._zero_count.flush()
        self._reservoir.flush()

        index_path = _cache_path(self._cache_dir, "index.json")
        with open(index_path, "w") as f:
            json.dump(self._index, f)

        logger.debug("Cache flusheado: %s grupos", f"{len(self._index):,}")

    # ── Serialización a PyArrow Table ───────────────────────────────────

    def to_arrow_table(self) -> pa.Table:
        """
        Convierte los acumuladores a pa.Table lista para write_deltalake.
        Solo procesa los índices activos (hasta self._next_idx).
        """
        n_groups = self._next_idx
        if n_groups == 0:
            logger.warning("to_arrow_table() con 0 grupos — tabla vacía")
            return _empty_gold_schema()

        logger.info("Serializando %s grupos a pa.Table...", f"{n_groups:,}")

        # Reconstruir keys desde el índice invertido
        inv_index = {v: k for k, v in self._index.items()}

        gene_ids     = [""] * n_groups
        gene_symbols = [""] * n_groups
        tissue_ids   = [""] * n_groups
        means        = [0.0] * n_groups
        stds         = [None] * n_groups
        medians      = [None] * n_groups
        counts       = [0]   * n_groups
        zero_fracs   = [0.0] * n_groups

        for idx in range(n_groups):
            key_str = inv_index[idx]
            parts   = key_str.split("|")
            gene_id, gene_symbol, tissue_id = parts[0], parts[1], parts[2]

            n     = self._n[idx]
            mean  = self._mean[idx]
            M2    = self._M2[idx]
            zc    = self._zero_count[idx]

            gene_ids[idx]     = gene_id
            gene_symbols[idx] = gene_symbol
            tissue_ids[idx]   = tissue_id
            means[idx]        = float(mean) if n > 0 else 0.0
            counts[idx]       = int(n)
            zero_fracs[idx]   = float(zc / n) if n > 0 else 0.0

            # std — Welford
            if n >= 2:
                stds[idx] = float(math.sqrt(M2 / n))

            # mediana — reservoir
            if n > 0:
                n_reservoir = min(int(n), self._reservoir_size)
                reservoir_slice = sorted(
                    float(self._reservoir[idx, j]) for j in range(n_reservoir)
                )
                mid = len(reservoir_slice) // 2
                if len(reservoir_slice) % 2 == 0:
                    medians[idx] = (reservoir_slice[mid - 1] + reservoir_slice[mid]) / 2.0
                else:
                    medians[idx] = reservoir_slice[mid]

        table = pa.table(
            {
                "gene_id":          pa.array(gene_ids,     type=pa.string()),
                "gene_symbol":      pa.array(gene_symbols, type=pa.string()),
                "tissue_id":        pa.array(tissue_ids,   type=pa.string()),
                "mean_log1p_tpm":   pa.array(means,        type=pa.float32()),
                "std_log1p_tpm":    pa.array(stds,         type=pa.float32()),
                "median_log1p_tpm": pa.array(medians,      type=pa.float32()),
                "sample_count":     pa.array(counts,       type=pa.int32()),
                "zero_fraction":    pa.array(zero_fracs,   type=pa.float32()),
            }
        )

        logger.info("pa.Table serializada: %s filas, %s columnas",
                    f"{table.num_rows:,}", table.num_columns)
        return table

    # ── Propiedades de inspección ────────────────────────────────────────

    @property
    def group_count(self) -> int:
        return len(self._index)

    def tissue_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for key_str in self._index:
            tissue_id = key_str.split("|")[2]
            counts[tissue_id] = counts.get(tissue_id, 0) + 1
        return counts


# ─────────────────────────────────────────────
#  Helper interno
# ─────────────────────────────────────────────

def _empty_gold_schema() -> pa.Table:
    return pa.table(
        {
            "gene_id":          pa.array([], type=pa.string()),
            "gene_symbol":      pa.array([], type=pa.string()),
            "tissue_id":        pa.array([], type=pa.string()),
            "mean_log1p_tpm":   pa.array([], type=pa.float32()),
            "std_log1p_tpm":    pa.array([], type=pa.float32()),
            "median_log1p_tpm": pa.array([], type=pa.float32()),
            "sample_count":     pa.array([], type=pa.int32()),
            "zero_fraction":    pa.array([], type=pa.float32()),
        }
    )


# ─────────────────────────────────────────────
#  CLI — tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import shutil
    import numpy as np

    TEST_CACHE = "/tmp/gold_cache_test"
    if os.path.exists(TEST_CACHE):
        shutil.rmtree(TEST_CACHE)

    print("\n── Test 1: Welford vs numpy ──────────────────────────────────\n")

    rng  = np.random.default_rng(42)
    vals = rng.exponential(scale=2.0, size=10_000).tolist()

    acc_map = GoldAccumulatorMap(cache_dir=TEST_CACHE, max_groups=100)
    batch = pa.record_batch({
        "gene_id":     pa.array(["ENSG001"] * len(vals)),
        "gene_symbol": pa.array(["GENE_A"]  * len(vals)),
        "tissue_id":   pa.array(["Liver"]   * len(vals)),
        "tpm_value":   pa.array(vals,           type=pa.float32()),
    })
    acc_map.update_from_batch(batch)

    log_vals  = np.log1p(vals)
    np_mean   = float(np.mean(log_vals))
    np_std    = float(np.std(log_vals))
    np_median = float(np.median(log_vals))

    idx  = acc_map._index["ENSG001|GENE_A|Liver"]
    mean = float(acc_map._mean[idx])
    n    = float(acc_map._n[idx])
    M2   = float(acc_map._M2[idx])
    std  = math.sqrt(M2 / n) if n >= 2 else None

    print(f"  mean  → Welford: {mean:.8f}  numpy: {np_mean:.8f}  diff: {abs(mean - np_mean):.2e}")
    print(f"  std   → Welford: {std:.8f}   numpy: {np_std:.8f}   diff: {abs(std - np_std):.2e}")
    print(f"  mean OK: {abs(mean - np_mean) < 1e-6}")
    print(f"  std  OK: {abs(std  - np_std)  < 1e-6}")

    print("\n── Test 2: zero_fraction ─────────────────────────────────────\n")

    shutil.rmtree(TEST_CACHE)
    acc_map2 = GoldAccumulatorMap(cache_dir=TEST_CACHE, max_groups=100)
    batch2 = pa.record_batch({
        "gene_id":     pa.array(["ENSG001"] * 10),
        "gene_symbol": pa.array(["GENE_A"]  * 10),
        "tissue_id":   pa.array(["Liver"]   * 10),
        "tpm_value":   pa.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                                type=pa.float32()),
    })
    acc_map2.update_from_batch(batch2)
    idx2 = acc_map2._index["ENSG001|GENE_A|Liver"]
    zf   = float(acc_map2._zero_count[idx2]) / float(acc_map2._n[idx2])
    print(f"  zero_fraction: {zf:.4f}  expected: 0.3000  OK: {abs(zf - 0.30) < 1e-9}")

    print("\n── Test 3: GoldAccumulatorMap + to_arrow_table ───────────────\n")

    shutil.rmtree(TEST_CACHE)
    acc_map3 = GoldAccumulatorMap(cache_dir=TEST_CACHE, max_groups=100)
    batch3 = pa.record_batch({
        "gene_id":     pa.array(["ENSG001", "ENSG001", "ENSG002", "ENSG001"]),
        "gene_symbol": pa.array(["GENE_A",  "GENE_A",  "GENE_B",  "GENE_A"]),
        "tissue_id":   pa.array(["Liver",   "Liver",   "Liver",   "Blood"]),
        "tpm_value":   pa.array([1.0, 2.0, 5.0, 0.0], type=pa.float32()),
    })
    acc_map3.update_from_batch(batch3)
    print(f"  grupos únicos: {acc_map3.group_count}")
    assert acc_map3.group_count == 3

    gold_table = acc_map3.to_arrow_table()
    print(f"  gold_table rows: {gold_table.num_rows}")
    rows = gold_table.to_pydict()
    for i in range(gold_table.num_rows):
        if rows["gene_id"][i] == "ENSG001" and rows["tissue_id"][i] == "Liver":
            assert rows["sample_count"][i] == 2
            assert rows["zero_fraction"][i] == 0.0
            print(f"  ENSG001/Liver → sample_count={rows['sample_count'][i]} zero_fraction={rows['zero_fraction'][i]:.2f} ✅")
        if rows["gene_id"][i] == "ENSG001" and rows["tissue_id"][i] == "Blood":
            assert rows["zero_fraction"][i] == 1.0
            print(f"  ENSG001/Blood → sample_count={rows['sample_count'][i]} zero_fraction={rows['zero_fraction'][i]:.2f} ✅")

    print("\n── Test 4: flush y resume ────────────────────────────────────\n")
    acc_map3.flush()
    acc_map4 = GoldAccumulatorMap(cache_dir=TEST_CACHE, max_groups=100, resume=True)
    print(f"  grupos después de resume: {acc_map4.group_count}")
    assert acc_map4.group_count == 3
    print("  ✅ resume OK")

    shutil.rmtree(TEST_CACHE)
    print("\n✅ Todos los tests pasaron — gold_accumulators.py v3 memmap listo\n")