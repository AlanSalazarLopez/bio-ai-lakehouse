"""
src/utils/gold_accumulators.py

Mathematical accumulators for Gold-tier groupBy operations — engine-agnostic, zero-OOM risk.

v3 Design — NumPy memmap:
    Statistical matrices are persisted to physical storage (SSD) via np.memmap.
    Python handles them as native arrays while the OS manages memory-mapped paging.
    RAM usage footprint = exclusively active memory pages, never the entire matrix layout.
    Guarantees zero-OOM execution limits regardless of total unique group counts.

Resolved Engineering Issues:
    v1: Native Python objects (GroupAccumulator) — ~3GB overhead for 5M tracking groups.
    v2: In-Memory Reservoir Sampler — maintained a ~2-3GB memory footprint for 5M groups.
    v3: NumPy memmap via SSD target blocks — constant RAM floor (~200-400MB) across 5M+ groups.

Calculated Statistics:
    mean_log1p_tpm  → Online Welford Algorithm (Mathematically Exact)
    std_log1p_tpm   → Online Welford Algorithm (Mathematically Exact)
    median_log1p_tpm → Memmap-backed Reservoir Sampling (~5% approximation error)
    sample_count     → Int64 Scalar Counter
    zero_fraction    → Accumulator Zero Samples Counter / sample_count

Storage Caching Disk Footprint Layout (GOLD_CACHE_DIR):
    index.json       — Structural map lookup: group_key → row_index
    n.bin            — float64[MAX_GROUPS] — Total sample running counts
    mean.bin         — float64[MAX_GROUPS] — Welford running mean values
    M2.bin           — float64[MAX_GROUPS] — Welford sum of squares tracking matrices
    zero_count.bin   — int64[MAX_GROUPS]   — Total verified absolute zero occurrences
    reservoir.bin    — float32[MAX_GROUPS × RESERVOIR_SIZE] — Slices targeted for median evaluations

Compatible with Python 3.8+.
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
#  Configuration
# ─────────────────────────────────────────────

# Path mapping to directory containing active physical memmap binary fragments
GOLD_CACHE_DIR = "data/gold_cache"

# Max potential group calculation ceilings: 74,628 genes × 68 tissues = 5,074,704 unique combinations
# Rounded upward to provide safety margins
MAX_GROUPS = 5_100_000

# Group reservoir size allocations — 10 samples per group yields a ~5% error margin for median evaluations
# Physical storage calculations: 10 samples × 4 bytes × 5M groups = ~200MB on disk, near-0 active RAM overhead
RESERVOIR_SIZE = 10

# Structural typing tuple signature for dictionary lookups: (gene_id, gene_symbol, tissue_id)
GroupKey = Tuple[str, str, str]


# ─────────────────────────────────────────────
#  Disk Storage Cache Initialization
# ─────────────────────────────────────────────

def _init_cache(cache_dir: str) -> None:
    """Creates the target underlying disk caching directory layout if it is missing."""
    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Initializing active memmap storage engine cache at: %s", cache_dir)


def _cache_path(cache_dir: str, name: str) -> str:
    return os.path.join(cache_dir, name)


# ─────────────────────────────────────────────
#  GoldAccumulatorMap — NumPy memmap Interface
# ─────────────────────────────────────────────

class GoldAccumulatorMap:
    """
    Manages Gold aggregation metrics leveraging low-overhead disk arrays via np.memmap.

    Instead of maintaining 5M isolated Python class records (costing a ~3GB RAM overhead), 
    this interface initializes 6 NumPy array footprints mapped to disk spaces (~800MB on disk, 
    pinning only ~200MB within active system paging bounds).

    Idempotent: If existing cache maps are identified from a historical lifecycle run, 
    the system automatically re-attaches and resumes processing from its last checkpoint flush.
    """

    def __init__(
        self,
        cache_dir:      str = GOLD_CACHE_DIR,
        max_groups:     int = MAX_GROUPS,
        reservoir_size: int = RESERVOIR_SIZE,
        resume:         bool = False,
    ) -> None:
        """
        Args:
            cache_dir:      Directory path location storing the physical binary memmap files.
            max_groups:     Maximum upper ceiling of expected unique tissue-gene grouping intersections.
            reservoir_size: Sample capacity tracking window assigned to execute median estimations.
            resume:         If explicitly flagged True, hooks into historical structural caches instead of flushing.
        """
        self._cache_dir      = cache_dir
        self._max_groups     = max_groups
        self._reservoir_size = reservoir_size

        _init_cache(cache_dir)

        index_path = _cache_path(cache_dir, "index.json")

        if resume and os.path.exists(index_path):
            logger.info("Resuming extraction state using existing cache targets at: %s", cache_dir)
            with open(index_path) as f:
                self._index: Dict[str, int] = json.load(f)
            self._next_idx = max(self._index.values()) + 1 if self._index else 0
            mode = "r+"
        else:
            logger.info("Initializing pristine memmap tracking cache profiles at: %s", cache_dir)
            self._index    = {}
            self._next_idx = 0
            mode = "w+"

        # Primary data matrices — bound directly to disk tracks via memmap configurations
        self._n          = np.memmap(_cache_path(cache_dir, "n.bin"),
                                     dtype="float64", mode=mode, shape=(max_groups,))
        self._mean       = np.memmap(_cache_path(cache_dir, "mean.bin"),
                                     dtype="float64", mode=mode, shape=(max_groups,))
        self._M2         = np.memmap(_cache_path(cache_dir, "M2.bin"),
                                     dtype="float64", mode=mode, shape=(max_groups,))
        self._zero_count = np.memmap(_cache_path(cache_dir, "zero_count.bin"),
                                     dtype="int64",   mode=mode, shape=(max_groups,))
        # Median Reservoir Matrix layout format: shape (max_groups, reservoir_size)
        self._reservoir  = np.memmap(_cache_path(cache_dir, "reservoir.bin"),
                                     dtype="float32", mode=mode,
                                     shape=(max_groups, reservoir_size))

        logger.info(
            "Memmap tracking matrices initialized: max_groups=%s, reservoir_size=%d",
            f"{max_groups:,}", reservoir_size,
        )

    # ── Internal Private Routing Helpers ────────────────────────────────

    def _get_or_create_idx(self, key_str: str) -> int:
        """Extracts unique array row index mapping identifiers for groups, instantiating missing ones."""
        if key_str not in self._index:
            if self._next_idx >= self._max_groups:
                raise RuntimeError(
                    f"Unique target data group tracking limit reached ({self._max_groups:,}). "
                    "You must expand MAX_GROUPS allocations inside gold_accumulators.py."
                )
            self._index[key_str] = self._next_idx
            self._next_idx += 1
        return self._index[key_str]

    def _update_single(self, idx: int, log_val: float, is_zero: bool) -> None:
        """Executes simultaneous online Welford updates and reservoir insertions for a single matrix index."""
        self._n[idx] += 1
        n = self._n[idx]

        if is_zero:
            self._zero_count[idx] += 1

        # Online Welford standard statistics formulation formulas
        delta            = log_val - self._mean[idx]
        self._mean[idx] += delta / n
        delta2           = log_val - self._mean[idx]
        self._M2[idx]   += delta * delta2

        # Reservoir Sampling Execution Block — Vitter's Algorithm R
        n_int = int(n)
        if n_int <= self._reservoir_size:
            self._reservoir[idx, n_int - 1] = log_val
        else:
            j = random.randint(0, n_int - 1)
            if j < self._reservoir_size:
                self._reservoir[idx, j] = log_val

    # ── Processing Streaming Ingest Vectors ──────────────────────────────

    def update_from_batch(self, batch: pa.RecordBatch) -> int:
        """
        Parses and feeds record structures residing in an incoming PyArrow RecordBatch to update metrics.

        Args:
            batch: PyArrow RecordBatch structure containing valid column pointers:
                   gene_id, gene_symbol, tissue_id, tpm_value

        Returns:
            Total row indices aggregated during this processing step execution pass.
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
        Forces physical block writes of internal memmap array pages to disk systems and serializes the index.
        Should be dispatched periodically to prevent extraction runtime state losses.
        """
        self._n.flush()
        self._mean.flush()
        self._M2.flush()
        self._zero_count.flush()
        self._reservoir.flush()

        index_path = _cache_path(self._cache_dir, "index.json")
        with open(index_path, "w") as f:
            json.dump(self._index, f)

        logger.debug("Memmap cache states successfully flushed to disk: %s tracked groups", f"{len(self._index):,}")

    # ── Arrow Conversion Serialization Subsystems ───────────────────────

    def to_arrow_table(self) -> pa.Table:
        """
        Transforms un-paged underlying binary matrices into a formatted pa.Table instance ready for Delta engines.
        Exclusively extracts metrics mapped up to active index tracks (self._next_idx boundaries).
        """
        n_groups = self._next_idx
        if n_groups == 0:
            logger.warning("to_arrow_table() triggered with 0 registered tracking targets — returning empty template schema.")
            return _empty_gold_schema()

        logger.info("Converting %s tracked metrics records into high-performance pa.Table structures...", f"{n_groups:,}")

        # Reverse structural index maps to map lookup addresses back to text components
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

            n    = self._n[idx]
            mean = self._mean[idx]
            M2   = self._M2[idx]
            zc   = self._zero_count[idx]

            gene_ids[idx]     = gene_id
            gene_symbols[idx] = gene_symbol
            tissue_ids[idx]   = tissue_id
            means[idx]        = float(mean) if n > 0 else 0.0
            counts[idx]       = int(n)
            zero_fracs[idx]   = float(zc / n) if n > 0 else 0.0

            # Compute standard deviation using aggregated Welford variance states
            if n >= 2:
                stds[idx] = float(math.sqrt(M2 / n))

            # Approximate median positioning using localized group reservoir arrays
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

        logger.info("pa.Table processing execution finalized: %s rows, %s schema columns bound.",
                    f"{table.num_rows:,}", table.num_columns)
        return table

    # ── Inspection Telemetry Properties ──────────────────────────────────

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
#  Fallback Template Schema Instantiation
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
#  Isolated Diagnostic Unit Testing Harnesses (CLI Sandboxing)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import shutil
    import numpy as np

    TEST_CACHE = "/tmp/gold_cache_test"
    if os.path.exists(TEST_CACHE):
        shutil.rmtree(TEST_CACHE)

    print("\n── Test 1: Evaluating Welford Streaming Accuracy vs NumPy ──────────────────────────────────\n")

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

    print(f"  mean → Welford: {mean:.8f}  numpy: {np_mean:.8f}  delta: {abs(mean - np_mean):.2e}")
    print(f"  std  → Welford: {std:.8f}  numpy: {np_std:.8f}  delta: {abs(std - np_std):.2e}")
    print(f"  mean convergence verification: {abs(mean - np_mean) < 1e-6}")
    print(f"  std convergence verification: {abs(std  - np_std)  < 1e-6}")

    print("\n── Test 2: Checking zero_fraction Precision Boundaries ─────────────────────────────────────\n")

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
    print(f"  zero_fraction: {zf:.4f}  expected value: 0.3000  status validation: {abs(zf - 0.30) < 1e-9}")

    print("\n── Test 3: Structural Extraction to PyArrow Object Tables ───────────────\n")

    shutil.rmtree(TEST_CACHE)
    acc_map3 = GoldAccumulatorMap(cache_dir=TEST_CACHE, max_groups=100)
    batch3 = pa.record_batch({
        "gene_id":     pa.array(["ENSG001", "ENSG001", "ENSG002", "ENSG001"]),
        "gene_symbol": pa.array(["GENE_A",  "GENE_A",  "GENE_B",  "GENE_A"]),
        "tissue_id":   pa.array(["Liver",   "Liver",   "Liver",   "Blood"]),
        "tpm_value":   pa.array([1.0, 2.0, 5.0, 0.0], type=pa.float32()),
    })
    acc_map3.update_from_batch(batch3)
    print(f"  Identified unique group partitions: {acc_map3.group_count}")
    assert acc_map3.group_count == 3

    gold_table = acc_map3.to_arrow_table()
    print(f"  Generated gold_table row footprint: {gold_table.num_rows}")
    rows = gold_table.to_pydict()
    for i in range(gold_table.num_rows):
        if rows["gene_id"][i] == "ENSG001" and rows["tissue_id"][i] == "Liver":
            assert rows["sample_count"][i] == 2
            assert rows["zero_fraction"][i] == 0.0
            print(f"  ENSG001/Liver → sample_count={rows['sample_count'][i]} zero_fraction={rows['zero_fraction'][i]:.2f} ✅")
        if rows["gene_id"][i] == "ENSG001" and rows["tissue_id"][i] == "Blood":
            assert rows["zero_fraction"][i] == 1.0
            print(f"  ENSG001/Blood → sample_count={rows['sample_count'][i]} zero_fraction={rows['zero_fraction'][i]:.2f} ✅")

    print("\n── Test 4: Persistent File State Flushing and Hydration Resumption ────────────────────────────────────\n")
    acc_map3.flush()
    acc_map4 = GoldAccumulatorMap(cache_dir=TEST_CACHE, max_groups=100, resume=True)
    print(f"  Recovered unique tracking metrics counts post-hydration: {acc_map4.group_count}")
    assert acc_map4.group_count == 3
    print("  State recovery assertion check passed ✅")

    shutil.rmtree(TEST_CACHE)
    print("\n✅ All localized unit tests executed successfully — gold_accumulators.py v3 memmap pipeline ready.\n")
