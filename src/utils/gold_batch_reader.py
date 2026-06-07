"""
src/utils/gold_batch_reader.py

Streaming Silver-tier layer parsing in batches for Gold processing — engine-agnostic, zero-Spark.

Problem: Silver layer consists of ~3,800 Parquet partitions totaling 1.5B rows. 
Loading this scale completely into RAM is impossible with only 2-3GB available system ceilings.

Solution: A generator pipeline that yields individual PyArrow RecordBatches sequentially. 
The caller wrapper (gold_transform.py) consumes and drops each isolated batch, locking down the 
maximum live active memory allocation footprint to a single batch at any given point in time.

Dynamic Batch-Size Strategy:
    - psutil monitors available hardware memory maps immediately before evaluating each file partition.
    - batch_size recalculates if the available resource context shifts past established thresholds.
    - Eliminates hardcoded row limits — self-adjusts seamlessly to real-world environments.

Compatibility:
    - Reads chunks via pq.ParquetFile().iter_batches() — mirrors Silver Phase 2 execution modes.
    - Automatically filters out internal engine metadata paths like _delta_log.
    - Strict Python 3.8 typing adherence (Optional[X] used over PEP 604 X | None notation).

Standard Deployment Example inside gold_transform.py:
    reader = SilverBatchReader(silver_root="data/silver/gtex/gene_expression_long")
    for batch in reader.iter_batches():
        acc_map.update_from_batch(batch)
"""

import logging
import pathlib
from typing import Generator, List, Optional

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Column targets required by Gold aggregation — leaves out sample_id (saves ~20% batch memory space)
GOLD_COLUMNS = ["gene_id", "gene_symbol", "tissue_id", "tpm_value"]

# Boundary safety constraints managing batch row thresholds
MIN_BATCH_ROWS = 10_000
MAX_BATCH_ROWS = 500_000

# Available memory resource fraction allocated per processing iteration step
RAM_BATCH_FRACTION = 0.15  # 15% — defensive threshold leaving safe overhead for data storage matrices

# Calculated runtime byte usage allocation estimate per row (4 strings + 1 float32 ≈ ~200 bytes)
BYTES_PER_ROW_ESTIMATE = 200


# ─────────────────────────────────────────────
#  Dynamic Batch-Size Computations — Pure Function
# ─────────────────────────────────────────────

def calculate_batch_size(available_ram_bytes: Optional[int] = None) -> int:
    """
    Computes optimal target chunk row allocations relative to currently available system memory maps.

    Args:
        available_ram_bytes: Physical system available memory capacity slice in bytes.
                             If None, queries real-time resource indicators using psutil.

    Returns:
        Optimized target size value clamped safely between MIN_BATCH_ROWS and MAX_BATCH_ROWS limits.

    Example context with 2GB available system memory:
        budget = 2GB × 0.15 = 307MB
        rows   = 307MB / 200 bytes = ~1,535,000 → clamped value → 500,000

    Example context with 512MB available system memory:
        budget = 512MB × 0.15 = 76MB
        rows   = 76MB / 200 bytes = ~380,000 → clamped value → 380,000
    """
    if available_ram_bytes is None:
        available_ram_bytes = psutil.virtual_memory().available

    budget_bytes = available_ram_bytes * RAM_BATCH_FRACTION
    raw_rows     = int(budget_bytes / BYTES_PER_ROW_ESTIMATE)

    batch_size = max(MIN_BATCH_ROWS, min(raw_rows, MAX_BATCH_ROWS))

    logger.debug(
        "Calculated batch_size: %s rows (available_ram=%.1fGB, system_budget=%.0fMB)",
        f"{batch_size:,}",
        available_ram_bytes / (1024 ** 3),
        budget_bytes / (1024 ** 2),
    )

    return batch_size


# ─────────────────────────────────────────────
#  Silver Storage Location Discovery
# ─────────────────────────────────────────────

def discover_silver_files(silver_root: str) -> List[pathlib.Path]:
    """
    Scans, extracts, and compiles all valid Parquet files while bypassing metadata logs (_delta_log).

    Args:
        silver_root: Root file path target representing the Silver Delta Lake location.

    Returns:
        Sorted array of file system path points, providing deterministic ordering for workflow tracing.

    Raises:
        FileNotFoundError: Triggered if the targeted entry directory path is invalid.
        RuntimeError: Triggered if discovery steps yield zero data targets.
    """
    root = pathlib.Path(silver_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Target Silver storage path location not found: {silver_root}\n"
            "Verify that silver_phase2_delta.py was successfully completed first."
        )

    files = sorted([
        p for p in root.rglob("*.parquet")
        if "_delta_log" not in str(p)
    ])

    if not files:
        raise RuntimeError(
            f"No valid Parquet files recovered under the target path context: {silver_root}\n"
            "The Silver partition might be completely un-allocated or corrupt."
        )

    logger.info("Silver Layer Discovery: Found %s valid Parquet file units at location: %s", f"{len(files):,}", silver_root)
    return files


# ─────────────────────────────────────────────
#  Silver Batch Stream Interface
# ─────────────────────────────────────────────

class SilverBatchReader:
    """
    Streaming generator engine yielding sequential PyArrow RecordBatches from Silver to Gold pipelines.

    Processes targets systematically across data splits.
    Maximum execution memory floor: restricts live allocations to a single batch in RAM at any given moment.

    Post-Iteration Metric Telemetry Fields:
        files_processed  : Total distinct file units processed.
        batches_yielded  : Cumulative batch elements dispatched to the caller runtime loop.
        rows_yielded     : Unified row volume extracted and transmitted.
    """

    def __init__(
        self,
        silver_root: str = "data/silver/gtex/gene_expression_long",
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            silver_root: Root file path target representing the Silver Delta Lake location.
            columns:     Target field arrays to extract. Default fallback layout: GOLD_COLUMNS
                         (implicitly bypasses sample_id references to conserve memory context overhead).
        """
        self.silver_root = silver_root
        self.columns     = columns or GOLD_COLUMNS

        # Metric tracking blocks — populated completely at termination of iter_batches() execution
        self.files_processed: int = 0
        self.batches_yielded: int = 0
        self.rows_yielded:    int = 0

        # Scan and verify entry files during execution setup — exits fast if directory routes are broken
        self._files = discover_silver_files(silver_root)

    @property
    def total_files(self) -> int:
        return len(self._files)

    def iter_batches(self) -> Generator[pa.RecordBatch, None, None]:
        """
        Primary engine generator stream loop. Delivers a sequential, isolated pa.RecordBatch element at each iteration step.

        Automatically triggers batch optimization updates every 100 partitions to safely dynamically adapt to 
        shifting environment states (e.g., competing workflows, accumulation layer memory shifts).

        Usage Pattern:
            for batch in reader.iter_batches():
                acc_map.update_from_batch(batch)
        """
        # Re-initialize evaluation metric fields before executing a sequence tracking loop pass
        self.files_processed = 0
        self.batches_yielded = 0
        self.rows_yielded    = 0

        batch_size   = calculate_batch_size()
        recalc_every = 100  # Evaluate system resource metrics every N files processed

        logger.info(
            "Launching streaming pipeline across Silver blocks: %s files identified, targeting baseline batch_size=%s rows, columns=%s",
            f"{self.total_files:,}", f"{batch_size:,}", self.columns,
        )

        for file_idx, parquet_path in enumerate(self._files):

            # Periodic batch volume boundary adjustments based on system memory changes
            if file_idx > 0 and file_idx % recalc_every == 0:
                new_batch_size = calculate_batch_size()
                if new_batch_size != batch_size:
                    logger.info(
                        "Recalibrated batch_size parameters dynamically adjusted: %s → %s (File index position: %s/%s)",
                        f"{batch_size:,}", f"{new_batch_size:,}",
                        f"{file_idx:,}", f"{self.total_files:,}",
                    )
                    batch_size = new_batch_size

            # Issue progress metrics at 500 file operational steps
            if file_idx % 500 == 0 and file_idx > 0:
                pct = file_idx / self.total_files * 100
                logger.info(
                    "Pipeline Progress Status: %s/%s files parsed (%.1f%% complete) — %s rows successfully dispatched.",
                    f"{file_idx:,}", f"{self.total_files:,}",
                    pct, f"{self.rows_yielded:,}",
                )

            try:
                pf = pq.ParquetFile(parquet_path)
                for batch in pf.iter_batches(
                    batch_size=batch_size,
                    columns=self.columns,
                ):
                    self.batches_yielded += 1
                    self.rows_yielded    += batch.num_rows
                    yield batch

            except Exception as e:
                # Catch, log anomalies, and preserve operational state — an isolated corrupted batch block must not halt the pipeline
                logger.warning(
                    "Bypassing defective file block structure to avoid pipeline halt: %s — Reason: %s", parquet_path, e
                )
                continue

            self.files_processed += 1

        logger.info(
            "Streaming extraction sequence successfully finalized: %s files handled, %s batches packed, %s cumulative rows dispatched.",
            f"{self.files_processed:,}",
            f"{self.batches_yielded:,}",
            f"{self.rows_yielded:,}",
        )

    def summary(self) -> str:
        """Returns structured string summaries tracking operational lineage metrics."""
        return (
            f"files={self.files_processed:,} "
            f"batches={self.batches_yielded:,} "
            f"rows={self.rows_yielded:,} "
            f"batch_size_initial={calculate_batch_size():,}"
        )


# ─────────────────────────────────────────────
#  Isolated Validation Harness Sandbox Testing (CLI Execution Target)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n── Test 1: Testing calculate_batch_size Boundary Logic ──────────────────────────────\n")

    # Simulate hypothetical RAM volume shifts
    test_cases = [
        512  * 1024 ** 2,   # 512 MB
        1    * 1024 ** 3,   # 1 GB
        2    * 1024 ** 3,   # 2 GB  ← Target operational environment baseline context
        8    * 1024 ** 3,   # 8 GB
    ]
    for ram in test_cases:
        bs = calculate_batch_size(ram)
        print(f"  Simulated RAM availability={ram / 1024**3:.1f}GB → Derived batch_size allocation blueprint={bs:,}")

    print("\n── Test 2: Evaluating batch_size Real-Time Hardware Profile ───────────────────────────\n")
    bs_real = calculate_batch_size()
    ram_gb  = psutil.virtual_memory().available / 1024 ** 3
    print(f"  Live environment reported available RAM capacity: {ram_gb:.2f} GB")
    print(f"  Computed runtime loop targeted batch_size matrix ceiling: {bs_real:,} rows")
    print(f"  Verifying boundary limit integrity constraints: {MIN_BATCH_ROWS <= bs_real <= MAX_BATCH_ROWS}")

    print("\n── Test 3: Checking discover_silver_files Path Extraction Performance ───────────────\n")
    silver_path = "data/silver/gtex/gene_expression_long"
    try:
        files = discover_silver_files(silver_path)
        print(f"  Recovered target data files matching criteria: {len(files):,}")
        print(f"  First partition block identified: {files[0].name}")
        print(f"  Final partition block identified: {files[-1].name}")
        print(f"  Confirming complete isolation of internal engine metadata blocks (_delta_log verification): {not any('_delta_log' in str(f) for f in files)}")
    except FileNotFoundError as e:
        print(f"  Silver directory tracks absent from current local environment profiling context — skipping execution step (Standard fallback path behavior for remote CI runners).")
        print(f"  Error structural layout detail logging: ({e})")
        sys.exit(0)

    print("\n── Test 4: Profiling iter_batches Sequence Streams (Constrained file slice evaluation) ────────────────\n")
    reader  = SilverBatchReader(silver_path)
    batches = 0
    rows    = 0

    # Overwrite internal array tracks with a limited 3-file target slice to run unit evaluation passes safely
    reader._files = reader._files[:3]

    for batch in reader.iter_batches():
        batches += 1
        rows    += batch.num_rows
        # Extract and verify the structural blueprint attributes inside the first generated chunk
        if batches == 1:
            cols = batch.schema.names
            print(f"  Extracted active schema fields layout from batch tracking instance: {cols}")
            assert "gene_id"    in cols, "❌ Structural extraction failure: gene_id absent from schema fields."
            assert "tpm_value"  in cols, "❌ Structural extraction failure: tpm_value absent from schema fields."
            assert "sample_id" not in cols, "❌ Structural constraint failure: sample_id field should have been skipped."
            print(f"  sample_id tracking isolation verified: ✅")

    print(f"  Cumulative array batch tracking units yielded: {batches}")
    print(f"  Integrated matrix line items counted: {rows:,}")
    print(f"  Telemetries reported for processed file records: {reader.files_processed}")
    print(f"  Constructed lineage block verification tracker summary: {reader.summary()}")
    print(f"\n✅ Pipeline layer components validated — gold_batch_reader.py operational workflow configurations complete.\n")
