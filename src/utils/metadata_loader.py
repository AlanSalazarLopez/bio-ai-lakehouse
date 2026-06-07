"""
src/utils/metadata_loader.py

Loads the GTEx sample metadata file and builds the SAMPID → SMTSD mapping
(sample_id → tissue_id) used during the Silver-tier ingestion join pass.

The gtex_metadata.txt file is a wide TSV file containing ~100 structural columns.
This module extracts only two critical fields: SAMPID (key) and SMTSD (target tissue dimension).

Primary Interface: load_tissue_mapping
    → returns dict {sample_id: tissue_id} ready for the Silver tier join operations

Secondary Interface: validate_tissue_mapping
    → validates that the derived map contains data and records baseline footprint telemetry

Strict Python 3.8 typing adherence (explicit Optional[X] over PEP 604 union structures).
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Target columns extracted from the wide metadata TSV — all remaining attributes are skipped
SAMPID_COL = "SAMPID"
SMTSD_COL  = "SMTSD"

# Default execution deployment track within the container file system
DEFAULT_METADATA_PATH = "data/raw/gtex_metadata.txt"


# ─────────────────────────────────────────────
#  Primary Ingestion Interface
# ─────────────────────────────────────────────

def load_tissue_mapping(
    path: str = DEFAULT_METADATA_PATH,
) -> Dict[str, str]:
    """
    Parses gtex_metadata.txt and returns a lookup mapping dictionary: {sample_id → tissue_id}.

    Slices only the target columns SAMPID and SMTSD — ignoring the remaining ~100 attributes.
    Records with empty or un-allocated values in either target column are omitted and logged as warnings.

    Args:
        path: File system location path pointing to the raw GTEx metadata TSV file.

    Returns:
        Dict[str, str] — {sample_id: tissue_id} map structure.
        Example block: {"GTEX-1117F-0005-SM-HL9SH": "Thyroid"}

    Raises:
        FileNotFoundError: Triggered if the target metadata file path is invalid or absent.
        KeyError: Triggered if specified target columns (SAMPID or SMTSD) are missing from the header line.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GTEx metadata asset source not found at destination path: {path}\n"
            f"Verify that the required source file is staged under: data/raw/"
        )

    mapping: Dict[str, str] = {}
    skipped_empty = 0
    total_rows    = 0

    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        columns     = header_line.split("\t")

        # Validate presence of schema identifier targets
        if SAMPID_COL not in columns:
            raise KeyError(
                f"Required key column identifier '{SAMPID_COL}' absent from schema metadata inside {path}. "
                f"Available heading tracks up to index limit: {columns[:10]}..."
            )
        if SMTSD_COL not in columns:
            raise KeyError(
                f"Required dimension column identifier '{SMTSD_COL}' absent from schema metadata inside {path}. "
                f"Available heading tracks up to index limit: {columns[:10]}..."
            )

        sampid_idx = columns.index(SAMPID_COL)
        smtsd_idx  = columns.index(SMTSD_COL)

        for line in f:
            total_rows += 1
            parts = line.rstrip("\n").split("\t")

            # Boundary protection logic avoiding index errors on fragmented lines
            if len(parts) <= max(sampid_idx, smtsd_idx):
                skipped_empty += 1
                continue

            sample_id = parts[sampid_idx].strip()
            tissue_id = parts[smtsd_idx].strip()

            if not sample_id or not tissue_id:
                skipped_empty += 1
                logger.debug(
                    "Dropping incomplete line item record — Captured inputs: SAMPID='%s' SMTSD='%s'",
                    sample_id, tissue_id
                )
                continue

            mapping[sample_id] = tissue_id

    if skipped_empty > 0:
        logger.warning(
            "metadata_loader tracking: Omitted %d/%d entries due to blank value blocks in SAMPID or SMTSD positions",
            skipped_empty, total_rows
        )

    logger.info(
        "metadata_loader tracking: Successfully compiled %d sample records from target location: %s",
        len(mapping), path
    )

    return mapping


# ─────────────────────────────────────────────
#  Mapping Evaluation Gate
# ─────────────────────────────────────────────

def validate_tissue_mapping(
    mapping: Dict[str, str],
    min_samples: int = 100,
) -> Tuple[bool, str]:
    """
    Evaluates the compiled mapping structure to guarantee sufficient sample density before pipeline execution.

    Args:
        mapping:     Extracted key-value lookup array resulting from load_tissue_mapping()
        min_samples: Floor constraint defining the minimum volume threshold of required tracking records (default 100)

    Returns:
        Tuple[bool, str] — (passed, audit_message_details)
    """
    if not mapping:
        return False, "Validation failure: Target map completely unpopulated — gtex_metadata.txt yielded 0 records."

    if len(mapping) < min_samples:
        return False, (
            f"Validation failure: Target sample map volume falls beneath safety parameters: {len(mapping)} records < required floor configuration ({min_samples})."
        )

    tissues       = set(mapping.values())
    tissue_counts = {}
    for tissue in mapping.values():
        tissue_counts[tissue] = tissue_counts.get(tissue, 0) + 1

    top_tissue    = max(tissue_counts, key=tissue_counts.__getitem__)
    bottom_tissue = min(tissue_counts, key=tissue_counts.__getitem__)

    msg = (
        f"Structural validation map verified: {len(mapping):,} samples parsed across {len(tissues)} distinct tissue sets | "
        f"Max density track={top_tissue} ({tissue_counts[top_tissue]:,} records) | "
        f"Min density track={bottom_tissue} ({tissue_counts[bottom_tissue]:,} records)"
    )

    logger.info("validate_tissue_mapping confirmation status: %s", msg)
    return True, msg


# ─────────────────────────────────────────────
#  Silver Ingestion Transformation Helpers
# ─────────────────────────────────────────────

def get_tissue_or_unknown(
    mapping: Dict[str, str],
    sample_id: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """
    Safely executes sample lookup queries against the memory lookup dictionary.
    Returns None (or a specific fallback block target) if no match is hit — handing control to the caller to flag quarantine.

    Usage blueprint within silver_transform.py:
        tissue = get_tissue_or_unknown(mapping, sample_id)
        if tissue is None:
            → routing downstream out to quarantine targets
    """
    return mapping.get(sample_id, fallback)


# ─────────────────────────────────────────────
#  Local Integration Validation Harness Sandbox (CLI Targets)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_METADATA_PATH

    print(f"\n── Loading dataset metadata from target destination: {path} ──────────")
    mapping = load_tissue_mapping(path)

    passed, msg = validate_tissue_mapping(mapping)
    status = "✅" if passed else "❌"
    print(f"{status} {msg}")

    # Display a brief 3-element sample tranche of the dictionary mapping
    print("\n── Structural Map Sample Elements Preview ──────────────────")
    for i, (sample_id, tissue_id) in enumerate(mapping.items()):
        print(f"  {sample_id} → {tissue_id}")
        if i >= 2:
            break

    # Execute isolated lookup behavior testing patterns
    print("\n── Evaluating get_tissue_or_unknown Behavior Patterns ────────────")
    test_id = "GTEX-FAKE-0000-SM-XXXXX"
    result  = get_tissue_or_unknown(mapping, test_id)
    print(f"  Querying simulated invalid sample_id instance target → {result}")

    first_sample = next(iter(mapping))
    result2 = get_tissue_or_unknown(mapping, first_sample)
    print(f"  Querying initial actual data sample tracking node → {result2}")
