"""
src/utils/quality_checks.py

Pure quality validation functions for the Bio-AI Lakehouse data pipeline.
Each automated evaluation gate yields a QualityReport mapping sufficient operational 
telemetry and metadata to feed data lineage targets without extra logic inside callers.

Design Principles:
- Pure functions — side-effect free, fully testable without active Spark sessions or I/O stubs
- Atomic verdict targets — separate run passes; the orchestration layer controls halting thresholds
- Explicit assertion parameters — zero hardcoded magic numbers inside verification logic
- Strict Python 3.8 compatibility compliance (explicit Optional[X] over PEP 604 union structures)

Typical Orchestration Blueprint:
    from src.utils.quality_checks import run_bronze_checks, run_silver_checks

    result = run_bronze_checks(df, expected_rows=74628)
    if not result.passed:
        raise PipelineQualityError(result.summary())
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Individual Validation Node Outputs
# ─────────────────────────────────────────────

@dataclass
class CheckResult:
    """
    Evaluation metrics wrapper for a single business rule pass.
    passed=False coupled with severity='critical' triggers an immediate hard pipeline crash.
    passed=False coupled with severity='warning' logs a tracking marker but allows progression.
    """
    name:       str
    passed:     bool
    severity:   str        # 'critical' | 'warning'
    expected:   str        # Targeted configuration constraint criteria
    actual:     str        # Live evaluation footprint metrics observed
    detail:     str = ""   # Supplementary descriptive error traces

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else ("❌ FAIL" if self.severity == "critical" else "⚠️  WARN")
        return (
            f"{status} [{self.name}] "
            f"expected={self.expected} actual={self.actual}"
            + (f" | {self.detail}" if self.detail else "")
        )


# ─────────────────────────────────────────────
#  Aggregated Quality Suite Assessment Reports
# ─────────────────────────────────────────────

@dataclass
class QualityReport:
    """
    Consolidated summary tracking block evaluating an entire platform processing tier.
    passed=True evaluates to True if and only if all critical validation checks succeed.
    Warning indicators are bypassed during master lifecycle gate calculations.
    """
    layer:      str                                # 'bronze' | 'silver' | 'gold'
    checks:     List[CheckResult] = field(default_factory=list)
    run_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "critical")

    @property
    def critical_failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "critical"]

    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    def summary(self) -> str:
        lines = [
            f"── Quality Report [{self.layer.upper()}] ────────────────",
            f"  Verdict Outcome  : {'APPROVED ✅' if self.passed else 'FAILED ❌'}",
            f"  Checks Tracked   : {len(self.checks)} total, "
            f"{sum(c.passed for c in self.checks)} passed, "
            f"{len(self.critical_failures)} critical failures, "
            f"{len(self.warnings)} warnings",
            f"  Executed At UTC  : {self.run_at_utc}",
            "  Detailed Checks Summary Logs:",
        ]
        for c in self.checks:
            lines.append(f"    {c.summary()}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializes current audit snapshot map directly to lineage JSON store formats."""
        return {
            "layer":      self.layer,
            "passed":      self.passed,
            "run_at_utc": self.run_at_utc,
            "checks": [
                {
                    "name":     c.name,
                    "passed":   c.passed,
                    "severity": c.severity,
                    "expected": c.expected,
                    "actual":   c.actual,
                    "detail":   c.detail,
                }
                for c in self.checks
            ],
        }


# ─────────────────────────────────────────────
#  Atomic Quality Evaluation Gates (Pure Functions)
# ─────────────────────────────────────────────

def check_row_count(
    actual_count: int,
    expected_count: int,
    severity: str = "critical",
) -> CheckResult:
    """
    Asserts structural entry metrics mirror baseline volume assumptions exactly.
    Application mappings:
    - Bronze tier: actual == 74,628
    - Silver tier: actual == Bronze_rows × sample_cols (minus quarantine partitions)
    """
    passed = actual_count == expected_count
    return CheckResult(
        name     = "row_count",
        passed   = passed,
        severity = severity,
        expected = str(expected_count),
        actual   = str(actual_count),
        detail   = f"variance footprint={actual_count - expected_count:+,}" if not passed else "",
    )


def check_nulls_in_column(
    null_count: int,
    column_name: str,
    max_allowed: int = 0,
    severity: str = "critical",
) -> CheckResult:
    """
    Validates structural column tracks do not contain unmapped null records beyond tolerance thresholds.
    max_allowed=0 mandates zero-tolerance absolute validation boundaries (Primary Key tracks).
    """
    passed = null_count <= max_allowed
    return CheckResult(
        name     = f"nulls_{column_name}",
        passed   = passed,
        severity = severity,
        expected = f"<= {max_allowed}",
        actual   = str(null_count),
        detail   = f"target attribute '{column_name}' containing {null_count} unexpected null values" if not passed else "",
    )


def check_duplicates_in_index(
    duplicate_count: int,
    index_name: str = "gene_id",
    severity: str = "critical",
) -> CheckResult:
    """
    Evaluates transactional tables to block identifier duplication inside master index definitions.
    On Bronze configurations, the indexing column represents the unique baseline Ensembl ID array.
    """
    passed = duplicate_count == 0
    return CheckResult(
        name     = f"duplicates_{index_name}",
        passed   = passed,
        severity = severity,
        expected = "0",
        actual   = str(duplicate_count),
        detail   = f"Detected {duplicate_count} duplicate Ensembl IDs inside the targeted primary index track" if not passed else "",
    )


def check_quarantine_threshold(
    quarantine_count: int,
    total_count: int,
    max_fraction: float = 0.01,
    severity: str = "critical",
) -> CheckResult:
    """
    Evaluates isolation volume densities against system operation safety ceilings.
    If quarantine_count / total_count > max_fraction → hard pipeline execution HALT.

    Default parameter: An anomaly volume breakout scaling beyond >1% drops the cluster loop.
    Under standard GTEx distribution rules with clean raw baselines, unmapped records should hit 0.
    """
    if total_count == 0:
        fraction = 0.0
    else:
        fraction = quarantine_count / total_count

    passed = fraction <= max_fraction
    return CheckResult(
        name     = "quarantine_threshold",
        passed   = passed,
        severity = severity,
        expected = f"<= {max_fraction:.1%}",
        actual   = f"{fraction:.2%} ({quarantine_count:,} / {total_count:,})",
        detail   = f"Excessive unmatched sample tracking volume discovered — verify configuration context in gtex_metadata.txt" if not passed else "",
    )


def check_zero_fraction(
    actual_fraction: float,
    baseline_fraction: float = 0.5189,
    tolerance: float = 0.05,
    severity: str = "warning",
) -> CheckResult:
    """
    Monitors unexpressed zero-value matrix distributions across Silver long transformations.
    Static Reference: ~51.89% unexpressed values across raw components (profiling_report.json).
    System window boundary: ±5% variance parameters account for metadata filtering offsets.
    Set to warning (non-critical) — structural zeros are biologically expected gene behaviors.
    """
    lower = baseline_fraction - tolerance
    upper = baseline_fraction + tolerance
    passed = lower <= actual_fraction <= upper
    return CheckResult(
        name     = "zero_fraction",
        passed   = passed,
        severity = severity,
        expected = f"{lower:.1%} – {upper:.1%}",
        actual   = f"{actual_fraction:.2%}",
        detail   = (
            f"Zero-expression ratio skew detected: value mapping resolves as {'critically high' if actual_fraction > upper else 'critically low'} "
            f"relative to target baseline criteria ({baseline_fraction:.1%})"
        ) if not passed else "",
    )


def check_lineage_closure(
    silver_rows: int,
    quarantine_rows: int,
    bronze_genes: int = 74_628,
    bronze_samples: int = 19_788,
    severity: str = "critical",
) -> CheckResult:
    """
    Audits execution mathematical closure rules post structural wide-to-long flattening loops.
    Assertion identity formula: silver_rows + quarantine_rows == bronze_genes × bronze_samples

    A validation drop indicates a silent structural record leak occurred inside transformation.
    This check acts as the primary logical gate validating the Silver aggregation stage.
    """
    expected_total = bronze_genes * bronze_samples
    actual_total   = silver_rows + quarantine_rows
    passed         = actual_total == expected_total

    return CheckResult(
        name     = "lineage_closure",
        passed   = passed,
        severity = severity,
        expected = f"{expected_total:,} ({bronze_genes:,} genes × {bronze_samples:,} samples)",
        actual   = f"{actual_total:,} (silver_layer={silver_rows:,} + quarantine_layer={quarantine_rows:,})",
        detail   = f"Mathematical alignment breach: undetected record leak of {actual_total - expected_total:+,} row values" if not passed else "",
    )


# ─────────────────────────────────────────────
#  Aggregated Validation Suites per Tier Layer
# ─────────────────────────────────────────────

def run_bronze_checks(
    df_index_duplicates: int,
    df_row_count: int,
    description_null_count: int,
    expected_rows: int = 74_628,
) -> QualityReport:
    """
    Executes the consolidated quality evaluation suite for raw Bronze parquet landing zones.

    Args:
        df_index_duplicates:    Sum total extracted via df.index.duplicated().sum()
        df_row_count:           Count metrics output via len(df)
        description_null_count: Null validation tracking count mapping via df['Description'].isnull().sum()
        expected_rows:          Standard volume baseline fixed configuration floor tracking target (74,628 for GTEx v11)

    Returns:
        QualityReport container. passed=True only when every critical constraint evaluates successfully.

    Blueprint Configuration usage:
        import pandas as pd
        df = pd.read_parquet('data/bronze/gtex/gene_tpm_raw.parquet', columns=['Description'])
        report = run_bronze_checks(
            df_index_duplicates    = df.index.duplicated().sum(),
            df_row_count          = len(df),
            description_null_count = df['Description'].isnull().sum(),
        )
        print(report.summary())
    """
    report = QualityReport(layer="bronze")

    report.checks.append(check_row_count(df_row_count, expected_rows))
    report.checks.append(check_duplicates_in_index(df_index_duplicates, index_name="Name"))
    report.checks.append(check_nulls_in_column(description_null_count, "Description"))

    _log_report(report)
    return report


def run_silver_checks(
    silver_row_count: int,
    quarantine_row_count: int,
    tissue_id_null_count: int,
    gene_id_null_count: int,
    sample_id_null_count: int,
    actual_zero_fraction: float,
    bronze_genes: int = 74_628,
    bronze_samples: int = 19_788,
    quarantine_threshold: float = 0.01,
) -> QualityReport:
    """
    Executes the consolidated quality validation suite for the reshaped long Silver Delta tier.

    Args:
        silver_row_count:        Record entry matrix counts targeted for Delta Silver tables
        quarantine_row_count:    Orphaned metadata fragments routed out to system isolation paths
        tissue_id_null_count:    Missing reference instances tracking inside the tissue_id column
        gene_id_null_count:      Missing reference instances tracking inside the gene_id column
        sample_id_null_count:    Missing reference instances tracking inside the sample_id column
        actual_zero_fraction:    Observed expression calculation limits inside computed tpm_value distributions
        bronze_genes:            Upstream matrix scaling factor mapping baseline variables (Default: 74,628)
        bronze_samples:          Upstream matrix scaling factor mapping baseline variables (Default: 19,788)
        quarantine_threshold:    Maximum allowed percentage limit before safety system faults fire (Default: 1%)

    Returns:
        QualityReport tracking state summaries with absolute verification checks recorded.
    """
    report = QualityReport(layer="silver")
    total  = silver_row_count + quarantine_row_count

    report.checks.append(check_nulls_in_column(gene_id_null_count,    "gene_id"))
    report.checks.append(check_nulls_in_column(sample_id_null_count,  "sample_id"))
    report.checks.append(check_nulls_in_column(tissue_id_null_count,  "tissue_id"))
    report.checks.append(check_quarantine_threshold(
        quarantine_row_count, total, max_fraction=quarantine_threshold
    ))
    report.checks.append(check_lineage_closure(
        silver_row_count, quarantine_row_count, bronze_genes, bronze_samples
    ))
    report.checks.append(check_zero_fraction(actual_zero_fraction))

    _log_report(report)
    return report


# ─────────────────────────────────────────────
#  Internal Logging Subroutines
# ─────────────────────────────────────────────

def _log_report(report: QualityReport) -> None:
    if report.passed:
        logger.info("Quality gate check pass [%s] APPROVED — %d constraint rules confirmed successfully",
                    report.layer.upper(), len(report.checks))
    else:
        logger.error("Quality gate check pass [%s] CRITICAL FAULT — Detected %d structural exception conditions",
                     report.layer.upper(), len(report.critical_failures))
        for failure in report.critical_failures:
            logger.error("  -> Trace Exception detail: %s", failure.summary())


# ─────────────────────────────────────────────
#  Pipeline Exception Definitions
# ─────────────────────────────────────────────

class PipelineQualityError(Exception):
    """
    Raised directly to halt runtime scheduling when an active processing block breaches critical quality floor limits.
    Injects complete historical diagnostic summary tables directly back to core logs.
    """
    def __init__(self, report: QualityReport):
        self.report = report
        super().__init__(
            f"Orchestration terminating: Quality validation floor criteria breached at layer [{report.layer.upper()}] with "
            f"{len(report.critical_failures)} unresolvable structural failures:\n"
            + report.summary()
        )


# ─────────────────────────────────────────────
#  Gold Optimization & Summary Verification Layer
# ─────────────────────────────────────────────

def check_gold_row_count(
    actual_count:   int,
    expected_genes: int = 74_628,
    expected_tissues: int = 68,
    severity: str = "critical",
) -> CheckResult:
    """
    Asserts structural entry volumes perfectly match the cross-joined dimensions product (genes × tissues).
    Tolerance boundary: Absolute strict zero-variance rule — compilation groups must align flawlessly.
    """
    expected = expected_genes * expected_tissues
    passed   = actual_count == expected
    return CheckResult(
        name     = "gold_row_count",
        passed   = passed,
        severity = severity,
        expected = f"{expected:,} ({expected_genes:,} genes × {expected_tissues} distinct tissue dimensions)",
        actual   = f"{actual_count:,}",
        detail   = f"Aggregation balance divergence detected: volume offset counts evaluate to={actual_count - expected:+,}" if not passed else "",
    )


def check_gold_tissue_count(
    actual_tissue_count:   int,
    expected_tissue_count: int = 68,
    severity: str = "critical",
) -> CheckResult:
    """
    Asserts aggregation profiles comprehensively map every single distinct target tissue segment (Default 68).
    Missing tissue indices imply filtering processing leaks occurred downstream from Silver staging tables.
    """
    passed = actual_tissue_count == expected_tissue_count
    return CheckResult(
        name     = "gold_tissue_count",
        passed   = passed,
        severity = severity,
        expected = str(expected_tissue_count),
        actual   = str(actual_tissue_count),
        detail   = (
            f"Dimensional validation failure: record profiling tracks show "
            f"{'unregistered missing' if actual_tissue_count < expected_tissue_count else 'unexpected excess overhead calculations numbering'} "
            f"{abs(actual_tissue_count - expected_tissue_count)} individual tissues"
        ) if not passed else "",
    )


def check_gold_min_sample_count(
    min_sample_count: int,
    min_allowed: int = 1,
    severity: str = "critical",
) -> CheckResult:
    """
    Ensures zero empty relational tracking nodes exist across compiled dimensional lookups.
    An active matrix record containing zero combined instances points to a functional bug inside matrix accumulators.
    """
    passed = min_sample_count >= min_allowed
    return CheckResult(
        name     = "gold_min_sample_count",
        passed   = passed,
        severity = severity,
        expected = f">= {min_allowed}",
        actual   = str(min_sample_count),
        detail   = "Invalid analytical tracking groups detected: structural groupings evaluate with 0 sample points — check engine sum logic logs" if not passed else "",
    )


def check_gold_zero_fraction_consistency(
    avg_zero_fraction:      float,
    silver_zero_fraction:   float = 0.5252,
    tolerance:              float = 0.05,
    severity:               str   = "warning",
) -> CheckResult:
    """
    Validates global mean non-expression metrics are mathematically consistent with Silver inputs.
    Baseline context: Expected target threshold anchors around ~52.52% derived from long-form verification traces.

    Severity evaluation set to warning — fractional drift marks changes in cluster density profiles, 
    not underlying storage corruption risks.
    """
    lower  = silver_zero_fraction - tolerance
    upper  = silver_zero_fraction + tolerance
    passed = lower <= avg_zero_fraction <= upper
    return CheckResult(
        name     = "gold_zero_fraction_consistency",
        passed   = passed,
        severity = severity,
        expected = f"{lower:.1%} – {upper:.1%}",
        actual   = f"{avg_zero_fraction:.2%}",
        detail   = (
            f"Aggregated Gold zero-expression metrics reflect unexpected skew: value registers as {'higher than standard profiles' if avg_zero_fraction > upper else 'lower than standard profiles'} "
            f"relative to Silver framework metrics target baseline ({silver_zero_fraction:.1%})"
        ) if not passed else "",
    )


# ─────────────────────────────────────────────
#  Gold Analytical Quality Framework Suite
# ─────────────────────────────────────────────

def run_gold_checks(
    gold_row_count:         int,
    tissue_count:           int,
    gene_id_null_count:     int,
    gene_symbol_null_count: int,
    tissue_id_null_count:   int,
    min_sample_count:       int,
    avg_zero_fraction:      float,
    expected_genes:         int   = 74_628,
    expected_tissues:       int   = 68,
    silver_zero_fraction:   float = 0.5252,
) -> QualityReport:
    """
    Executes the comprehensive analytical validation quality suite for aggregated Gold consumption layers.

    Args:
        gold_row_count:         Calculated array volumes committed to downstream Delta Gold storage targets
        tissue_count:           Total volume count of distinct localized tissue definitions discovered
        gene_id_null_count:     Evaluation null metrics checking the primary structural matrix key
        gene_symbol_null_count: Evaluation null metrics checking the primary structural matrix key
        tissue_id_null_count:   Evaluation null metrics checking the primary structural matrix key
        min_sample_count:       Calculated floor validation density found within individual grouping maps
        avg_zero_fraction:      Evaluated arithmetic mean tracking global silent expressions across arrays
        expected_genes:         Total baseline gene constraints tracked via upstream systems (74,628)
        expected_tissues:       Total tissue groupings target verified within transformation models (68)
        silver_zero_fraction:   Historical baseline parameters pulled directly from validation states (52.52%)

    Returns:
        QualityReport containing complete runtime operational tracking variables.

    Critical Boundaries (Halts pipeline scheduling instantly upon error conditions):
        - gold_row_count           : Demands flawless matrix dimensional density alignment
        - gold_tissue_count        : Demands exact confirmation that all 68 tissue structures exist
        - Key integrity constraints: Absolute zero-tolerance null value parameters inside identifiers
        - gold_min_sample_count    : Forbids tracking calculations over missing or unpopulated groupings

    Warning Evaluation Constraints (Logs operational profile drifts without breaking process execution):
        - zero_fraction_consistency: Validates matrix zero tracking falls inside expected bounds (±5%)
    """
    report = QualityReport(layer="gold")

    # Critical Identifiers — Index Integrity Null Scans
    report.checks.append(check_nulls_in_column(gene_id_null_count,     "gene_id"))
    report.checks.append(check_nulls_in_column(gene_symbol_null_count, "gene_symbol"))
    report.checks.append(check_nulls_in_column(tissue_id_null_count,   "tissue_id"))

    # Critical Identifiers — Relational Volume Structural Audits
    report.checks.append(check_gold_row_count(
        gold_row_count, expected_genes, expected_tissues
    ))
    report.checks.append(check_gold_tissue_count(tissue_count, expected_tissues))
    report.checks.append(check_gold_min_sample_count(min_sample_count))

    # Warning Indicators — Analytical Trend Context Mapping
    report.checks.append(check_gold_zero_fraction_consistency(
        avg_zero_fraction, silver_zero_fraction
    ))

    _log_report(report)
    return report


# ─────────────────────────────────────────────
#  Local Integration Validation Harness Sandbox (CLI Targets)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Sandbox Execution Trace: Bronze Quality Passes (Profiling Benchmarks) ===\n")
    bronze_report = run_bronze_checks(
        df_index_duplicates    = 0,
        df_row_count           = 74_628,
        description_null_count = 0,
    )
    print(bronze_report.summary())

    print("\n=== Sandbox Execution Trace: Silver Quality Passes (Post-Reshape Simulation) ===\n")
    silver_report = run_silver_checks(
        silver_row_count      = 74_628 * 19_788,
        quarantine_row_count  = 0,
        tissue_id_null_count  = 0,
        gene_id_null_count    = 0,
        sample_id_null_count  = 0,
        actual_zero_fraction  = 0.5189,
    )
    print(silver_report.summary())

    print("\n=== Sandbox Execution Trace: Simulated Silver Validation Failure Scenario (Data Leak) ===\n")
    fail_report = run_silver_checks(
        silver_row_count      = 74_628 * 19_788 - 1000,
        quarantine_row_count  = 0,
        tissue_id_null_count  = 0,
        gene_id_null_count    = 0,
        sample_id_null_count  = 0,
        actual_zero_fraction  = 0.5189,
    )
    print(fail_report.summary())

    print("\n=== Sandbox Execution Trace: Gold Quality Passes (Target Design Baselines) ===\n")
    gold_report = run_gold_checks(
        gold_row_count         = 74_628 * 68,
        tissue_count           = 68,
        gene_id_null_count     = 0,
        gene_symbol_null_count = 0,
        tissue_id_null_count   = 0,
        min_sample_count       = 5,
        avg_zero_fraction      = 0.5252,
    )
    print(gold_report.summary())

    print("\n=== Sandbox Execution Trace: Simulated Gold Validation Failure Scenario (Volume Mismatch) ===\n")
    gold_fail = run_gold_checks(
        gold_row_count         = 74_628 * 68 - 500,
        tissue_count           = 68,
        gene_id_null_count     = 0,
        gene_symbol_null_count = 0,
        tissue_id_null_count   = 0,
        min_sample_count       = 5,
        avg_zero_fraction      = 0.5252,
    )
    print(gold_fail.summary())
