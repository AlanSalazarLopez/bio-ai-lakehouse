"""
src/utils/quality_checks.py

Funciones puras de validación de calidad para el pipeline Bio-AI Lakehouse.
Cada check retorna un QualityResult con el veredicto y metadata suficiente
para actualizar el data lineage sin lógica adicional en el caller.

Diseño:
- Funciones puras — sin side-effects, testeables sin Spark ni filesystem
- Un resultado por check — el caller decide si parar o continuar
- Threshold explícito en cada check — sin valores mágicos
- Compatible con Python 3.8 (Optional[X], no X | None)

Uso típico:
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
#  Resultado de un check individual
# ─────────────────────────────────────────────

@dataclass
class CheckResult:
    """
    Resultado de una validación individual.
    passed=False con severity='critical' detiene el pipeline.
    passed=False con severity='warning' se loguea pero no detiene.
    """
    name:       str
    passed:     bool
    severity:   str        # 'critical' | 'warning'
    expected:   str        # lo que esperábamos
    actual:     str        # lo que encontramos
    detail:     str = ""   # contexto adicional

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else ("❌ FAIL" if self.severity == "critical" else "⚠️  WARN")
        return (
            f"{status} [{self.name}] "
            f"expected={self.expected} actual={self.actual}"
            + (f" | {self.detail}" if self.detail else "")
        )


# ─────────────────────────────────────────────
#  Resultado agregado de una suite de checks
# ─────────────────────────────────────────────

@dataclass
class QualityReport:
    """
    Resultado agregado de una suite de checks.
    passed=True solo si todos los checks críticos pasaron.
    Los warnings no afectan passed.
    """
    layer:      str                        # 'bronze' | 'silver' | 'gold'
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
            f"  resultado    : {'APROBADO ✅' if self.passed else 'FALLIDO ❌'}",
            f"  checks       : {len(self.checks)} total, "
            f"{sum(c.passed for c in self.checks)} passed, "
            f"{len(self.critical_failures)} critical failures, "
            f"{len(self.warnings)} warnings",
            f"  ejecutado    : {self.run_at_utc}",
            "  detalle:",
        ]
        for c in self.checks:
            lines.append(f"    {c.summary()}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Para serializar al data lineage JSON."""
        return {
            "layer":      self.layer,
            "passed":     self.passed,
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
#  Checks individuales — funciones puras
# ─────────────────────────────────────────────

def check_row_count(
    actual_count: int,
    expected_count: int,
    severity: str = "critical",
) -> CheckResult:
    """
    Verifica que el número de filas coincide con lo esperado.
    Úsalo para:
    - Bronze: actual == 74,628
    - Silver: actual == Bronze_rows × sample_cols (± quarantine)
    """
    passed = actual_count == expected_count
    return CheckResult(
        name     = "row_count",
        passed   = passed,
        severity = severity,
        expected = str(expected_count),
        actual   = str(actual_count),
        detail   = f"diferencia={actual_count - expected_count:+,}" if not passed else "",
    )


def check_nulls_in_column(
    null_count: int,
    column_name: str,
    max_allowed: int = 0,
    severity: str = "critical",
) -> CheckResult:
    """
    Verifica que una columna key no tenga nulls por encima del threshold.
    max_allowed=0 significa cero tolerancia (keys primarias).
    """
    passed = null_count <= max_allowed
    return CheckResult(
        name     = f"nulls_{column_name}",
        passed   = passed,
        severity = severity,
        expected = f"<= {max_allowed}",
        actual   = str(null_count),
        detail   = f"columna '{column_name}' tiene {null_count} nulls" if not passed else "",
    )


def check_duplicates_in_index(
    duplicate_count: int,
    index_name: str = "gene_id",
    severity: str = "critical",
) -> CheckResult:
    """
    Verifica que el índice (key primaria) no tenga duplicados.
    En Bronze, el índice es Name (Ensembl ID) — debe ser único.
    """
    passed = duplicate_count == 0
    return CheckResult(
        name     = f"duplicates_{index_name}",
        passed   = passed,
        severity = severity,
        expected = "0",
        actual   = str(duplicate_count),
        detail   = f"{duplicate_count} Ensembl IDs duplicados en índice" if not passed else "",
    )


def check_quarantine_threshold(
    quarantine_count: int,
    total_count: int,
    max_fraction: float = 0.01,
    severity: str = "critical",
) -> CheckResult:
    """
    Verifica que los registros en cuarentena no superen el threshold.
    Si quarantine_count / total_count > max_fraction → pipeline STOP.

    Default: >1% de registros en cuarentena detiene el pipeline.
    En GTEx con nulls=0%, esperamos quarantine=0. El 1% es el límite de seguridad.
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
        detail   = "demasiados unmatched samples — verificar gtex_metadata.txt" if not passed else "",
    )


def check_zero_fraction(
    actual_fraction: float,
    baseline_fraction: float = 0.5189,
    tolerance: float = 0.05,
    severity: str = "warning",
) -> CheckResult:
    """
    Verifica que la fracción de zeros en Silver sea consistente con Bronze.
    Baseline: 51.89% de zeros en Bronze (profiling_report.json).
    Tolerance: ±5% es aceptable dado el reshape y el join.
    Severity warning (no crítico) — los zeros son biológicamente válidos.
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
            f"fracción de zeros {'muy alta' if actual_fraction > upper else 'muy baja'} "
            f"vs baseline {baseline_fraction:.1%}"
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
    Verifica que el lineage matemático cierra después del reshape.
    Fórmula: silver_rows + quarantine_rows == bronze_genes × bronze_samples

    Si no cierra → filas se perdieron silenciosamente durante el reshape.
    Este es el check más importante de Silver.
    """
    expected_total = bronze_genes * bronze_samples
    actual_total   = silver_rows + quarantine_rows
    passed         = actual_total == expected_total

    return CheckResult(
        name     = "lineage_closure",
        passed   = passed,
        severity = severity,
        expected = f"{expected_total:,} ({bronze_genes:,} × {bronze_samples:,})",
        actual   = f"{actual_total:,} (silver={silver_rows:,} + quarantine={quarantine_rows:,})",
        detail   = f"diferencia={actual_total - expected_total:+,} filas" if not passed else "",
    )


# ─────────────────────────────────────────────
#  Suites — conjuntos de checks por capa
# ─────────────────────────────────────────────

def run_bronze_checks(
    df_index_duplicates: int,
    df_row_count: int,
    description_null_count: int,
    expected_rows: int = 74_628,
) -> QualityReport:
    """
    Suite completa de quality checks para Bronze.

    Args:
        df_index_duplicates:   resultado de df.index.duplicated().sum()
        df_row_count:          resultado de len(df)
        description_null_count: resultado de df['Description'].isnull().sum()
        expected_rows:         74,628 por defecto (GTEx v11)

    Returns:
        QualityReport con passed=True si todos los críticos pasan.

    Ejemplo:
        import pandas as pd
        df = pd.read_parquet('data/bronze/gtex/gene_tpm_raw.parquet',
                             columns=['Description'])
        report = run_bronze_checks(
            df_index_duplicates   = df.index.duplicated().sum(),
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
    Suite completa de quality checks para Silver post-reshape.

    Args:
        silver_row_count:       filas escritas en Delta Silver
        quarantine_row_count:   filas enviadas a cuarentena
        tissue_id_null_count:   nulls en columna tissue_id
        gene_id_null_count:     nulls en columna gene_id
        sample_id_null_count:   nulls en columna sample_id
        actual_zero_fraction:   fracción real de zeros en tpm_value
        bronze_genes:           74,628 por defecto
        bronze_samples:         19,788 por defecto
        quarantine_threshold:   fracción máxima aceptable en cuarentena (default 1%)

    Returns:
        QualityReport con passed=True si todos los críticos pasan.
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
#  Helper interno
# ─────────────────────────────────────────────

def _log_report(report: QualityReport) -> None:
    if report.passed:
        logger.info("Quality gate [%s] APROBADO — %d checks passed",
                    report.layer.upper(), len(report.checks))
    else:
        logger.error("Quality gate [%s] FALLIDO — %d critical failures",
                     report.layer.upper(), len(report.critical_failures))
        for failure in report.critical_failures:
            logger.error("  %s", failure.summary())


# ─────────────────────────────────────────────
#  Excepción para el pipeline
# ─────────────────────────────────────────────

class PipelineQualityError(Exception):
    """
    Se lanza cuando un quality gate crítico falla.
    El mensaje incluye el QualityReport completo para diagnóstico.
    """
    def __init__(self, report: QualityReport):
        self.report = report
        super().__init__(
            f"Quality gate [{report.layer.upper()}] falló con "
            f"{len(report.critical_failures)} fallo(s) crítico(s):\n"
            + report.summary()
        )


# ─────────────────────────────────────────────
#  Gold checks — funciones puras
# ─────────────────────────────────────────────

def check_gold_row_count(
    actual_count:   int,
    expected_genes: int = 74_628,
    expected_tissues: int = 68,
    severity: str = "critical",
) -> CheckResult:
    """
    Verifica que Gold tenga exactamente genes × tejidos filas.
    Tolerancia: ±0 — Gold debe ser exacto porque es un groupBy completo.
    """
    expected = expected_genes * expected_tissues
    passed   = actual_count == expected
    return CheckResult(
        name     = "gold_row_count",
        passed   = passed,
        severity = severity,
        expected = f"{expected:,} ({expected_genes:,} genes × {expected_tissues} tejidos)",
        actual   = f"{actual_count:,}",
        detail   = f"diferencia={actual_count - expected:+,}" if not passed else "",
    )


def check_gold_tissue_count(
    actual_tissue_count:   int,
    expected_tissue_count: int = 68,
    severity: str = "critical",
) -> CheckResult:
    """
    Verifica que Gold tenga exactamente 68 tejidos únicos.
    Si faltan tejidos → algún tissue_id de Silver no llegó a Gold.
    """
    passed = actual_tissue_count == expected_tissue_count
    return CheckResult(
        name     = "gold_tissue_count",
        passed   = passed,
        severity = severity,
        expected = str(expected_tissue_count),
        actual   = str(actual_tissue_count),
        detail   = (
            f"{'faltan' if actual_tissue_count < expected_tissue_count else 'sobran'} "
            f"{abs(actual_tissue_count - expected_tissue_count)} tejidos"
        ) if not passed else "",
    )


def check_gold_min_sample_count(
    min_sample_count: int,
    min_allowed: int = 1,
    severity: str = "critical",
) -> CheckResult:
    """
    Verifica que ningún grupo gen×tejido tenga sample_count == 0.
    Un grupo con 0 muestras indica un bug en los acumuladores.
    min_allowed=1 — todo grupo debe tener al menos 1 muestra.
    """
    passed = min_sample_count >= min_allowed
    return CheckResult(
        name     = "gold_min_sample_count",
        passed   = passed,
        severity = severity,
        expected = f">= {min_allowed}",
        actual   = str(min_sample_count),
        detail   = "existen grupos con 0 muestras — bug en acumuladores" if not passed else "",
    )


def check_gold_zero_fraction_consistency(
    avg_zero_fraction:      float,
    silver_zero_fraction:   float = 0.5252,
    tolerance:              float = 0.05,
    severity:               str   = "warning",
) -> CheckResult:
    """
    Verifica que el promedio de zero_fraction en Gold sea consistente
    con la zero_fraction de Silver (52.52% medido en el quality gate Silver).

    Severity warning — una desviación no indica corrupción de datos,
    solo una posible diferencia en la distribución por tejido.
    Silver zero_fraction: 52.52% (confirmado en log del quality gate).
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
            f"avg zero_fraction Gold {'muy alta' if avg_zero_fraction > upper else 'muy baja'} "
            f"vs Silver baseline {silver_zero_fraction:.1%}"
        ) if not passed else "",
    )


# ─────────────────────────────────────────────
#  Suite Gold
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
    Suite completa de quality checks para Gold post-agregación.

    Args:
        gold_row_count:         filas escritas en Delta Gold
        tissue_count:           tejidos únicos en Gold
        gene_id_null_count:     nulls en columna gene_id
        gene_symbol_null_count: nulls en columna gene_symbol
        tissue_id_null_count:   nulls en columna tissue_id
        min_sample_count:       mínimo de sample_count en toda la tabla
        avg_zero_fraction:      promedio de zero_fraction en toda la tabla
        expected_genes:         74,628 por defecto (GTEx v11)
        expected_tissues:       68 por defecto (confirmado en Silver)
        silver_zero_fraction:   baseline de Silver (52.52%)

    Returns:
        QualityReport con passed=True si todos los críticos pasan.

    Checks críticos (detienen el pipeline si fallan):
        - gold_row_count           : genes × tejidos exacto
        - gold_tissue_count        : 68 tejidos presentes
        - nulls en las 3 keys      : cero tolerancia
        - gold_min_sample_count    : ningún grupo con 0 muestras

    Checks warning (se loguean pero no detienen):
        - zero_fraction_consistency: consistente con Silver ±5%
    """
    report = QualityReport(layer="gold")

    # Críticos — keys sin nulls
    report.checks.append(check_nulls_in_column(gene_id_null_count,     "gene_id"))
    report.checks.append(check_nulls_in_column(gene_symbol_null_count, "gene_symbol"))
    report.checks.append(check_nulls_in_column(tissue_id_null_count,   "tissue_id"))

    # Críticos — integridad estructural
    report.checks.append(check_gold_row_count(
        gold_row_count, expected_genes, expected_tissues
    ))
    report.checks.append(check_gold_tissue_count(tissue_count, expected_tissues))
    report.checks.append(check_gold_min_sample_count(min_sample_count))

    # Warning — consistencia con Silver
    report.checks.append(check_gold_zero_fraction_consistency(
        avg_zero_fraction, silver_zero_fraction
    ))

    _log_report(report)
    return report

# ─────────────────────────────────────────────
#  CLI — smoke test sin dependencias externas
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Smoke test Bronze (números reales del profiling) ===\n")
    bronze_report = run_bronze_checks(
        df_index_duplicates    = 0,
        df_row_count           = 74_628,
        description_null_count = 0,
    )
    print(bronze_report.summary())

    print("\n=== Smoke test Silver (simulado post-reshape) ===\n")
    silver_report = run_silver_checks(
        silver_row_count      = 74_628 * 19_788,
        quarantine_row_count  = 0,
        tissue_id_null_count  = 0,
        gene_id_null_count    = 0,
        sample_id_null_count  = 0,
        actual_zero_fraction  = 0.5189,
    )
    print(silver_report.summary())

    print("\n=== Smoke test Silver con fallo crítico (lineage no cierra) ===\n")
    fail_report = run_silver_checks(
        silver_row_count      = 74_628 * 19_788 - 1000,
        quarantine_row_count  = 0,
        tissue_id_null_count  = 0,
        gene_id_null_count    = 0,
        sample_id_null_count  = 0,
        actual_zero_fraction  = 0.5189,
    )
    print(fail_report.summary())

    print("\n=== Smoke test Gold (números esperados) ===\n")
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

    print("\n=== Smoke test Gold con fallo crítico (row count incorrecto) ===\n")
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