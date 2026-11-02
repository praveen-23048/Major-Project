"""Unit tests for validation-only baseline diagnostics."""

from pathlib import Path

import pandas as pd

from src.baseline import BaselineConfig
from src.diagnostics import (
    _js_divergence,
    _psi,
    analyze_categorical_distributions,
    analyze_class_distribution,
    analyze_numeric_distributions,
    generate_diagnostic_report,
    run_convergence_experiments,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    training = pd.DataFrame(
        {
            "TransactionDT": range(12),
            "TransactionID": range(12),
            "isFraud": [0, 1] * 6,
            "amount": [1.0, 2.0, None, 4.0, 5.0, 6.0] * 2,
            "device": ["a", "a", "b", "b", "a", None] * 2,
        }
    )
    validation = pd.DataFrame(
        {
            "TransactionDT": range(12, 18),
            "TransactionID": range(12, 18),
            "isFraud": [0, 0, 1, 0, 0, 1],
            "amount": [7.0, None, 9.0, 10.0, 11.0, 12.0],
            "device": ["a", "c", "c", "b", None, "c"],
        }
    )
    return training, validation


def test_distribution_and_class_analyses_handle_missing_and_new_values() -> None:
    training, validation = _frames()
    numeric = analyze_numeric_distributions(training, validation)
    categorical = analyze_categorical_distributions(training, validation)
    class_distribution = analyze_class_distribution(training, validation)
    assert {"ks_statistic", "psi"}.issubset(numeric.columns)
    assert categorical.loc[0, "new_validation_categories"] == 1
    assert class_distribution["validation_fraud_prevalence"] < class_distribution["train_fraud_prevalence"]


def test_psi_and_js_are_zero_for_identical_distributions() -> None:
    values = pd.Series([1.0, 2.0, 3.0])
    categories = pd.Series(["a", "b", "a"])
    assert _psi(values, values) == 0.0
    assert _js_divergence(categories, categories) == 0.0


def test_convergence_experiment_and_report_use_validation_only(tmp_path: Path) -> None:
    training, validation = _frames()
    results = run_convergence_experiments(
        training,
        validation,
        BaselineConfig(max_iter=5),
        c_values=[0.1],
        max_iter_values=[5],
    )
    assert len(results) == 1
    assert {"converged", "n_iter", "validation_pr_auc"}.issubset(results.columns)
    summary = generate_diagnostic_report(
        training,
        validation,
        tmp_path,
        BaselineConfig(max_iter=5),
        c_values=[0.1],
        max_iter_values=[5],
    )
    assert summary["selection_rule"].startswith("highest validation")
    assert (tmp_path / "diagnostic_summary.json").is_file()
    assert (tmp_path / "plots" / "convergence_comparison.png").is_file()
