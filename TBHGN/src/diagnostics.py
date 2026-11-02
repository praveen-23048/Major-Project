"""Baseline reliability diagnostics for the chronological fraud split."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.baseline import (
    BaselineConfig,
    _features_and_target,
    build_baseline_pipeline,
    evaluate_predictions,
)

def _numeric_columns(frame: pd.DataFrame, config: BaselineConfig) -> list[str]:
    excluded = {config.target_column, config.temporal_column, *config.identifier_columns}
    return frame.drop(columns=[column for column in excluded if column in frame]).select_dtypes(
        include=["number"]
    ).columns.tolist()


def _categorical_columns(frame: pd.DataFrame, config: BaselineConfig) -> list[str]:
    excluded = {config.target_column, config.temporal_column, *config.identifier_columns}
    return frame.drop(columns=[column for column in excluded if column in frame]).select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()


def _ks_statistic(first: pd.Series, second: pd.Series) -> float:
    first_values = first.dropna().to_numpy(dtype=float)
    second_values = second.dropna().to_numpy(dtype=float)
    if not len(first_values) or not len(second_values):
        return 0.0
    combined = np.sort(np.concatenate([first_values, second_values]))
    first_cdf = np.searchsorted(np.sort(first_values), combined, side="right") / len(first_values)
    second_cdf = np.searchsorted(np.sort(second_values), combined, side="right") / len(second_values)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def _psi(first: pd.Series, second: pd.Series, bins: int = 10) -> float:
    first_values = first.dropna().to_numpy(dtype=float)
    second_values = second.dropna().to_numpy(dtype=float)
    if not len(first_values) or not len(second_values):
        return 0.0
    edges = np.unique(np.quantile(first_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    first_counts = np.histogram(first_values, edges)[0].astype(float)
    second_counts = np.histogram(second_values, edges)[0].astype(float)
    smoothing = 1e-6
    first_rates = (first_counts + smoothing) / (first_counts.sum() + smoothing * len(first_counts))
    second_rates = (second_counts + smoothing) / (second_counts.sum() + smoothing * len(second_counts))
    return float(np.sum((second_rates - first_rates) * np.log(second_rates / first_rates)))


def _js_divergence(first: pd.Series, second: pd.Series) -> float:
    first_rates = first.value_counts(normalize=True, dropna=False)
    second_rates = second.value_counts(normalize=True, dropna=False)
    index = first_rates.index.union(second_rates.index)
    p = first_rates.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    q = second_rates.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    midpoint = (p + q) / 2
    p_mask = p > 0
    q_mask = q > 0
    first_kl = np.sum(p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask]))
    second_kl = np.sum(q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask]))
    return float((first_kl + second_kl) / 2)


def analyze_numeric_distributions(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    config: BaselineConfig = BaselineConfig(),
) -> pd.DataFrame:
    """Compare numeric feature summaries and train/validation distributions."""
    rows: list[dict[str, Any]] = []
    for column in _numeric_columns(training, config):
        train_values = training[column]
        validation_values = validation[column]
        rows.append(
            {
                "feature": column,
                "train_missing_pct": float(train_values.isna().mean() * 100),
                "validation_missing_pct": float(validation_values.isna().mean() * 100),
                "train_mean": float(train_values.mean()),
                "validation_mean": float(validation_values.mean()),
                "train_std": float(train_values.std()),
                "validation_std": float(validation_values.std()),
                "train_median": float(train_values.median()),
                "validation_median": float(validation_values.median()),
                "train_min": float(train_values.min()),
                "validation_min": float(validation_values.min()),
                "train_max": float(train_values.max()),
                "validation_max": float(validation_values.max()),
                "ks_statistic": _ks_statistic(train_values, validation_values),
                "psi": _psi(train_values, validation_values),
            }
        )
    return pd.DataFrame(rows)


def analyze_categorical_distributions(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    config: BaselineConfig = BaselineConfig(),
) -> pd.DataFrame:
    """Compare category coverage and Jensen-Shannon divergence."""
    rows: list[dict[str, Any]] = []
    for column in _categorical_columns(training, config):
        train_values = training[column]
        validation_values = validation[column]
        train_categories = set(train_values.dropna().unique())
        validation_categories = set(validation_values.dropna().unique())
        rows.append(
            {
                "feature": column,
                "train_missing_pct": float(train_values.isna().mean() * 100),
                "validation_missing_pct": float(validation_values.isna().mean() * 100),
                "train_unique_count": len(train_categories),
                "validation_unique_count": len(validation_categories),
                "new_validation_categories": len(validation_categories - train_categories),
                "train_only_categories": len(train_categories - validation_categories),
                "unknown_validation_rate_pct": float(
                    (
                        validation_values.notna()
                        & ~validation_values.isin(train_categories)
                    ).mean()
                    * 100
                ),
                "js_divergence": _js_divergence(train_values, validation_values),
            }
        )
    return pd.DataFrame(rows)


def analyze_missing_values(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    config: BaselineConfig = BaselineConfig(),
) -> pd.DataFrame:
    """Return missing-value rates for all candidate model features."""
    excluded = {config.target_column, config.temporal_column, *config.identifier_columns}
    columns = [column for column in training.columns if column not in excluded]
    return pd.DataFrame(
        [
            {
                "feature": column,
                "train_missing_pct": float(training[column].isna().mean() * 100),
                "validation_missing_pct": float(validation[column].isna().mean() * 100),
                "absolute_difference_pct": float(
                    abs(training[column].isna().mean() - validation[column].isna().mean())
                    * 100
                ),
            }
            for column in columns
        ]
    )


def analyze_class_distribution(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    config: BaselineConfig = BaselineConfig(),
) -> dict[str, float]:
    """Compare fraud prevalence without using the test partition."""
    train_prevalence = float(training[config.target_column].mean())
    validation_prevalence = float(validation[config.target_column].mean())
    return {
        "train_fraud_prevalence": train_prevalence,
        "validation_fraud_prevalence": validation_prevalence,
        "absolute_difference": abs(train_prevalence - validation_prevalence),
    }


def analyze_temporal_distribution(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    config: BaselineConfig = BaselineConfig(),
    windows: int = 10,
) -> pd.DataFrame:
    """Summarize chronological ranges and fraud prevalence by time windows."""
    combined = pd.concat(
        [
            training[[config.temporal_column, config.target_column]].assign(partition="train"),
            validation[[config.temporal_column, config.target_column]].assign(partition="validation"),
        ],
        ignore_index=True,
    )
    edges = np.unique(
        np.quantile(combined[config.temporal_column], np.linspace(0, 1, windows + 1))
    )
    if len(edges) < 2:
        return pd.DataFrame()
    combined["time_window"] = pd.cut(
        combined[config.temporal_column], bins=edges, include_lowest=True, duplicates="drop"
    )
    return (
        combined.groupby(["partition", "time_window"], observed=True)[config.target_column]
        .agg(row_count="size", fraud_prevalence="mean")
        .reset_index()
    )


def run_convergence_experiments(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    config: BaselineConfig = BaselineConfig(),
    c_values: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    max_iter_values: Iterable[int] = (300,),
) -> pd.DataFrame:
    """Run configurable Logistic Regression experiments on validation only."""
    train_features, train_target = _features_and_target(training, config)
    validation_features, validation_target = _features_and_target(validation, config)
    rows: list[dict[str, Any]] = []
    for max_iter in max_iter_values:
        for regularization_strength in c_values:
            candidate = BaselineConfig(
                **{
                    **config.__dict__,
                    "regularization_strength": regularization_strength,
                    "max_iter": max_iter,
                }
            )
            pipeline = build_baseline_pipeline(training, candidate)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(train_features, train_target)
            elapsed = time.perf_counter() - started
            probabilities = pipeline.predict_proba(validation_features)[:, 1]
            metrics = evaluate_predictions(validation_target, probabilities)
            classifier = pipeline.named_steps["classifier"]
            convergence_warnings = [
                str(item.message)
                for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]
            rows.append(
                {
                    "solver": candidate.solver,
                    "C": regularization_strength,
                    "max_iter": max_iter,
                    "tol": candidate.tolerance,
                    "converged": not convergence_warnings,
                    "n_iter": int(classifier.n_iter_[0]),
                    "training_seconds": elapsed,
                    "convergence_warnings": " | ".join(convergence_warnings),
                    "validation_pr_auc": metrics["pr_auc"],
                    "validation_precision": metrics["precision"],
                    "validation_recall": metrics["recall"],
                    "validation_f1": metrics["f1"],
                    "validation_confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                }
            )
    return pd.DataFrame(rows)


def generate_diagnostic_report(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    output_directory: Path,
    config: BaselineConfig = BaselineConfig(),
    c_values: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    max_iter_values: Iterable[int] = (300,),
    run_note: str | None = None,
) -> dict[str, Any]:
    """Generate diagnostic tables, plots, and a consolidated JSON summary."""
    output_directory.mkdir(parents=True, exist_ok=True)
    plots = output_directory / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    convergence = run_convergence_experiments(
        training, validation, config, c_values, max_iter_values
    )
    numeric = analyze_numeric_distributions(training, validation, config)
    categorical = analyze_categorical_distributions(training, validation, config)
    missing = analyze_missing_values(training, validation, config)
    temporal = analyze_temporal_distribution(training, validation, config)
    class_distribution = analyze_class_distribution(training, validation, config)
    convergence.to_csv(output_directory / "convergence_results.csv", index=False)
    convergence.sort_values("C").to_csv(
        output_directory / "regularization_comparison.csv", index=False
    )
    numeric.to_csv(output_directory / "numeric_distribution_analysis.csv", index=False)
    categorical.to_csv(output_directory / "categorical_distribution_analysis.csv", index=False)
    missing.to_csv(output_directory / "missing_value_analysis.csv", index=False)
    temporal.to_csv(output_directory / "temporal_analysis.csv", index=False)
    _plot_convergence(convergence, plots / "convergence_comparison.png")
    _plot_convergence(convergence, plots / "regularization_comparison.png")
    _plot_temporal(temporal, plots / "fraud_prevalence_over_time.png")
    _plot_numeric(numeric, plots / "feature_distribution_comparison.png")
    _plot_missing(missing, plots / "missing_value_comparison.png")
    best = convergence.loc[convergence["validation_pr_auc"].idxmax()].to_dict()
    summary = {
        "selection_rule": "highest validation PR-AUC; test data was not used",
        "experiment_grid": {
            "C": list(c_values),
            "max_iter": list(max_iter_values),
            "note": "The runner default is bounded to one C=1.0 fit at max_iter=100; expand explicitly for expensive checks.",
        },
        "best_validation_configuration": _json_safe(best),
        "class_distribution": class_distribution,
        "numeric_features_with_largest_ks": _json_safe(
            numeric.nlargest(10, "ks_statistic").to_dict(orient="records")
        ),
        "categorical_features_with_largest_js": _json_safe(
            categorical.nlargest(10, "js_divergence").to_dict(orient="records")
        ),
        "interpretation": (
            "Distribution differences are covariate-shift indicators only; "
            "they do not establish concept drift without comparing feature-target relationships."
        ),
    }
    if run_note is not None:
        summary["run_note"] = run_note
    (output_directory / "diagnostic_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    return summary


def _plot_convergence(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for max_iter, group in frame.groupby("max_iter"):
        axis.plot(group["C"], group["validation_pr_auc"], marker="o", label=f"max_iter={max_iter}")
    axis.set_xscale("log")
    axis.set(xlabel="C", ylabel="Validation PR-AUC", title="Validation regularization comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_temporal(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    for partition, group in frame.groupby("partition"):
        axis.plot(group.index, group["fraud_prevalence"], marker="o", label=partition)
    axis.set(xlabel="Chronological window", ylabel="Fraud prevalence", title="Fraud prevalence over time")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_numeric(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    if not frame.empty:
        top = frame.nlargest(10, "ks_statistic")
        axis.bar(top["feature"].astype(str), top["ks_statistic"])
        axis.tick_params(axis="x", rotation=75)
    axis.set(title="Largest numeric train/validation KS differences", ylabel="KS statistic")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_missing(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    if not frame.empty:
        top = frame.nlargest(10, "absolute_difference_pct")
        axis.bar(top["feature"].astype(str), top["absolute_difference_pct"])
        axis.tick_params(axis="x", rotation=75)
    axis.set(title="Largest missing-value rate differences", ylabel="Absolute difference (%)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value
