"""Chronological scikit-learn baseline for the IEEE-CIS transaction data."""

from dataclasses import dataclass, replace
import json
import warnings
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split proportions and reproducibility settings."""

    train_fraction: float = 0.70
    validation_fraction: float = 0.15

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Train and validation fractions must leave test rows.")


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the reproducible baseline."""

    target_column: str = "isFraud"
    temporal_column: str = "TransactionDT"
    identifier_columns: tuple[str, ...] = ("TransactionID",)
    random_state: int = 42
    solver: str = "saga"
    regularization_strength: float = 1.0
    max_iter: int = 300
    tolerance: float = 1e-3
    categorical_min_frequency: int = 10
    categorical_max_categories: int = 100


def chronological_split(
    dataframe: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
    temporal_column: str = "TransactionDT",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by time and return non-overlapping train, validation, and test frames."""
    if dataframe.empty:
        raise ValueError("Cannot split an empty dataframe.")
    if temporal_column not in dataframe.columns:
        raise ValueError(f"Missing temporal split column: {temporal_column}")
    if dataframe[temporal_column].isna().any():
        raise ValueError(f"Temporal split column contains missing values: {temporal_column}")

    ordered = dataframe.sort_values(temporal_column, kind="mergesort").reset_index(drop=True)
    row_count = len(ordered)
    train_end = int(row_count * config.train_fraction)
    validation_end = int(row_count * (config.train_fraction + config.validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError("Dataset is too small for the configured chronological split.")
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def _feature_columns(
    dataframe: pd.DataFrame, config: BaselineConfig
) -> tuple[list[str], list[str]]:
    excluded = {config.target_column, config.temporal_column, *config.identifier_columns}
    feature_names = [column for column in dataframe.columns if column not in excluded]
    if not feature_names:
        raise ValueError("No predictive feature columns remain after exclusions.")

    numeric = dataframe[feature_names].select_dtypes(include=["number"]).columns.tolist()
    categorical = dataframe[feature_names].select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    supported = set(numeric) | set(categorical)
    unsupported = [column for column in feature_names if column not in supported]
    if unsupported:
        raise ValueError(
            "Unsupported feature dtypes; convert these columns explicitly: "
            + ", ".join(unsupported)
        )
    if not numeric and not categorical:
        raise ValueError("No supported predictive feature columns remain.")
    return numeric, categorical


def build_baseline_pipeline(
    training_frame: pd.DataFrame, config: BaselineConfig = BaselineConfig()
) -> Pipeline:
    """Build a preprocessing-and-classifier pipeline using training columns only."""
    numeric_columns, categorical_columns = _feature_columns(training_frame, config)
    transformers: list[tuple[str, Any, Iterable[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=config.categorical_min_frequency,
                                max_categories=config.categorical_max_categories,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        n_jobs=1,
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        C=config.regularization_strength,
        max_iter=config.max_iter,
        random_state=config.random_state,
        solver=config.solver,
        tol=config.tolerance,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _features_and_target(
    frame: pd.DataFrame, config: BaselineConfig
) -> tuple[pd.DataFrame, pd.Series]:
    if config.target_column not in frame.columns:
        raise ValueError(f"Missing target column: {config.target_column}")
    if frame[config.target_column].isna().any():
        raise ValueError("Target column contains missing values.")
    numeric, categorical = _feature_columns(frame, config)
    features = frame[numeric + categorical].copy()
    if numeric:
        features[numeric] = features[numeric].astype(np.float32)
    target = frame[config.target_column].astype(int)
    if target.nunique() < 2:
        raise ValueError("Training data must contain both fraud classes.")
    return features, target


def evaluate_predictions(
    target: pd.Series, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Calculate imbalance-aware metrics and confusion-matrix values."""
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    report = classification_report(
        target,
        predictions,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )
    return {
        "class_prevalence": float(target.mean()),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "threshold": threshold,
        "row_count": int(len(target)),
    }


def preprocessing_diagnostics(
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    config: BaselineConfig,
) -> dict[str, dict[str, float]]:
    """Summarize train-fitted feature coverage without using evaluation targets."""
    numeric, categorical = _feature_columns(training_frame, config)
    diagnostics: dict[str, dict[str, float]] = {
        "numeric_missing_rate": {
            column: float(evaluation_frame[column].isna().mean())
            for column in numeric
        },
        "categorical_unknown_rate": {},
    }
    for column in categorical:
        known_values = set(training_frame[column].dropna().unique())
        values = evaluation_frame[column]
        diagnostics["categorical_unknown_rate"][column] = float(
            (~values.isna() & ~values.isin(known_values)).mean()
        )
    return diagnostics


def save_confusion_matrix(metrics: dict[str, Any], output_path: Path, title: str) -> None:
    """Save a labeled confusion-matrix plot."""
    matrix = np.asarray(metrics["confusion_matrix"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def train_and_evaluate(
    dataframe: pd.DataFrame,
    report_directory: Path = Path("reports"),
    model_directory: Path = Path("models"),
    split_config: SplitConfig = SplitConfig(),
    config: BaselineConfig = BaselineConfig(),
    tune_validation: bool = True,
) -> dict[str, Any]:
    """Select on validation, then evaluate the frozen selection once on test."""
    print(f"Sorting {len(dataframe):,} rows by {config.temporal_column}...")
    train, validation, test = chronological_split(
        dataframe, split_config, config.temporal_column
    )
    print(
        "Partitions: "
        f"train={len(train):,}, validation={len(validation):,}, test={len(test):,}"
    )
    features_train, target_train = _features_and_target(train, config)
    validation_features, validation_target = _features_and_target(validation, config)
    candidates = (
        [config, replace(config, regularization_strength=0.1)]
        if tune_validation
        else [config]
    )
    candidate_results: list[dict[str, Any]] = []
    fitted_candidates: list[tuple[float, Pipeline, dict[str, Any]]] = []
    for candidate in candidates:
        print(
            "Fitting Logistic Regression "
            f"(solver={candidate.solver}, C={candidate.regularization_strength})..."
        )
        pipeline = build_baseline_pipeline(train, candidate)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(features_train, target_train)
        probabilities = pipeline.predict_proba(validation_features)[:, 1]
        validation_metrics = evaluate_predictions(validation_target, probabilities)
        classifier = pipeline.named_steps["classifier"]
        convergence_warnings = [
            str(item.message)
            for item in caught_warnings
            if issubclass(item.category, ConvergenceWarning)
        ]
        diagnostics = {
            "solver": candidate.solver,
            "regularization_strength": candidate.regularization_strength,
            "max_iter": candidate.max_iter,
            "tolerance": candidate.tolerance,
            "convergence_warnings": convergence_warnings,
            "n_iter": classifier.n_iter_.tolist(),
            "validation": validation_metrics,
        }
        candidate_results.append(diagnostics)
        fitted_candidates.append((candidate.regularization_strength, pipeline, diagnostics))

    _, pipeline, selected_diagnostics = max(
        fitted_candidates,
        key=lambda item: item[2]["validation"]["pr_auc"],
    )
    print(
        "Selected configuration using validation PR-AUC: "
        f"C={selected_diagnostics['regularization_strength']}"
    )
    test_features, test_target = _features_and_target(test, config)
    test_probabilities = pipeline.predict_proba(test_features)[:, 1]
    evaluations = {
        "validation": selected_diagnostics["validation"],
        "test": evaluate_predictions(test_target, test_probabilities),
    }
    for name in evaluations:
        save_confusion_matrix(
            evaluations[name],
            report_directory / f"baseline_confusion_matrix_{name}.png",
            f"Baseline {name.title()} Confusion Matrix",
        )

    report_directory.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "LogisticRegression(class_weight='balanced', solver='saga')",
        "selection": {
            "method": "highest validation PR-AUC; test evaluated once after selection",
            "candidates": candidate_results,
            "selected": selected_diagnostics,
        },
        "split": {
            "train_fraction": split_config.train_fraction,
            "validation_fraction": split_config.validation_fraction,
            "test_fraction": 1 - split_config.train_fraction - split_config.validation_fraction,
            "temporal_column": config.temporal_column,
            "train_max_time": train[config.temporal_column].item()
            if len(train) == 1
            else train[config.temporal_column].max(),
            "validation_min_time": validation[config.temporal_column].min(),
            "validation_max_time": validation[config.temporal_column].max(),
            "test_min_time": test[config.temporal_column].min(),
        },
        "excluded_columns": [
            config.target_column,
            config.temporal_column,
            *config.identifier_columns,
        ],
        "warning": "Accuracy alone is insufficient for this imbalanced fraud dataset.",
        "evaluations": evaluations,
        "preprocessing_diagnostics": {
            "validation": preprocessing_diagnostics(train, validation, config),
            "test": preprocessing_diagnostics(train, test, config),
        },
    }
    serializable_metrics = _json_safe(metrics)
    (report_directory / "baseline_metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2),
        encoding="utf-8",
    )
    model_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_directory / "baseline_pipeline.joblib")
    return serializable_metrics


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
