"""Fraud-only CTGAN augmentation for the chronological training partition."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import platform
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from src.baseline import SplitConfig, chronological_split


DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "C1",
    "C2",
    "C4",
    "C5",
    "C6",
    "D1",
    "D2",
    "D3",
    "D4",
    "V12",
    "V14",
    "V17",
    "V34",
    "V38",
    "V40",
    "V42",
    "V62",
    "V70",
    "ProductCD",
    "card4",
    "card6",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
)
MISSING_CATEGORY = "__CTGAN_MISSING__"
INTEGER_LIKE_COLUMNS = frozenset(
    {"card1", "card2", "card3", "card5", "addr1", "addr2"}
    | {f"C{i}" for i in range(1, 15)}
    | {f"D{i}" for i in range(1, 16)}
    | {f"V{i}" for i in range(1, 340)}
)


class GeneratorProtocol(Protocol):
    def fit(self, data: pd.DataFrame, discrete_columns: list[str]) -> None: ...

    def sample(self, num_samples: int) -> pd.DataFrame: ...


@dataclass(frozen=True)
class CTGANConfig:
    """Reproducible, modest first-pass CTGAN experiment settings."""

    feature_columns: tuple[str, ...] = field(default_factory=lambda: DEFAULT_FEATURE_COLUMNS)
    sample_count: int = 2_000
    random_state: int = 42
    epochs: int = 50
    batch_size: int = 500
    generator_dim: tuple[int, ...] = (256, 256)
    discriminator_dim: tuple[int, ...] = (256, 256)
    target_column: str = "isFraud"
    temporal_column: str = "TransactionDT"
    identifier_column: str = "TransactionID"

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty.")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        forbidden = {
            self.target_column,
            self.temporal_column,
            self.identifier_column,
        }
        invalid = sorted(forbidden.intersection(self.feature_columns))
        if invalid:
            raise ValueError("CTGAN features cannot include: " + ", ".join(invalid))


def fraud_training_partition(
    dataframe: pd.DataFrame,
    split_config: SplitConfig = SplitConfig(),
    target_column: str = "isFraud",
    temporal_column: str = "TransactionDT",
) -> pd.DataFrame:
    """Return only fraudulent rows from the baseline's chronological train split."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Missing target column: {target_column}")
    train, _, _ = chronological_split(dataframe, split_config, temporal_column)
    fraud = train.loc[train[target_column].astype(int).eq(1)].copy()
    if fraud.empty:
        raise ValueError("The chronological training partition contains no fraud rows.")
    return fraud


def select_ctgan_features(
    fraud_training: pd.DataFrame, config: CTGANConfig
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Validate and select the configured columns, retaining their original dtypes."""
    missing = [column for column in config.feature_columns if column not in fraud_training.columns]
    if missing:
        raise ValueError("Missing CTGAN feature columns: " + ", ".join(missing))
    selected = fraud_training.loc[:, list(config.feature_columns)].copy()
    unsupported = [
        column
        for column in selected.columns
        if not (
            pd.api.types.is_numeric_dtype(selected[column])
            or pd.api.types.is_object_dtype(selected[column])
            or pd.api.types.is_string_dtype(selected[column])
            or isinstance(selected[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(selected[column])
        )
    ]
    if unsupported:
        raise ValueError("Unsupported CTGAN feature dtypes: " + ", ".join(unsupported))
    return selected, {column: str(dtype) for column, dtype in selected.dtypes.items()}


def _discrete_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(frame[column])
        )
    ]


def _prepare_generator_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Make CTGAN-compatible values using statistics from fraud training only."""
    prepared = frame.copy()
    numeric = prepared.select_dtypes(include=["number"]).columns
    categorical = [column for column in prepared.columns if column not in numeric]
    for column in numeric:
        median = prepared[column].median()
        if pd.isna(median):
            raise ValueError(f"Cannot impute numeric CTGAN feature with no values: {column}")
        prepared[column] = prepared[column].fillna(median)
    for column in categorical:
        prepared[column] = prepared[column].astype("object").fillna(MISSING_CATEGORY)
    if prepared.isna().any().any():
        raise ValueError("CTGAN input still contains null values after imputation.")
    return prepared


def build_ctgan(config: CTGANConfig) -> GeneratorProtocol:
    """Construct the optional CTGAN dependency lazily."""
    try:
        from ctgan import CTGAN
    except ImportError as error:
        raise RuntimeError(
            "CTGAN is not installed. Install the pinned project dependencies first."
        ) from error
    return CTGAN(
        epochs=config.epochs,
        batch_size=config.batch_size,
        generator_dim=config.generator_dim,
        discriminator_dim=config.discriminator_dim,
        enable_gpu=False,
        verbose=False,
    )


def _restore_dtypes(frame: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    restored = frame.copy()
    for column, dtype in dtypes.items():
        if column not in restored:
            raise ValueError(f"Generated data is missing feature column: {column}")
        try:
            if dtype == "category":
                restored[column] = restored[column].astype("category")
            elif dtype == "bool":
                restored[column] = restored[column].astype(bool)
            else:
                restored[column] = restored[column].astype(dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Could not restore dtype {dtype} for {column}") from error
    return restored.loc[:, list(dtypes)]


def generate_synthetic_fraud(
    generator: GeneratorProtocol,
    fraud_features: pd.DataFrame,
    dtypes: dict[str, str],
    config: CTGANConfig,
) -> pd.DataFrame:
    """Fit a generator on fraud-only rows and return labeled synthetic fraud."""
    if len(fraud_features) < 2:
        raise ValueError("CTGAN requires at least two fraud training rows.")
    np.random.seed(config.random_state)
    generator.fit(
        _prepare_generator_input(fraud_features),
        _discrete_columns(fraud_features),
    )
    synthetic = generator.sample(config.sample_count)
    if not isinstance(synthetic, pd.DataFrame):
        raise ValueError("CTGAN generator returned a non-DataFrame sample.")
    synthetic = _restore_dtypes(synthetic, dtypes)
    synthetic = postprocess_synthetic_fraud(synthetic, fraud_features, config)
    synthetic[config.target_column] = np.int8(1)
    return synthetic


def postprocess_synthetic_fraud(
    synthetic: pd.DataFrame,
    real_features: pd.DataFrame,
    config: CTGANConfig,
) -> pd.DataFrame:
    """Constrain generated values to the observed fraud-training schema."""
    processed = synthetic.loc[:, list(config.feature_columns)].copy()
    numeric = real_features.select_dtypes(include=["number"]).columns
    for column in numeric:
        lower = real_features[column].min()
        upper = real_features[column].max()
        processed[column] = processed[column].clip(lower=lower, upper=upper)
        if column in INTEGER_LIKE_COLUMNS:
            processed[column] = processed[column].round()
        processed[column] = processed[column].astype(real_features[column].dtype)
    return processed


def synthetic_quality_report(
    real_features: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: CTGANConfig,
) -> dict[str, Any]:
    """Run basic schema, range, null, and overlap checks."""
    feature_columns = list(config.feature_columns)
    if list(synthetic.columns) != feature_columns + [config.target_column]:
        raise ValueError("Synthetic output columns do not match the configured schema.")
    numeric = real_features.select_dtypes(include=["number"]).columns
    range_checks = {
        column: {
            "real_min": float(real_features[column].min()),
            "real_max": float(real_features[column].max()),
            "synthetic_min": float(synthetic[column].min()),
            "synthetic_max": float(synthetic[column].max()),
            "within_real_range": bool(
                synthetic[column].between(
                    real_features[column].min(), real_features[column].max()
                ).all()
            ),
        }
        for column in numeric
    }
    categorical = [column for column in feature_columns if column not in numeric]
    allowed_categories = {
        column: sorted(
            set(synthetic[column].dropna().astype(str))
            - set(real_features[column].dropna().astype(str))
        )
        for column in categorical
    }
    allowed_categories = {
        column: [
            value for value in values if value != MISSING_CATEGORY
        ]
        for column, values in allowed_categories.items()
    }
    real_keys = real_features.astype("string").fillna("<NA>").astype(str).agg("|".join, axis=1)
    synthetic_keys = synthetic[feature_columns].astype("string").fillna("<NA>").astype(str).agg("|".join, axis=1)
    return {
        "real_row_count": int(len(real_features)),
        "synthetic_row_count": int(len(synthetic)),
        "null_counts": synthetic[feature_columns].isna().sum().astype(int).to_dict(),
        "duplicate_synthetic_rows": int(synthetic[feature_columns].duplicated().sum()),
        "exact_overlap_with_real_rows": int(synthetic_keys.isin(set(real_keys)).sum()),
        "all_rows_labeled_fraud": bool(synthetic[config.target_column].eq(1).all()),
        "numeric_range_checks": range_checks,
        "unseen_categories": allowed_categories,
        "preprocessing_sentinel_counts": {
            column: int(synthetic[column].eq(MISSING_CATEGORY).sum())
            for column in categorical
        },
        "integer_like_violations": {
            column: int(
                (synthetic[column].dropna() != synthetic[column].dropna().round()).sum()
            )
            for column in numeric
            if column in INTEGER_LIKE_COLUMNS
        },
    }


def run_ctgan_augmentation(
    dataframe: pd.DataFrame,
    model_path: Path,
    sample_path: Path,
    report_path: Path,
    config: CTGANConfig = CTGANConfig(),
    split_config: SplitConfig = SplitConfig(),
    generator: GeneratorProtocol | None = None,
) -> dict[str, Any]:
    """Fit, persist, sample, and report one reproducible CTGAN experiment."""
    fraud = fraud_training_partition(
        dataframe, split_config, config.target_column, config.temporal_column
    )
    real_features, dtypes = select_ctgan_features(fraud, config)
    fitted_generator = generator or build_ctgan(config)
    raw_generator = fitted_generator
    synthetic = generate_synthetic_fraud(raw_generator, real_features, dtypes, config)
    quality = synthetic_quality_report(real_features, synthetic, config)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_generator, model_path)
    synthetic.to_csv(sample_path, index=False)
    report = {
        "experiment": "fraud-only-ctgan",
        "python_version": platform.python_version(),
        "random_state": config.random_state,
        "feature_columns": list(config.feature_columns),
        "feature_dtypes": dtypes,
        "training_rows": int(len(fraud)),
        "synthetic_rows": int(len(synthetic)),
        "split": {
            "train_fraction": split_config.train_fraction,
            "validation_fraction": split_config.validation_fraction,
            "test_fraction": 1 - split_config.train_fraction - split_config.validation_fraction,
        },
        "ctgan": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "generator_dim": list(config.generator_dim),
            "discriminator_dim": list(config.discriminator_dim),
        },
        "quality": quality,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
