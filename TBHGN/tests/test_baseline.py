"""Focused tests for chronological splitting and training-only preprocessing."""

import pandas as pd
import pytest

from src.baseline import (
    BaselineConfig,
    SplitConfig,
    build_baseline_pipeline,
    chronological_split,
    preprocessing_diagnostics,
)


def test_chronological_split_orders_rows_and_keeps_time_boundaries() -> None:
    frame = pd.DataFrame(
        {"TransactionDT": [30, 10, 40, 20], "isFraud": [0, 1, 0, 1], "value": [3, 1, 4, 2]}
    )
    train, validation, test = chronological_split(
        frame, SplitConfig(0.5, 0.25)
    )

    assert train["TransactionDT"].tolist() == [10, 20]
    assert validation["TransactionDT"].tolist() == [30]
    assert test["TransactionDT"].tolist() == [40]
    assert train["TransactionDT"].max() < validation["TransactionDT"].min()
    assert validation["TransactionDT"].max() < test["TransactionDT"].min()


def test_pipeline_fits_imputation_and_categories_only_from_training() -> None:
    training = pd.DataFrame(
        {
            "TransactionDT": [1, 2, 3, 4],
            "TransactionID": [11, 12, 13, 14],
            "isFraud": [0, 1, 0, 1],
            "amount": [10.0, None, 30.0, 40.0],
            "device": ["known", "known", None, "known"],
        }
    )
    pipeline = build_baseline_pipeline(training, BaselineConfig(max_iter=5))
    pipeline.fit(training[["amount", "device"]], training["isFraud"])

    unseen = pd.DataFrame({"amount": [None], "device": ["future-only-category"]})
    prediction = pipeline.predict(unseen)
    onehot = pipeline.named_steps["preprocessor"].named_transformers_["categorical"]
    assert prediction.shape == (1,)
    assert "future-only-category" not in onehot.named_steps["onehot"].categories_[0]


def test_pipeline_caps_high_cardinality_categories() -> None:
    training = pd.DataFrame(
        {
            "TransactionDT": range(20),
            "isFraud": [0, 1] * 10,
            "merchant": [f"merchant-{index}" for index in range(20)],
        }
    )
    config = BaselineConfig(
        max_iter=5,
        categorical_min_frequency=1,
        categorical_max_categories=5,
    )
    pipeline = build_baseline_pipeline(training, config)
    pipeline.fit(training[["merchant"]], training["isFraud"])

    onehot = pipeline.named_steps["preprocessor"].named_transformers_["categorical"]
    assert len(onehot.named_steps["onehot"].get_feature_names_out()) <= 5


def test_logistic_regression_uses_sparse_compatible_solver_and_scaling() -> None:
    training = pd.DataFrame(
        {
            "TransactionDT": range(8),
            "isFraud": [0, 1] * 4,
            "amount": [1.0, 1000.0, 2.0, 900.0, 3.0, 800.0, 4.0, 700.0],
            "device": ["a", "b"] * 4,
        }
    )
    pipeline = build_baseline_pipeline(
        training,
        BaselineConfig(max_iter=10, regularization_strength=0.1),
    )
    classifier = pipeline.named_steps["classifier"]
    numeric = pipeline.named_steps["preprocessor"].transformers[0][1]
    assert classifier.solver == "saga"
    assert numeric.named_steps["scaler"].with_mean is True
    features = training[["amount"]].astype("float32")
    assert features["amount"].dtype == "float32"


def test_preprocessing_diagnostics_identify_unseen_categories_without_targets() -> None:
    training = pd.DataFrame(
        {"TransactionDT": [1, 2], "isFraud": [0, 1], "device": ["known", "known"]}
    )
    later = pd.DataFrame(
        {"TransactionDT": [3, 4], "isFraud": [1, 0], "device": ["known", "new"]}
    )
    diagnostics = preprocessing_diagnostics(training, later, BaselineConfig())
    assert diagnostics["categorical_unknown_rate"]["device"] == 0.5


def test_unsupported_feature_dtype_fails_explicitly() -> None:
    frame = pd.DataFrame(
        {"TransactionDT": [1, 2], "isFraud": [0, 1], "event_time": pd.to_datetime(["2020-01-01", "2020-01-02"])}
    )
    with pytest.raises(ValueError, match="Unsupported feature dtypes"):
        build_baseline_pipeline(frame)
