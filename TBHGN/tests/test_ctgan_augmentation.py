from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.ctgan_augmentation import (
    CTGANConfig,
    MISSING_CATEGORY,
    fraud_training_partition,
    postprocess_synthetic_fraud,
    run_ctgan_augmentation,
    select_ctgan_features,
)


class FakeGenerator:
    def fit(self, data: pd.DataFrame, discrete_columns: list[str]) -> None:
        self.data = data.copy()
        self.columns = list(data.columns)
        self.discrete_columns = discrete_columns

    def sample(self, num_samples: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "amount": [10.0] * num_samples,
                "device": ["known"] * num_samples,
            }
        )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": range(1, 11),
            "TransactionDT": range(10, 0, -1),
            "isFraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "amount": [float(value) for value in range(10)],
            "device": ["known"] * 10,
        }
    )


def test_fraud_training_partition_uses_only_chronological_training_rows() -> None:
    fraud = fraud_training_partition(_frame())
    assert fraud["TransactionDT"].tolist() == [1, 3, 5, 7]
    assert fraud["isFraud"].eq(1).all()


def test_feature_selection_rejects_reserved_columns() -> None:
    with pytest.raises(ValueError, match="cannot include"):
        CTGANConfig(feature_columns=("amount", "TransactionID"))


def test_run_ctgan_saves_schema_samples_model_and_report(tmp_path: Path) -> None:
    config = CTGANConfig(
        feature_columns=("amount", "device"),
        sample_count=3,
        epochs=1,
        batch_size=2,
    )
    report = run_ctgan_augmentation(
        _frame(),
        tmp_path / "model.joblib",
        tmp_path / "synthetic.csv",
        tmp_path / "report.json",
        config,
        generator=FakeGenerator(),
    )
    generated = pd.read_csv(tmp_path / "synthetic.csv")
    assert list(generated.columns) == ["amount", "device", "isFraud"]
    assert generated["amount"].dtype.kind == "f"
    assert generated["isFraud"].eq(1).all()
    assert report["training_rows"] == 4
    assert report["synthetic_rows"] == 3
    assert joblib.load(tmp_path / "model.joblib").discrete_columns == ["device"]


def test_feature_selection_reports_original_dtypes() -> None:
    selected, dtypes = select_ctgan_features(
        _frame().loc[[_ for _ in range(5)]],
        CTGANConfig(feature_columns=("amount", "device")),
    )
    assert list(selected.columns) == ["amount", "device"]
    assert dtypes == {"amount": "float64", "device": "str"}


def test_ctgan_input_replaces_numeric_and_categorical_missing_values(
    tmp_path: Path,
) -> None:
    frame = _frame().loc[:, ["amount", "device"]].copy()
    frame.loc[3, "amount"] = None
    frame.loc[3, "device"] = None
    generator = FakeGenerator()
    config = CTGANConfig(
        feature_columns=("amount", "device"), sample_count=2, epochs=1, batch_size=2
    )
    run_ctgan_augmentation(
        _frame().assign(amount=frame["amount"], device=frame["device"]),
        tmp_path / "model.joblib",
        tmp_path / "synthetic.csv",
        tmp_path / "report.json",
        config,
        generator=generator,
    )
    assert generator.discrete_columns == ["device"]
    assert generator.columns == ["amount", "device"]
    assert not generator.data.isna().any().any()
    assert MISSING_CATEGORY in generator.data["device"].values


def test_postprocessing_clips_ranges_and_rounds_integer_like_features() -> None:
    real = pd.DataFrame({"card1": [100, 200], "amount": [1.0, 10.0]})
    generated = pd.DataFrame({"card1": [99.6], "amount": [-4.0]})
    config = CTGANConfig(feature_columns=("card1", "amount"), sample_count=1)
    processed = postprocess_synthetic_fraud(generated, real, config)
    assert processed.loc[0, "card1"] == 100
    assert processed.loc[0, "amount"] == 1.0
