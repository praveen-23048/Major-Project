"""Tests for bounded real-data TBHGN history construction."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scripts.run_tbhgn_real_experiment import build_prior_histories, load_bounded_chronological


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    source = pd.DataFrame(
        {
            "TransactionDT": [1, 3, 5, 7],
            "card1": ["a", "a", "b", "a"],
        }
    )
    target = pd.DataFrame(
        {
            "TransactionDT": [4, 8],
            "card1": ["a", "a"],
        }
    )
    source_features = np.arange(8, dtype=np.float32).reshape(4, 2)
    target_features = np.arange(4, dtype=np.float32).reshape(2, 2) + 100
    return target, source, target_features, source_features


def test_histories_are_strictly_prior_and_aligned() -> None:
    target, source, target_features, source_features = _frames()
    histories, deltas, masks = build_prior_histories(
        target, source, target_features, source_features, max_history=2
    )
    assert histories.shape == (2, 2, 2)
    assert torch.equal(histories[0, 1], torch.tensor([2.0, 3.0]))
    assert torch.equal(histories[1, 1], torch.tensor([6.0, 7.0]))
    assert deltas[0, 1].item() == 1
    assert deltas[1, 1].item() == 1
    assert torch.all(masks == 1)


def test_history_padding_repeats_earliest_prior_and_no_future_row() -> None:
    target = pd.DataFrame({"TransactionDT": [2], "card1": ["a"]})
    source = pd.DataFrame({"TransactionDT": [1, 10], "card1": ["a", "a"]})
    target_features = np.array([[99.0]], dtype=np.float32)
    source_features = np.array([[1.0], [10.0]], dtype=np.float32)
    histories, deltas, masks = build_prior_histories(
        target, source, target_features, source_features, max_history=3
    )
    assert histories[0, :, 0].tolist() == [1.0, 1.0, 1.0]
    assert deltas[0].tolist() == [1.0, 1.0, 1.0]
    assert masks[0].tolist() == [1.0, 1.0, 1.0]


def test_missing_proxy_has_zero_history_and_mask() -> None:
    target = pd.DataFrame({"TransactionDT": [2], "card1": [None]})
    source = pd.DataFrame({"TransactionDT": [1], "card1": ["a"]})
    histories, deltas, masks = build_prior_histories(
        target,
        source,
        np.array([[99.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        max_history=2,
    )
    assert np.allclose(histories.numpy(), 0)
    assert np.allclose(deltas.numpy(), 0)
    assert np.allclose(masks.numpy(), 0)


def test_equal_timestamp_is_not_admitted_as_prior_history() -> None:
    target = pd.DataFrame({"TransactionDT": [2], "card1": ["a"]})
    source = pd.DataFrame({"TransactionDT": [1, 2, 3], "card1": ["a", "a", "a"]})
    histories, deltas, masks = build_prior_histories(
        target,
        source,
        np.array([[99.0]], dtype=np.float32),
        np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        max_history=2,
    )
    assert histories[0, :, 0].tolist() == [1.0, 1.0]
    assert deltas[0].tolist() == [1.0, 1.0]
    assert masks[0].tolist() == [1.0, 1.0]


def test_bounded_loader_preserves_chronological_partition_sizes(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "TransactionID": list(range(1, 22)),
            "TransactionDT": list(range(10, 201, 10)) + [140],
            "isFraud": [0, 1] * 10 + [0],
            "card1": list("aabbccddee") * 2 + ["f"],
        }
    )
    path = tmp_path / "transactions.csv"
    frame.to_csv(path, index=False)
    train, validation, metadata = load_bounded_chronological(
        path, training_rows=4, validation_rows=2, chunksize=3
    )
    assert train["TransactionDT"].tolist() == [10, 20, 30, 40]
    assert validation["TransactionDT"].tolist() == [140, 150]
    assert metadata["full_rows"] == 21
