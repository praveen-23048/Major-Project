"""Bounded synthetic tests for the Day 2 TBHGN prototype."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402

from src.graph_builder import TEMPORAL_EDGE  # noqa: E402
from src.tbhgn_prototype import (  # noqa: E402
    FeatureOrientedExtraction,
    TBHGNConfig,
    TBHGNPrototype,
    run_bounded_training,
)


def _graph() -> HeteroData:
    graph = HeteroData()
    graph["transaction"].x = torch.randn(8, 5)
    graph["transaction"].transaction_dt = torch.arange(8)
    graph["transaction"].num_nodes = 8
    graph[TEMPORAL_EDGE].edge_index = torch.tensor(
        [list(range(7)), list(range(1, 8))], dtype=torch.long
    )
    return graph


def test_model_forward_shape_and_probability_range() -> None:
    model = TBHGNPrototype(5, TBHGNConfig(hidden_dim=8, representation_dim=8))
    assert model.toem.representation_dim == 16
    logits = model(torch.randn(3, 5), torch.randn(3, 2, 5), torch.ones(3, 2))
    assert logits.shape == (3,)
    assert torch.all((torch.sigmoid(logits) >= 0) & (torch.sigmoid(logits) <= 1))


def test_foem_returns_bidirectional_transaction_representation() -> None:
    foem = FeatureOrientedExtraction(feature_dim=5, hidden_dim=8)
    representation = foem(torch.randn(2, 5))
    assert representation.shape == (2, 16)


def test_bounded_training_reports_initial_and_final_validation_metrics() -> None:
    model, history = run_bounded_training(
        _graph(),
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32),
        torch.randn(4, 5),
        torch.randn(4, 2, 5),
        torch.ones(4, 2),
        torch.tensor([0, 1, 0, 1], dtype=torch.float32),
        TBHGNConfig(hidden_dim=8, representation_dim=8, epochs=1),
    )
    assert isinstance(model, TBHGNPrototype)
    assert len(history) == 2
    assert all(0 <= row["pr_auc"] <= 1 for row in history)
