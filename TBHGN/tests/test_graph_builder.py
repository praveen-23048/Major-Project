"""Leakage and shape tests for the PyG graph builder."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.graph_builder import (  # noqa: E402
    MERCHANT_EDGE,
    TEMPORAL_EDGE,
    USER_EDGE,
    build_training_graph_with_audit,
    validate_temporal_edges,
)


def test_training_graph_schema_temporal_order_and_split_isolation() -> None:
    train = pd.DataFrame(
        {
            "TransactionID": [10, 11, 12, 13],
            "TransactionDT": [30, 10, 20, 40],
            "card1": ["u1", "u1", "u2", "u1"],
            "ProductCD": ["A", "A", "B", "A"],
        }
    )
    validation = pd.DataFrame(
        {"TransactionID": [14], "TransactionDT": [50], "card1": ["u3"], "ProductCD": ["C"]}
    )
    test = pd.DataFrame(
        {"TransactionID": [15], "TransactionDT": [60], "card1": ["u4"], "ProductCD": ["D"]}
    )
    graph, audit = build_training_graph_with_audit(
        train,
        validation,
        test,
        np.ones((len(train), 3), dtype=np.float32),
    )

    assert graph["transaction"].num_nodes == 4
    assert graph["user"].num_nodes == 2
    assert graph["merchant"].num_nodes == 2
    assert graph[USER_EDGE].edge_index.shape == (2, 4)
    assert graph[MERCHANT_EDGE].edge_index.shape == (2, 4)
    assert graph[TEMPORAL_EDGE].edge_index.shape == (2, 2)
    validate_temporal_edges(graph)
    assert audit.leakage_free
    assert audit.validation_ids_in_graph == ()
    assert audit.test_ids_in_graph == ()


def test_validation_overlap_is_rejected() -> None:
    frame = pd.DataFrame(
        {"TransactionID": [1], "TransactionDT": [1], "card1": ["u"], "ProductCD": ["A"]}
    )
    with pytest.raises(ValueError, match="overlap"):
        build_training_graph_with_audit(
            frame,
            frame,
            frame.iloc[0:0],
            np.ones((1, 2), dtype=np.float32),
        )
