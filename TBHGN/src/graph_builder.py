"""Leakage-safe heterogeneous graph construction for the TBHGN prototype.

The public split builder intentionally accepts the chronological training,
validation, and test frames separately. Only the training frame is converted
to a graph. Validation and test rows are used solely for an audit that proves
their transaction identifiers are absent from the graph.

Install the optional graph dependencies before importing this module:
``pip install torch torch-geometric``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch import Tensor
    from torch_geometric.data import HeteroData
except ImportError as error:  # pragma: no cover - depends on environment
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    HeteroData = None  # type: ignore[assignment,misc]
    _PYG_IMPORT_ERROR = error


USER_EDGE = ("user", "made", "transaction")
MERCHANT_EDGE = ("transaction", "executed_at", "merchant")
TEMPORAL_EDGE = ("transaction", "temporal_successor", "transaction")


@dataclass(frozen=True)
class GraphBuildConfig:
    """Column and graph-construction settings."""

    transaction_id_column: str = "TransactionID"
    temporal_column: str = "TransactionDT"
    user_columns: tuple[str, ...] = ("card1", "card2", "card3", "card5")
    merchant_column: str = "ProductCD"


@dataclass(frozen=True)
class GraphSplitAudit:
    """Evidence that only training rows entered the graph."""

    training_transaction_count: int
    validation_transaction_count: int
    test_transaction_count: int
    training_graph_transaction_count: int
    validation_ids_in_graph: tuple[object, ...]
    test_ids_in_graph: tuple[object, ...]

    @property
    def leakage_free(self) -> bool:
        return not self.validation_ids_in_graph and not self.test_ids_in_graph


def _require_pyg() -> None:
    if HeteroData is None or torch is None:
        raise ImportError(
            "PyTorch Geometric is required for graph construction. "
            "Install optional dependencies with: "
            "python -m pip install -r requirements-pyg.txt"
        ) from _PYG_IMPORT_ERROR


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {', '.join(missing)}")


def _as_feature_tensor(features: object, row_count: int) -> Tensor:
    _require_pyg()
    if isinstance(features, pd.DataFrame):
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
            raise TypeError("transaction_features DataFrame must contain numeric columns.")
        values = features.to_numpy(dtype=np.float32, copy=False)
    elif isinstance(features, np.ndarray):
        values = features
    elif torch is not None and isinstance(features, torch.Tensor):
        tensor = features.detach().to(dtype=torch.float32, device="cpu")
        if tensor.ndim != 2 or tensor.shape[0] != row_count:
            raise ValueError("transaction_features must have one row per training transaction.")
        return tensor.contiguous()
    else:
        raise TypeError("transaction_features must be a numeric DataFrame, NumPy array, or torch.Tensor.")
    if values.ndim != 2 or values.shape[0] != row_count:
        raise ValueError("transaction_features must have one row per training transaction.")
    if not np.isfinite(values).all():
        raise ValueError("transaction_features contains NaN or infinite values.")
    return torch.as_tensor(values, dtype=torch.float32)


def _first_user_column(frame: pd.DataFrame, config: GraphBuildConfig) -> str:
    for column in config.user_columns:
        if column in frame.columns:
            return column
    raise ValueError(
        "None of the configured user identifier columns are present: "
        + ", ".join(config.user_columns)
    )


def _stable_mapping(values: pd.Series) -> dict[object, int]:
    # Sorting by repr gives deterministic indices even when raw IDs have mixed types.
    unique_values = pd.unique(values.astype(object))
    if any(pd.isna(value) for value in unique_values):
        raise ValueError("Graph identifiers cannot contain missing values.")
    return {
        value: index
        for index, value in enumerate(sorted(unique_values, key=lambda value: repr(value)))
    }


def _edge_tensor(source: list[int], target: list[int]) -> Tensor:
    _require_pyg()
    if not source:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([source, target], dtype=torch.long)


def build_training_graph(
    training_frame: pd.DataFrame,
    transaction_features: object,
    config: GraphBuildConfig = GraphBuildConfig(),
) -> HeteroData:
    """Build a training-only ``HeteroData`` graph.

    One transaction node is created for each training row. User and merchant
    nodes are deduplicated using deterministic mappings. Temporal edges are
    directed from each user's earlier transaction to their later transaction;
    equal timestamps are ordered by the stable input order.
    """
    _require_pyg()
    if training_frame.empty:
        raise ValueError("training_frame must not be empty.")
    _require_columns(
        training_frame,
        [config.transaction_id_column, config.temporal_column, config.merchant_column],
        "graph",
    )
    user_column = _first_user_column(training_frame, config)
    _require_columns(training_frame, [user_column], "user")
    if training_frame[config.transaction_id_column].duplicated().any():
        raise ValueError("Training transaction identifiers must be unique.")
    if training_frame[config.temporal_column].isna().any():
        raise ValueError("Training temporal values cannot be missing.")

    features = _as_feature_tensor(transaction_features, len(training_frame))
    user_mapping = _stable_mapping(training_frame[user_column])
    merchant_mapping = _stable_mapping(training_frame[config.merchant_column])
    user_indices = [user_mapping[value] for value in training_frame[user_column]]
    merchant_indices = [merchant_mapping[value] for value in training_frame[config.merchant_column]]

    data = HeteroData()
    data["transaction"].x = features
    data["transaction"].transaction_id = torch.as_tensor(
        training_frame[config.transaction_id_column].to_numpy(), dtype=torch.long
    )
    data["transaction"].transaction_dt = torch.as_tensor(
        training_frame[config.temporal_column].to_numpy(), dtype=torch.long
    )
    data["user"].num_nodes = len(user_mapping)
    data["merchant"].num_nodes = len(merchant_mapping)
    data[USER_EDGE].edge_index = _edge_tensor(user_indices, list(range(len(training_frame))))
    data[MERCHANT_EDGE].edge_index = _edge_tensor(list(range(len(training_frame))), merchant_indices)

    temporal_source: list[int] = []
    temporal_target: list[int] = []
    for _, group in training_frame.assign(_row_index=np.arange(len(training_frame))).groupby(
        user_column, sort=False, dropna=False
    ):
        ordered = group.sort_values(config.temporal_column, kind="mergesort")["_row_index"].tolist()
        temporal_source.extend(ordered[:-1])
        temporal_target.extend(ordered[1:])
    data[TEMPORAL_EDGE].edge_index = _edge_tensor(temporal_source, temporal_target)
    data["user"].raw_id = list(user_mapping.keys())
    data["merchant"].raw_id = list(merchant_mapping.keys())
    return data


def build_training_graph_with_audit(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    transaction_features: object,
    config: GraphBuildConfig = GraphBuildConfig(),
) -> tuple[HeteroData, GraphSplitAudit]:
    """Build only the training graph and prove validation/test IDs are absent."""
    _require_columns(
        training_frame,
        [config.transaction_id_column],
        "training transaction",
    )
    _require_columns(validation_frame, [config.transaction_id_column], "validation transaction")
    _require_columns(test_frame, [config.transaction_id_column], "test transaction")
    graph = build_training_graph(training_frame, transaction_features, config)
    graph_ids = set(training_frame[config.transaction_id_column].tolist())
    validation_overlap = tuple(
        sorted(graph_ids.intersection(validation_frame[config.transaction_id_column].tolist()), key=repr)
    )
    test_overlap = tuple(
        sorted(graph_ids.intersection(test_frame[config.transaction_id_column].tolist()), key=repr)
    )
    audit = GraphSplitAudit(
        training_transaction_count=len(training_frame),
        validation_transaction_count=len(validation_frame),
        test_transaction_count=len(test_frame),
        training_graph_transaction_count=int(graph["transaction"].num_nodes),
        validation_ids_in_graph=validation_overlap,
        test_ids_in_graph=test_overlap,
    )
    if not audit.leakage_free:
        raise ValueError("Validation or test transaction IDs overlap the training graph.")
    return graph, audit


def validate_temporal_edges(
    graph: HeteroData,
) -> None:
    """Raise if any temporal successor edge points backward in time."""
    _require_pyg()
    edge_index = graph[TEMPORAL_EDGE].edge_index
    timestamps = graph["transaction"].transaction_dt
    if edge_index.numel() and not torch.all(
        timestamps[edge_index[0]] <= timestamps[edge_index[1]]
    ):
        raise ValueError("Temporal successor edges contain a backward-in-time edge.")


__all__ = [
    "GraphBuildConfig",
    "GraphSplitAudit",
    "USER_EDGE",
    "MERCHANT_EDGE",
    "TEMPORAL_EDGE",
    "build_training_graph",
    "build_training_graph_with_audit",
    "validate_temporal_edges",
]
