"""Bounded TBHGN prototype for the leakage-free Day 1 graph.

The model is deliberately small and modular:

FOEM encodes each transaction's feature values as a sequence and attends over
features. TOEM applies a time-sensitive GRU to strictly prior transactions.
RIM combines the raw transaction representation, FOEM output, and TOEM context.

Validation tensors are passed separately from the training ``HeteroData``.
This is intentional: validation and test nodes must never be inserted into the
training graph or its categorical mappings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from src.graph_builder import TEMPORAL_EDGE


@dataclass(frozen=True)
class TBHGNConfig:
    """Small CPU-friendly prototype settings."""

    hidden_dim: int = 32
    representation_dim: int = 32
    epochs: int = 5
    learning_rate: float = 1e-3
    max_history: int = 10
    random_seed: int = 42


class FeatureOrientedExtraction(nn.Module):
    """Paper-aligned bidirectional gated feature encoder and attention."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.forward_update = nn.Linear(hidden_dim + 1, hidden_dim)
        self.forward_reset = nn.Linear(hidden_dim + 1, hidden_dim)
        self.forward_candidate = nn.Linear(hidden_dim + 1, hidden_dim)
        self.backward_update = nn.Linear(hidden_dim + 1, hidden_dim)
        self.backward_reset = nn.Linear(hidden_dim + 1, hidden_dim)
        self.backward_candidate = nn.Linear(hidden_dim + 1, hidden_dim)
        self.query_projection = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.key_projection = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.attention_vector = nn.Parameter(torch.empty(hidden_dim * 2))
        self.attention_bias = nn.Parameter(torch.zeros(hidden_dim * 2))
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0))

    def _encode_direction(self, values: Tensor, reverse: bool) -> Tensor:
        batch_size, feature_count = values.shape
        hidden = values.new_zeros(batch_size, self.hidden_dim)
        states: list[Tensor] = []
        indices = range(feature_count - 1, -1, -1) if reverse else range(feature_count)
        update_layer = self.backward_update if reverse else self.forward_update
        reset_layer = self.backward_reset if reverse else self.forward_reset
        candidate_layer = self.backward_candidate if reverse else self.forward_candidate
        for index in indices:
            current = values[:, index].unsqueeze(-1)
            combined = torch.cat((hidden, current), dim=-1)
            update = torch.sigmoid(update_layer(combined))
            reset = torch.sigmoid(reset_layer(combined))
            candidate = torch.tanh(candidate_layer(torch.cat((reset * hidden, current), dim=-1)))
            hidden = (1.0 - update) * hidden + update * candidate
            states.append(hidden)
        if reverse:
            states.reverse()
        return torch.stack(states, dim=1)

    def forward(self, transaction_features: Tensor) -> Tensor:
        if transaction_features.ndim != 2 or transaction_features.shape[1] != self.feature_dim:
            raise ValueError("transaction_features must have shape [batch, features].")
        forward_states = self._encode_direction(transaction_features, reverse=False)
        backward_states = self._encode_direction(transaction_features, reverse=True)
        states = torch.cat((forward_states, backward_states), dim=-1)
        queries = self.query_projection(states).unsqueeze(2)
        keys = self.key_projection(states).unsqueeze(1)
        scores = torch.tanh(queries + keys + self.attention_bias).matmul(
            self.attention_vector
        )
        feature_count = transaction_features.shape[1]
        previous_mask = torch.tril(
            torch.ones(feature_count, feature_count, device=states.device, dtype=torch.bool),
            diagonal=-1,
        )
        scores = scores.masked_fill(~previous_mask.unsqueeze(0), float("-inf"))
        scores[:, 0, :] = float("-inf")
        scores[:, 0, 0] = 0.0
        weights = torch.softmax(scores, dim=-1)
        attended = torch.bmm(weights, states)
        return attended.mean(dim=1)


class TimeSensitiveGRUCell(nn.Module):
    """Time-sensitive gated unit from equations (11)-(15) of the paper."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_state = nn.Linear(hidden_dim + input_dim + 1, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim + input_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(hidden_dim + input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(hidden_dim + input_dim, hidden_dim)

    def forward(self, inputs: Tensor, hidden: Tensor, delta_seconds: Tensor) -> Tensor:
        time_input = torch.cat((hidden, inputs, delta_seconds.unsqueeze(-1)), dim=-1)
        time_state = torch.tanh(self.time_state(time_input))
        gate_input = torch.cat((hidden, inputs, time_state), dim=-1)
        update = torch.sigmoid(self.update_gate(gate_input))
        reset = torch.sigmoid(self.reset_gate(gate_input))
        candidate = torch.tanh(self.candidate(torch.cat((reset * hidden, inputs), dim=-1)))
        return (1.0 - update) * hidden + update * candidate


class TransactionOrientedExtraction(nn.Module):
    """Time-sensitive encoder with paper-style pairwise transaction attention."""

    def __init__(self, representation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.representation_dim = representation_dim
        self.cell = TimeSensitiveGRUCell(representation_dim, hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_vector = nn.Parameter(torch.empty(hidden_dim))
        self.attention_bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0))

    def forward(
        self,
        history_features: Tensor,
        deltas: Tensor,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        if history_features.ndim != 3 or deltas.ndim != 2:
            raise ValueError("History must be [batch, steps, features] and deltas [batch, steps].")
        if history_features.shape[:2] != deltas.shape:
            raise ValueError("History and delta dimensions do not match.")
        if history_features.shape[2] != self.representation_dim:
            raise ValueError("History features must be FOEM representations.")
        if history_mask is not None and history_mask.shape != deltas.shape:
            raise ValueError("History mask and delta dimensions do not match.")
        hidden = history_features.new_zeros(history_features.shape[0], self.cell.hidden_dim)
        states: list[Tensor] = []
        for step in range(history_features.shape[1]):
            hidden = self.cell(
                history_features[:, step],
                hidden,
                deltas[:, step].clamp_min(0),
            )
            states.append(hidden)
        sequence = torch.stack(states, dim=1)
        query = self.query_projection(sequence[:, -1:]).expand_as(sequence)
        keys = self.key_projection(sequence)
        scores = torch.tanh(query + keys + self.attention_bias).matmul(self.attention_vector)
        if history_mask is not None:
            scores = scores.masked_fill(history_mask <= 0, float("-inf"))
            empty_rows = history_mask.sum(dim=1) == 0
            scores[empty_rows, 0] = 0.0
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class RepresentationInteraction(nn.Module):
    """RIM: nonlinear interaction of raw, feature, and behavior representations."""

    def __init__(self, feature_dim: int, feature_rep_dim: int, behavior_dim: int, output_dim: int) -> None:
        super().__init__()
        self.raw_projection = nn.Linear(feature_dim, output_dim)
        self.feature_projection = nn.Linear(feature_rep_dim, output_dim, bias=False)
        self.behavior_projection = nn.Linear(behavior_dim, output_dim, bias=False)
        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, raw: Tensor, key_features: Tensor, behavior: Tensor) -> Tensor:
        enhanced = torch.tanh(
            self.raw_projection(raw)
            + self.feature_projection(key_features)
            + self.behavior_projection(behavior)
        )
        return self.classifier(enhanced).squeeze(-1)


class TBHGNPrototype(nn.Module):
    """FOEM -> TOEM -> RIM fraud-probability model."""

    def __init__(self, feature_dim: int, config: TBHGNConfig = TBHGNConfig()) -> None:
        super().__init__()
        self.foem = FeatureOrientedExtraction(feature_dim, config.hidden_dim)
        self.toem = TransactionOrientedExtraction(config.hidden_dim * 2, config.hidden_dim)
        self.rim = RepresentationInteraction(
            feature_dim,
            config.hidden_dim * 2,
            config.hidden_dim,
            config.representation_dim,
        )

    def forward(
        self,
        transaction_features: Tensor,
        history_features: Tensor,
        deltas: Tensor,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        key_features = self.foem(transaction_features)
        if history_features.ndim != 3 or history_features.shape[2] != transaction_features.shape[1]:
            raise ValueError("history_features must contain raw transaction features.")
        batch_size, steps, feature_count = history_features.shape
        history_representations = self.foem(
            history_features.reshape(batch_size * steps, feature_count)
        ).reshape(batch_size, steps, -1)
        behavior = self.toem(history_representations, deltas, history_mask)
        return self.rim(transaction_features, key_features, behavior)


def _validate_batch(features: Tensor, history: Tensor, deltas: Tensor, labels: Tensor) -> None:
    if features.ndim != 2 or history.ndim != 3 or deltas.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected features [N,F], history [N,H,F], deltas [N,H], labels [N].")
    if features.shape[0] != history.shape[0] or features.shape[0] != deltas.shape[0] or features.shape[0] != labels.shape[0]:
        raise ValueError("All batch tensors must contain the same number of transactions.")
    if history.shape[1] != deltas.shape[1] or history.shape[2] != features.shape[1]:
        raise ValueError("History dimensions must match transaction feature dimensions.")


def metrics_from_logits(logits: Tensor, labels: Tensor) -> dict[str, float]:
    """Return imbalance-aware validation metrics from model logits."""
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    targets = labels.detach().cpu().numpy().astype(int)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
    }


def _training_inputs_from_graph(
    graph: HeteroData,
    labels: Tensor,
    max_history: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create strictly-prior history tensors from Day 1 temporal edges."""
    from src.graph_builder import validate_temporal_edges

    validate_temporal_edges(graph)
    features = graph["transaction"].x.float()
    timestamps = graph["transaction"].transaction_dt.float()
    edge_index = graph[TEMPORAL_EDGE].edge_index
    predecessors: list[list[int]] = [[] for _ in range(features.shape[0])]
    for source, target in edge_index.t().tolist():
        predecessors[target].append(source)
    histories = torch.zeros(
        features.shape[0], max_history, features.shape[1], dtype=features.dtype
    )
    deltas = torch.zeros(features.shape[0], max_history, dtype=features.dtype)
    for transaction_index, prior_nodes in enumerate(predecessors):
        prior_nodes = prior_nodes[-max_history:]
        if prior_nodes:
            padded_nodes = [prior_nodes[0]] * (max_history - len(prior_nodes)) + prior_nodes
        else:
            padded_nodes = []
        for position, prior_index in enumerate(padded_nodes):
            histories[transaction_index, position] = features[prior_index]
            deltas[transaction_index, position] = max(
                0.0, float(timestamps[transaction_index] - timestamps[prior_index])
            )
    return features, histories, deltas, labels.float()


def run_bounded_training(
    training_graph: HeteroData,
    training_labels: Tensor,
    validation_features: Tensor,
    validation_history: Tensor,
    validation_deltas: Tensor,
    validation_labels: Tensor,
    config: TBHGNConfig = TBHGNConfig(),
) -> tuple[TBHGNPrototype, list[dict[str, float]]]:
    """Train on graph transactions and evaluate on isolated validation tensors."""
    torch.manual_seed(config.random_seed)
    if training_graph["transaction"].num_nodes != len(training_labels):
        raise ValueError("training_labels must align with training graph transactions.")
    train_features, train_history, train_deltas, train_labels = _training_inputs_from_graph(
        training_graph, training_labels, config.max_history
    )
    _validate_batch(validation_features, validation_history, validation_deltas, validation_labels)
    model = TBHGNPrototype(train_features.shape[1], config)
    positive = train_labels.sum()
    if positive <= 0 or positive >= len(train_labels):
        raise ValueError("Training labels must contain both fraud classes.")
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=((len(train_labels) - positive) / positive).detach()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        initial = metrics_from_logits(
            model(validation_features, validation_history, validation_deltas),
            validation_labels,
        )
    initial["epoch"] = 0.0
    history.append(initial)
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_features, train_history, train_deltas)
        loss = loss_function(logits, train_labels)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_metrics = metrics_from_logits(
                model(validation_features, validation_history, validation_deltas),
                validation_labels,
            )
        validation_metrics["epoch"] = float(epoch)
        validation_metrics["loss"] = float(loss.detach())
        history.append(validation_metrics)
    return model, history


def _demo() -> None:
    """Run a tiny isolated demonstration without reading the full dataset."""
    from torch_geometric.data import HeteroData

    torch.manual_seed(42)
    feature_count = 6
    training_count = 12
    validation_count = 4
    graph = HeteroData()
    graph["transaction"].x = torch.randn(training_count, feature_count)
    graph["transaction"].transaction_dt = torch.arange(training_count)
    graph["transaction"].num_nodes = training_count
    graph[TEMPORAL_EDGE].edge_index = torch.tensor(
        [list(range(training_count - 1)), list(range(1, training_count))],
        dtype=torch.long,
    )
    training_labels = torch.tensor([0, 1] * (training_count // 2), dtype=torch.float32)
    validation_features = torch.randn(validation_count, feature_count)
    validation_history = torch.randn(validation_count, 3, feature_count)
    validation_deltas = torch.ones(validation_count, 3)
    validation_labels = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    _, history = run_bounded_training(
        graph,
        training_labels,
        validation_features,
        validation_history,
        validation_deltas,
        validation_labels,
        TBHGNConfig(epochs=1),
    )
    print(f"Initial validation PR-AUC: {history[0]['pr_auc']:.4f}")


if __name__ == "__main__":
    _demo()
