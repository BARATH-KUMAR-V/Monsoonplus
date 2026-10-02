"""Spatial block over the road graph. GCN is the default; GAT is a one-line swap.

The swap, in full:

    model = MonsoonPlusNet(graph_layer="gat")      # instead of "gcn"

or from the CLI:

    python -m ml.training.train --graph-layer gat

Everything else -- encoders, gate, head, checkpoint layout -- is identical, which is
what makes the GCN-vs-GAT ablation in ``ml/training/ablations.py`` a fair comparison.

Dense twins (no torch_geometric)
--------------------------------
``"dense_gcn"`` and ``"dense_gat"`` run the same two-round block on pure-PyTorch layers
from ``ml/models/dense_graph.py``. They are numerically equivalent to ``"gcn"`` /
``"gat"`` and share parameter names, so a PyG-trained checkpoint loads straight into
its dense twin (``load_checkpoint(path, graph_layer="dense_gcn")``). Use them when
torch_geometric is unavailable or when exporting to ONNX.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.dense_graph import dense_adjacency, dense_conv, gcn_normalise

# torch_geometric is imported lazily, inside the branch that needs it, so the dense
# variants work on a machine that does not have it installed.
PYG_LAYERS = ("gcn", "gat")
DENSE_LAYERS = ("dense_gcn", "dense_gat")
GRAPH_LAYERS = PYG_LAYERS + DENSE_LAYERS


class SpatialBlock(nn.Module):
    """Two message-passing rounds with a residual connection.

    Two rounds is deliberate: Chennai's graph has a diameter of about 4, so two hops
    lets congestion at Velachery reach the OMR segments without over-smoothing a
    10-node graph into a single value.

    The batch is folded into the node dimension as one big disconnected graph
    (PyG's standard trick), so edge_index is reused unchanged across the batch.
    """

    def __init__(
        self,
        hidden: int,
        graph_layer: str = "gcn",
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        graph_layer = graph_layer.lower()
        if graph_layer not in GRAPH_LAYERS:
            raise ValueError(f"graph_layer must be one of {GRAPH_LAYERS}, got {graph_layer!r}")

        self.graph_layer = graph_layer
        self.dropout = dropout
        self.dense = graph_layer in DENSE_LAYERS
        # The family decides the maths; the "dense_" prefix only decides the backend.
        self.family = graph_layer.replace("dense_", "")

        if self.dense:
            self.conv1 = dense_conv(self.family, hidden, heads=heads, dropout=dropout)
            self.conv2 = dense_conv(self.family, hidden, heads=heads, dropout=dropout)
        else:
            try:
                from torch_geometric.nn import GATConv, GCNConv
            except ImportError as error:  # pragma: no cover - environment dependent
                raise ImportError(
                    f"graph_layer={graph_layer!r} needs torch_geometric "
                    "(pip install torch_geometric). To run without it use "
                    f"graph_layer='dense_{graph_layer}', which is numerically equivalent."
                ) from error

            if graph_layer == "gcn":
                self.conv1 = GCNConv(hidden, hidden, add_self_loops=True)
                self.conv2 = GCNConv(hidden, hidden, add_self_loops=True)
            else:
                # heads * out = hidden keeps the width identical to the GCN variant, so a
                # checkpoint's downstream shapes do not change between the two.
                if hidden % heads:
                    raise ValueError(f"hidden {hidden} must divide by heads {heads} for GAT")
                self.conv1 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
                self.conv2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def _conv(
        self,
        conv: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # GATConv learns its own attention and takes no edge_weight argument.
        if self.family == "gcn":
            return conv(x, edge_index, edge_weight=edge_weight)
        return conv(x, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, nodes, hidden = x.shape

        if self.dense:
            return self._forward_dense(x, edge_index, edge_weight)

        big_edge_index, big_edge_weight = batch_graph(
            edge_index, edge_weight, nodes, batch, x.device
        )
        flat = x.reshape(batch * nodes, hidden)

        h = self._conv(self.conv1, flat, big_edge_index, big_edge_weight)
        h = self.norm1(F.gelu(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h2 = self._conv(self.conv2, h, big_edge_index, big_edge_weight)
        h2 = self.norm2(F.gelu(h2))

        return (flat + h2).reshape(batch, nodes, hidden)

    def _forward_dense(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Same block as the PyG path, but on ``(B, N, H)`` with an ``(N, N)`` adjacency.

        No batch folding is needed: the adjacency is shared by every sample, so the
        einsum inside each layer broadcasts over the batch dimension.
        """
        nodes = x.shape[1]
        adj = dense_adjacency(edge_index, edge_weight, nodes, x.device, x.dtype)
        if self.family == "gcn":
            adj = gcn_normalise(adj)  # once per forward, shared by both rounds

        def conv(layer: nn.Module, h: torch.Tensor) -> torch.Tensor:
            if self.family == "gcn":
                return layer(h, adj, normalised=True)
            return layer(h, adj)

        h = conv(self.conv1, x)
        h = self.norm1(F.gelu(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h2 = conv(self.conv2, h)
        h2 = self.norm2(F.gelu(h2))
        return x + h2


def batch_graph(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    nodes: int,
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Tile one graph ``batch`` times into a single disconnected graph.

    Offsetting each copy by ``i * nodes`` keeps the copies from exchanging messages,
    which is the whole point -- sample 3's Velachery must not influence sample 4's.
    """
    edge_index = edge_index.to(device)
    offsets = (torch.arange(batch, device=device) * nodes).repeat_interleave(
        edge_index.size(1)
    )
    big_edge_index = edge_index.repeat(1, batch) + offsets.unsqueeze(0)
    big_edge_weight = (
        edge_weight.to(device).repeat(batch) if edge_weight is not None else None
    )
    return big_edge_index, big_edge_weight
