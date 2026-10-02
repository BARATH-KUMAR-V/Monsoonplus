"""Pure-PyTorch GCN and GAT layers -- no torch_geometric required.

Where this came from
--------------------
MonsoonPlus_v3 shipped its graph layers as dense ``einsum`` code over an ``(N, N)``
adjacency instead of PyG's sparse ``edge_index`` kernels. This module is that idea,
rebuilt so it is a *drop-in twin* of the PyG layers the main model already uses:

* Same maths. ``DenseGCNConv`` reproduces ``GCNConv(add_self_loops=True)`` including its
  symmetric ``D^-1/2 (A + I) D^-1/2`` normalisation over the weighted Gaussian-kernel
  adjacency; ``DenseGATConv`` reproduces ``GATConv`` (self-loops, LeakyReLU(0.2)
  scores, softmax over neighbours, heads concatenated, bias).
* Same parameter names and shapes (``lin.weight``, ``bias``, ``att_src``, ``att_dst``).
  A checkpoint trained with the PyG layers therefore loads into the dense twin with
  ``strict=True`` and produces the same outputs -- which is what lets the shipped
  ``monsoonplus_final.pt`` run on a machine with no torch_geometric at all
  (``load_checkpoint(path, graph_layer="dense_gcn")``).
* ``tests/test_dense_graph.py`` pins all of the above numerically, so "twin" is a tested
  claim and not a hope.

Why bother
----------
1. torch_geometric is the only awkward dependency (version drift, Windows installs).
   Inference and demos should not need it.
2. The whole model is plain tensor ops (matmul, softmax, masked-fill, GRU), which is the
   easy case for ONNX. Checked: the shipped checkpoint, loaded with
   ``graph_layer="dense_gcn"`` and exported with the legacy tracer at opset 17 (graph
   baked in), runs in onnxruntime and matches the PyTorch outputs to ~4e-6 km/h -- a
   possible route to in-browser inference later. (Not wired into the frontend.)
3. On a 10-node road graph a dense ``(N, N)`` product is as cheap as a sparse one.

Dense is the wrong choice for a graph with tens of thousands of nodes (the ``N x N``
matrix grows quadratically); METR-LA's 207 sensors are still comfortably small.

Conventions
-----------
``adj[i, j]`` is the weight of the edge ``j -> i`` (row = receiving node), matching how
PyG scatters messages into ``edge_index[1]``. Node features are ``(batch, nodes, hidden)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dense_adjacency(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    num_nodes: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Scatter a PyG-style ``edge_index`` (+ optional weights) into an ``(N, N)`` matrix.

    ``adj[target, source] = weight``. Duplicate edges are summed, which is also what
    PyG's message passing does. Self-loops already present are kept as given; the layers
    below add their own, exactly as GCNConv/GATConv do.
    """
    device = device or edge_index.device
    edge_index = edge_index.to(device)
    if edge_weight is None:
        weights = torch.ones(edge_index.size(1), device=device, dtype=dtype)
    else:
        weights = edge_weight.to(device=device, dtype=dtype)
    adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
    adj.index_put_((edge_index[1], edge_index[0]), weights, accumulate=True)
    return adj


def gcn_normalise(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """``D^-1/2 (A + I) D^-1/2``, as ``torch_geometric.nn.conv.gcn_conv.gcn_norm`` computes.

    Existing diagonal entries are replaced by a unit self-loop (PyG removes then re-adds
    them), and the degree is the sum of *incoming* weights -- identical to the row sum
    in this module's ``adj[target, source]`` convention.
    """
    n = adj.size(-1)
    if add_self_loops:
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
        adj = adj * (1.0 - eye) + eye
    degree = adj.sum(dim=-1)
    inv_sqrt = degree.pow(-0.5)
    inv_sqrt = torch.where(torch.isinf(inv_sqrt), torch.zeros_like(inv_sqrt), inv_sqrt)
    return inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)


class DenseGCNConv(nn.Module):
    """Twin of ``GCNConv(in, out, add_self_loops=True)``: ``out = A_hat @ (x W^T) + b``.

    Parameter names match PyG's (``lin.weight``, ``bias``) so state dicts transfer.
    The normalised adjacency is recomputed per call from the raw one; for a fixed graph
    pass a pre-normalised matrix with ``normalised=True`` to skip the work.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.lin.weight)  # glorot, as GCNConv initialises

    def forward(self, x: torch.Tensor, adj: torch.Tensor, normalised: bool = False) -> torch.Tensor:
        a_hat = adj if normalised else gcn_normalise(adj)
        support = self.lin(x)                                   # (B, N, out)
        return torch.einsum("ij,bjf->bif", a_hat, support) + self.bias


class DenseGATConv(nn.Module):
    """Twin of ``GATConv(in, out // heads, heads=heads)`` with ``concat=True``.

    Attention runs over the edges ``adj`` has (``adj > 0``) plus a self-loop on every
    node; edge *weights* are ignored, as in GATConv, which learns its own weighting.
    Parameter names match PyG's: ``lin.weight`` ``(heads*C, in)``, ``att_src`` and
    ``att_dst`` ``(1, heads, C)``, ``bias`` ``(heads*C,)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels_per_head: int,
        heads: int = 4,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels_per_head
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_channels, heads * out_channels_per_head, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels_per_head))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels_per_head))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels_per_head))

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        h = self.lin(x).view(batch, nodes, self.heads, self.out_channels)  # (B, N, H, C)

        alpha_src = (h * self.att_src).sum(dim=-1)  # (B, N, H)  score as a *source*
        alpha_dst = (h * self.att_dst).sum(dim=-1)  # (B, N, H)  score as a *target*

        # score[b, h, i, j] for the edge j -> i
        scores = alpha_dst.permute(0, 2, 1).unsqueeze(-1) + alpha_src.permute(0, 2, 1).unsqueeze(-2)
        scores = F.leaky_relu(scores, self.negative_slope)

        eye = torch.eye(nodes, device=x.device, dtype=torch.bool)
        mask = ((adj > 0) | eye).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        scores = scores.masked_fill(~mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        out = torch.einsum("bhij,bjhc->bihc", attention, h)  # (B, N, H, C)
        return out.reshape(batch, nodes, self.heads * self.out_channels) + self.bias


def dense_conv(kind: str, hidden: int, heads: int = 4, dropout: float = 0.1) -> nn.Module:
    """Factory used by ``SpatialBlock``: ``kind`` is ``"gcn"`` or ``"gat"``."""
    if kind == "gcn":
        return DenseGCNConv(hidden, hidden)
    if kind == "gat":
        if hidden % heads:
            raise ValueError(f"hidden {hidden} must divide by heads {heads} for GAT")
        return DenseGATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
    raise ValueError(f"unknown dense layer kind {kind!r}")


__all__ = [
    "DenseGATConv",
    "DenseGCNConv",
    "dense_adjacency",
    "dense_conv",
    "gcn_normalise",
]

