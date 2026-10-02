"""Adjacency construction for both graphs: METR-LA's 207 sensors and Chennai's 10 segments.

Why the Gaussian kernel matters (this was a real bug, not a nicety)
------------------------------------------------------------------
A first attempt weighted each Chennai edge by ``1 / travel_time``, which produced
weights around 0.003-0.008. ``GCNConv`` adds self-loops with weight 1.0, so after
normalisation each node's own feature carried >99% of the message and the graph
contributed essentially nothing -- the "GNN" was a per-node MLP wearing a costume.

The fix is the same thresholded Gaussian kernel METR-LA's own adjacency uses:

    W[i, j] = exp(-(d[i, j] / sigma)^2)   if that value >= threshold, else 0

with ``sigma`` the standard deviation of the finite distances. That puts neighbour
weights in the 0.1-1.0 range, comparable to the self-loop, so neighbouring roads
actually influence each other. ``assert_graph_contributes`` guards the regression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from config.segments import (
    JUNCTION_BY_ID,
    MODEL_CFG,
    SEGMENTS,
    SEGMENT_IDS,
    haversine_km,
    segment_midpoints,
)

THRESHOLD = float(MODEL_CFG["adjacency_threshold"])


def gaussian_kernel_adjacency(
    distances: np.ndarray,
    threshold: float = THRESHOLD,
    sigma: Optional[float] = None,
    sigma_mode: str = "mean",
) -> np.ndarray:
    """Thresholded Gaussian kernel over a distance matrix.

    ``distances`` may contain ``inf`` for unconnected pairs. The diagonal is left at 0
    -- GCNConv/GATConv add their own self-loops, and a doubled self-loop skews the
    normalisation.

    Choosing ``sigma``
    ------------------
    METR-LA's published adjacency uses ``sigma = std(finite distances)``, which works
    there because 207 sensors spread over LA give a wide distance spread. On the
    10-segment Chennai graph every connected pair sits 2-5 km apart, so the standard
    deviation is small, ``d / sigma`` becomes large, and ``exp(-(d/sigma)^2)`` underflows
    below the threshold -- producing an adjacency with zero edges. ``sigma_mode="mean"``
    scales by the typical distance instead, which keeps weights in a usable 0.1-0.8
    band on both graphs.

    Pass ``sigma_mode="std"`` to reproduce the METR-LA convention exactly.
    """
    distances = np.asarray(distances, dtype=np.float64)
    finite = distances[np.isfinite(distances) & (distances > 0)]
    if sigma is None:
        if finite.size == 0:
            sigma = 1.0
        elif sigma_mode == "std":
            sigma = float(finite.std())
        elif sigma_mode == "mean":
            sigma = float(finite.mean())
        else:
            raise ValueError(f"unknown sigma_mode {sigma_mode!r}; use 'mean' or 'std'")
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    with np.errstate(over="ignore"):
        weights = np.exp(-np.square(distances / sigma))
    weights[~np.isfinite(weights)] = 0.0
    weights[weights < threshold] = 0.0
    np.fill_diagonal(weights, 0.0)
    return weights.astype(np.float32)


def chennai_distance_matrix() -> np.ndarray:
    """Road-network distance between segment midpoints, in km.

    Two segments are "connected" when they share a junction; everything else is inf
    so the kernel zeroes it. Sharing a junction is what lets congestion actually
    propagate, which is the physical relationship we want the GNN to model.
    """
    n = len(SEGMENTS)
    midpoints = segment_midpoints()
    distances = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)

    for i, seg_i in enumerate(SEGMENTS):
        ends_i = {seg_i.a, seg_i.b}
        for j, seg_j in enumerate(SEGMENTS):
            if i == j:
                continue
            if ends_i & {seg_j.a, seg_j.b}:
                distances[i, j] = haversine_km(midpoints[i], midpoints[j])
    return distances


def chennai_adjacency(threshold: float = THRESHOLD) -> np.ndarray:
    """Chennai's 10-segment adjacency. ``sigma_mode="mean"`` -- see the kernel docstring."""
    return gaussian_kernel_adjacency(
        chennai_distance_matrix(), threshold=threshold, sigma_mode="mean"
    )


def metr_la_distance_matrix(
    sensor_ids: list[str], distances_csv: Path
) -> Optional[np.ndarray]:
    """Build METR-LA's distance matrix from the published sensor distance table.

    The CSV is ``from,to,cost`` over detector ids. Returns None when the file is
    absent so the caller can fall back to a correlation adjacency.
    """
    if not distances_csv.exists():
        return None

    index = {sid: i for i, sid in enumerate(sensor_ids)}
    n = len(sensor_ids)
    distances = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)

    rows = 0
    with distances_csv.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if "," not in header:
            return None
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            src, dst, cost = parts[0], parts[1], parts[2]
            i, j = index.get(src), index.get(dst)
            if i is None or j is None:
                continue
            try:
                distances[i, j] = float(cost)
            except ValueError:
                continue
            rows += 1
    if rows == 0:
        return None
    return distances


def correlation_adjacency(speeds: np.ndarray, top_k: int = 8) -> np.ndarray:
    """Fallback adjacency: connect each sensor to its k most correlated peers.

    Used only when the distance table is unavailable. Weaker than the real road
    graph, and the training log says so when it happens.
    """
    observed = speeds > 0
    filled = np.where(observed, speeds, np.nan)
    column_means = np.nanmean(filled, axis=0)
    filled = np.where(np.isnan(filled), column_means[None, :], filled)

    corr = np.corrcoef(filled.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, -np.inf)

    n = corr.shape[0]
    weights = np.zeros((n, n), dtype=np.float32)
    k = min(top_k, n - 1)
    for i in range(n):
        for j in np.argsort(corr[i])[-k:]:
            value = max(0.0, float(corr[i, j]))
            weights[i, j] = value
            weights[j, i] = max(weights[j, i], value)
    np.fill_diagonal(weights, 0.0)
    return weights


def dense_to_edge_index(weights: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense weight matrix -> PyG ``(edge_index, edge_weight)``."""
    rows, cols = np.nonzero(weights)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(weights[rows, cols], dtype=torch.float32)
    return edge_index, edge_weight


def graph_stats(weights: np.ndarray) -> dict:
    """Numbers used by the tests and printed in the training log."""
    off_diagonal = weights[~np.eye(weights.shape[0], dtype=bool)]
    nonzero = off_diagonal[off_diagonal > 0]
    return {
        "nodes": int(weights.shape[0]),
        "edges": int((off_diagonal > 0).sum()),
        "mean_edge_weight": float(nonzero.mean()) if nonzero.size else 0.0,
        "max_edge_weight": float(nonzero.max()) if nonzero.size else 0.0,
        "min_edge_weight": float(nonzero.min()) if nonzero.size else 0.0,
        "mean_degree": float((off_diagonal > 0).sum() / weights.shape[0]),
    }


def assert_graph_contributes(weights: np.ndarray, min_mean_weight: float = 0.05) -> None:
    """Fail loudly if edge weights are so small the self-loop would dominate.

    This is the regression guard for the bug described in the module docstring. With
    GCN's self-loop weight of 1.0, a mean neighbour weight below ~0.05 means messages
    from neighbours are noise next to a node's own features.
    """
    stats = graph_stats(weights)
    if stats["edges"] == 0:
        raise ValueError("adjacency has no edges: the graph cannot contribute anything")
    if stats["mean_edge_weight"] < min_mean_weight:
        raise ValueError(
            f"mean edge weight {stats['mean_edge_weight']:.4f} < {min_mean_weight}: "
            "GCN self-loops (weight 1.0) would dominate aggregation. Rescale the "
            "adjacency with gaussian_kernel_adjacency() instead of raw 1/travel_time."
        )


if __name__ == "__main__":
    adjacency = chennai_adjacency()
    stats = graph_stats(adjacency)
    print("Chennai segment graph")
    print(f"  nodes {stats['nodes']}  edges {stats['edges']}  mean degree {stats['mean_degree']:.1f}")
    print(
        f"  edge weight: min {stats['min_edge_weight']:.3f} "
        f"mean {stats['mean_edge_weight']:.3f} max {stats['max_edge_weight']:.3f}"
    )
    assert_graph_contributes(adjacency)
    print("  self-loop domination check: PASS")

    print("\n  adjacency (rows = segments):")
    print("        " + " ".join(f"{s:>5}" for s in SEGMENT_IDS))
    for sid, row in zip(SEGMENT_IDS, adjacency):
        print(f"  {sid:>4}  " + " ".join(f"{v:5.2f}" for v in row))

    # Show what the old 1/travel-time weighting would have produced.
    from config.segments import segment_lengths_km

    lengths = np.array(segment_lengths_km())
    naive = np.zeros_like(adjacency)
    distances = chennai_distance_matrix()
    for i in range(len(SEGMENTS)):
        for j in range(len(SEGMENTS)):
            if i != j and np.isfinite(distances[i, j]):
                naive[i, j] = 1.0 / (lengths[i] / 30.0 * 3600.0)  # seconds
    naive_stats = graph_stats(naive)
    print(
        f"\n  for contrast, 1/travel_time weighting gives mean edge weight "
        f"{naive_stats['mean_edge_weight']:.4f} -- swamped by the 1.0 self-loop"
    )
