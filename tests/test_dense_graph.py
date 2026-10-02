"""The pure-PyTorch graph layers must be numerical twins of the PyG ones.

"Twin" is a claim, so it is tested: same adjacency normalisation, same layer outputs
given the same weights, same parameter names (so a PyG-trained checkpoint loads into the
dense backend), and the dense backend must work with torch_geometric absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from config.segments import (
    NUM_SATELLITE_FEATURES,
    NUM_TRAFFIC_FEATURES,
    NUM_WEATHER_FEATURES,
    REPO_ROOT,
)
from ml.data.graph import chennai_adjacency, dense_to_edge_index
from ml.models.dense_graph import (
    DenseGATConv,
    DenseGCNConv,
    dense_adjacency,
    gcn_normalise,
)
from ml.models.graph_layers import SpatialBlock
from ml.models.monsoonplus_net import build_model, load_checkpoint

pyg = pytest.importorskip("torch_geometric", reason="equivalence tests compare against PyG")
from torch_geometric.nn import GATConv, GCNConv  # noqa: E402
from torch_geometric.nn.conv.gcn_conv import gcn_norm  # noqa: E402

CHECKPOINT = REPO_ROOT / "ml" / "checkpoints" / "monsoonplus_final.pt"
N, HIDDEN, HEADS = 10, 64, 4


@pytest.fixture(scope="module")
def graph():
    edge_index, edge_weight = dense_to_edge_index(chennai_adjacency())
    return edge_index, edge_weight


def _features(batch: int = 3, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, N, HIDDEN, generator=generator)


def test_dense_adjacency_scatters_edges_as_target_by_source(graph):
    edge_index, edge_weight = graph
    adj = dense_adjacency(edge_index, edge_weight, N)
    for k in range(edge_index.size(1)):
        source, target = int(edge_index[0, k]), int(edge_index[1, k])
        assert adj[target, source].item() == pytest.approx(float(edge_weight[k]))


def test_gcn_normalisation_matches_pyg(graph):
    edge_index, edge_weight = graph
    pyg_index, pyg_weight = gcn_norm(edge_index, edge_weight, N, add_self_loops=True)
    expected = torch.zeros(N, N)
    expected[pyg_index[1], pyg_index[0]] = pyg_weight

    got = gcn_normalise(dense_adjacency(edge_index, edge_weight, N))
    assert torch.allclose(got, expected, atol=1e-6)


def test_dense_gcn_matches_gcnconv(graph):
    edge_index, edge_weight = graph
    reference = GCNConv(HIDDEN, HIDDEN, add_self_loops=True).eval()
    twin = DenseGCNConv(HIDDEN, HIDDEN).eval()
    # identical parameter names and shapes means a plain state_dict copy must work
    twin.load_state_dict(reference.state_dict(), strict=True)
    torch.nn.init.normal_(reference.bias, std=0.3)  # a non-zero bias exercises that path
    twin.load_state_dict(reference.state_dict(), strict=True)

    x = _features()
    adj = dense_adjacency(edge_index, edge_weight, N)
    got = twin(x, adj)
    expected = torch.stack([reference(sample, edge_index, edge_weight=edge_weight) for sample in x])
    assert torch.allclose(got, expected, atol=1e-5)


def test_dense_gat_matches_gatconv(graph):
    edge_index, _ = graph
    reference = GATConv(HIDDEN, HIDDEN // HEADS, heads=HEADS, dropout=0.0).eval()
    twin = DenseGATConv(HIDDEN, HIDDEN // HEADS, heads=HEADS, dropout=0.0).eval()
    torch.nn.init.normal_(reference.bias, std=0.3)
    twin.load_state_dict(reference.state_dict(), strict=True)

    x = _features(seed=1)
    adj = dense_adjacency(edge_index, None, N)
    got = twin(x, adj)
    expected = torch.stack([reference(sample, edge_index) for sample in x])
    assert torch.allclose(got, expected, atol=1e-5)


@pytest.mark.parametrize("family", ["gcn", "gat"])
def test_spatial_block_dense_twin_matches_pyg(family, graph):
    edge_index, edge_weight = graph
    pyg_block = SpatialBlock(HIDDEN, graph_layer=family, heads=HEADS).eval()
    dense_block = SpatialBlock(HIDDEN, graph_layer=f"dense_{family}", heads=HEADS).eval()
    dense_block.load_state_dict(pyg_block.state_dict(), strict=True)

    x = _features(batch=4, seed=2)
    with torch.no_grad():
        expected = pyg_block(x, edge_index, edge_weight)
        got = dense_block(x, edge_index, edge_weight)
    assert torch.allclose(got, expected, atol=1e-5)


def _batch(n_batch: int = 2):
    generator = torch.Generator().manual_seed(3)
    traffic = torch.rand(n_batch, N, 24, NUM_TRAFFIC_FEATURES, generator=generator) * 30 + 5
    weather = torch.rand(n_batch, N, 24, NUM_WEATHER_FEATURES, generator=generator) * 10
    satellite = torch.randn(n_batch, N, NUM_SATELLITE_FEATURES, generator=generator)
    return traffic, weather, satellite


@pytest.mark.parametrize("family", ["gcn", "gat"])
def test_whole_model_state_dict_transfers_to_dense_twin(family, graph):
    edge_index, edge_weight = graph
    torch.manual_seed(7)
    pyg_model = build_model(graph_layer=family).eval()
    dense_model = build_model(graph_layer=f"dense_{family}").eval()
    dense_model.load_state_dict(pyg_model.state_dict(), strict=True)

    inputs = _batch()
    with torch.no_grad():
        expected = pyg_model(*inputs, edge_index, edge_weight)
        got = dense_model(*inputs, edge_index, edge_weight)
    assert torch.allclose(got.prediction, expected.prediction, atol=1e-4)
    assert torch.allclose(got.gate, expected.gate, atol=1e-6)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="no trained checkpoint in ml/checkpoints")
def test_shipped_checkpoint_runs_on_the_dense_backend(graph):
    """The headline use: the PyG-trained checkpoint, run without torch_geometric layers."""
    edge_index, edge_weight = graph
    original, _ = load_checkpoint(CHECKPOINT)
    twin, _ = load_checkpoint(CHECKPOINT, graph_layer="dense_gcn")
    assert twin.config.graph_layer == "dense_gcn"

    inputs = _batch()
    with torch.no_grad():
        expected = original(*inputs, edge_index, edge_weight)
        got = twin(*inputs, edge_index, edge_weight)
    assert torch.allclose(got.prediction, expected.prediction, atol=1e-4)


def test_swapping_gcn_for_gat_at_load_time_fails_loudly(tmp_path):
    model = build_model(graph_layer="gcn")
    path = tmp_path / "gcn.pt"
    torch.save({"config": model.config.to_dict(), "state_dict": model.state_dict()}, path)
    with pytest.raises(RuntimeError):
        load_checkpoint(path, graph_layer="gat")


def test_dense_backend_trains(graph):
    """Gradients reach every parameter, including the attention vectors."""
    edge_index, edge_weight = graph
    for family in ("dense_gcn", "dense_gat"):
        model = build_model(graph_layer=family).train()
        out = model(*_batch(), edge_index, edge_weight)
        out.prediction.sum().backward()
        graph_params = {n: p for n, p in model.named_parameters() if n.startswith("spatial.conv")}
        assert graph_params
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in graph_params.values()), family


def test_dense_backend_works_without_torch_geometric(monkeypatch, graph):
    """Importing the dense variants must not touch torch_geometric at all."""
    edge_index, edge_weight = graph
    # a None entry in sys.modules makes `import torch_geometric...` raise ImportError
    for name in [m for m in sys.modules if m == "torch_geometric" or m.startswith("torch_geometric.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torch_geometric", None)
    monkeypatch.setitem(sys.modules, "torch_geometric.nn", None)

    model = build_model(graph_layer="dense_gcn").eval()
    with torch.no_grad():
        out = model(*_batch(), edge_index, edge_weight)
    assert torch.isfinite(out.prediction).all()

    with pytest.raises(ImportError, match="dense_gcn"):
        build_model(graph_layer="gcn")


def test_modality_masked_equals_manual_availability_mask(graph):
    edge_index, edge_weight = graph
    torch.manual_seed(11)
    model = build_model().eval()
    inputs = _batch()
    with torch.no_grad():
        masked = model.modality_masked(*inputs, edge_index, edge_weight, drop=("satellite",))
        manual = model(*inputs, edge_index, edge_weight, availability=torch.tensor([1.0, 1.0, 0.0]))
        full = model(*inputs, edge_index, edge_weight)
    assert torch.allclose(masked.prediction, manual.prediction)
    assert masked.gate[..., 2].abs().max().item() == 0.0  # dropped modality gets no weight
    assert not torch.allclose(masked.prediction, full.prediction)


def test_modality_masked_never_drops_traffic(graph):
    edge_index, edge_weight = graph
    model = build_model().eval()
    with pytest.raises(ValueError, match="traffic"):
        model.modality_masked(*_batch(), edge_index, edge_weight, drop=("traffic",))


def test_repo_root_is_importable():
    assert Path(REPO_ROOT).exists()
