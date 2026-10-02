"""Model shape, graph maths, gate behaviour and the transfer-learning guarantee."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config.segments import (
    HISTORY_STEPS,
    HORIZONS_MINUTES,
    NUM_SATELLITE_FEATURES,
    NUM_TRAFFIC_FEATURES,
    NUM_WEATHER_FEATURES,
    SEGMENTS,
)
from ml.data.graph import (
    assert_graph_contributes,
    chennai_adjacency,
    dense_to_edge_index,
    gaussian_kernel_adjacency,
    graph_stats,
)
from ml.models.fusion import GatedFusion, gate_summary
from ml.models.monsoonplus_net import ModelConfig, MonsoonPlusNet, build_model


@pytest.fixture(scope="module")
def graph():
    return dense_to_edge_index(chennai_adjacency())


def make_batch(batch=3, nodes=len(SEGMENTS)):
    torch.manual_seed(0)
    return (
        torch.rand(batch, nodes, HISTORY_STEPS, NUM_TRAFFIC_FEATURES) * 30 + 5,
        torch.rand(batch, nodes, HISTORY_STEPS, NUM_WEATHER_FEATURES) * 20,
        torch.rand(batch, nodes, NUM_SATELLITE_FEATURES),
    )


# ------------------------------------------------------------------ graph --


def test_chennai_adjacency_is_symmetric_and_hollow():
    adjacency = chennai_adjacency()
    assert np.allclose(adjacency, adjacency.T), "a road graph must be undirected"
    assert np.allclose(np.diag(adjacency), 0.0), "GCNConv adds its own self-loops"


def test_graph_edge_weights_are_not_swamped_by_self_loops():
    """The regression guard for the real bug: tiny edge weights make the GNN a no-op.

    GCNConv's self-loop has weight 1.0. If neighbour weights average ~0.003 (which is
    what 1/travel_time_seconds produces) the graph contributes nothing at all.
    """
    adjacency = chennai_adjacency()
    stats = graph_stats(adjacency)
    assert stats["edges"] > 0
    assert stats["mean_edge_weight"] > 0.05, stats
    assert_graph_contributes(adjacency)  # must not raise


def test_naive_inverse_travel_time_weighting_is_rejected():
    """The guard has to actually fire on the bad weighting, or it is decoration."""
    bad = np.full((10, 10), 0.004)
    np.fill_diagonal(bad, 0.0)
    with pytest.raises(ValueError, match="self-loops"):
        assert_graph_contributes(bad)


def test_gaussian_kernel_sigma_mode_matters_on_a_small_graph():
    """std-based sigma collapses on a narrow distance spread; mean-based does not.

    This is why chennai_adjacency() passes sigma_mode='mean' while METR-LA uses 'std'.
    """
    distances = np.full((6, 6), np.inf)
    np.fill_diagonal(distances, 0.0)
    rng = np.random.default_rng(0)
    for i in range(6):
        for j in range(6):
            if i != j:
                distances[i, j] = 3.0 + rng.uniform(-0.3, 0.3)  # very narrow spread

    by_std = gaussian_kernel_adjacency(distances, sigma_mode="std")
    by_mean = gaussian_kernel_adjacency(distances, sigma_mode="mean")
    assert graph_stats(by_std)["edges"] == 0, "narrow spread should collapse the std kernel"
    assert graph_stats(by_mean)["edges"] > 0


# ------------------------------------------------------------------ model --


def test_forward_shapes(graph):
    edge_index, edge_weight = graph
    model = build_model()
    traffic, weather, satellite = make_batch()
    out = model(traffic, weather, satellite, edge_index, edge_weight)

    assert out.prediction.shape == (3, len(SEGMENTS), len(HORIZONS_MINUTES))
    assert out.gate.shape == (3, len(SEGMENTS), 3)
    assert torch.isfinite(out.prediction).all()


def test_gate_is_a_distribution(graph):
    edge_index, edge_weight = graph
    model = build_model()
    traffic, weather, satellite = make_batch()
    out = model(traffic, weather, satellite, edge_index, edge_weight)

    sums = out.gate.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert (out.gate >= 0).all()

    percentages = gate_summary(out.gate)
    assert set(percentages) == {"traffic", "weather", "satellite"}
    assert abs(sum(percentages.values()) - 100.0) < 0.5


@pytest.mark.parametrize("availability_shape", ["flat", "batch", "per_node"])
def test_availability_mask_zeroes_absent_modalities(graph, availability_shape):
    """Masking must actually remove a modality, not merely down-weight it.

    This is what makes phase-1 pretraining honest: METR-LA has no weather channel, so
    the weather branch must receive exactly zero gate weight rather than being fed
    zeros and pretended to have trained.
    """
    edge_index, edge_weight = graph
    model = build_model()
    traffic, weather, satellite = make_batch()
    batch, nodes = traffic.shape[0], traffic.shape[1]

    masks = {
        "flat": torch.tensor([1.0, 0.0, 0.0]),
        "batch": torch.tensor([[1.0, 0.0, 0.0]]).repeat(batch, 1),
        "per_node": torch.tensor([[[1.0, 0.0, 0.0]]]).repeat(batch, nodes, 1),
    }
    out = model(
        traffic, weather, satellite, edge_index, edge_weight, availability=masks[availability_shape]
    )
    assert torch.allclose(out.gate[..., 1], torch.zeros_like(out.gate[..., 1]), atol=1e-6)
    assert torch.allclose(out.gate[..., 2], torch.zeros_like(out.gate[..., 2]), atol=1e-6)
    assert torch.allclose(out.gate[..., 0], torch.ones_like(out.gate[..., 0]), atol=1e-5)


def test_absent_modalities_receive_no_gradient(graph):
    """If a masked-out encoder still got gradient, 'trimodal' would be a false claim."""
    edge_index, edge_weight = graph
    model = build_model()
    traffic, weather, satellite = make_batch()

    out = model(
        traffic,
        weather,
        satellite,
        edge_index,
        edge_weight,
        availability=torch.tensor([1.0, 0.0, 0.0]),
    )
    out.prediction.sum().backward()

    weather_grads = [p.grad for p in model.weather_encoder.parameters() if p.grad is not None]
    assert weather_grads, "expected grad tensors to exist (even if zero)"
    assert all(torch.allclose(g, torch.zeros_like(g)) for g in weather_grads)

    traffic_grads = [p.grad for p in model.traffic_encoder.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in traffic_grads), "traffic branch must still learn"


def test_gate_responds_to_rain(graph):
    """The gate must be input-dependent, or the novelty claim is architectural fiction."""
    edge_index, edge_weight = graph
    model = build_model()
    model.eval()
    traffic, weather, satellite = make_batch()

    dry = weather.clone()
    dry[..., 0] = 0.0
    wet = weather.clone()
    wet[..., 0] = 80.0

    with torch.no_grad():
        gate_dry = model(traffic, dry, satellite, edge_index, edge_weight).gate
        gate_wet = model(traffic, wet, satellite, edge_index, edge_weight).gate

    assert not torch.allclose(gate_dry, gate_wet, atol=1e-4), (
        "the gate produced identical weights for 0 mm/h and 80 mm/h -- it is not "
        "conditioning on rainfall at all"
    )


def test_prediction_is_residual_around_last_speed(graph):
    """An untrained model should sit near persistence, because the head predicts a delta."""
    edge_index, edge_weight = graph
    model = build_model()
    model.eval()
    traffic, weather, satellite = make_batch()

    with torch.no_grad():
        out = model(traffic, weather, satellite, edge_index, edge_weight)
    last_speed = traffic[:, :, -1, 0]
    difference = (out.prediction - last_speed.unsqueeze(-1)).abs().mean()
    assert difference < 15.0, "an untrained residual head should not be far from persistence"


# ------------------------------------------------------- transfer learning --


def test_checkpoint_transfers_across_node_counts(graph, tmp_path):
    """207 METR-LA sensors -> 10 Chennai segments with the SAME weights.

    Node count must never appear in a weight shape. This is the transfer-learning
    claim, and load_state_dict(strict=True) is the test: strict=False would hide a
    shape mismatch rather than fix it.
    """
    big = build_model()
    state = big.state_dict()

    small = MonsoonPlusNet(ModelConfig())
    small.load_state_dict(state, strict=True)  # must not raise

    edge_index, edge_weight = graph
    traffic, weather, satellite = make_batch(batch=2, nodes=len(SEGMENTS))
    with torch.no_grad():
        out = small(traffic, weather, satellite, edge_index, edge_weight)
    assert out.prediction.shape[1] == len(SEGMENTS)

    # And the same weights on a 207-node graph.
    big_adjacency = np.zeros((207, 207), dtype=np.float32)
    for i in range(206):
        big_adjacency[i, i + 1] = big_adjacency[i + 1, i] = 0.6
    big_edge_index, big_edge_weight = dense_to_edge_index(big_adjacency)
    traffic, weather, satellite = make_batch(batch=2, nodes=207)
    with torch.no_grad():
        out = big(traffic, weather, satellite, big_edge_index, big_edge_weight)
    assert out.prediction.shape[1] == 207


def test_gat_is_a_drop_in_swap(graph):
    """The documented one-line swap must actually produce a working model."""
    edge_index, edge_weight = graph
    traffic, weather, satellite = make_batch()
    for layer in ("gcn", "gat"):
        model = build_model(graph_layer=layer)
        out = model(traffic, weather, satellite, edge_index, edge_weight)
        assert out.prediction.shape == (3, len(SEGMENTS), len(HORIZONS_MINUTES))


def test_rejects_unknown_graph_layer():
    with pytest.raises(ValueError, match="graph_layer"):
        build_model(graph_layer="transformer")


def test_batched_graph_does_not_leak_between_samples(graph):
    """Sample 3's Velachery must not influence sample 4's. If the batch tiling is wrong,
    changing one sample's inputs would change another sample's output."""
    edge_index, edge_weight = graph
    model = build_model()
    model.eval()
    traffic, weather, satellite = make_batch(batch=4)

    with torch.no_grad():
        baseline = model(traffic, weather, satellite, edge_index, edge_weight).prediction

    perturbed = traffic.clone()
    perturbed[0] += 12.0  # only sample 0
    with torch.no_grad():
        after = model(perturbed, weather, satellite, edge_index, edge_weight).prediction

    assert not torch.allclose(baseline[0], after[0]), "sample 0 should have changed"
    assert torch.allclose(baseline[1:], after[1:], atol=1e-5), "other samples leaked"
