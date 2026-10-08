"""NIR: networks exported by snnTorch run the same in sparx, and come back out unchanged."""

from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.dynamics import LIFCell, decay
from sparx.nir import from_nir, to_nir
from sparx.nn import LI, LIF, Flatten, Neuron, Recurrent

nir = pytest.importorskip("nir")
FIXTURES = Path(__file__).parent / "fixtures"
# snnTorch reads and writes NIR with forward Euler at dt = 1e-4 s.
SNNTORCH = {"dt": 1e-4, "discretization": "euler"}


def nhwc(x):
    """A PyTorch tensor `[T, B, C, H, W]` in flax's layout, `[T, B, H, W, C]`."""
    return np.moveaxis(x, 2, -1)


def assert_exports_back(graph, again):
    """Every node of `again` that `graph` names has `graph`'s type and parameters."""
    for name, node in graph.nodes.items():
        other = again.nodes[name]
        assert type(other) is type(node)
        for field in ("weight", "bias", "tau", "v_threshold", "v_leak", "v_reset",
                      "stride", "padding", "dilation", "groups", "start_dim", "end_dim"):
            if hasattr(node, field):
                # Observed 9.6e-8 relative.
                np.testing.assert_allclose(np.asarray(getattr(other, field), float),
                                           np.asarray(getattr(node, field), float), rtol=1e-6)
        for port in ("input_type", "output_type"):
            for key, shape in getattr(node, port).items():
                np.testing.assert_array_equal(getattr(other, port)[key], shape)
        if isinstance(node, nir.LIF):
            # The input scale r is folded into the preceding layer on import,
            # so it comes back as the Euler convention's tau / dt, and the
            # weights carry snnTorch's r * dt / tau, which it writes as 1.
            np.testing.assert_allclose(other.r, node.tau / 1e-4, rtol=1e-6)  # observed 4.8e-8 relative
    assert sorted(again.edges) == sorted(graph.edges)


def test_a_network_snntorch_exported_runs_spike_for_spike():
    graph = nir.read(FIXTURES / "snntorch.nir")
    data = np.load(FIXTURES / "nir.npz")
    model, variables = from_nir(graph, **SNNTORCH)
    spikes = model.apply(variables, jnp.asarray(data["inputs"]))
    np.testing.assert_array_equal(np.asarray(spikes), data["spikes"])
    assert data["spikes"].sum() > 30


def test_the_graph_exports_back_unchanged():
    graph = nir.read(FIXTURES / "snntorch.nir")
    model, variables = from_nir(graph, **SNNTORCH)
    assert_exports_back(graph, to_nir(model, variables, **SNNTORCH))


def test_a_conv_network_snntorch_exported_runs_spike_for_spike():
    graph = nir.read(FIXTURES / "snntorch_conv.nir")
    data = np.load(FIXTURES / "nir_conv.npz")
    model, variables = from_nir(graph, **SNNTORCH)
    x = jnp.asarray(nhwc(data["inputs"]))
    # The first convolution's spikes, [T, B, H, W, C] here and NCHW in snnTorch.
    head = nn.Sequential(model.layers[:2])
    hidden = head.apply({"params": {"layers_0": variables["params"]["layers_0"]}}, x)
    np.testing.assert_array_equal(np.asarray(hidden), nhwc(data["hidden"]))
    np.testing.assert_array_equal(np.asarray(model.apply(variables, x)), data["spikes"])
    assert data["hidden"].sum() > 500 and data["spikes"].sum() > 30


def test_the_conv_network_needs_pytorchs_flatten_order():
    # With flax's own H, W, C order, the dense layer after the flatten reads
    # the wrong features, so the fixture tells the two orders apart.
    graph = nir.read(FIXTURES / "snntorch_conv.nir")
    data = np.load(FIXTURES / "nir_conv.npz")
    model, variables = from_nir(graph, **SNNTORCH)
    def hwc(x):
        return x.reshape(*x.shape[:2], -1)

    layers = [hwc if isinstance(layer, Flatten) else layer for layer in model.layers]
    spikes = nn.Sequential(layers).apply(variables, jnp.asarray(nhwc(data["inputs"])))
    assert not np.array_equal(np.asarray(spikes), data["spikes"])


def test_the_conv_graph_exports_back_unchanged():
    graph = nir.read(FIXTURES / "snntorch_conv.nir")
    model, variables = from_nir(graph, **SNNTORCH)
    shape = graph.nodes["input"].input_type["input"]
    again = to_nir(model, variables, **SNNTORCH, input_shape=(*shape[1:], shape[0]))
    assert_exports_back(graph, again)


def test_a_recurrent_network_snntorch_exported_runs_spike_for_spike():
    graph = nir.read(FIXTURES / "snntorch_rleaky.nir")
    data = np.load(FIXTURES / "nir_rleaky.npz")
    model, variables = from_nir(graph, **SNNTORCH)
    assert isinstance(model.layers[1], Recurrent)
    x = jnp.asarray(data["inputs"])
    head = nn.Sequential(model.layers[:2])
    params = variables["params"]
    hidden = head.apply({"params": {k: params[k] for k in ("layers_0", "layers_1")}}, x)
    np.testing.assert_array_equal(np.asarray(hidden), data["hidden"])
    np.testing.assert_array_equal(np.asarray(model.apply(variables, x)), data["spikes"])
    assert data["hidden"].sum() > 60 and data["spikes"].sum() > 30


def test_the_recurrent_graph_exports_back_with_its_feedback_bias_moved():
    graph = nir.read(FIXTURES / "snntorch_rleaky.nir")
    model, variables = from_nir(graph, **SNNTORCH)
    again = to_nir(model, variables, **SNNTORCH)
    w_rec = graph.nodes["1.w_rec"]
    # sparx's feedback has no bias; snnTorch's constant feedback bias joins
    # the input layer's bias, and the feedback comes back as a Linear node.
    assert isinstance(w_rec, nir.Affine) and np.any(w_rec.bias != 0)
    assert isinstance(again.nodes["1.w_rec"], nir.Linear)
    np.testing.assert_allclose(again.nodes["1.w_rec"].weight, w_rec.weight, rtol=1e-6)  # observed 0 relative
    # Observed 0 relative.
    np.testing.assert_allclose(again.nodes["0"].bias, graph.nodes["0"].bias + w_rec.bias, rtol=1e-6)
    rest = {name: node for name, node in graph.nodes.items() if name not in ("0", "1.w_rec")}
    assert_exports_back(nir.NIRGraph(nodes=rest, edges=graph.edges, type_check=False), again)


def test_a_nested_recurrent_subgraph_imports_like_the_flat_one():
    graph = nir.read(FIXTURES / "snntorch_rleaky.nir")
    data = np.load(FIXTURES / "nir_rleaky.npz")
    size = graph.nodes["1.lif"].tau.shape
    inner = nir.NIRGraph(
        nodes={"input": nir.Input(input_type={"input": np.array(size)}), "lif": graph.nodes["1.lif"],
               "w_rec": graph.nodes["1.w_rec"], "output": nir.Output(output_type={"output": np.array(size)})},
        edges=[("input", "lif"), ("lif", "w_rec"), ("w_rec", "lif"), ("lif", "output")])
    nodes = {name: node for name, node in graph.nodes.items() if not name.startswith("1.")}
    nested = nir.NIRGraph(nodes={**nodes, "1": inner},
                          edges=[("input", "0"), ("0", "1"), ("1", "2"), ("2", "3"), ("3", "output")])
    model, variables = from_nir(nested, **SNNTORCH)
    np.testing.assert_array_equal(np.asarray(model.apply(variables, jnp.asarray(data["inputs"]))),
                                  data["spikes"])


def test_a_conv_node_computes_pytorchs_convolution():
    # NIR 1.0.8 types a Conv2d by its kernel's height and as if ungrouped, so
    # this graph, a non-square grouped kernel, is built without its check.
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(6, 2, 3, 2)).astype(np.float32)  # [C_out, C_in / groups, kH, kW]
    bias = rng.normal(size=6).astype(np.float32)
    conv = nir.Conv2d(input_shape=None, weight=weight, stride=(2, 1), padding=(1, 0), dilation=(1, 2),
                      groups=2, bias=bias)
    graph = nir.NIRGraph(nodes={"input": nir.Input(input_type={"input": np.array([4, 7, 6])}), "conv": conv,
                                "output": nir.Output(output_type={"output": np.array([6, 4, 4])})},
                         edges=[("input", "conv"), ("conv", "output")], type_check=False)
    model, variables = from_nir(graph, dt=1e-3)
    x = rng.normal(size=(3, 2, 4, 7, 6)).astype(np.float32)  # [T, B, C, H, W]
    expected = jax.lax.conv_general_dilated(
        x.reshape(6, 4, 7, 6), weight, window_strides=(2, 1), padding=[(1, 1), (0, 0)], rhs_dilation=(1, 2),
        dimension_numbers=("NCHW", "OIHW", "NCHW"), feature_group_count=2,
        precision=jax.lax.Precision.HIGHEST) + bias[:, None, None]
    out = model.apply(variables, jnp.asarray(nhwc(x)))
    np.testing.assert_allclose(np.asarray(out), nhwc(np.asarray(expected).reshape(3, 2, 6, 4, 4)),
                               rtol=1e-5, atol=1e-5)  # observed 0


def test_sparx_round_trips_through_nir_exactly():
    model = nn.Sequential([nn.Dense(7), LIF(tau=3.0, reset="zero"), nn.Dense(2), LIF(tau=5.0, reset="zero")])
    x = jnp.asarray(np.random.default_rng(0).random((20, 3, 4)), jnp.float32)
    variables = model.init(jax.random.key(0), x)
    variables = jax.tree.map(lambda w: 3 * w, variables)
    graph = to_nir(model, variables, dt=1e-3)
    back, back_variables = from_nir(graph, dt=1e-3)
    np.testing.assert_array_equal(np.asarray(back.apply(back_variables, x)),
                                  np.asarray(model.apply(variables, x)))


@pytest.mark.parametrize("discretization", ["exact", "euler"])
def test_a_classifier_with_a_leaky_readout_round_trips(discretization):
    # NIR's LI is the LIF without a threshold; the readout's membrane comes back exactly.
    model = nn.Sequential([nn.Dense(7), LIF(tau=3.0, reset="zero"), nn.Dense(2), LI(tau=5.0)])
    x = jnp.asarray(np.random.default_rng(0).random((20, 3, 4)), jnp.float32)
    variables = jax.tree.map(lambda w: 3 * w, model.init(jax.random.key(0), x))
    graph = to_nir(model, variables, dt=1e-3, discretization=discretization)
    assert type(graph.nodes["3"]).__name__ == "LI"
    back, back_variables = from_nir(graph, dt=1e-3, discretization=discretization)
    assert isinstance(back.layers[3], LI)
    expected = np.asarray(model.apply(variables, x))
    np.testing.assert_array_equal(np.asarray(back.apply(back_variables, x)), expected)
    assert np.abs(expected).max() > 1


def test_a_sparx_conv_and_recurrent_stack_round_trips_exactly(tmp_path):
    model = nn.Sequential([
        nn.Conv(4, (3, 3), strides=(2, 1), padding="SAME"), LIF(tau=3.0, reset="zero"),
        nn.Conv(3, (3, 3), padding=((0, 0), (2, 2)), kernel_dilation=(1, 2)), LIF(tau=4.0, reset="zero"),
        Flatten(), nn.Dense(6), Recurrent(LIF(tau=5.0, reset="zero")),
        nn.Dense(4), LIF(tau=4.0, reset="zero"),
    ])
    x = jnp.asarray(np.random.default_rng(0).random((20, 3, 9, 6, 2)), jnp.float32)
    variables = jax.tree.map(lambda w: 3 * w, model.init(jax.random.key(0), x))
    graph = to_nir(model, variables, dt=1e-3, input_shape=(9, 6, 2))
    assert [type(graph.nodes[k]).__name__ for k in ("0", "4", "6.lif", "6.w_rec")] == [
        "Conv2d", "Flatten", "LIF", "Linear"]
    np.testing.assert_array_equal(graph.nodes["input"].input_type["input"], [2, 9, 6])
    np.testing.assert_array_equal(graph.nodes["1"].tau.shape, [4, 5, 6])  # [C, H, W]
    nir.write(tmp_path / "sparx.nir", graph)  # reading checks the graph's types
    back, back_variables = from_nir(nir.read(tmp_path / "sparx.nir"), dt=1e-3)
    expected = np.asarray(model.apply(variables, x))
    np.testing.assert_array_equal(np.asarray(back.apply(back_variables, x)), expected)
    assert expected.sum() > 30


class HardLIF(Neuron):
    """A layer of one's own, not sparx's LIF, whose model is a hard-reset `LIFCell`."""

    def build(self, x):
        return LIFCell(decay(4.0), threshold=1.0, reset="zero")


def test_a_layer_exports_as_the_model_it_builds():
    # NIR export reads a layer's model (`sparx.nn.Modelled`), not its class, so a layer of one's own
    # that builds a LIFCell exports as LIF(tau=4.0) does.
    x = jnp.ones((2, 1, 3))
    graphs = []
    for neuron in (HardLIF(), LIF(tau=4.0, reset="zero")):
        model = nn.Sequential([nn.Dense(2), neuron])
        graphs.append(to_nir(model, model.init(jax.random.key(0), x), dt=1e-3))
    ours, theirs = (graph.nodes["1"] for graph in graphs)
    assert type(ours) is type(theirs)
    for field in ("tau", "r", "v_threshold", "v_reset"):
        np.testing.assert_array_equal(getattr(ours, field), getattr(theirs, field))


def test_unsupported_layers_are_refused():
    model = nn.Sequential([nn.Dense(4), LIF(tau=2.0)])  # a soft reset NIR cannot state
    variables = model.init(jax.random.key(0), jnp.zeros((2, 1, 3)))
    with pytest.raises(NotImplementedError, match="reset"):
        to_nir(model, variables, dt=1e-3)


@pytest.mark.parametrize(("layers", "match"), [
    ([nn.Conv(2, (3, 3)), Flatten(ndim=1), nn.Dense(2)], "whole shape"),
    ([nn.Conv(2, (2, 2), padding="SAME")], "both sides"),
    ([nn.Conv(2, (3, 3)), nn.Dense(2)], "flat inputs"),
])
def test_layouts_nir_cannot_state_are_refused(layers, match):
    model = nn.Sequential(layers)
    variables = model.init(jax.random.key(0), jnp.zeros((2, 1, 6, 6, 1)))
    with pytest.raises(NotImplementedError, match=match):
        to_nir(model, variables, dt=1e-3, input_shape=(6, 6, 1))
