"""NIR: a network exported by snnTorch runs the same in sparx, and comes back out unchanged."""

from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.nir import from_nir, to_nir
from sparx.nn import LIF

nir = pytest.importorskip("nir")
FIXTURES = Path(__file__).parent / "fixtures"


def test_a_network_snntorch_exported_runs_spike_for_spike():
    # snnTorch reads and writes NIR with forward Euler at dt = 1e-4 s.
    graph = nir.read(FIXTURES / "snntorch.nir")
    data = np.load(FIXTURES / "nir.npz")
    model, variables = from_nir(graph, dt=1e-4, discretization="euler")
    spikes = model.apply(variables, jnp.asarray(data["inputs"]))
    np.testing.assert_array_equal(np.asarray(spikes), data["spikes"])
    assert data["spikes"].sum() > 30


def test_the_graph_exports_back_unchanged():
    graph = nir.read(FIXTURES / "snntorch.nir")
    model, variables = from_nir(graph, dt=1e-4, discretization="euler")
    again = to_nir(model, variables, dt=1e-4, discretization="euler")
    for name, node in graph.nodes.items():
        if isinstance(node, (nir.Input, nir.Output)):
            continue
        other = again.nodes[name]
        assert type(other) is type(node)
        for field in ("weight", "bias", "tau", "v_threshold", "v_leak", "v_reset"):
            if hasattr(node, field):
                np.testing.assert_allclose(getattr(other, field), getattr(node, field), rtol=1e-6)
        if isinstance(node, nir.LIF):
            # The input scale r is folded into the preceding layer on import,
            # so it comes back as the Euler convention's tau / dt, and the
            # weights carry snnTorch's r * dt / tau, which it writes as 1.
            np.testing.assert_allclose(other.r, node.tau / 1e-4, rtol=1e-6)
    assert sorted(again.edges) == sorted(graph.edges)


def test_sparx_round_trips_through_nir_exactly():
    model = nn.Sequential([nn.Dense(7), LIF(tau=3.0, reset="zero"), nn.Dense(2), LIF(tau=5.0, reset="zero")])
    x = jnp.asarray(np.random.default_rng(0).random((20, 3, 4)), jnp.float32)
    variables = model.init(jax.random.key(0), x)
    variables = jax.tree.map(lambda w: 3 * w, variables)
    graph = to_nir(model, variables, dt=1e-3)
    back, back_variables = from_nir(graph, dt=1e-3)
    np.testing.assert_array_equal(np.asarray(back.apply(back_variables, x)),
                                  np.asarray(model.apply(variables, x)))


def test_unsupported_layers_are_refused():
    model = nn.Sequential([nn.Dense(4), LIF(tau=2.0)])  # a soft reset NIR cannot state
    variables = model.init(jax.random.key(0), jnp.zeros((2, 1, 3)))
    with pytest.raises(NotImplementedError, match="reset"):
        to_nir(model, variables, dt=1e-3)
