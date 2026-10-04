"""Parity with SpikingJelly (commit c6cb8e46, torch 2.14.1+cpu).

`tools/make_reference_fixtures.py` ran SpikingJelly's own modules on fixed
inputs and saved their spikes and gradients. Each case's membranes stay at
least 1e-4 from threshold, so spikes are compared exactly. Gradients are
float32 sums taken in a different order by the two frameworks; the largest
differences observed, on CPU at float32, are noted at each tolerance.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.cells import LIFCell, run
from sparx.nn import PSN, MaskedPSN, SlidingPSN
from sparx.surrogate import ATan

FIXTURES = np.load(Path(__file__).parent / "fixtures" / "spikingjelly.npz")


def case(name):
    prefix = f"{name}/"
    return {key[len(prefix):]: FIXTURES[key] for key in FIXTURES.files if key.startswith(prefix)}


@pytest.mark.parametrize(("name", "reset", "detach"), [
    ("lif_soft", "subtract", False),
    ("lif_hard", "zero", False),
    ("lif_soft_detached", "subtract", True),
])
def test_lif_matches_spikingjelly_lifnode_without_input_decay(name, reset, detach):
    # SpikingJelly's LIFNode(decay_input=False) charges v + (v_reset - v) / tau + x
    # with v_reset 0, which is LIFCell with decay 1 - 1 / tau.
    c = case(name)
    cell = LIFCell(1 - 1 / float(c["tau"]), 1.0, reset, ATan(2.0), detach)

    def weighted(x):
        spikes, state = run(cell, x)
        return jnp.sum(spikes * c["weights"]), (spikes, state)

    grad_x, (spikes, state) = jax.grad(weighted, has_aux=True)(jnp.asarray(c["x"]))
    np.testing.assert_array_equal(spikes, c["spikes"])
    np.testing.assert_allclose(state.v, c["v"], rtol=1e-6, atol=1e-6)  # observed 6.0e-8
    np.testing.assert_allclose(grad_x, c["grad_x"], rtol=1e-5, atol=1e-6)  # observed 3.6e-7


def _psn_parity(name, layer, **call):
    c = case(name)
    params = {"params": {"weight": jnp.asarray(c["weight"]), "bias": jnp.asarray(c["bias"])}}

    def weighted(params, x):
        spikes = layer.apply(params, x, **call)
        return jnp.sum(spikes * c["weights"]), spikes

    (grads, grad_x), spikes = jax.grad(weighted, argnums=(0, 1), has_aux=True)(
        params, jnp.asarray(c["x"]))
    np.testing.assert_array_equal(spikes, c["spikes"])
    # Observed at most 1.2e-6 on any gradient (SlidingPSN's bias, which sums every position).
    np.testing.assert_allclose(grad_x, c["grad_x"], rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(grads["params"]["weight"], c["grad_weight"], rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(grads["params"]["bias"], c["grad_bias"], rtol=1e-5, atol=2e-6)


def test_psn_matches_spikingjelly():
    _psn_parity("psn", PSN(precision=jax.lax.Precision.HIGHEST))


@pytest.mark.parametrize("masking", [0.4, 1.0])
def test_masked_psn_matches_spikingjelly(masking):
    _psn_parity(f"masked_psn_{masking}", MaskedPSN(k=3, precision=jax.lax.Precision.HIGHEST),
                masking=masking)


@pytest.mark.parametrize("init", ["exp", "kaiming"])
def test_sliding_psn_matches_spikingjelly(init):
    _psn_parity(f"sliding_psn_{init}", SlidingPSN(k=3, precision=jax.lax.Precision.HIGHEST))


def test_psn_initialization_follows_spikingjelly():
    x = jnp.ones((16, 2, 3))
    params = PSN().init(jax.random.key(0), x)["params"]
    weight = np.asarray(params["weight"])
    # kaiming_uniform_(a=sqrt(5)) on [16, 16]: uniform within 1 / sqrt(16).
    assert np.abs(weight).max() <= 0.25 and np.abs(weight).max() > 0.2
    np.testing.assert_array_equal(params["bias"], -np.ones(16))
    sliding = SlidingPSN(k=4).init(jax.random.key(0), x)["params"]
    np.testing.assert_array_equal(sliding["weight"], [0.125, 0.25, 0.5, 1.0])
    assert float(sliding["bias"]) == -1.0
    kaiming = SlidingPSN(k=4, exponential_init=False).init(jax.random.key(0), x)["params"]["weight"]
    assert np.abs(kaiming).max() <= 0.5 and len(set(np.asarray(kaiming).tolist())) == 4
