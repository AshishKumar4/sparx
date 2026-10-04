from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.losses import per_step_cross_entropy, rate_mse
from sparx.nn import LI, LIF, RATES
from sparx.rates import firing_rates, rate_penalty

SNNTORCH = np.load(Path(__file__).parent / "fixtures" / "snntorch.npz")


def test_per_step_cross_entropy_matches_snntorch_ce_rate_loss():
    out = per_step_cross_entropy(jnp.asarray(SNNTORCH["loss_outputs"]), jnp.asarray(SNNTORCH["loss_labels"]))
    np.testing.assert_allclose(out, SNNTORCH["ce_rate"], rtol=1e-6)


def test_rate_mse_is_snntorch_mse_count_loss_over_steps():
    spikes = SNNTORCH["loss_spikes"]
    out = rate_mse(jnp.asarray(spikes), jnp.asarray(SNNTORCH["loss_labels"]))
    steps = spikes.shape[0]
    np.testing.assert_allclose(out, SNNTORCH["mse_count"].mean(-1) / steps, rtol=1e-6)


def test_rate_mse_is_zero_at_the_target_rates():
    labels = jnp.asarray([2, 0])
    spikes = jnp.zeros((10, 2, 3)).at[:8, 0, 2].set(1).at[:2, 0, :2].set(1)
    spikes = spikes.at[:8, 1, 0].set(1).at[:2, 1, 1:].set(1)
    np.testing.assert_allclose(rate_mse(spikes, labels), [0, 0], atol=1e-7)


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = LIF(tau=3.0)(nn.Dense(8)(x))
        x = LIF(tau=3.0)(nn.Dense(5)(x))
        return LI()(nn.Dense(2)(x))


def _sown(seed=0):
    x = jnp.asarray(np.random.default_rng(seed).normal(0.5, 1.0, (12, 6, 4)), jnp.float32)
    net = Net()
    params = net.init(jax.random.key(0), x)
    return net, params, x


def test_firing_rates_name_every_spiking_layer_with_its_mean_rate():
    net, params, x = _sown()
    _, sown = net.apply(params, x, mutable=[RATES])
    rates = firing_rates(sown)
    assert set(rates) == {"LIF_0", "LIF_1"}
    (first,) = sown[RATES]["LIF_0"]["rate"]
    np.testing.assert_allclose(rates["LIF_0"], first.mean())


def test_a_layer_called_twice_reports_each_call():
    class Twice(nn.Module):
        @nn.compact
        def __call__(self, x):
            lif = LIF()
            return lif(lif(x))

    _, sown = Twice().apply({}, jnp.ones((4, 2, 3)), mutable=[RATES])
    assert set(firing_rates(sown)) == {"LIF_0", "LIF_0#1"}


def test_rate_penalty_is_zero_inside_the_band_and_grows_outside():
    sown = {"spike_rates": {"a": {"rate": (jnp.asarray([[0.2, 0.0, 0.9], [0.4, 0.0, 0.7]]),)}}}
    # Per-neuron batch means are 0.3, 0.0 and 0.8.
    assert float(rate_penalty(sown, lower=0.0, upper=1.0)) == 0
    # Neuron 1 sits 0.1 under the band and neuron 2 0.2 over it.
    np.testing.assert_allclose(rate_penalty(sown, lower=0.1, upper=0.6), (0.1**2 + 0.2**2) / 3, rtol=1e-6)


def test_rate_penalty_trains_silent_neurons_to_fire():
    net, params, x = _sown(1)

    def penalty(params):
        _, sown = net.apply(params, x * 0.0, mutable=[RATES])  # no input current: silence
        return rate_penalty(sown, lower=0.2)

    grads = jax.grad(penalty)(params)["params"]
    # Raising the biases raises every rate, so descent must raise them.
    assert np.all(np.asarray(grads["Dense_0"]["bias"]) < 0)


def test_rate_penalty_refuses_an_empty_collection():
    with pytest.raises(ValueError, match="no firing rates"):
        rate_penalty({})
