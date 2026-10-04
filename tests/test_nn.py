import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reference

import sparx
from sparx.cells import LIFCell, run
from sparx.nn import ALIF, IF, LI, LIF, RATES, STATE, Izhikevich, Recurrent, Synaptic

T, B, D = 24, 4, 6


def inputs(seed=0, shape=(T, B, D)):
    return jnp.asarray(np.random.default_rng(seed).normal(0.4, 1.0, shape), jnp.float32)


class Net(nn.Module):
    """Two spiking layers and a readout, the shape of a small classifier."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = LIF(tau=3.0, learn_tau=True)(x)
        x = nn.Dense(16)(x)
        x = Recurrent(ALIF(tau=5.0, tau_adapt=20.0, beta=0.2, learn_tau=True))(x)
        return LI(tau=4.0)(nn.Dense(3)(x))


def test_lif_layer_is_the_lif_cell_with_decay_from_tau():
    x = inputs()
    layer = LIF(tau=4.0, threshold=0.7, reset="zero")
    out = layer.apply(layer.init(jax.random.key(0), x), x)
    expected, _ = reference.lif(np.asarray(x, np.float64), math.exp(-1 / 4.0), 0.7, "zero")
    np.testing.assert_array_equal(out, expected)


def test_if_layer_integrates_without_leak():
    x = inputs(1) * 0.3
    out = IF().apply({}, x)
    expected, _ = reference.lif(np.asarray(x, np.float64), 1.0)
    np.testing.assert_array_equal(out, expected)


def test_learned_decays_start_at_tau_and_receive_gradients():
    x = inputs(2)
    layer = LIF(tau=3.0, learn_tau=True)
    variables = layer.init(jax.random.key(0), x)
    decay = jax.nn.sigmoid(variables["params"]["decay"])
    np.testing.assert_allclose(decay, np.full(D, math.exp(-1 / 3.0)), rtol=1e-6)
    np.testing.assert_array_equal(layer.apply(variables, x), LIF(tau=3.0).apply({}, x))
    grad = jax.grad(lambda v: jnp.sum(layer.apply(v, x)))(variables)["params"]["decay"]
    assert np.all(np.isfinite(grad)) and np.any(grad != 0)


def test_synaptic_and_alif_layers_learn_both_time_constants():
    x = inputs(3)
    for layer, names in ((Synaptic(learn_tau=True), {"decay", "synapse_decay"}),
                         (ALIF(learn_tau=True), {"decay", "adapt_decay"})):
        variables = layer.init(jax.random.key(0), x)
        assert set(variables["params"]) == names
        grads = jax.grad(lambda v, layer=layer: jnp.sum(layer.apply(v, x)))(variables)["params"]
        assert all(np.any(np.asarray(g) != 0) for g in grads.values())


def test_init_creates_parameters_only():
    variables = Net().init(jax.random.key(0), inputs())
    assert set(variables) == {"params"}


def test_streaming_in_chunks_equals_one_call():
    x = inputs(4)
    net = Net()
    params = net.init(jax.random.key(0), x)
    whole = net.apply(params, x)

    outputs, carried = [], {}
    for chunk in (x[:1], x[1:10], x[10:11], x[11:]):
        out, carried = net.apply({**params, **carried}, chunk, mutable=[STATE])
        outputs.append(out)
    np.testing.assert_allclose(jnp.concatenate(outputs), whole, rtol=1e-6, atol=1e-6)


def test_a_call_without_the_state_collection_starts_at_rest():
    x = inputs(5)
    net = Net()
    params = net.init(jax.random.key(0), x)
    _, carried = net.apply(params, x, mutable=[STATE])
    # Handing the carried state back without making it mutable must not
    # continue from it: only a mutable collection is read.
    np.testing.assert_array_equal(net.apply({**params, **carried}, x), net.apply(params, x))
    continued, _ = net.apply({**params, **carried}, x, mutable=[STATE])
    assert not np.allclose(continued, net.apply(params, x))


def test_spike_rates_are_the_time_averaged_spikes_of_each_spiking_layer():
    x = inputs(6)

    class Probe(nn.Module):
        @nn.compact
        def __call__(self, x):
            spikes = LIF(tau=3.0)(x)
            LI()(spikes)
            return spikes

    probe = Probe()
    spikes, sown = probe.apply({}, x, mutable=[RATES])
    rates = sown[RATES]
    assert set(rates) == {"LIF_0"}  # the readout fires no spikes
    (rate,) = rates["LIF_0"]["rate"]
    assert rate.shape == (B, D)
    np.testing.assert_allclose(rate, jnp.mean(spikes, axis=0), rtol=1e-6)


def test_recurrent_parameters_live_under_the_layer_wherever_it_is_built():
    x = inputs(7)
    outside = Recurrent(LIF(learn_tau=True))
    assert set(outside.init(jax.random.key(0), x)["params"]) == {"neuron", "recurrent"}
    variables = Net().init(jax.random.key(0), x)
    assert set(variables["params"]["Recurrent_0"]) == {"neuron", "recurrent"}
    assert "ALIF_0" not in variables["params"]


def test_recurrent_layer_is_the_recurrent_cell():
    x = inputs(8) * 0.5
    layer = Recurrent(LIF(tau=4.0), precision=jax.lax.Precision.HIGHEST)
    variables = layer.init(jax.random.key(1), x)
    weight = variables["params"]["recurrent"]
    expected = reference.recurrent_lif(np.asarray(x, np.float64), np.asarray(weight, np.float64),
                                       math.exp(-1 / 4.0))
    np.testing.assert_array_equal(layer.apply(variables, x), expected)


def test_izhikevich_layer_runs_with_its_defaults():
    x = jnp.full((400, 1, 2), 10.0)
    assert float(Izhikevich().apply({}, x).sum()) > 0


def test_conv_networks_run_over_time_and_batch_axes():
    x = (jax.random.uniform(jax.random.key(0), (5, 2, 8, 8, 1)) < 0.3).astype(jnp.float32)

    class ConvNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Conv(4, (3, 3))(x)
            x = nn.BatchNorm(use_running_average=False)(x)
            x = LIF()(x)
            x = nn.max_pool(x, (2, 2), (2, 2))
            return LI()(nn.Dense(10)(x.reshape(*x.shape[:2], -1)))

    variables = ConvNet().init(jax.random.key(1), x)
    out, _ = ConvNet().apply(variables, x, mutable=["batch_stats"])
    assert out.shape == (5, 2, 10)


def test_layers_compose_with_jit_vmap_and_grad():
    x = inputs(9)
    net = Net()
    params = net.init(jax.random.key(0), x)
    loss = jax.jit(lambda p, x: jnp.sum(net.apply(p, x) ** 2))
    grads = jax.grad(loss)(params, x)
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(grads))
    # vmap over a leading axis of independent streams, each time-major.
    streams = jnp.stack([x, x * 0.5])
    batched = jax.vmap(lambda x: net.apply(params, x))(streams)
    np.testing.assert_allclose(batched[1], net.apply(params, x * 0.5), rtol=1e-6, atol=1e-6)


def test_bf16_networks_spike_in_bf16():
    x = inputs(10).astype(jnp.bfloat16)
    out = LIF().apply({}, x)
    assert out.dtype == jnp.bfloat16
    np.testing.assert_array_equal(out, run(LIFCell(math.exp(-1 / 2.0)), x)[0])


@pytest.mark.parametrize("tau", [0.0, -1.0])
def test_nonpositive_time_constants_are_refused(tau):
    with pytest.raises(ValueError, match="positive"):
        LIF(tau=tau).apply({}, inputs())


def test_positional_arguments_name_the_neuron_not_the_unroll():
    assert LIF(3.0).tau == 3.0
    assert Recurrent(ALIF()).neuron == ALIF()
    assert sparx.nn.decay(2.0) == math.exp(-0.5)
