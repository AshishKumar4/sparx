"""Parity with SpikingJelly (commit c6cb8e46, torch 2.14.1+cpu).

`tools/make_reference_fixtures.py` ran SpikingJelly's own modules on fixed
inputs and saved their spikes and gradients. Each case's membranes stay at
least 1e-4 from threshold, so spikes are compared exactly. Gradients are
float32 sums taken in a different order by the two frameworks; the largest
differences observed, on CPU at float32, are noted at each tolerance.
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.dynamics import LICell, LIFCell, Serial, run
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
        out, state = run(cell, x)
        spikes = out.fired
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
    # Observed 9.5e-7.
    np.testing.assert_allclose(grads["params"]["weight"], c["grad_weight"], rtol=1e-5, atol=2e-6)
    # Observed 1.2e-6.
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


SNNTORCH = np.load(Path(__file__).parent / "fixtures" / "snntorch.npz")


def _snntorch_case(key):
    return {name: SNNTORCH[f"{key}/{name}"] for name in ("spikes", "grad_x")}


@pytest.mark.parametrize("reset", ["subtract", "zero"])
@pytest.mark.parametrize(("name", "cell"), [
    ("leaky", lambda reset: LIFCell(0.8, 1.0, reset, ATan(2.0))),
    ("synaptic", lambda reset: Serial(LICell(0.6), LIFCell(0.8, 1.0, reset, ATan(2.0)))),
])
def test_lif_and_synaptic_match_snntorch_with_an_immediate_reset(name, cell, reset):
    # snnTorch's immediate reset subtracts the threshold times the live spike,
    # so the gradient flows through it, as with sparx's detach_reset=False. On
    # this input no membrane stays above threshold after its reset, where the
    # two models are the same (the next test is the case where they are not).
    expected = _snntorch_case(f"neuron_x/{name}_{reset}_immediate")

    def weighted(x):
        spikes = run(cell(reset), x)[0].fired
        return jnp.sum(spikes * SNNTORCH["neuron_weights"]), spikes

    grad_x, spikes = jax.grad(weighted, has_aux=True)(jnp.asarray(SNNTORCH["neuron_x"]))
    np.testing.assert_array_equal(spikes, expected["spikes"])
    assert spikes.sum() > 40
    np.testing.assert_allclose(grad_x, expected["grad_x"], rtol=1e-5, atol=1e-6)  # observed 8.9e-7


def test_snntorch_loses_spikes_after_an_overshoot_where_sparx_fires():
    # A soft reset that leaves the membrane at or above threshold should fire
    # again at the next step whenever the membrane is at threshold. snnTorch
    # subtracts the next reset before it tests for a spike, so such a neuron
    # needs twice the threshold. Every neuron where the two part ways does so
    # on the step right after its own overshoot (docs/fidelity.md).
    import reference
    x = SNNTORCH["overshoot_x"].astype(np.float64)
    spikes, post = reference.lif(x, 0.8, 1.0, "subtract")
    theirs = SNNTORCH["overshoot_x/leaky_subtract_immediate/spikes"]
    mismatch = (spikes != theirs).reshape(x.shape[0], -1)
    overshoot = np.zeros_like(post, bool)
    overshoot[1:] = post[:-1] >= 1.0
    overshoot = overshoot.reshape(x.shape[0], -1)
    diverged = [n for n in range(mismatch.shape[1]) if mismatch[:, n].any()]
    assert diverged
    for n in diverged:
        first = int(np.argmax(mismatch[:, n]))
        assert overshoot[first, n] and spikes.reshape(x.shape[0], -1)[first, n] == 1 and \
            theirs.reshape(x.shape[0], -1)[first, n] == 0


@pytest.mark.parametrize("name", ["leaky", "synaptic"])
def test_snntorchs_default_delayed_reset_is_a_different_model(name):
    # snnTorch's default subtracts the threshold one step after the spike,
    # without decaying it, which no continuous LIF discretizes to. Sparx resets
    # on the spiking step (docs/fidelity.md). The spike trains part ways.
    delayed = _snntorch_case(f"neuron_x/{name}_subtract_delayed")["spikes"]
    immediate = _snntorch_case(f"neuron_x/{name}_subtract_immediate")["spikes"]
    assert (delayed != immediate).sum() >= 5


DCLS = np.load(Path(__file__).parent / "fixtures" / "dcls.npz")


def _dcls_params(positions):
    # DCLS's kernel index k reads x[t - (K - 1 - k)] once the input is padded
    # by K - 1 on the left, and its Gaussian is centered at k = P + K // 2,
    # so the delay is K - 1 - (P + K // 2). Weights are [out, in] there.
    kernel = 7
    delay = (kernel - 1) - (positions + kernel // 2)
    return {"params": {"kernel": jnp.asarray(DCLS["weight"].T), "delay": jnp.asarray(delay.T),
                       "bias": jnp.asarray(DCLS["bias"])}}


@pytest.mark.parametrize("mode", ["gauss", "rounded"])
def test_delayed_dense_matches_dcls_delays(mode):
    from sparx.nn import DelayedDense
    layer = DelayedDense(5, max_delay=6, precision=jax.lax.Precision.HIGHEST)
    # DCLS's width is |SIG| + 0.27; at evaluation it reads rounded positions.
    params = _dcls_params(DCLS["P"] if mode == "gauss" else DCLS["rounded_P"])
    sigma = float(DCLS["SIG"]) + 0.27 if mode == "gauss" else 0

    def weighted(params, x):
        out = layer.apply(params, x, sigma)
        return jnp.sum(out * DCLS["weights"]), out

    (grads, grad_x), out = jax.grad(weighted, argnums=(0, 1), has_aux=True)(params, jnp.asarray(DCLS["x"]))
    expected = {name[len(mode) + 1:]: DCLS[name] for name in DCLS.files if name.startswith(f"{mode}/")}
    # Observed at most 2.4e-7 on the outputs and 1.4e-6 on any gradient.
    np.testing.assert_allclose(out, expected["out"], rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(grad_x, expected["grad_x"], rtol=1e-5, atol=4e-6)  # observed 2.4e-7
    # Observed 1.4e-6.
    np.testing.assert_allclose(grads["params"]["kernel"], expected["grad_weight"].T, rtol=1e-5, atol=4e-6)
    if mode == "gauss":
        # d/d delay = -d/d P.
        # Observed 8.9e-8.
        np.testing.assert_allclose(grads["params"]["delay"], -expected["grad_P"].T, rtol=1e-4, atol=4e-6)


def _sew_variables(key):
    def conv(name):  # torch [out, in, h, w] -> flax [h, w, in, out]
        return jnp.asarray(FIXTURES[f"{key}/{name}.weight"].transpose(2, 3, 1, 0))

    def bn(name):
        return ({"scale": jnp.asarray(FIXTURES[f"{key}/{name}.weight"]),
                 "bias": jnp.asarray(FIXTURES[f"{key}/{name}.bias"])},
                {"mean": jnp.asarray(FIXTURES[f"{key}/{name}.running_mean"]),
                 "var": jnp.asarray(FIXTURES[f"{key}/{name}.running_var"])})

    params, stats = {}, {}
    pairs = [("first", "conv1", "bn1"), ("second", "conv2", "bn2")]
    if f"{key}/downsample.0.weight" in FIXTURES.files:
        pairs.append(("downsample", "downsample.0", "downsample.1"))
    for ours, conv_name, bn_name in pairs:
        params[f"{ours}_conv"] = {"kernel": conv(conv_name)}
        params[f"{ours}_bn"], stats[f"{ours}_bn"] = bn(bn_name)
    return {"params": params, "batch_stats": stats}


@pytest.mark.parametrize("connect", ["add", "and", "iand"])
@pytest.mark.parametrize(("key", "features", "strides"), [("sew_same", 8, 1), ("sew_down", 16, 2)])
def test_sew_block_matches_spikingjelly(key, features, strides, connect):
    from sparx.models import SEWBlock
    from sparx.nn import IF
    case = f"{key}_{connect}"
    block = SEWBlock(features, strides, connect, IF(reset="zero", detach_reset=True))
    # Both run in float64, so a reordered convolution sum cannot flip a spike.
    with jax.enable_x64(new_val=True):
        out = block.apply(_sew_variables(case), jnp.asarray(FIXTURES[f"{case}/x"]), train=False)
        np.testing.assert_array_equal(np.asarray(out), FIXTURES[f"{case}/out"])


SNN_DELAYS = np.load(Path(__file__).parent / "fixtures" / "snn_delays.npz")
_DCLS_KERNEL = 5  # the fixture's dilated kernel: delays of 0 to 4 steps


def _snn_delays_network():
    from sparx.models import SpikingMLP
    from sparx.nn import LIF
    # SpikingJelly's decay_input=False LIF keeps 1 - 1 / tau of the membrane a step.
    tau = -1 / math.log(1 - 1 / float(SNN_DELAYS["tau"]))
    neuron = LIF(tau=tau, reset="zero", surrogate=ATan(5.0), detach_reset=True)
    return SpikingMLP(hidden=(5, 5), classes=3, neuron=neuron, delays=(4, 4, 4), extend=True, batch_norm=True,
                      use_bias=False, readout_tau=tau)


def _snn_delays_params(positions):
    params = {}
    for layer, name in enumerate(("delayed_0", "delayed_1", "readout")):
        # DCLS centers its Gaussian at kernel index P + K // 2, which reads K - 1 - that steps back.
        delay = (_DCLS_KERNEL - 1) - (SNN_DELAYS[positions.format(layer)] + _DCLS_KERNEL // 2)
        params[name] = {"kernel": jnp.asarray(SNN_DELAYS[f"layer{layer}/weight"].T),
                        "delay": jnp.asarray(delay.T)}
    for layer in range(2):
        params[f"norm_{layer}"] = {"scale": jnp.ones(5), "bias": jnp.zeros(5)}
    return params


def test_a_fully_delayed_network_trains_as_snn_delays():
    from sparx.losses import softmax_sum_cross_entropy
    from sparx.nn import RATES
    net = _snn_delays_network()
    stats = {f"norm_{layer}": {"mean": jnp.zeros(5), "var": jnp.ones(5)} for layer in range(2)}
    labels = jnp.asarray(SNN_DELAYS["labels"])
    sigma = float(SNN_DELAYS["sig"]) + 0.27  # DCLS's effective width

    def loss(params, x):
        out, updated = net.apply({"params": params, "batch_stats": stats}, x, train=True, sigma=sigma,
                                 mutable=["batch_stats", RATES])
        return jnp.mean(softmax_sum_cross_entropy(out, labels)), (out, updated)

    (value, (out, updated)), (grads, grad_x) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(
        _snn_delays_params("layer{}/P"), jnp.asarray(SNN_DELAYS["x"]))
    assert all(float(rate.mean()) > 0.05 for rate in jax.tree.leaves(updated[RATES]))  # hidden layers fire
    # Each delayed synapse appends 2 steps, so 9 input steps reach the readout as 15. Observed at
    # most 1.2e-7 on the outputs and the loss, 2.1e-7 on the input gradient, 5.7e-7 on any other.
    np.testing.assert_allclose(out, SNN_DELAYS["train/out"], rtol=1e-5, atol=1e-5)  # observed 1.2e-7
    np.testing.assert_allclose(value, SNN_DELAYS["train/loss"], rtol=1e-6)  # observed 7.9e-8 relative
    np.testing.assert_allclose(grad_x, SNN_DELAYS["train/grad_x"], rtol=1e-5, atol=1e-6)  # observed 1.9e-7
    for layer, name in enumerate(("delayed_0", "delayed_1", "readout")):
        np.testing.assert_allclose(grads[name]["kernel"], SNN_DELAYS[f"train/grad_weight{layer}"].T,
                                   rtol=1e-5, atol=1e-6)  # observed 4.8e-7
        # d/d delay = -d/d P.
        np.testing.assert_allclose(grads[name]["delay"], -SNN_DELAYS[f"train/grad_P{layer}"].T,
                                   rtol=1e-5, atol=1e-6)  # observed 9.7e-8
    for layer in range(2):
        norm = f"norm_{layer}"
        np.testing.assert_allclose(grads[norm]["scale"], SNN_DELAYS[f"train/grad_bn_weight{layer}"],
                                   rtol=1e-5, atol=1e-6)  # observed 4.3e-7
        np.testing.assert_allclose(grads[norm]["bias"], SNN_DELAYS[f"train/grad_bn_bias{layer}"],
                                   rtol=1e-5, atol=1e-6)  # observed 4.2e-7
        running = updated["batch_stats"][norm]
        np.testing.assert_allclose(running["mean"], SNN_DELAYS[f"train/running_mean{layer}"],
                                   rtol=1e-5, atol=1e-7)  # observed 1.5e-8
        # torch keeps the unbiased variance of the n = steps * batch values, flax the biased one.
        n = (9 + 2 * (layer + 1)) * 4
        unbiased = 0.9 + (np.asarray(running["var"]) - 0.9) * n / (n - 1)
        # Observed 6.5e-8 relative.
        np.testing.assert_allclose(unbiased, SNN_DELAYS[f"train/running_var{layer}"], rtol=1e-5)


def test_a_fully_delayed_network_evaluates_as_snn_delays():
    from sparx.losses import softmax_sum_cross_entropy
    net = _snn_delays_network()
    # Their evaluation reads rounded positions and the running statistics.
    stats = {f"norm_{layer}": {"mean": jnp.asarray(SNN_DELAYS[f"train/running_mean{layer}"]),
                               "var": jnp.asarray(SNN_DELAYS[f"train/running_var{layer}"])}
             for layer in range(2)}
    variables = {"params": _snn_delays_params("eval/P{}"), "batch_stats": stats}
    out = net.apply(variables, jnp.asarray(SNN_DELAYS["x"]), train=False, sigma=0)
    # Observed at most 2.4e-7.
    np.testing.assert_allclose(out, SNN_DELAYS["eval/out"], rtol=1e-5, atol=1e-5)
    loss = jnp.mean(softmax_sum_cross_entropy(out, jnp.asarray(SNN_DELAYS["labels"])))
    np.testing.assert_allclose(loss, SNN_DELAYS["eval/loss"], rtol=1e-6)  # observed 1.8e-7 relative


def test_event_binning_reproduces_snn_delays_frames():
    from sparx.datasets import bin_events
    frames = SNN_DELAYS["events/frames"]
    steps = frames.shape[0]
    ours = bin_events(SNN_DELAYS["events/times"], SNN_DELAYS["events/units"], steps, steps * 0.01, 700,
                      binning="events")
    np.testing.assert_array_equal(ours, frames)
    # The grid keeps the silences their binning drops, so it needs more steps.
    grid = bin_events(SNN_DELAYS["events/times"], SNN_DELAYS["events/units"], 40, 0.4, 700)
    assert np.flatnonzero(grid.sum(1)).max() >= steps


def test_schedules_stepped_once_an_epoch_are_snn_delays_torch_schedulers():
    from dew.training.optim import Cosine

    from sparx.optim import ExponentialDecay, OneCycle, stepped
    epochs, per_epoch = int(SNN_DELAYS["schedule/epochs"]), 3
    steps = np.arange(epochs) * per_epoch + 1  # a step inside each epoch

    def values(schedule):
        return np.asarray(jax.vmap(stepped(schedule, epochs * per_epoch, per_epoch))(jnp.asarray(steps)))

    # OneCycleLR(max_lr=5e-3, total_steps=epochs) and its Adam momentum cycle.
    np.testing.assert_allclose(values(OneCycle(peak=5e-3, start=2e-4, end=2e-8)), SNN_DELAYS["schedule/lr_w"],
                               rtol=1e-5)  # observed 7.8e-6 relative
    np.testing.assert_allclose(values(OneCycle(peak=0.85, start=0.95, end=0.95)), SNN_DELAYS["schedule/b1"],
                               rtol=1e-6)  # observed 6.0e-8 relative
    # CosineAnnealingLR(T_max=epochs) on the positions' rate, 100 times the weights' 1e-3.
    np.testing.assert_allclose(values(Cosine(peak=0.1, warmup_steps=0)), SNN_DELAYS["schedule/lr_pos"],
                               rtol=1e-4, atol=1e-8)  # observed 9.6e-9
    # decrease_sig: DCLS's raw width from 12 to 0.23 over the first quarter; sparx's width adds 0.27.
    width = ExponentialDecay(start=12.0, end=0.23, decay_steps=epochs // 4, offset=0.27)
    # Observed 4.1e-7 relative.
    np.testing.assert_allclose(values(width), SNN_DELAYS["schedule/sig"] + 0.27, rtol=1e-5)
