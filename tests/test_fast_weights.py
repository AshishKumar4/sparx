from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct

import sparx.nn as snn
from sparx.dynamics import (
    DecayingHebb,
    LIFCell,
    ModulatedHebb,
    OjaHebb,
    PlasticRecurrentCell,
    RateCell,
    RetroactiveHebb,
    SynapticInput,
    run,
)

MICONI = np.load(Path(__file__).parent / "fixtures" / "miconi.npz")

# Each case's parameters, in the fixture's names, and the rule built from them.
RULES = {
    "hebb": (("weight", "alpha", "eta"), lambda p: DecayingHebb(p["eta"][0])),
    "oja": (("i2h_weight", "i2h_bias", "weight", "alpha", "eta"), lambda p: OjaHebb(p["eta"][0])),
    "modulated": (("i2h_weight", "i2h_bias", "weight", "alpha", "modulator", "modulator_bias", "fanout",
                   "fanout_bias"),
                  lambda p: ModulatedHebb(p["modulator"][0], p["modulator_bias"][0], p["fanout"][:, 0],
                                          p["fanout_bias"], clip=2.0)),
    "retroactive": (("i2h_weight", "i2h_bias", "weight", "alpha", "eta", "modulator", "modulator_bias"),
                    lambda p: RetroactiveHebb(p["modulator"][0], p["modulator_bias"][0], p["eta"][0],
                                              clip=1.0)),
}


def case(name):
    return {key.split("/", 1)[1]: MICONI[key] for key in MICONI.files if key.startswith(f"{name}/")}


def episode(name, given, parameters):
    """Miconi's network as sparx builds it: a tanh unit without leak behind a plastic recurrence.

    Returns the case's loss and, per step, the activity and what the rule keeps.
    """
    inputs = jnp.asarray(given["inputs"])
    drive = inputs if name == "hebb" else inputs @ parameters["i2h_weight"] + parameters["i2h_bias"]
    model = PlasticRecurrentCell(RateCell(0.0), parameters["weight"], parameters["alpha"],
                                 RULES[name][1](parameters), jax.lax.Precision.HIGHEST)

    def step(state, x):
        state, out = model.step(state, SynapticInput(jump=x), 1.0)
        return state, (out.value, state.trace)

    _, (outputs, traces) = jax.lax.scan(step, model.init_state(drive.shape[1:], drive.dtype), drive)
    if name == "hebb":  # simple.py's loss: the squared error of the last step on the pattern's bits
        loss = jnp.sum((outputs[-1, 0, :given["target"].size] - given["target"]) ** 2)
    else:
        loss = jnp.sum(outputs * given["coefficients"])
    return loss, (outputs, traces)


@pytest.mark.parametrize("name", list(RULES))
def test_a_plastic_recurrence_is_miconis_network_step_by_step_and_in_its_gradient(name):
    given = case(name)
    with jax.enable_x64(new_val=True):
        parameters = {key: jnp.asarray(given[key]) for key in RULES[name][0]}
        (loss, (outputs, traces)), grads = jax.value_and_grad(
            lambda p: episode(name, given, p), has_aux=True)(parameters)
        alone, _ = episode(name, given, {**parameters, "alpha": jnp.zeros_like(parameters["alpha"])})
    hebb = traces.hebb if name == "retroactive" else traces
    if name in ("hebb", "oja"):  # their scripts keep one trace for their one example
        hebb = hebb[:, 0]
    # Observed (activity, trace, then gradients over each array's largest entry): hebb 9.4e-16, 1.4e-15,
    # 3.9e-15; oja 3.1e-16, 2.2e-16, 5.2e-16; modulated 1.3e-14, 4.5e-14, 1.0e-14; retroactive 8.9e-16,
    # 1.4e-15 and its eligibility 5.1e-16, 2.6e-15.
    np.testing.assert_allclose(outputs, given["outputs"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(hebb, given["traces"], rtol=0, atol=1e-12)
    if name == "retroactive":
        np.testing.assert_allclose(traces.eligibility, given["eligibilities"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(loss, given["loss"], rtol=1e-12)
    for key in RULES[name][0]:
        expected = given[f"grad_{key}"]
        np.testing.assert_allclose(grads[key], expected, rtol=0, atol=1e-12 * np.abs(expected).max(),
                                   err_msg=key)
    # The trace shapes the activity: without it the loss differs.
    assert abs(float(alone) - float(given["loss"])) > 0.1


def test_the_bounded_rules_clip_part_of_the_trace():
    # The fixtures were drawn so the clip holds some entries and not others.
    for name, bound in (("modulated", 2.0), ("retroactive", 1.0)):
        held = np.mean((np.abs(case(name)["traces"]) == bound).any(axis=0))
        assert 0.05 < held < 0.95, name


def test_a_decaying_trace_over_two_half_steps_is_one_whole_step():
    # keep = (1 - eta) ** dt: the trace decays as a rate per unit of time, whatever the step.
    rng = np.random.default_rng(0)
    with jax.enable_x64(new_val=True):
        trace, pre, post = (jnp.asarray(rng.normal(size=shape)) for shape in ((4, 4), (4,), (4,)))
        for rule in (DecayingHebb(0.3), RetroactiveHebb(jnp.zeros(4), 0.0, 0.3)):
            start = rule.init_trace((4,), jnp.float64)
            start = jax.tree.map(lambda zero: zero + trace, start)
            halves = rule.update(rule.update(start, pre, post, 0.5), pre, post, 0.5)
            whole = rule.update(start, pre, post, 1.0)
            for a, b in zip(jax.tree.leaves(halves), jax.tree.leaves(whole), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-15)


def test_a_retroactive_trace_credits_coactivity_the_modulator_arrives_after():
    # Coactivity with the modulator at zero leaves the weights alone and
    # fills the eligibility; a modulator that comes later writes it in.
    rule = RetroactiveHebb(jnp.ones(2), 0.0, 0.5)
    trace = rule.init_trace((2,), jnp.float32)
    active, silent = jnp.asarray([1.0, -1.0]), jnp.zeros(2)
    trace = rule.update(trace, active, active, 1.0)  # m = tanh(0) = 0
    assert np.all(trace.hebb == 0) and np.any(trace.eligibility != 0)
    reward = rule.update(trace, silent, jnp.asarray([2.0, 0.0]), 1.0)
    np.testing.assert_allclose(reward.hebb, np.tanh(2.0) * trace.eligibility, rtol=1e-6)


def test_a_plastic_spiking_layer_keeps_its_trace_in_float32_under_bf16_spikes():
    rng = np.random.default_rng(1)
    xs = jnp.asarray(rng.normal(0.4, 0.8, (30, 2, 5)), jnp.bfloat16)
    model = PlasticRecurrentCell(LIFCell(0.8), jnp.asarray(rng.normal(0, 0.3, (5, 5)), jnp.float32),
                                 jnp.full((5,), 0.5, jnp.float32), DecayingHebb(0.2))
    out, state = run(model, xs)
    assert out.value.dtype == jnp.bfloat16 and state.output.dtype == jnp.bfloat16
    assert state.trace.dtype == jnp.float32 and state.trace.shape == (2, 5, 5)
    # Spikes are 0 and 1, so the trace is a decaying count of coincidences, between 0 and 1.
    assert 0 < float(state.trace.max()) <= 1 and float(state.trace.min()) >= 0


TRACES = {"decaying": snn.DecayingTrace(), "oja": snn.OjaTrace(), "modulated": snn.ModulatedTrace(),
          "retroactive": snn.RetroactiveTrace()}


@pytest.mark.parametrize("name", list(TRACES))
def test_a_plastic_layer_runs_the_cell_its_parameters_build(name):
    x = jnp.asarray(np.random.default_rng(2).normal(0, 1.0, (12, 3, 5)), jnp.float32)
    layer = snn.Plastic(snn.Rate(tau=0), rule=TRACES[name])
    variables = layer.init(jax.random.key(0), x)
    params = variables["params"]
    assert params["recurrent"].shape == params["alpha"].shape == (5, 5)
    modulator = {"modulator", "modulator_bias"}
    assert set(params["rule"]) == {"decaying": {"eta"}, "oja": {"eta"}, "retroactive": {"eta", *modulator},
                                   "modulated": {"fanout", "fanout_bias", *modulator}}[name]
    built = layer.apply(variables, x, method="model")
    assert isinstance(built, PlasticRecurrentCell) and built.inner.decay == 0.0
    expected, _ = run(built, x)
    np.testing.assert_array_equal(layer.apply(variables, x), expected.value)


def test_a_plastic_layer_fed_in_chunks_carries_its_trace():
    x = jnp.asarray(np.random.default_rng(3).normal(0, 1.0, (20, 2, 4)), jnp.float32)
    layer = snn.Plastic(snn.LIF(), rule=snn.RetroactiveTrace(eta=0.3))
    variables = layer.init(jax.random.key(1), x)
    whole = layer.apply(variables, x)
    head, state = layer.apply(variables, x[:7], mutable=["state"])
    tail, state = layer.apply({**variables, **state}, x[7:], mutable=["state"])
    np.testing.assert_array_equal(jnp.concatenate([head, tail]), whole)
    assert state["state"]["carry"].trace.hebb.shape == (2, 4, 4)


@struct.dataclass
class Frozen:
    """A rule whose trace never moves from zero: a plastic layer that is not plastic."""

    def init_trace(self, shape, dtype):
        return jnp.zeros((*shape, shape[-1]), dtype)

    def hebb(self, trace):
        return trace

    def update(self, trace, pre, post, dt):
        return trace


class FrozenTrace(snn.HebbianTrace):
    def build(self, features):
        return Frozen()


def test_a_rule_of_ones_own_plugs_into_the_layer():
    # The trace stays zero, so the layer is Recurrent with the same weights.
    x = jnp.asarray(np.random.default_rng(4).normal(0.4, 0.8, (15, 2, 6)), jnp.float32)
    plastic = snn.Plastic(snn.LIF(), rule=FrozenTrace())
    variables = plastic.init(jax.random.key(2), x)
    recurrent = {"params": {"recurrent": variables["params"]["recurrent"]}}
    np.testing.assert_array_equal(plastic.apply(variables, x), snn.Recurrent(snn.LIF()).apply(recurrent, x))


def test_a_plastic_layer_steps_at_its_neurons_dt():
    x = jnp.zeros((3, 1, 2))
    with pytest.raises(ValueError, match="give both one dt"):
        snn.Plastic(snn.Rate(tau=0, dt=0.5)).init(jax.random.key(0), x)


# Miconi et al.'s (2018) pattern completion (their simple/simple.py), small:
# two patterns of 8 bits, each shown twice for 3 steps with 2 blank steps
# after, then one of them with half its bits zeroed, which the network must
# fill in on the last step from what this episode showed it.
BITS, PATTERNS, SHOWN, BLANK, CYCLES, TESTED = 8, 2, 3, 2, 2, 3


def episodes(key, batch):
    """Inputs `[T, B, BITS + 1]` (a last unit of constant input, their bias neuron, all times their gain
    of 20), each episode's pattern to complete `[B, BITS]`, and which of its bits are shown `[B, BITS]`."""
    keys = jax.random.split(key, 4)
    balanced = jnp.where(jnp.arange(BITS) < BITS // 2, -1.0, 1.0)

    def shuffled(k, values, n):
        return jax.vmap(lambda kk: jax.random.permutation(kk, values))(jax.random.split(k, n))

    patterns = jax.vmap(lambda k: shuffled(k, balanced, PATTERNS))(jax.random.split(keys[0], batch))
    target = patterns[jnp.arange(batch), jax.random.randint(keys[1], (batch,), 0, PATTERNS)]
    shown = shuffled(keys[2], jnp.where(jnp.arange(BITS) < BITS // 2, 0.0, 1.0), batch)
    steps = []
    for cycle in range(CYCLES):
        order = shuffled(jax.random.fold_in(keys[3], cycle), jnp.arange(PATTERNS), batch)
        for i in range(PATTERNS):
            pattern = patterns[jnp.arange(batch), order[:, i]]
            steps += [pattern] * SHOWN + [jnp.zeros_like(pattern)] * BLANK
    steps += [target * shown] * TESTED
    x = jnp.stack(steps)
    return 20.0 * jnp.concatenate([x, jnp.ones((*x.shape[:2], 1))], axis=-1), target, shown


def completion_error(layer, steps=300):
    """The share of zeroed bits the trained layer gets wrong on fresh episodes."""
    params = layer.init(jax.random.key(0), episodes(jax.random.key(1), 2)[0])["params"]
    optimizer = optax.adam(1e-2)

    def loss(p, key):  # their loss: the last step's squared error on the pattern's bits
        x, target, _ = episodes(key, 32)
        return jnp.mean(jnp.sum((layer.apply({"params": p}, x)[-1, :, :BITS] - target) ** 2, axis=-1))

    @jax.jit
    def update(p, s, key):
        updates, s = optimizer.update(jax.grad(loss)(p, key), s, p)
        return optax.apply_updates(p, updates), s

    state = optimizer.init(params)
    for step in range(steps):
        params, state = update(params, state, jax.random.key(100 + step))
    x, target, shown = episodes(jax.random.key(2), 512)
    wrong = (jnp.sign(layer.apply({"params": params}, x)[-1, :, :BITS]) != target) & (shown == 0)
    return float(wrong.sum() / (shown == 0).sum())


def test_fast_weights_learn_to_complete_a_pattern_the_episode_showed():
    # Observed after 300 steps: decaying 0.051, oja 0.042, modulated 0.052,
    # retroactive 0.040; the same network without a trace 0.223.
    small = nn.initializers.normal(0.01)  # their simple.py's draws
    fixed = completion_error(snn.Recurrent(snn.Rate(tau=0), kernel_init=small))
    for name, trace in TRACES.items():
        error = completion_error(snn.Plastic(snn.Rate(tau=0), rule=trace, kernel_init=small))
        assert error < 0.1 < 0.15 < fixed, (name, error, fixed)
