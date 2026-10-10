"""Homeostasis: intrinsic plasticity and synaptic scaling, each against its reference.

Intrinsic plasticity is SORN's rule (Lazar, Pipa and Triesch 2009, equation 7), checked step by step
against a NumPy transcription on binary threshold units; synaptic scaling is van Rossum, Bi and
Turrigiano's (2000, equation 3), checked against a NumPy integration of their equations.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.dynamics import (
    Exponential,
    IntrinsicPlasticity,
    LeakyIntegrateAndFire,
    LICell,
    LIFCell,
    PairSTDP,
    Receptor,
    Rules,
    SynapticInput,
    SynapticScaling,
    run,
)
from sparx.graph import Network, PoissonInput, Population, SpikeRaster, StateMonitor, simulate
from sparx.nn import LIF, Homeostatic


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(new_val=True):
        yield


def test_intrinsic_plasticity_is_sorns_rule_on_binary_threshold_units():
    # SORN's units fire when their input reaches the threshold, x_i(t+1) = H(input - T_i), and after each
    # step T_i += eta_IP (x_i - H_IP). A LIF cell with no leak is that unit.
    rng = np.random.default_rng(0)
    inputs = rng.normal(0.5, 0.4, (400, 6))
    eta, target, start = 0.01, 0.1, 0.5
    out, final = run(IntrinsicPlasticity(LIFCell(0.0, start, reset="zero"), target, eta), jnp.asarray(inputs))
    threshold, spikes = np.full(6, start), []
    for x in inputs:
        s = (x - threshold >= 0).astype(float)
        threshold = threshold + eta * (s - target)
        spikes.append(s)
    np.testing.assert_array_equal(np.asarray(out.value), np.array(spikes))
    np.testing.assert_allclose(start + np.asarray(final.shift), threshold, rtol=0, atol=1e-12)
    assert 0 < np.mean(spikes) < 1


@pytest.mark.parametrize("target", [0.05, 0.1, 0.3])
def test_intrinsic_plasticity_holds_a_leaky_neuron_at_its_target_rate(target):
    # Whatever the drive, the threshold settles where the neuron fires `target` of its steps.
    rng = np.random.default_rng(1)
    drive = jnp.asarray(rng.normal(0.3, 0.3, (30_000, 3)) * np.array([0.5, 1.0, 2.0]))
    out, _ = run(IntrinsicPlasticity(LIFCell(0.9), target, 0.05), drive)
    rates = np.asarray(out.value)[-10_000:].mean(0)
    np.testing.assert_allclose(rates, target, atol=0.1 * target)


def test_intrinsic_plasticity_moves_a_physical_neurons_threshold_in_millivolts():
    # The same rule on a physical LIF in mV and ms: its v_th rises until it fires at 20 Hz (0.02 per ms)
    # on a current that alone drives it at about 100 Hz.
    neuron = IntrinsicPlasticity(LeakyIntegrateAndFire(), target=0.02, eta=0.5, field="v_th")
    out, final = run(neuron, SynapticInput(current=jnp.full((200_000, 1), 500.0)), dt=0.1)
    rate = np.asarray(out.value)[-50_000:].sum() / (50_000 * 0.1)
    assert abs(rate - 0.02) < 0.002, rate
    assert float(final.shift[0]) > 10


def test_intrinsic_plasticity_holds_a_networks_population_at_its_target_rate():
    # Poisson input drives 20 neurons at about 35 Hz; their thresholds rise until they fire at 5 Hz. A
    # voltage monitor reads through the wrapper to the membrane.
    neuron = IntrinsicPlasticity(LeakyIntegrateAndFire(), target=0.005, eta=1.0, field="v_th")
    network = Network((Population("n", 20, neuron, {"ex": Receptor(Exponential(5.0))}),),
                      inputs=(PoissonInput("n", rate=1000.0, weight=30.0, receptor="ex"),), dt=0.1)
    result = simulate(network, network.init(jax.random.key(0)), duration=4000.0, key=jax.random.key(1),
                      monitors={"spikes": SpikeRaster("n"), "v": StateMonitor("n", neurons=(0,))})
    spikes = np.asarray(result.records["spikes"], np.float64)
    early, late = spikes[:1000].mean() / 0.1, spikes[-20_000:].mean() / 0.1  # spikes per ms
    assert early > 0.015 and abs(late - 0.005) < 0.001, (early, late)
    voltage = np.asarray(result.records["v"])
    assert voltage.shape == (40_000, 1) and np.all(np.isfinite(voltage)) and voltage.max() > -50


def test_intrinsic_plasticity_wraps_a_conductance_based_population():
    # At eta = 0 the wrapper is the model it wraps, conductances and all; at eta > 0 its thresholds rise.
    receptors = {"ampa": Receptor(Exponential(5.0), "conductance")}

    def spikes(neuron):
        network = Network((Population("n", 10, neuron, receptors),),
                          inputs=(PoissonInput("n", rate=1000.0, weight=2.0, receptor="ampa"),), dt=0.1)
        result = simulate(network, network.init(jax.random.key(0)), duration=300.0, key=jax.random.key(1),
                          monitors={"spikes": SpikeRaster("n")})
        return np.asarray(result.records["spikes"])

    plain = spikes(LeakyIntegrateAndFire())
    still = IntrinsicPlasticity(LeakyIntegrateAndFire(), 0.005, 0.0, "v_th")
    np.testing.assert_array_equal(spikes(still), plain)
    adapted = spikes(IntrinsicPlasticity(LeakyIntegrateAndFire(), 0.005, 1.0, "v_th"))
    assert plain.sum() > 50 and adapted.sum() < plain.sum()
    dimensionless = Population("n", 2, IntrinsicPlasticity(LIFCell(0.9), 0.1), receptors)
    with pytest.raises(ValueError, match="IntrinsicPlasticity has no reversal potentials"):
        Network((dimensionless,)).init(jax.random.key(0))


def test_intrinsic_plasticity_keeps_its_drift_in_the_states_dtype():
    # Parameters in float64 drive a float32 population without changing the carried state's dtype.
    neuron = IntrinsicPlasticity(LIFCell(0.9), jnp.full(3, 0.1, jnp.float64), jnp.full(3, 0.01, jnp.float64))
    _, final = run(neuron, jnp.ones((20, 3), jnp.float32))
    assert final.shift.dtype == jnp.float32 and float(final.shift[0]) > 0


def test_intrinsic_plasticity_drifts_in_float32_on_low_precision_inputs():
    # bfloat16 inputs and the same values in float32 give the same spikes and the same drift.
    inputs = jnp.asarray(np.random.default_rng(8).normal(0.4, 0.5, (200, 4)), jnp.bfloat16)
    neuron = IntrinsicPlasticity(LIFCell(0.9), 0.1, 0.03)
    low, low_final = run(neuron, inputs)
    full, full_final = run(neuron, inputs.astype(jnp.float32))
    np.testing.assert_array_equal(np.asarray(low.value, np.float32), np.asarray(full.value))
    assert low_final.shift.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(low_final.shift), np.asarray(full_final.shift))


def test_intrinsic_plasticity_refuses_a_graded_model_and_a_missing_field():
    with pytest.raises(ValueError, match="graded"):
        IntrinsicPlasticity(LICell(0.9), 0.1).init_state((3,), jnp.float32)
    with pytest.raises(ValueError, match="no field"):
        IntrinsicPlasticity(LIFCell(0.9), 0.1, field="v_th").init_state((3,), jnp.float32)


def scaling_reference(weights, post, spikes, dt, tau, goal, beta, gamma):
    """van Rossum et al.'s equation 3 solved over each step with the sensor held at its value after the
    step's spikes: with d = goal - a, the integral grows as E + d s over the step, and
    log(w' / w) = dt (beta d + gamma E) + gamma d dt^2 / 2."""
    activity, error, w = np.zeros(spikes.shape[1]), np.zeros(spikes.shape[1]), weights.copy()
    for s in spikes:
        activity = activity * np.exp(-dt / tau) + s / tau
        d = goal - activity
        w = w * np.exp(dt * (beta * d + gamma * error) + gamma * d * dt**2 / 2)[post]
        error = error + dt * d
    return w, activity


def scan_rule(rule, traces, weights, pre_spikes, post_spikes, pre, post, dt):
    def step(carry, spikes):
        traces, weights = carry
        return rule.step(traces, weights, *spikes, pre, post, dt, modulators={}), None

    (traces, weights), _ = jax.lax.scan(step, (traces, weights), (pre_spikes, post_spikes))
    return traces, weights


def test_synaptic_scaling_is_van_rossum_bi_and_turrigianos_equations():
    rng = np.random.default_rng(2)
    pre, post = np.array([0, 1, 2, 0, 1, 2, 3]), np.array([0, 0, 0, 1, 1, 2, 2])
    weights = rng.uniform(0.5, 2.0, 7)
    spikes = (rng.random((2_000, 3)) < np.array([0.005, 0.03, 0.01])).astype(float)
    rule = SynapticScaling(tau=200.0, goal=0.015, beta=0.05, gamma=1e-4)
    traces, w = scan_rule(rule, rule.init_state(4, 3, 7, jnp.float64), jnp.asarray(weights),
                          jnp.zeros((2_000, 4)), jnp.asarray(spikes), jnp.asarray(pre), jnp.asarray(post),
                          0.5)
    expected, activity = scaling_reference(weights, post, spikes, 0.5, 200.0, 0.015, 0.05, 1e-4)
    np.testing.assert_allclose(np.asarray(w), expected, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(traces.activity), activity, rtol=1e-12)
    # A neuron firing below its goal had its weights scaled up, one above it down.
    assert w[0] > weights[0] and w[3] < weights[3]


def test_synaptic_scaling_keeps_the_ratios_of_a_neurons_weights():
    rng = np.random.default_rng(3)
    weights = jnp.asarray(rng.uniform(0.1, 3.0, 5))
    rule = SynapticScaling(tau=50.0, goal=0.02, beta=0.1)
    fired = jnp.asarray(rng.random((3_000, 1)) < 0.05, jnp.float64)
    _, w = scan_rule(rule, rule.init_state(5, 1, 5, jnp.float64), weights, jnp.zeros((3_000, 5)), fired,
                     jnp.arange(5), jnp.zeros(5, int), 1.0)
    np.testing.assert_allclose(np.asarray(w / w[0]), np.asarray(weights / weights[0]), rtol=1e-12)
    assert float(w[0] / weights[0]) < 0.9


def test_synaptic_scaling_brings_a_driven_neuron_to_its_goal_rate():
    # A LIF neuron driven through scaled weights by 50 inputs firing at 10 Hz settles at the 20 Hz goal,
    # from too weak and too strong a start.
    rng = np.random.default_rng(4)
    pre, post = jnp.arange(50), jnp.zeros(50, int)
    rule = SynapticScaling(tau=500.0, goal=0.02, beta=0.02, gamma=0.0)
    neuron = LIFCell(np.exp(-1 / 20.0), 1.0, reset="zero")
    inputs = jnp.asarray(rng.random((60_000, 50)) < 0.01, jnp.float64)

    def settled(start):
        def step(carry, x):
            cell, traces, w = carry
            cell, out = neuron.step(cell, SynapticInput(jump=jnp.reshape(x @ w, (1,))), 1.0)
            traces, w = rule.step(traces, w, x, out.value, pre, post, 1.0, modulators={})
            return (cell, traces, w), out.value[0]

        carry = (neuron.init_state((1,), jnp.float64), rule.init_state(50, 1, 50, jnp.float64),
                 jnp.full(50, start))
        _, spikes = jax.lax.scan(step, carry, inputs)
        return float(np.asarray(spikes)[-20_000:].mean())

    for start in (0.02, 0.6):
        assert abs(settled(start) - 0.02) < 0.004, start


def test_scaling_above_soft_bounded_stdps_bound_leaves_it_at_the_bound():
    # Scaling raises a weight already at PairSTDP's bound while its neuron is silent; at the next
    # postsynaptic spike STDP holds it at the bound, where (1 - w)^0.5 alone would be NaN.
    scaling = SynapticScaling(tau=20.0, goal=0.05, beta=0.5, gamma=0.0)
    rule = Rules((PairSTDP(mu_plus=0.5, w_max=1.0), scaling))
    traces, w = rule.init_state(1, 1, 1, jnp.float64), jnp.ones(1)
    index = jnp.zeros(1, int)
    traces, w = rule.step(traces, w, jnp.zeros(1), jnp.zeros(1), index, index, 1.0, modulators={})
    assert float(w[0]) > 1
    _, w = rule.step(traces, w, jnp.zeros(1), jnp.ones(1), index, index, 1.0, modulators={})
    assert np.isfinite(float(w[0])) and float(w[0]) <= 1.0


def test_rules_step_each_rule_on_the_weights_the_one_before_left():
    rng = np.random.default_rng(5)
    pre, post = jnp.asarray([0, 1, 2]), jnp.asarray([0, 0, 1])
    stdp, scaling = PairSTDP(lambda_=0.05, w_max=2.0), SynapticScaling(tau=20.0, goal=0.05, beta=0.2)
    pre_spikes = jnp.asarray(rng.random((200, 3)) < 0.2, jnp.float64)
    post_spikes = jnp.asarray(rng.random((200, 2)) < 0.2, jnp.float64)
    start = jnp.asarray([0.5, 1.0, 1.5])
    both = Rules((stdp, scaling))
    _, w = scan_rule(both, both.init_state(3, 2, 3, jnp.float64), start, pre_spikes, post_spikes, pre, post,
                     1.0)

    def each(carry, spikes):
        (one, two), weights = carry
        one, weights = stdp.step(one, weights, *spikes, pre, post, 1.0, modulators={})
        two, weights = scaling.step(two, weights, *spikes, pre, post, 1.0, modulators={})
        return ((one, two), weights), None

    traces = (stdp.init_state(3, 2, 3, jnp.float64), scaling.init_state(3, 2, 3, jnp.float64))
    (_, expected), _ = jax.lax.scan(each, (traces, start), (pre_spikes, post_spikes))
    np.testing.assert_allclose(np.asarray(w), np.asarray(expected), rtol=1e-13)
    assert not np.allclose(np.asarray(w), np.asarray(start))


def test_a_homeostatic_layer_carries_its_thresholds_drift():
    layer = Homeostatic(LIF(tau=3.0), target=0.05, eta=0.02)
    x = jnp.asarray(np.random.default_rng(7).normal(1.0, 0.5, (200, 2, 5)), jnp.float32)
    variables = layer.init(jax.random.key(0), x[:1])
    spikes, state = layer.apply(variables, x, mutable=["state"])
    assert spikes.shape == x.shape and float(jnp.min(state["state"]["carry"].shift)) > 0
    # Fed in two chunks with the state carried, it is the whole run.
    first, state = layer.apply(variables, x[:80], mutable=["state"])
    rest, _ = layer.apply({**variables, **state}, x[80:], mutable=["state"])
    np.testing.assert_array_equal(np.asarray(jnp.concatenate([first, rest])), np.asarray(spikes))


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        return Homeostatic(LIF(tau=3.0), target=0.1)(nn.Dense(4)(x))


def test_a_homeostatic_layer_composes_with_flax_layers():
    x = jnp.ones((10, 2, 3))
    variables = Net().init(jax.random.key(0), x)
    assert Net().apply(variables, x).shape == (10, 2, 4)
