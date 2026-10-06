"""Biophysical models against their ground truth: analytic solutions and fine float64 integrations."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from sparx.dynamics import (
    LIF,
    Alpha,
    Arrivals,
    BiExponential,
    Delta,
    Exponential,
    LIFCell,
    MgBlock,
    PointNeuron,
    Receptor,
    SynapticInput,
    Term,
    run,
)
from sparx.dynamics.core import response


def lif_period(neuron: LIF, current: float) -> float:
    """The analytic interspike interval of a current-driven LIF reset to rest: refractory plus
    the time the membrane takes to climb from E_L to v_th toward E_L + R I."""
    r = neuron.tau_m / neuron.c_m
    target = neuron.e_l + r * current
    return neuron.t_ref + neuron.tau_m * math.log((target - neuron.v_reset) / (target - neuron.v_th))


def test_one_long_step_equals_many_short_ones_below_threshold():
    # The update is the exact solution for inputs constant over the step, so
    # its accuracy does not depend on dt.
    neuron = LIF()
    with jax.enable_x64(new_val=True):
        current = jnp.asarray([10.0, 40.0])  # steady states -51.5 and -59.3 mV
        g = {"ampa": jnp.asarray([2.0, 0.5]), "gaba_a": jnp.asarray([1.0, 3.0])}
        def held(steps):
            return SynapticInput(jnp.broadcast_to(current, (steps, 2)), conductance=
                                 {k: jnp.broadcast_to(v, (steps, 2)) for k, v in g.items()})

        long, long_state = run(neuron, held(1), dt=5.0)
        short, short_state = run(neuron, held(500), dt=0.01)
    assert float(long.value.sum() + short.value.sum()) == 0
    np.testing.assert_allclose(long_state.v, short_state.v, rtol=1e-12)  # observed 1.4e-15 relative


def test_constant_conductances_relax_to_their_analytic_steady_state():
    neuron = LIF()
    g_l = neuron.c_m / neuron.tau_m
    g_e, g_i = 3.0, 6.0
    v_inf = (g_l * neuron.e_l + g_e * 0.0 + g_i * -80.0) / (g_l + g_e + g_i)
    tau = neuron.c_m / (g_l + g_e + g_i)
    with jax.enable_x64(new_val=True):
        _, state = run(neuron, SynapticInput(0.0, conductance={"ampa": jnp.full((300, 1), g_e),
                                                         "gaba_a": jnp.full((300, 1), g_i)}), dt=0.1)
    expected = v_inf + (neuron.e_l - v_inf) * math.exp(-30.0 / tau)
    np.testing.assert_allclose(state.v[0], expected, rtol=1e-12)  # observed 2.5e-16 relative
    assert v_inf < neuron.v_th  # the test never fires


@pytest.mark.parametrize("current", [250.0, 400.0, 900.0])
def test_firing_period_is_the_analytic_one_within_a_step(current):
    neuron = LIF()
    dt = 0.01
    with jax.enable_x64(new_val=True):
        spikes, _ = run(neuron, SynapticInput(jnp.full((40_000, 1), current)), dt=dt)
    steps = np.flatnonzero(np.asarray(spikes.value[:, 0]))
    assert len(steps) > 5
    periods = np.diff(steps) * dt
    expected = lif_period(neuron, current)
    # Refractoriness counts whole steps and a spike is seen at the first step
    # boundary past the crossing, so a period exceeds the analytic one by
    # less than one step.
    assert np.all(periods >= expected - 1e-9) and np.all(periods < expected + dt)


def test_in_step_spike_times_are_closer_than_the_grid():
    neuron = LIF()
    dt = 0.5
    current = 400.0
    with jax.enable_x64(new_val=True):
        spikes, _ = run(neuron, SynapticInput(jnp.full((200, 1), current)), dt=dt)
    first = int(np.flatnonzero(np.asarray(spikes.value[:, 0]))[0])
    on_grid = (first + 1) * dt
    precise = (first + float(spikes.offset[first, 0])) * dt
    exact = lif_period(neuron, current) - neuron.t_ref  # from rest, no refractory before the first
    assert abs(precise - exact) < 0.1 * abs(on_grid - exact) + 1e-6


def test_refractoriness_holds_for_t_ref_over_dt_steps():
    neuron = LIF(t_ref=2.0)
    dt = 0.1
    spikes, _ = run(neuron, SynapticInput(jnp.full((400, 1), 5000.0)), dt=dt)
    steps = np.flatnonzero(np.asarray(spikes.value[:, 0]))
    # The membrane is held at reset for round(t_ref / dt) = 20 steps, then
    # climbs from reset to threshold toward E_L + R I, which takes
    # tau ln((target - v_reset) / (target - v_th)) = 0.40 ms: 5 more steps.
    target = neuron.e_l + neuron.tau_m / neuron.c_m * 5000.0
    climb = neuron.tau_m * math.log((target - neuron.v_reset) / (target - neuron.v_th))
    assert set(np.diff(steps).tolist()) == {round(neuron.t_ref / dt) + math.ceil(climb / dt)}


def test_the_surrogate_passes_gradients_to_the_input_current():
    neuron = LIF()

    def rate(current):
        spikes, _ = run(neuron, SynapticInput(jnp.full((2000, 1), current)), dt=0.1)
        return jnp.sum(spikes.value)

    assert float(jax.grad(rate)(400.0)) > 0


def _impulse(synapse, steps=4000, dt=0.01, weight=3.0):
    """A synapse's output over time after one arrival of `weight` at t = 0."""
    with jax.enable_x64(new_val=True):
        state = synapse.step(synapse.init_state((1,), jnp.float64), jnp.asarray([weight]), dt)

        def step(state, _):
            value = sum(term.amplitude for term in synapse.output(state))[0]
            return synapse.step(state, jnp.zeros(1), dt), value

        _, values = jax.lax.scan(step, state, length=steps)
    return np.asarray(values), dt


@pytest.mark.parametrize("synapse", [Alpha(2.0), BiExponential(0.5, 5.0), BiExponential(1.0, 1.5)])
def test_peaked_synapses_peak_at_the_weight(synapse):
    values, dt = _impulse(synapse)
    assert abs(values.max() - 3.0) < 3e-5  # sampled every 0.01 ms, so the peak falls between samples
    if isinstance(synapse, Alpha):
        assert abs(np.argmax(values) * dt - 2.0) <= dt


def test_exponential_synapse_jumps_by_the_weight_and_decays():
    values, dt = _impulse(Exponential(4.0))
    # Observed 1.6e-13 relative.
    np.testing.assert_allclose(values, 3.0 * np.exp(-np.arange(len(values)) * dt / 4.0), rtol=1e-12)


def test_bi_exponential_needs_a_rise_faster_than_its_decay():
    with pytest.raises(ValueError, match="Alpha"):
        BiExponential(5.0, 5.0)


@pytest.mark.parametrize(("tau_s", "tau_m"),
                         [(2.0, 10.0), (10.0, 10.0), (10.0, 9.999), (5.0, 0.5), (0.05, 20.0)])
@pytest.mark.parametrize("dt", [0.1, 1.0])
def test_the_response_integral_is_exact_at_equal_and_distant_time_constants(tau_s, tau_m, dt):
    def current(s):
        return (1.7 - 0.6 * s) * math.exp(-s / tau_s)

    with jax.enable_x64(new_val=True):
        term = Term(jnp.asarray(1.7), jnp.asarray(-0.6), tau_s)
        exact = float(response(term, tau_m, dt))
        mean = float(term.mean(dt))
    expected, _ = quad(lambda s: current(s) * math.exp(-(dt - s) / tau_m), 0, dt, epsabs=0, epsrel=1e-13)
    np.testing.assert_allclose(exact, expected, rtol=1e-12)  # observed 2.1e-16 relative
    # Observed 1.7e-16 relative.
    np.testing.assert_allclose(mean, quad(current, 0, dt, epsabs=0, epsrel=1e-13)[0] / dt, rtol=1e-12)


def test_mg_block_is_jahr_and_stevens():
    block = MgBlock()
    # Observed 1.8e-8 relative.
    np.testing.assert_allclose(float(block(jnp.asarray(0.0))), 1 / (1 + 1 / 3.57), rtol=1e-6)
    assert float(block(jnp.asarray(-80.0))) < 0.05 < 0.5 < float(block(jnp.asarray(-10.0)))
    assert float(MgBlock(mg=0.0)(jnp.asarray(-80.0))) == 1.0


def test_nmda_conductance_is_scaled_by_the_block_at_the_start_of_the_step():
    neuron = LIF()
    with jax.enable_x64(new_val=True):
        state = neuron.init_state((1,), jnp.float64)
        nmda, _ = neuron.step(state, SynapticInput(conductance={"nmda": jnp.asarray([4.0])}), 0.1)
        g = 4.0 * float(MgBlock()(jnp.asarray(neuron.e_l)))
        ampa, _ = neuron.step(state, SynapticInput(conductance={"ampa": jnp.asarray([g])}), 0.1)
    np.testing.assert_allclose(nmda.v, ampa.v, rtol=1e-12)  # observed 0 relative


@pytest.mark.parametrize(("reset", "after_spike"), [("zero", 0.0), ("subtract", 0.75)])
def test_a_jump_after_the_threshold_is_lost_only_to_a_reset_that_sets_the_membrane(reset, after_spike):
    # A jump of 1.5 before the threshold fires the neuron; 0.25 lands after
    # the test. A zero reset sets the membrane and overwrites it; a
    # subtraction keeps it (1.5 - 1 + 0.25). A neuron that did not fire keeps it.
    neuron = PointNeuron(LIFCell(1.0, reset=reset), {"now": Receptor(Delta()),
                                                     "late": Receptor(Delta(after_threshold=True))})
    for now, fired, v in [(1.5, 1.0, after_spike), (0.5, 0.0, 0.75)]:
        arrivals = Arrivals(spikes={"now": jnp.array([now]), "late": jnp.array([0.25])})
        state, spikes = neuron.step(neuron.init_state((1,), jnp.float32), arrivals, 1.0)
        assert float(spikes.value[0]) == fired
        assert float(state.neuron.v[0]) == v
