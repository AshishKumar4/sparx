"""Biophysical models against their ground truth: analytic solutions and fine float64 integrations."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.dynamics import LIF, SynapticInput, integrate


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
            return SynapticInput(jnp.broadcast_to(current, (steps, 2)),
                                 {k: jnp.broadcast_to(v, (steps, 2)) for k, v in g.items()})

        long, long_state = integrate(neuron, held(1), 5.0)
        short, short_state = integrate(neuron, held(500), 0.01)
    assert float(long.fired.sum() + short.fired.sum()) == 0
    np.testing.assert_allclose(long_state.v, short_state.v, rtol=1e-12)


def test_constant_conductances_relax_to_their_analytic_steady_state():
    neuron = LIF()
    g_l = neuron.c_m / neuron.tau_m
    g_e, g_i = 3.0, 6.0
    v_inf = (g_l * neuron.e_l + g_e * 0.0 + g_i * -80.0) / (g_l + g_e + g_i)
    tau = neuron.c_m / (g_l + g_e + g_i)
    with jax.enable_x64(new_val=True):
        _, state = integrate(neuron, SynapticInput(0.0, {"ampa": jnp.full((300, 1), g_e),
                                                         "gaba_a": jnp.full((300, 1), g_i)}), 0.1)
    expected = v_inf + (neuron.e_l - v_inf) * math.exp(-30.0 / tau)
    np.testing.assert_allclose(state.v[0], expected, rtol=1e-12)
    assert v_inf < neuron.v_th  # the test never fires


@pytest.mark.parametrize("current", [250.0, 400.0, 900.0])
def test_firing_period_is_the_analytic_one_within_a_step(current):
    neuron = LIF()
    dt = 0.01
    with jax.enable_x64(new_val=True):
        spikes, _ = integrate(neuron, SynapticInput(jnp.full((40_000, 1), current)), dt)
    steps = np.flatnonzero(np.asarray(spikes.fired[:, 0]))
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
        spikes, _ = integrate(neuron, SynapticInput(jnp.full((200, 1), current)), dt)
    first = int(np.flatnonzero(np.asarray(spikes.fired[:, 0]))[0])
    on_grid = (first + 1) * dt
    precise = (first + float(spikes.offset[first, 0])) * dt
    exact = lif_period(neuron, current) - neuron.t_ref  # from rest, no refractory before the first
    assert abs(precise - exact) < 0.1 * abs(on_grid - exact) + 1e-6


def test_refractoriness_holds_for_t_ref_over_dt_steps():
    neuron = LIF(t_ref=2.0)
    dt = 0.1
    spikes, _ = integrate(neuron, SynapticInput(jnp.full((400, 1), 5000.0)), dt)
    steps = np.flatnonzero(np.asarray(spikes.fired[:, 0]))
    # The membrane is held at reset for round(t_ref / dt) = 20 steps, then
    # climbs from reset to threshold toward E_L + R I, which takes
    # tau ln((target - v_reset) / (target - v_th)) = 0.40 ms: 5 more steps.
    target = neuron.e_l + neuron.tau_m / neuron.c_m * 5000.0
    climb = neuron.tau_m * math.log((target - neuron.v_reset) / (target - neuron.v_th))
    assert set(np.diff(steps).tolist()) == {round(neuron.t_ref / dt) + math.ceil(climb / dt)}


def test_the_surrogate_passes_gradients_to_the_input_current():
    neuron = LIF()

    def rate(current):
        spikes, _ = integrate(neuron, SynapticInput(jnp.full((2000, 1), current)), 0.1)
        return jnp.sum(spikes.fired)

    assert float(jax.grad(rate)(400.0)) > 0
