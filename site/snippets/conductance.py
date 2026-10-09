import jax
import jax.numpy as jnp

import sparx
from sparx.dynamics import (
    Arrivals,
    Exponential,
    LeakyIntegrateAndFire,
    PointNeuron,
    Receptor,
)

jax.config.update("jax_enable_x64", True)
dt, steps, arrive = 0.1, 5000, 4000      # one input at 400 ms


def psp(receptor, kind, weight, held):
    """The input's peak effect (mV) on a membrane held at `held`."""
    cell = PointNeuron(LeakyIntegrateAndFire(v_th=jnp.inf),  # no spikes
                       {receptor: Receptor(Exponential(5.0), kind)})
    current = jnp.full((steps,), 10.0 * (held + 60.0))  # g_L is 10 nS
    spikes = jnp.zeros(steps).at[arrive].set(weight)
    (_, v), _ = sparx.run(cell, Arrivals(current, {receptor: spikes}),
                          dt=dt, record=lambda s: s.neuron.v)
    change = v[arrive:] - v[arrive - 1]
    return float(change[jnp.argmax(jnp.abs(change))])


# 3 nS of AMPA at rest, 60 mV from its reversal, passes 180 pA.
for held in (-80.0, -60.0, -40.0, -20.0, 0.0):
    current = psp("ex", "current", 180.0, held)
    ampa = psp("ampa", "conductance", 3.0, held)
    gaba = psp("gaba_a", "conductance", 3.0, held)
    print(f"held at {held:3.0f} mV: 180 pA {current:+.2f} mV,"
          f" AMPA {ampa:+.2f} mV, GABA-A {gaba:+.2f} mV")
