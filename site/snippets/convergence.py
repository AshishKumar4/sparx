from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np

import sparx
from sparx.dynamics import (
    Arrivals,
    Exponential,
    LeakyIntegrateAndFire,
    PointNeuron,
    Receptor,
)

jax.config.update("jax_enable_x64", True)
ms = np.arange(1, 201)                           # 200 ms
weights = {"ampa": 6.0 * (ms % 3 == 0),          # nS, every 3 ms
           "gaba_a": 20.0 * (ms % 7 == 0)}       # every 7 ms


def membrane(dt, hold):
    """The membrane at the end of each ms; spikes land on ms edges."""
    cell = PointNeuron(
        LeakyIntegrateAndFire(v_th=jnp.inf),
        {"ampa": Receptor(Exponential(2.0), "conductance"),
         "gaba_a": Receptor(Exponential(5.0), "conductance")},
        hold=hold)
    per = round(1 / dt)
    spikes = {}
    for name, w in weights.items():
        spikes[name] = np.zeros(200 * per)
        spikes[name][per - 1::per] = w
    (_, v), _ = sparx.run(cell, Arrivals(0.0, spikes), dt=dt,
                          record=lambda s: s.neuron.v)
    return np.asarray(v)[per - 1::per]


truth = membrane(1 / 1024, "mean")
for hold in ("start", "mean"):
    errors = [np.abs(membrane(dt, hold) - truth).max()
              for dt in (1 / 2, 1 / 4, 1 / 8, 1 / 16)]
    ratios = [a / b for a, b in pairwise(errors)]
    print(f"hold={hold}:", ", ".join(f"{e:.1e}" for e in errors),
          "mV; each halving divides it by",
          ", ".join(f"{r:.1f}" for r in ratios))
