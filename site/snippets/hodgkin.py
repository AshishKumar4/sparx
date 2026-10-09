import jax
import jax.numpy as jnp

import sparx
from sparx.dynamics import HodgkinHuxley, SynapticInput

jax.config.update("jax_enable_x64", True)
cell = HodgkinHuxley()                   # NEST's hh_psc_alpha, 100 pF
dt, steps = 0.1, 20_000                  # 2 s


def rate(pA):
    """Spikes per second over the last second of a constant current."""
    out, _ = sparx.run(cell, SynapticInput(jnp.full((steps,), pA)),
                       dt=dt)
    return out.value[steps // 2:].sum()


currents = jnp.arange(0.0, 1501.0, 50.0)
for i, r in zip(currents, jax.vmap(rate)(currents), strict=True):
    print(f"{int(i):5d} pA: {int(r):3d} Hz")
