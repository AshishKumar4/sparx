import jax
import jax.numpy as jnp
import numpy as np

import sparx
from sparx.dynamics import LeakyIntegrateAndFire, SynapticInput

jax.config.update("jax_enable_x64", True)
# 20 ms and 200 pF give a leak of 10 nS; the threshold is out of
# reach, so the membrane only charges and leaks.
cell = LeakyIntegrateAndFire(tau_m=20.0, c_m=200.0, e_l=-60.0,
                             v_th=0.0)


def charge(dt):
    """The membrane over 120 ms with 150 pA from 10 ms to 70 ms."""
    k = jnp.arange(round(120 / dt))
    on = (k >= round(10 / dt)) & (k < round(70 / dt))
    current = jnp.where(on, 150.0, 0.0)
    _, v = sparx.run(cell, SynapticInput(current), dt=dt,
                     record=lambda s: s.v)[0]
    return np.asarray(v)


fine, coarse = charge(0.1), charge(5.0)
# The same instants: every 50th fine step is a coarse step.
print(np.abs(fine[49::50] - coarse).max(), "mV apart")
print(coarse[13], "mV at 70 ms; the exact value is",
      -60 + 15 * (1 - np.exp(-60 / 20)))
