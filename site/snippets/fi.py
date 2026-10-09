import jax
import jax.numpy as jnp

import sparx
from sparx.dynamics import LeakyIntegrateAndFire, SynapticInput

jax.config.update("jax_enable_x64", True)
# 20 ms, 200 pF: R = 100 MOhm. Threshold 10 mV above rest, 5 ms
# refractory, reset to rest.
cell = LeakyIntegrateAndFire(tau_m=20.0, c_m=200.0, e_l=-60.0,
                             v_th=-50.0, v_reset=-60.0, t_ref=5.0)
dt, steps = 0.1, 20_000                    # 2 s


def rate(pA):
    """Spikes per second over the last second of a constant current."""
    out, _ = sparx.run(cell, SynapticInput(jnp.full((steps,), pA)),
                       dt=dt)
    return out.value[steps // 2:].sum()    # one second


currents = jnp.arange(120.0, 620.0, 100.0)
measured = jax.vmap(rate)(currents)

RI = currents * 0.1                        # mV, with R = 0.1 mV/pA
formula = 1000 / (5.0 + 20.0 * jnp.log(RI / (RI - 10.0)))
for i, m, f in zip(currents, measured, formula, strict=True):
    print(f"{int(i)} pA: {int(m)} Hz stepped, {float(f):.1f} Hz exact")
