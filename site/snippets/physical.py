import jax.numpy as jnp

import sparx
from sparx.dynamics import LeakyIntegrateAndFire, SynapticInput

cell = LeakyIntegrateAndFire(tau_m=20.0, c_m=200.0, e_l=-60.0, v_th=-50.0,
                             v_reset=-60.0, t_ref=5.0)  # ms, pF, mV
current = jnp.full((3000,), 260.0)                      # pA, 300 ms at 0.1 ms
(spikes, v), _ = sparx.run(cell, SynapticInput(current), dt=0.1, record=lambda s: s.v)
print(int(spikes.value.sum()), "spikes in 300 ms")
