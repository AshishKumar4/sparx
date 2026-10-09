import jax.numpy as jnp

import sparx
from sparx.dynamics import SynapticInput, izhikevich_2003

# a, b, c and d of one class of the 2003 paper
cell = izhikevich_2003("chattering")
# 400 ms at 0.1 ms, the current switched on after 20 ms
current = jnp.where(jnp.arange(4000) > 200, 10.0, 0.0)
(spikes, v), _ = sparx.run(cell, SynapticInput(current), dt=0.1,
                           record=lambda s: s.v)
print(int(spikes.value.sum()), "spikes")
