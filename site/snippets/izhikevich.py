import jax.numpy as jnp

import sparx
from sparx.dynamics import SynapticInput, izhikevich_2003

cell = izhikevich_2003("chattering")            # a, b, c, d of a class of the 2003 paper
current = jnp.where(jnp.arange(4000) > 200, 10.0, 0.0)   # 400 ms at 0.1 ms
(spikes, v), _ = sparx.run(cell, SynapticInput(current), dt=0.1, record=lambda s: s.v)
print(int(spikes.value.sum()), "spikes")
