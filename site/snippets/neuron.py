import jax.numpy as jnp

import sparx
from sparx.dynamics import LIFCell, decay

cell = LIFCell(decay=decay(tau=12.0), threshold=1.0, reset="subtract")
drive = jnp.full((300,), 0.12)              # the input of each step
spikes, state = sparx.run(cell, drive)      # spikes.value is 1 where it fired
print(int(spikes.value.sum()), "spikes in 300 steps")
