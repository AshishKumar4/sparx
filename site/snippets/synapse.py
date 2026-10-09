import jax.numpy as jnp

import sparx
from sparx.dynamics import LICell, LIFCell, Serial, decay

# A synaptic current that decays in 5 steps, charging a membrane that decays in 10.
neuron = Serial(LICell(decay(tau=5.0)), LIFCell(decay(tau=10.0), threshold=1.0))
weights = jnp.array([0.2, 0.2, -0.3])                 # A and B excite, C inhibits

spikes_in = jnp.zeros((300, 3))                        # 300 steps of 1 ms, three inputs
spikes_in = spikes_in.at[jnp.array([20, 90, 110, 190, 192]), jnp.array([0, 0, 1, 0, 1])].set(1.0)

(out, v), _ = sparx.run(neuron, spikes_in @ weights, record=lambda state: state[1].v)
print(jnp.flatnonzero(out.value))                      # [194]: only A and B together fire it
