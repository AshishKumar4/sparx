import flax.linen as nn
import jax.numpy as jnp

import sparx


class Net(nn.Module):
    @nn.compact
    def __call__(self, spikes):                  # [T, B, 3]
        current = nn.Dense(1, use_bias=False)(spikes)
        return sparx.nn.Synaptic(tau=10.0, tau_synapse=5.0)(current)


times, which = jnp.array([190, 192]), jnp.array([0, 1])
spikes_in = jnp.zeros((300, 1, 3)).at[times, 0, which].set(1.0)
kernel = jnp.array([[0.2], [0.2], [-0.3]])
params = {"params": {"Dense_0": {"kernel": kernel}}}
out = Net().apply(params, spikes_in)
print(jnp.flatnonzero(out[:, 0, 0]))             # [194]
