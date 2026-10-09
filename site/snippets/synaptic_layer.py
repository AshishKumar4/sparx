import flax.linen as nn
import jax.numpy as jnp

import sparx


class Net(nn.Module):
    @nn.compact
    def __call__(self, spikes):                         # [T, B, 3]
        current = nn.Dense(1, use_bias=False)(spikes)    # one weight per input
        return sparx.nn.Synaptic(tau=10.0, tau_synapse=5.0)(current)


spikes_in = jnp.zeros((300, 1, 3)).at[jnp.array([190, 192]), 0, jnp.array([0, 1])].set(1.0)
params = {"params": {"Dense_0": {"kernel": jnp.array([[0.2], [0.2], [-0.3]])}}}
print(jnp.flatnonzero(Net().apply(params, spikes_in)[:, 0, 0]))   # [194]
