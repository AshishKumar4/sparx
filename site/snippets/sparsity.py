import flax.linen as nn
import jax

import sparx


class Net(nn.Module):
    @nn.compact
    def __call__(self, spikes):                  # [T, B, 784]
        x = sparx.nn.LIF(tau=2.0)(nn.Dense(256)(spikes))
        return sparx.nn.LI(tau=2.0)(nn.Dense(10)(x))


# Dim images, mean intensity 0.125, as 8 steps of spikes
images = jax.random.uniform(jax.random.key(0), (32, 784)) * 0.25
spikes = sparx.encode.RateEncoder(steps=8)(jax.random.key(1), images)
net = Net()
params = net.init(jax.random.key(2), spikes)
_, sown = net.apply(params, spikes, mutable=["spike_rates"])

# Spikes per synapse per inference: each input spike crosses 256
# synapses, each hidden spike 10.
steps = spikes.shape[0]
rate = sparx.firing_rates(sown)["LIF_0"]
print({"input to hidden": float(spikes.mean()) * steps,
       "hidden to output": float(rate) * steps})
