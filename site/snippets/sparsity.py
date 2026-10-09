import flax.linen as nn
import jax

import sparx


class Net(nn.Module):
    @nn.compact
    def __call__(self, spikes):                                  # [T, B, 784]
        x = sparx.nn.LIF(tau=2.0)(nn.Dense(256)(spikes))
        return sparx.nn.LI(tau=2.0)(nn.Dense(10)(x))


images = jax.random.uniform(jax.random.key(0), (32, 784)) * 0.25   # dim images, mean 0.125
spikes = sparx.encode.RateEncoder(steps=8)(jax.random.key(1), images)
net = Net()
params = net.init(jax.random.key(2), spikes)
_, sown = net.apply(params, spikes, mutable=["spike_rates"])

steps = spikes.shape[0]
per_synapse = {
    "input -> hidden": float(spikes.mean()) * steps,                     # every input spike crosses 256 synapses
    "hidden -> output": float(sparx.firing_rates(sown)["LIF_0"]) * steps,
}
print(per_synapse)    # spikes per synapse per inference; compare with the break-even above
