import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

import sparx


class Net(nn.Module):
    @nn.compact
    def __call__(self, spikes):                         # [T, B, 784]
        x = sparx.nn.LIF(tau=2.0)(nn.Dense(256)(spikes))
        return sparx.nn.LI(tau=2.0)(nn.Dense(10)(x))    # membrane [T, B, 10]


net = Net()
images = jax.random.uniform(jax.random.key(0), (32, 784))
labels = jnp.zeros(32, jnp.int32)
spikes = sparx.encode.RateEncoder(steps=8)(jax.random.key(1), images)
params = net.init(jax.random.key(2), spikes)


def loss(params):
    logits = jnp.mean(net.apply(params, spikes), axis=0)
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


grads = jax.grad(loss)(params)                          # through the spikes, by their surrogate
