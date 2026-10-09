import jax
import jax.numpy as jnp
import optax

from sparx.nn import LI, DelayedDense

x = jnp.zeros((80, 1, 3)).at[jnp.array([5, 18, 30]), 0, jnp.arange(3)].set(1.0)  # A, B, C fire once
layer, readout = DelayedDense(1, max_delay=45, use_bias=False), LI(tau=4.0)
params = layer.init(jax.random.key(0), x, 8.0)["params"]


def loss(params, sigma):                     # minus the readout at step 50
    return -readout.apply({}, layer.apply({"params": params}, x, sigma))[50, 0, 0]


adam = optax.masked(optax.adam(0.6), {"kernel": False, "delay": True})   # learn the delays only
state = adam.init(params)
grad = jax.jit(jax.grad(loss))
for sigma in jnp.geomspace(8.0, 0.5, 260):   # the Gaussians narrow as they train
    updates, state = adam.update(grad(params, sigma), state)
    params = optax.apply_updates(params, updates)
print(params["delay"][:, 0].round(), -loss(params, 0))   # deployed: each delay rounded
