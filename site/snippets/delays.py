import jax
import jax.numpy as jnp
import optax

from sparx.nn import LI, DelayedDense

x = jnp.zeros((80, 1, 3)).at[jnp.array([5, 18, 30]), 0, jnp.arange(3)].set(1.0)  # A, B, C fire once
layer, readout = DelayedDense(1, max_delay=45, use_bias=False), LI(tau=4.0)
kernel = jnp.full((3, 1), 0.6)
delay = jnp.array([[4.0], [16.0], [22.0]])


def loss(delay, sigma):                      # minus the readout at step 50
    y = layer.apply({"params": {"kernel": kernel, "delay": delay}}, x, sigma)
    return -readout.apply({}, y)[50, 0, 0]


adam = optax.adam(0.6)
state = adam.init(delay)
grad = jax.jit(jax.grad(loss))
for sigma in jnp.geomspace(8.0, 0.5, 260):   # the Gaussians narrow as they train
    updates, state = adam.update(grad(delay, sigma), state)
    delay = jnp.clip(optax.apply_updates(delay, updates), 0, 45)
print(delay[:, 0].round(), -loss(delay.round(), 0))   # deployed: each delay rounded
