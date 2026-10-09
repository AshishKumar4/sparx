import jax
import jax.numpy as jnp
import optax

from sparx.nn import LI, DelayedDense

# A, B and C fire once each, at steps 5, 18 and 30
x = jnp.zeros((80, 1, 3))
x = x.at[jnp.array([5, 18, 30]), 0, jnp.arange(3)].set(1.0)
layer = DelayedDense(1, max_delay=45, use_bias=False)
readout = LI(tau=4.0)
kernel = jnp.full((3, 1), 0.6)
delay = jnp.array([[4.0], [16.0], [22.0]])


def loss(delay, sigma):              # minus the readout at step 50
    params = {"kernel": kernel, "delay": delay}
    y = layer.apply({"params": params}, x, sigma)
    return -readout.apply({}, y)[50, 0, 0]


adam = optax.adam(0.6)
state = adam.init(delay)
grad = jax.jit(jax.grad(loss))
# The Gaussians narrow as the delays train.
for sigma in jnp.geomspace(8.0, 0.5, 260):
    updates, state = adam.update(grad(delay, sigma), state)
    delay = jnp.clip(optax.apply_updates(delay, updates), 0, 45)

# Deployed, each delay is rounded to a whole step.
print(delay[:, 0].round(), -loss(delay.round(), 0))
