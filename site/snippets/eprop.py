import jax
import jax.numpy as jnp

from sparx.dynamics import LIFCell, decay
from sparx.learn import EPropParams, bptt_loss, eprop
from sparx.surrogate import Triangle

keys = jax.random.split(jax.random.key(0), 4)
T, inputs, size = 200, 20, 50
u = (jax.random.uniform(keys[0], (T, 1, inputs)) < 0.05) * 1.0
t = jnp.arange(T)
target = jnp.sin(2 * jnp.pi * t / 100)[:, None, None]
params = EPropParams(jax.random.normal(keys[1], (inputs, size)),
                     0.15 * jax.random.normal(keys[2], (size, size)),
                     0.05 * jax.random.normal(keys[3], (size, 1)),
                     jnp.zeros(1))
cell = LIFCell(decay(tau=20.0), surrogate=Triangle(scale=0.3),
               detach_reset=True)


def loss(y, target):
    return 0.5 * jnp.sum((y - target) ** 2)


_, online = eprop(cell, params, u, target, loss, tau=10.0)


def through_time(p, cut):
    return bptt_loss(cell, p, u, target, loss, tau=10.0,
                     cut_recurrence=cut)


full = jax.grad(through_time)(params, False)
cut = jax.grad(through_time)(params, True)


def cosine(a, b):
    a = jnp.concatenate([x.ravel() for x in a])
    b = jnp.concatenate([x.ravel() for x in b])
    return float(a @ b / jnp.linalg.norm(a) / jnp.linalg.norm(b))


print("e-prop vs BPTT:", cosine(online, full))
print("e-prop vs BPTT with the recurrence cut:", cosine(online, cut))
