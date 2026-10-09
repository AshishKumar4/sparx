import jax
import jax.numpy as jnp
import optax

import sparx
from sparx.dynamics import LIFCell, decay
from sparx.surrogate import ATan

T, N = 200, 40
inputs = jax.random.uniform(jax.random.key(0), (T, N)) < 0.04
trains = inputs.astype(jnp.float32)
# Fire at these steps
target = jnp.zeros(T).at[jnp.array([40, 90, 150])].set(1.0)
cell = LIFCell(decay=decay(tau=10.0), threshold=1.0,
               surrogate=ATan())
keep = decay(tau=10.0)              # an exponential filter of 10 steps


def smooth(spikes):
    return jax.lax.scan(lambda f, s: (keep * f + s,) * 2, 0.0, spikes)[1]


def loss(w):
    out, _ = sparx.run(cell, trains @ w)     # spikes exactly 0 or 1
    return jnp.mean((smooth(out.value) - smooth(target)) ** 2)


w = 0.35 * jax.random.normal(jax.random.key(1), (N,))
adam = optax.adam(0.04)
state = adam.init(w)
grad = jax.jit(jax.grad(loss))       # the slope comes from ATan
for _ in range(400):
    updates, state = adam.update(grad(w), state)
    w = optax.apply_updates(w, updates)
fired = sparx.run(cell, trains @ w)[0].value
print("fires at", jnp.flatnonzero(fired))
