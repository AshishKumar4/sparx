import jax
import jax.numpy as jnp

from sparx.surrogate import ATan, FastSigmoid, spike

v = jnp.array([-1.0, -0.1, 0.0, 0.1, 1.0])       # membrane - threshold
print(spike(v, ATan()))                           # [0. 0. 1. 1. 1.]

# The forward pass is the step; the gradient is the surrogate's.
for surrogate in (ATan(), FastSigmoid(25.0)):
    slope = jax.vmap(jax.grad(lambda x, s=surrogate: spike(x, s)))(v)
    print(type(surrogate).__name__, slope)
