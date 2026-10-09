import jax
import jax.numpy as jnp

from sparx.dynamics import LIFCell, MembraneState, SynapticInput, decay
from sparx.surrogate import ATan

x = 0.12 + 0.2 * jax.random.uniform(jax.random.key(0), (120,))


def last_membrane(x, w, cell):
    """Run a neuron whose own spike feeds back with weight w."""
    def step(carry, xt):
        state, spike = carry
        drive = SynapticInput(jump=xt + w * spike)
        state, out = cell.step(state, drive, 1.0)
        return (state, out.value), None

    start = (MembraneState(jnp.zeros(())), jnp.zeros(()))
    (state, _), _ = jax.lax.scan(step, start, x)
    return state.v


cell = LIFCell(decay=decay(tau=10.0), surrogate=ATan(),
               detach_reset=True)
for w in (0.0, 0.5):
    grad = jax.grad(last_membrane)(x, w, cell)   # d v_T / d x_t
    print(f"w={w}: {float(abs(grad[0])):.1e} at step 0, "
          f"{float(abs(grad[-1])):.1e} at the last step")
