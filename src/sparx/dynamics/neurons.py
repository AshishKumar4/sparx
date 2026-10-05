"""Neuron models in physical units (ms, mV, pA, nS, pF; see `sparx.dynamics.core`).

Every model here has a refractory period after a spike, measured on the step
grid as NEST's `iaf_*` models count it: a neuron that fires holds its
reset voltage for `t_ref` ms, `round(t_ref / dt)` steps. Spikes pass
gradients through a surrogate, as in `sparx.cells`, and the reset sets the
voltage exactly while its gradient follows the spike.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from sparx.dynamics.core import Spikes, SynapticInput, crossing, exact_linear, membrane_dtype
from sparx.surrogate import ATan, Surrogate, spike

__all__ = ["LIF", "RECEPTORS", "LIFState"]

RECEPTORS: Mapping[str, float] = {"ampa": 0.0, "nmda": 0.0, "gaba_a": -80.0, "gaba_b": -95.0}
"""Reversal potentials (mV) of the common receptors, the defaults models read
conductances against: AMPA and NMDA 0 mV, GABA-A -80 mV (Cl-), GABA-B
-95 mV (K+), as in Brette et al.'s simulator benchmarks (2007) and the
cortical models built on them."""


def _reset(v: jax.Array, fired: jax.Array, to: jax.Array | float) -> jax.Array:
    """`to` where fired, exactly, with the gradient of `v + fired * (to - v)`."""
    return jnp.where(fired > 0, to, v) + (fired - jax.lax.stop_gradient(fired)) * (to - v)


class LIFState(NamedTuple):
    v: jax.Array
    refractory: jax.Array
    """Milliseconds of refractoriness left."""


@struct.dataclass
class LIF:
    """Leaky integrate-and-fire with current and conductance input.

        C dv/dt = -g_L (v - E_L) + sum_k g_k (E_k - v) + I,    g_L = C / tau_m

    At `v >= v_th` the neuron fires, `v` is set to `v_reset` and held there
    for `t_ref`. With the inputs constant over a step the equation is linear
    in `v`, and the update is its exact solution: `v` relaxes toward
    `(g_L E_L + sum g_k E_k + I) / (g_L + sum g_k)` with time constant
    `C / (g_L + sum g_k)`. Conductances are read against `reversal`, by
    receptor name. The defaults are the cortical cell of Brette et al.'s
    benchmarks (2007): 20 ms, 200 pF, rest -60 mV, threshold -50 mV, reset
    -60 mV, 5 ms refractory.
    """

    tau_m: jax.Array | float = 20.0
    c_m: jax.Array | float = 200.0
    e_l: jax.Array | float = -60.0
    v_th: jax.Array | float = -50.0
    v_reset: jax.Array | float = -60.0
    t_ref: float = struct.field(pytree_node=False, default=5.0)
    reversal: Mapping[str, float] = struct.field(pytree_node=False, default_factory=lambda: dict(RECEPTORS))
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> LIFState:
        dtype = membrane_dtype(dtype)
        return LIFState(jnp.full(shape, self.e_l, dtype), jnp.zeros(shape, dtype))

    def step(self, state: LIFState, inputs: SynapticInput, dt: float) -> tuple[LIFState, Spikes]:
        g_l = self.c_m / self.tau_m
        g_total = g_l + sum((g for g in inputs.conductance.values()), jnp.zeros(()))
        drive = g_l * self.e_l + inputs.current + sum(
            (g * self.reversal[name] for name, g in inputs.conductance.items()), jnp.zeros(()))
        integrated = exact_linear(state.v, drive / g_total, self.c_m / g_total, dt)
        held = state.refractory > dt / 2
        v = jnp.where(held, self.v_reset, integrated)
        fired = spike(v - self.v_th, self.surrogate) * (1 - held)
        offset = jnp.where(fired > 0, crossing(state.v, v, self.v_th), 1.0)
        refractory = jnp.where(fired > 0, self.t_ref, jnp.maximum(state.refractory - dt, 0))
        dtype = state.v.dtype
        return (LIFState(_reset(v, fired, self.v_reset).astype(dtype), refractory.astype(dtype)),
                Spikes(fired, offset))
