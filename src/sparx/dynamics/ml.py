"""The dimensionless neuron models deep spiking networks train with.

Each model meets the contract of `sparx.dynamics.core`,

    model.init_state(shape, dtype) -> state
    model.step(state, SynapticInput, dt) -> (state, Spikes)

and `sparx.dynamics.run` scans one over time. Time is counted in steps:
a decay is what is left after one unit of time, a step of `dt` decays by
`decay ** dt`, and `dt = 1`, the deep-learning convention, applies the
decay as it is, with `decay = exp(-1 / tau)` for a time constant of `tau`
steps (`sparx.dynamics.decay`):

    v[t] = decay * v[t-1] + x[t]
    s[t] = H(v[t] - threshold)
    v[t] <- reset(v[t], s[t])

The input `x` is the step's `SynapticInput.jump`, a jump of the membrane
that lands before the threshold test, so the membrane crosses at the end of
the step and every spike's `offset` is 1. A dimensionless membrane has no
capacitance or reversal potentials, so these models refuse currents and
conductances. The jump enters unscaled (snnTorch's `Leaky`);
SpikingJelly's `LIFNode` scales it by `1 - decay`, which a preceding linear
layer absorbs. The reset reads the spike of the same step.

A model is a Flax struct dataclass. Its numerical constants (decays,
thresholds, weights) are pytree leaves, so they can be traced,
differentiated and sharded; its choices (reset rule, surrogate) are static
fields. The membrane runs in float32 (float64 when the input is float64)
whatever the input's dtype (`membrane_dtype`); spikes come back in the
input's dtype, which holds 0 and 1 exactly. `sparx.nn` builds these models
from module attributes and parameters; pure JAX code can use them directly.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from sparx.dynamics.core import NeuronModel, Reset, Spikes, SynapticInput, fire, membrane_dtype
from sparx.surrogate import ATan, Surrogate

__all__ = [
    "ALIFCell",
    "ALIFState",
    "LICell",
    "LIFCell",
    "MembraneState",
    "RecurrentCell",
    "RecurrentState",
    "Serial",
]


def _jump(inputs: SynapticInput) -> jax.Array:
    """The step's input to a dimensionless membrane, its jump; anything else is refused."""
    held = inputs.current
    if inputs.currents or inputs.conductance or not (isinstance(held, float | int) and held == 0):
        raise ValueError("a dimensionless model takes its input as a jump (SynapticInput.jump), "
                         "not as currents or conductances")
    return jnp.asarray(inputs.jump)


def _at_end(fired: jax.Array) -> Spikes:
    return Spikes(fired, jnp.ones_like(fired))


class MembraneState(NamedTuple):
    v: jax.Array


@struct.dataclass
class LIFCell:
    """Leaky integrate-and-fire. `decay=1` integrates without leak (IF)."""

    decay: jax.Array | float
    threshold: jax.Array | float = 1.0
    reset: Reset = struct.field(pytree_node=False, default="subtract")
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())
    detach_reset: bool = struct.field(pytree_node=False, default=False)

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> MembraneState:
        return MembraneState(jnp.zeros(shape, membrane_dtype(dtype)))

    def step(self, state: MembraneState, inputs: SynapticInput, dt: float) -> tuple[MembraneState, Spikes]:
        x = _jump(inputs)
        v = self.decay ** dt * state.v + x
        v, s = fire(v, self.threshold, self.surrogate, self.reset, detach_reset=self.detach_reset)
        return MembraneState(v), _at_end(s.astype(x.dtype))


@struct.dataclass
class LICell:
    """A leaky integrator that never fires; it reports its membrane as `Spikes.fired`.

    The usual readout of a spiking classifier: the last layer integrates the
    spikes it receives and the loss reads its membrane. As the first model
    of a `Serial` it is a synapse's decaying current.
    """

    decay: jax.Array | float

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> MembraneState:
        return MembraneState(jnp.zeros(shape, membrane_dtype(dtype)))

    def step(self, state: MembraneState, inputs: SynapticInput, dt: float) -> tuple[MembraneState, Spikes]:
        v = self.decay ** dt * state.v + _jump(inputs)
        return MembraneState(v), _at_end(v)


@struct.dataclass
class Serial[First, Second]:
    """Two models in series: each step, the first's `Spikes.fired` is the second's input jump.

    `Serial(LICell(synapse_decay), LIFCell(decay))` is the current-based
    LIF, whose input charges a decaying synaptic current that the membrane
    integrates,

        i[t] = synapse_decay * i[t-1] + x[t]
        v[t] = decay * v[t-1] + i[t]

    snnTorch's `Synaptic` and the CUBA neurons of Zenke and Vogels (2021).
    Longer chains nest. The spikes are the second model's, in the dtype it
    gives them. In physical units the counterpart is a `PointNeuron` with an
    `Exponential` synapse, where an arrival shapes the membrane from the
    next step on.
    """

    first: NeuronModel[First]
    second: NeuronModel[Second]

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> tuple[First, Second]:
        return self.first.init_state(shape, dtype), self.second.init_state(shape, dtype)

    def step(self, state: tuple[First, Second], inputs: SynapticInput,
             dt: float) -> tuple[tuple[First, Second], Spikes]:
        first, out = self.first.step(state[0], inputs, dt)
        second, spikes = self.second.step(state[1], SynapticInput(jump=out.fired), dt)
        return (first, second), spikes


class ALIFState(NamedTuple):
    v: jax.Array
    a: jax.Array
    r: jax.Array
    """Steps of refractoriness left."""


@struct.dataclass
class ALIFCell:
    """LIF with an adaptive threshold that rises with each spike and decays back.

        theta[t] = threshold + beta * a[t-1]
        s[t] = H(v[t] - theta[t])
        v[t] <- v[t] - s[t] * threshold      (a soft reset by the baseline)
        a[t] = adapt_decay * a[t-1] + s[t]

    The adaptive neurons of Bellec et al., "A solution to the learning dilemma
    for recurrent networks of spiking neurons" (Nature Communications 2020),
    whose adaptation time constants of hundreds of steps give a recurrent
    network memory beyond its membranes'. As in their equations and code, a
    spike subtracts the baseline threshold, not the adaptive one. Their reset
    lands one step later, undecayed; the reset here lands at the spike, like
    every model of this family, which makes this their model with a reset of
    `decay * threshold` (docs/fidelity.md). Gradients flow through `a`.

    `refractory` is their `n_refractory` as a duration in the unit of `dt`,
    `round(refractory / dt)` steps: a spike and the silence after it span
    that many steps, during which the membrane integrates but cannot fire,
    and no gradient passes the spike (they use 2 to 5 at 1 ms a step). 0
    and 1 step leave the neuron free to fire on the next step.
    """

    decay: jax.Array | float
    adapt_decay: jax.Array | float
    beta: jax.Array | float = 1.8
    threshold: jax.Array | float = 1.0
    reset: Reset = struct.field(pytree_node=False, default="subtract")
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())
    detach_reset: bool = struct.field(pytree_node=False, default=False)
    refractory: float = struct.field(pytree_node=False, default=0)

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> ALIFState:
        zeros = jnp.zeros(shape, membrane_dtype(dtype))
        return ALIFState(zeros, zeros, zeros)

    def step(self, state: ALIFState, inputs: SynapticInput, dt: float) -> tuple[ALIFState, Spikes]:
        x = _jump(inputs)
        steps = round(self.refractory / dt)
        theta = self.threshold + self.beta * state.a
        if steps > 1:
            # An infinite threshold fires nothing and resets nothing; the
            # surrogates that vanish far from threshold pass no gradient there.
            theta = jnp.where(state.r > 0, jnp.inf, theta)
        v = self.decay ** dt * state.v + x
        v, s = fire(v, theta, self.surrogate, self.reset, subtract=self.threshold,
                    detach_reset=self.detach_reset)
        r = jax.lax.stop_gradient(jnp.clip(state.r + steps * s - 1, 0, max(steps, 0)))
        return ALIFState(v, self.adapt_decay ** dt * state.a + s, r), _at_end(s.astype(x.dtype))


class RecurrentState[State](NamedTuple):
    inner: State
    spikes: jax.Array


@struct.dataclass
class RecurrentCell[State]:
    """Feed a model's spikes back to its own input jump through `weight`, `[F, F]`.

        output[t] = inner.step(x[t] + output[t-1] @ weight)

    Wrapping `ALIFCell` gives the recurrent adaptive network (LSNN) of Bellec
    et al. (2020). The product runs at `precision`, the matrix product
    precision of `jax.lax.dot`.
    """

    inner: NeuronModel[State]
    weight: jax.Array
    precision: jax.lax.Precision | None = struct.field(pytree_node=False, default=None)

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> RecurrentState[State]:
        return RecurrentState(self.inner.init_state(shape, dtype), jnp.zeros(shape, dtype))

    def step(self, state: RecurrentState[State], inputs: SynapticInput,
             dt: float) -> tuple[RecurrentState[State], Spikes]:
        feedback = jnp.matmul(state.spikes.astype(self.weight.dtype), self.weight, precision=self.precision)
        fed = dataclasses.replace(inputs, jump=inputs.jump + feedback)
        inner, spikes = self.inner.step(state.inner, fed, dt)
        # The feedback promotes the step's input, so the spikes are cast back
        # to the dtype the carry started with, the input's.
        fired = spikes.fired.astype(state.spikes.dtype)
        return RecurrentState(inner, fired), Spikes(fired, spikes.offset)
