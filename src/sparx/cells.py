"""Neuron dynamics as pure JAX: a cell is its constants, a state and a step.

A cell is a Flax struct dataclass. Its numerical constants (decays,
thresholds, weights) are pytree leaves, so they can be traced, differentiated
and sharded; its choices (reset rule, surrogate) are static fields. Every cell
has

    cell.init_state(shape, dtype) -> state      # the neurons at rest
    cell.step(state, x) -> (state, output)      # one time step

and `run(cell, xs)` scans the step over a time-major input `[T, ...]`,
returning the outputs `[T, ...]` and the final state, which a later call
continues from. `sparx.nn` builds these cells from module attributes and
parameters; pure JAX code can use them directly.

All cells use one discrete-time convention, with `dt = 1` and a decay per step
`decay = exp(-1 / tau)`:

    v[t] = decay * v[t-1] + x[t]
    s[t] = H(v[t] - threshold)
    v[t] <- reset(v[t], s[t])

The input enters unscaled (snnTorch's `Leaky`); SpikingJelly's `LIFNode`
scales it by `1 - decay`, which a preceding linear layer absorbs. The reset
reads the spike of the same step.

The membrane runs in float32 (float64 when the input is float64) whatever the
input's dtype, since a bf16 membrane loses the small inputs it integrates over
long sequences; spikes come back in the input's dtype, which holds 0 and 1
exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from flax import struct

from sparx.surrogate import ATan, Surrogate, spike

__all__ = [
    "ALIFCell",
    "ALIFState",
    "Cell",
    "IzhikevichCell",
    "IzhikevichState",
    "LICell",
    "LIFCell",
    "RecurrentCell",
    "RecurrentState",
    "Reset",
    "SynapticCell",
    "SynapticState",
    "fire",
    "membrane_dtype",
    "run",
]

type Reset = Literal["subtract", "zero", "none"]
"""What a spike does to the membrane: `subtract` the threshold (soft reset,
which keeps the overshoot), set it to `zero` (hard reset), or leave it, `none`."""


class Cell[State](Protocol):
    """One population of neurons, advanced one time step at a time."""

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> State:
        """The neurons at rest, for inputs of per-step shape `shape`."""
        ...

    def step(self, state: State, x: jax.Array) -> tuple[State, jax.Array]:
        """Advance one step on input current `x`; return the new state and the output."""
        ...


def membrane_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """The dtype a membrane integrates in for inputs of `dtype`: float32 or wider."""
    return jnp.promote_types(dtype, jnp.float32)


def fire(v: jax.Array, threshold: jax.Array | float, reset: Reset, surrogate: Surrogate,
         detach_reset: bool) -> tuple[jax.Array, jax.Array]:
    """Spike where `v` reaches `threshold`, then reset; return the membrane and the spikes.

    `detach_reset` stops the gradient through the reset, so the surrogate
    reaches the membrane only through the spike output (Zenke and Vogels,
    "The Remarkable Robustness of Surrogate Gradient Learning", 2021, find it
    trains more reliably).
    """
    s = spike(v - threshold, surrogate)
    r = jax.lax.stop_gradient(s) if detach_reset else s
    if reset == "subtract":
        v = v - r * threshold
    elif reset == "zero":
        v = v * (1 - r)
    elif reset != "none":
        raise ValueError(f"reset must be subtract, zero or none, not {reset!r}")
    return v, s


class LIFState(NamedTuple):
    v: jax.Array


@struct.dataclass
class LIFCell:
    """Leaky integrate-and-fire. `decay=1` integrates without leak (IF)."""

    decay: jax.Array | float
    threshold: jax.Array | float = 1.0
    reset: Reset = struct.field(pytree_node=False, default="subtract")
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())
    detach_reset: bool = struct.field(pytree_node=False, default=False)

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> LIFState:
        return LIFState(jnp.zeros(shape, membrane_dtype(dtype)))

    def step(self, state: LIFState, x: jax.Array) -> tuple[LIFState, jax.Array]:
        v = self.decay * state.v + x
        v, s = fire(v, self.threshold, self.reset, self.surrogate, self.detach_reset)
        return LIFState(v), s.astype(x.dtype)


@struct.dataclass
class LICell:
    """A leaky integrator that never fires; its output is the membrane.

    The usual readout of a spiking classifier: the last layer integrates the
    spikes it receives and the loss reads its membrane.
    """

    decay: jax.Array | float

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> LIFState:
        return LIFState(jnp.zeros(shape, membrane_dtype(dtype)))

    def step(self, state: LIFState, x: jax.Array) -> tuple[LIFState, jax.Array]:
        v = self.decay * state.v + x
        return LIFState(v), v


class SynapticState(NamedTuple):
    i: jax.Array
    v: jax.Array


@struct.dataclass
class SynapticCell:
    """Current-based LIF: the input charges a decaying synaptic current that
    the membrane integrates.

        i[t] = synapse_decay * i[t-1] + x[t]
        v[t] = decay * v[t-1] + i[t]

    snnTorch's `Synaptic` and the CUBA neurons of Zenke and Vogels (2021).
    """

    decay: jax.Array | float
    synapse_decay: jax.Array | float
    threshold: jax.Array | float = 1.0
    reset: Reset = struct.field(pytree_node=False, default="subtract")
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())
    detach_reset: bool = struct.field(pytree_node=False, default=False)

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> SynapticState:
        zeros = jnp.zeros(shape, membrane_dtype(dtype))
        return SynapticState(zeros, zeros)

    def step(self, state: SynapticState, x: jax.Array) -> tuple[SynapticState, jax.Array]:
        i = self.synapse_decay * state.i + x
        v = self.decay * state.v + i
        v, s = fire(v, self.threshold, self.reset, self.surrogate, self.detach_reset)
        return SynapticState(i, v), s.astype(x.dtype)


class ALIFState(NamedTuple):
    v: jax.Array
    a: jax.Array


@struct.dataclass
class ALIFCell:
    """LIF with an adaptive threshold that rises with each spike and decays back.

        theta[t] = threshold + beta * a[t-1]
        s[t] = H(v[t] - theta[t]),  v reset against theta[t]
        a[t] = adapt_decay * a[t-1] + s[t]

    The adaptive neurons of Bellec et al., "A solution to the learning dilemma
    for recurrent networks of spiking neurons" (Nature Communications 2020),
    whose adaptation time constants of hundreds of steps give a recurrent
    network memory beyond its membranes'. Gradients flow through `a`.
    """

    decay: jax.Array | float
    adapt_decay: jax.Array | float
    beta: jax.Array | float = 1.8
    threshold: jax.Array | float = 1.0
    reset: Reset = struct.field(pytree_node=False, default="subtract")
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())
    detach_reset: bool = struct.field(pytree_node=False, default=False)

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> ALIFState:
        zeros = jnp.zeros(shape, membrane_dtype(dtype))
        return ALIFState(zeros, zeros)

    def step(self, state: ALIFState, x: jax.Array) -> tuple[ALIFState, jax.Array]:
        theta = self.threshold + self.beta * state.a
        v = self.decay * state.v + x
        v, s = fire(v, theta, self.reset, self.surrogate, self.detach_reset)
        return ALIFState(v, self.adapt_decay * state.a + s), s.astype(x.dtype)


class IzhikevichState(NamedTuple):
    v: jax.Array
    u: jax.Array


@struct.dataclass
class IzhikevichCell:
    """Izhikevich's two-variable neuron, Euler-integrated with step `dt` (ms).

        v' = 0.04 v^2 + 5 v + 140 - u + x
        u' = a (b v - u)
        at v >= 30: v <- c, u <- u + d

    Izhikevich, "Simple Model of Spiking Neurons" (IEEE TNN 2003). The
    defaults are its regular-spiking cortical cell; inputs are in its units,
    where a constant 10 drives tonic spiking. The membrane starts at `c`.
    """

    a: jax.Array | float = 0.02
    b: jax.Array | float = 0.2
    c: jax.Array | float = -65.0
    d: jax.Array | float = 8.0
    dt: float = struct.field(pytree_node=False, default=0.5)
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> IzhikevichState:
        v = jnp.full(shape, self.c, membrane_dtype(dtype))
        return IzhikevichState(v, self.b * v)

    def step(self, state: IzhikevichState, x: jax.Array) -> tuple[IzhikevichState, jax.Array]:
        v, u = state
        v = v + self.dt * (0.04 * v * v + 5 * v + 140 - u + x)
        u = u + self.dt * self.a * (self.b * v - u)
        s = spike(v - 30.0, self.surrogate)
        return IzhikevichState(v + s * (self.c - v), u + s * self.d), s.astype(x.dtype)


class RecurrentState[State](NamedTuple):
    inner: State
    spikes: jax.Array


@struct.dataclass
class RecurrentCell[State]:
    """Feed a cell's spikes back to its own input through `weight`, `[F, F]`.

        output[t] = inner.step(x[t] + output[t-1] @ weight)

    Wrapping `ALIFCell` gives the recurrent adaptive network (LSNN) of Bellec
    et al. (2020). The product runs at `precision`, the matrix product
    precision of `jax.lax.dot`.
    """

    inner: Cell[State]
    weight: jax.Array
    precision: jax.lax.Precision | None = struct.field(pytree_node=False, default=None)

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> RecurrentState[State]:
        return RecurrentState(self.inner.init_state(shape, dtype), jnp.zeros(shape, dtype))

    def step(self, state: RecurrentState[State], x: jax.Array) -> tuple[RecurrentState[State], jax.Array]:
        feedback = jnp.matmul(state.spikes.astype(self.weight.dtype), self.weight, precision=self.precision)
        inner, s = self.inner.step(state.inner, x + feedback)
        # The feedback promotes the step's input, so the spikes are cast back:
        # the carry keeps the dtype it started with and the output has x's.
        return RecurrentState(inner, s.astype(state.spikes.dtype)), s.astype(x.dtype)


def run[State](cell: Cell[State], xs: jax.Array, state: State | None = None, *,
               unroll: int | bool = 1) -> tuple[jax.Array, State]:
    """Scan `cell` over the leading (time) axis of `xs`; return outputs `[T, ...]` and the final state.

    `state` None starts the neurons at rest. A run over `xs[:k]` then one over
    `xs[k:]` from its final state equals one run over `xs`. `unroll` is
    `jax.lax.scan`'s: how many steps one loop iteration holds.
    """
    if xs.ndim < 1:
        raise ValueError("run takes a time-major input [T, ...]")
    if state is None:
        state = cell.init_state(xs.shape[1:], xs.dtype)
    step: Callable[[State, jax.Array], tuple[State, jax.Array]] = cell.step
    state, outputs = jax.lax.scan(step, state, xs, unroll=unroll)
    return outputs, state
