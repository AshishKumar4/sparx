"""The contract every biophysical model meets, and the arithmetic they share.

Units are a convention, checked where values are read in (design.md 4.4):

    time         ms        voltage      mV
    current      pA        conductance  nS
    capacitance  pF        rate         1/ms

so `pF * mV / ms = pA` and `nS * mV = pA` hold without conversion factors.

A neuron model advances one step of `dt` ms from its state and the synaptic
input it receives that step. The input separates what is a current from
what is a conductance, because a conductance acts through the neuron's own
voltage and reversal potential:

    model.init_state(shape, dtype) -> state
    model.step(state, SynapticInput, dt) -> (state, Spikes)

A step covers `(t, t + dt]`. Synaptic currents are waveforms over it, sums
of `(a + b s) exp(-s / tau)` for `s` in `[0, dt]` (`Term`), which cover the
exponential, alpha and bi-exponential synapses; a model whose membrane is
linear integrates them exactly. Conductances are held at their value at the
start of the step. Spikes that arrive at the end of the step, at `t + dt`,
are added to the synapses after the membrane has moved, so they shape the
next step; a voltage jump (a delta synapse) lands before the threshold test.
This is NEST's order, and Brian2's for a spike sent with no delay.

Where the subthreshold dynamics are linear in the voltage for the step's
inputs (LIF, current- or conductance-based), the update is their exact
solution over the step, as NEST's `iaf_psc_*` and Brian2's `exact` and
`exponential_euler` methods integrate them; nonlinear models name their
scheme.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from flax import struct

__all__ = [
    "NeuronModel",
    "Spikes",
    "SynapticInput",
    "Term",
    "crossing",
    "exact_linear",
    "integrate",
    "membrane_dtype",
    "response",
]


def membrane_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """The dtype state integrates in for inputs of `dtype`: float32 or wider."""
    return jnp.promote_types(dtype, jnp.float32)


class Term(NamedTuple):
    """A current `(amplitude + slope * s) * exp(-s / tau)` over a step, `s` in `[0, dt]` ms from its start.

    `amplitude` in pA, `slope` in pA/ms, `tau` in ms.
    """

    amplitude: jax.Array
    slope: jax.Array
    tau: jax.Array | float

    def at(self, s: jax.Array | float) -> jax.Array:
        return (self.amplitude + self.slope * s) * jnp.exp(-s / self.tau)

    def mean(self, dt: float) -> jax.Array:
        """The term's average over `[0, dt]`: exact."""
        x = -dt / jnp.asarray(self.tau)
        return self.amplitude * _phi1(x) + self.slope * dt * _psi(x)


@struct.dataclass
class SynapticInput:
    """What a population receives in one step.

    `current` (pA) adds to the membrane equation as it is, held over the
    step. `currents` are synaptic current waveforms over the step (`Term`).
    `conductance` (nS) holds one total conductance per receptor, keyed by
    the receptor's name, which the neuron model pairs with its reversal
    potential, so an excitatory and an inhibitory conductance pull the
    voltage toward different targets. `jump` (mV) is added to the voltage at
    the end of the step, before the threshold test.
    """

    current: jax.Array | float = 0.0
    currents: tuple[Term, ...] = ()
    conductance: Mapping[str, jax.Array] = struct.field(default_factory=dict)
    jump: jax.Array | float = 0.0

    def current_at(self, s: jax.Array | float) -> jax.Array:
        """The total current (pA) `s` ms into the step."""
        return sum((term.at(s) for term in self.currents), jnp.asarray(self.current))


class Spikes(NamedTuple):
    """A step's spikes, and when within the step each happened.

    `fired` is 0 or 1 in the input dtype. `offset` is the fraction of the
    step, in [0, 1], at which the membrane crossed threshold, from linear
    interpolation of the voltage across the step (Hansel et al., Neural
    Computation 1998); 1 where nothing fired. A spike's time is
    `(step + offset) * dt` from the start of the run.
    """

    fired: jax.Array
    offset: jax.Array


class NeuronModel[State](Protocol):
    """One population of neurons in physical units, advanced one step at a time."""

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> State: ...

    def step(self, state: State, inputs: SynapticInput, dt: float) -> tuple[State, Spikes]: ...


def exact_linear(v: jax.Array, target: jax.Array, tau: jax.Array | float, dt: float) -> jax.Array:
    """`v` after `dt` of `dv/dt = (target - v) / tau` with `target` and `tau` constant: exact."""
    return target + (v - target) * jnp.exp(-dt / tau)


def _series(x: jax.Array, coefficients: list[float]) -> jax.Array:
    out = jnp.zeros_like(x)
    for c in reversed(coefficients):
        out = out * x + c
    return out


_PHI1 = [1 / math.factorial(n + 1) for n in range(14)]
_PSI = [1 / (math.factorial(n) * (n + 2)) for n in range(14)]


def _phi1(x: jax.Array) -> jax.Array:
    """`(e^x - 1) / x`, by series near 0."""
    near = jnp.abs(x) < 0.5
    safe = jnp.where(near, 1.0, x)
    return jnp.where(near, _series(x, _PHI1), jnp.expm1(safe) / safe)


def _psi(x: jax.Array) -> jax.Array:
    """`(e^x (x - 1) + 1) / x^2`, by series near 0."""
    near = jnp.abs(x) < 0.5
    safe = jnp.where(near, 1.0, x)
    return jnp.where(near, _series(x, _PSI), (jnp.exp(safe) * (safe - 1) + 1) / safe ** 2)


def response(term: Term, tau: jax.Array | float, dt: float) -> jax.Array:
    """`integral_0^dt term(s) exp(-(dt - s) / tau) ds`: exact, in pA ms.

    The voltage a membrane with time constant `tau` (held over the step)
    gains over the step from the current `term`, times its capacitance.
    With `k = 1 / tau_s - 1 / tau` and `x = -k dt`, the integral is
    `exp(-dt / tau) dt (a phi1(x) + b dt psi(x))` with
    `phi1(x) = (e^x - 1) / x` and `psi(x) = (e^x (x - 1) + 1) / x^2`, which
    are summed as series near `x = 0` (equal time constants included),
    where the closed forms cancel.
    """
    x = -dt * (1 / jnp.asarray(term.tau) - 1 / jnp.asarray(tau))
    return jnp.exp(-dt / tau) * dt * (term.amplitude * _phi1(x) + term.slope * dt * _psi(x))


def crossing(before: jax.Array, after: jax.Array, threshold: jax.Array | float) -> jax.Array:
    """Where in the step a membrane going from `before` to `after` crossed `threshold`, as a fraction.

    Linear interpolation of the voltage across the step; 0 when it was already
    at or above threshold at the start, clipped to [0, 1].
    """
    rise = after - before
    fraction = (threshold - before) / jnp.where(rise > 0, rise, 1)
    return jnp.clip(jnp.where(rise > 0, fraction, 0), 0, 1)


def integrate[State](model: NeuronModel[State], inputs, dt: float, state: State | None = None,
                     record: Callable[[State], Any] | None = None):
    """Scan `model` over time-major `inputs`: every leaf `[T, ...]`, or a scalar held over time.

    `inputs` is what `model.step` takes for one step (a `SynapticInput`
    for a neuron model). Returns the step outputs stacked over time and the
    final state, which a later call continues from. The population's shape
    is the per-step shape of the first input with a time axis. With
    `record`, each step's output is paired with `record(state)` after the
    step (the membrane voltage, say).
    """
    leaves = [jnp.asarray(leaf) for leaf in jax.tree.leaves(inputs)]
    timed = [leaf for leaf in leaves if leaf.ndim > 0]
    if not timed:
        raise ValueError("integrate needs at least one input with a leading time axis")
    steps, shape, dtype = timed[0].shape[0], timed[0].shape[1:], timed[0].dtype
    if state is None:
        state = model.init_state(shape, dtype)
    held = jax.tree.map(lambda leaf: jnp.broadcast_to(jnp.asarray(leaf, dtype), (steps, *shape)), inputs)

    def step(state, inputs):
        state, out = model.step(state, inputs, dt)
        return state, out if record is None else (out, record(state))

    state, out = jax.lax.scan(step, state, held)
    return out, state
