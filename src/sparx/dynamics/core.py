"""The contract every neuron model meets, the runner that scans one over time, and the arithmetic they share.

A neuron model advances one step of `dt` from its state and the synaptic
input it receives that step:

    model.init_state(shape, dtype) -> state
    model.step(state, SynapticInput, dt) -> (state, Spikes)

and `run(model, inputs)` scans the step over time-major inputs. Two
families meet this contract. The physical models (`sparx.dynamics.neurons`)
are in units, checked where values are read in (design.md 4.4):

    time         ms        voltage      mV
    current      pA        conductance  nS
    capacitance  pF        rate         1/ms

so `pF * mV / ms = pA` and `nS * mV = pA` hold without conversion factors.
The dimensionless models deep networks train with (`sparx.dynamics.ml`)
count time in steps, `dt = 1` by default, and take their input as a jump
of the membrane.

The input separates what is a current from what is a conductance, because
a conductance acts through the neuron's own voltage and reversal potential.
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
from typing import Literal, NamedTuple, Protocol, overload

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from sparx.surrogate import Surrogate, spike

__all__ = [
    "Model",
    "NeuronModel",
    "Reset",
    "Spikes",
    "SynapticInput",
    "Term",
    "crossing",
    "decay",
    "exact_linear",
    "fire",
    "jump_after_threshold",
    "membrane_dtype",
    "response",
    "rk4",
    "run",
    "substeps",
]


def membrane_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """The dtype state integrates in for inputs of `dtype`: float32 or wider.

    A bf16 membrane loses the small inputs it integrates over long sequences.
    """
    return jnp.promote_types(dtype, jnp.float32)


def decay(tau: float, dt: float = 1.0) -> float:
    """`exp(-dt / tau)`: what a time constant `tau` leaves of a value after a step `dt` in the same unit."""
    if tau <= 0:
        raise ValueError(f"a time constant must be positive, not {tau}")
    return math.exp(-dt / tau)


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

    `fired` is 0 or 1. `offset` is the fraction of the step, in [0, 1], at
    which the membrane crossed threshold, from linear interpolation of the
    voltage across the step (Hansel et al., Neural Computation 1998); 1
    where nothing fired. A spike's time is `(step + offset) * dt` from the
    start of the run. A model that never fires (`sparx.dynamics.ml.LICell`)
    reports its membrane as `fired`, the output a readout reads and the
    next model of a `Serial` receives.
    """

    fired: jax.Array
    offset: jax.Array


class Model[State, Inputs](Protocol):
    """Anything `run` steps: a neuron model, or a neuron with the synapses onto it (`PointNeuron`)."""

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> State:
        """The population at rest, for inputs of per-step shape `shape` and dtype `dtype`."""
        ...

    def step(self, state: State, inputs: Inputs, dt: float) -> tuple[State, Spikes]:
        """Advance one step of `dt` on `inputs`; return the new state and the step's spikes."""
        ...


class NeuronModel[State](Model[State, SynapticInput], Protocol):
    """One population of neurons, advanced one step at a time on the synaptic input it receives.

    Besides stepping, a model answers the two questions a `PointNeuron`
    asks of it between steps, so the synapses can follow the neuron
    without reading its state's fields: where it is refractory, and how a
    voltage jump that lands after the threshold test changes it.
    """

    def is_refractory(self, state: State, dt: float) -> jax.Array:
        """Where the neuron cannot fire in the coming step of `dt`, as booleans; all False for a model
        without refractoriness."""
        ...

    def after_threshold(self, state: State, jump: jax.Array, fired: jax.Array) -> State:
        """`state` after a voltage jump that lands past the step's threshold test, before its reset.

        `fired` is the step's `Spikes.fired`. A model whose spike resets the
        membrane loses the jump where it fired, since the reset that follows
        overwrites it; one that does not reset keeps it.
        """
        ...


type Reset = Literal["subtract", "zero", "none"]
"""What a spike does to a dimensionless membrane: `subtract` the threshold (soft reset,
which keeps the overshoot), set it to `zero` (hard reset), or leave it, `none`."""


def fire(v: jax.Array, threshold: jax.Array | float, surrogate: Surrogate, reset: Reset | jax.Array | float,
         *, subtract: jax.Array | float | None = None,
         detach_reset: bool = False) -> tuple[jax.Array, jax.Array]:
    """Spike where `v` reaches `threshold`, then reset; return the membrane and the spikes.

    `reset` is the voltage a membrane that fired is set to, or a rule:
    `"subtract"` takes away `subtract` (the threshold unless given),
    `"zero"` sets it to 0 and `"none"` leaves it. A set voltage is exact,
    and its gradient is that of `v + s (reset - v)`, so the surrogate's
    gradient passes the reset as it passes a subtraction. `detach_reset`
    stops the gradient through the reset, so the surrogate reaches the
    membrane only through the spike output, as in SpyTorch's tutorials and
    SpikingJelly's `detach_reset`. An infinite threshold fires nothing and
    resets nothing, which is how a refractory neuron is held.
    """
    s = spike(v - threshold, surrogate)
    r = jax.lax.stop_gradient(s) if detach_reset else s
    if isinstance(reset, str):
        if reset == "subtract":
            return v - r * (threshold if subtract is None else subtract), s
        if reset == "none":
            return v, s
        if reset != "zero":
            raise ValueError(f"reset must be subtract, zero or none, not {reset!r}")
        reset = 0.0
    return jnp.where(r > 0, reset, v) + (r - jax.lax.stop_gradient(r)) * (reset - v), s


def jump_after_threshold(v: jax.Array, jump: jax.Array, fired: jax.Array | None) -> jax.Array:
    """`v` plus a jump that lands after the threshold test, in `v`'s dtype.

    Where `fired`, the jump is lost, since the reset that follows it sets
    the voltage; `fired` None keeps it everywhere, for a model whose spike
    does not overwrite the membrane.
    """
    if fired is not None:
        jump = jnp.where(fired > 0, 0.0, jump)
    return (v + jump).astype(v.dtype)


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


def substeps(dt: float, longest: float | None) -> int:
    """How many equal substeps of `dt` keep each at most `longest` ms long (one when `longest` is None)."""
    return 1 if longest is None else max(1, math.ceil(dt / longest - 1e-9))


def rk4[Y](f: Callable[[jax.Array | float, Y], Y], y: Y, h: float, steps: int = 1, start: float = 0.0) -> Y:
    """`steps` classical Runge-Kutta steps of `h` ms of `dy/ds = f(s, y)` from `s = start`; `y` a pytree."""

    def add(y, k, c):
        return jax.tree.map(lambda a, b: a + c * b, y, k)

    def one(i, y):
        s = start + i * h
        k1 = f(s, y)
        k2 = f(s + h / 2, add(y, k1, h / 2))
        k3 = f(s + h / 2, add(y, k2, h / 2))
        k4 = f(s + h, add(y, k3, h))
        return jax.tree.map(lambda y, a, b, c, d: y + h / 6 * (a + 2 * b + 2 * c + d), y, k1, k2, k3, k4)

    return one(0, y) if steps == 1 else jax.lax.fori_loop(0, steps, one, y)


def crossing(before: jax.Array, after: jax.Array, threshold: jax.Array | float) -> jax.Array:
    """Where in the step a membrane going from `before` to `after` crossed `threshold`, as a fraction.

    Linear interpolation of the voltage across the step; 0 when it was already
    at or above threshold at the start, clipped to [0, 1].
    """
    rise = after - before
    fraction = (threshold - before) / jnp.where(rise > 0, rise, 1)
    return jnp.clip(jnp.where(rise > 0, fraction, 0), 0, 1)


@overload
def run[State, Inputs](model: Model[State, Inputs], inputs: Inputs | jax.Array | np.ndarray,
                       state: State | None = None, *, dt: float = 1.0, record: None = None,
                       unroll: int | bool = 1) -> tuple[Spikes, State]: ...


@overload
def run[State, Inputs, Record](model: Model[State, Inputs], inputs: Inputs | jax.Array | np.ndarray,
                               state: State | None = None, *, dt: float = 1.0,
                               record: Callable[[State], Record],
                               unroll: int | bool = 1) -> tuple[tuple[Spikes, Record], State]: ...


def run[State, Inputs, Record](model: Model[State, Inputs], inputs: Inputs | jax.Array | np.ndarray,
                               state: State | None = None, *, dt: float = 1.0,
                               record: Callable[[State], Record] | None = None,
                               unroll: int | bool = 1) -> tuple[Spikes | tuple[Spikes, Record], State]:
    """Scan `model` over time-major `inputs`; return its spikes `[T, ...]` and the final state.

    `inputs` is what `model.step` takes for one step, with every leaf
    `[T, ...]` or a scalar held over time: a `SynapticInput` for a neuron
    model, `Arrivals` for a `PointNeuron`. An array `[T, ...]` stands for
    `SynapticInput(jump=...)`, the dimensionless family's input. The
    population's per-step shape and its dtype are those of the first input
    with a time axis. `state` None starts the population at rest; a run over
    the first `k` steps and one over the rest from its final state equal one
    run over all. With `record`, each step's spikes are paired with
    `record(state)` after the step (the membrane voltage, say). `unroll` is
    `jax.lax.scan`'s: how many steps one loop iteration holds.
    """
    given = SynapticInput(jump=jnp.asarray(inputs)) if isinstance(inputs, jax.Array | np.ndarray) else inputs
    leaves, tree = jax.tree.flatten(given)
    timed = [i for i, leaf in enumerate(leaves) if jnp.ndim(leaf) > 0]
    if not timed:
        raise ValueError("run takes time-major inputs: at least one leaf [T, ...]")
    first = jnp.asarray(leaves[timed[0]])
    steps, shape, dtype = first.shape[0], first.shape[1:], first.dtype
    if state is None:
        state = model.init_state(shape, dtype)
    # Held scalars stay outside the scan as they were given, so an absent
    # input reads as Python's 0.0 in every step.
    xs = [jnp.broadcast_to(jnp.asarray(leaves[i], dtype), (steps, *shape)) for i in timed]

    def step(state: State, xs_t: list[jax.Array]) -> tuple[State, Spikes | tuple[Spikes, Record]]:
        now = list(leaves)
        for i, x in zip(timed, xs_t, strict=True):
            now[i] = x
        state, spikes = model.step(state, jax.tree.unflatten(tree, now), dt)
        return state, spikes if record is None else (spikes, record(state))

    state, out = jax.lax.scan(step, state, xs, unroll=unroll)
    return out, state
