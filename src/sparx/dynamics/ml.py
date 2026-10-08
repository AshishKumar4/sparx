"""The dimensionless neuron models deep spiking networks train with.

Each model meets the contract of `sparx.dynamics.core`,

    model.init_state(shape, dtype) -> state
    model.step(state, SynapticInput, dt) -> (state, Output)

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
whatever the input's dtype (`dew.nn.precision.at_least_fp32`); spikes come back in the
input's dtype, which holds 0 and 1 exactly. Two models here are graded,
their output a real value each step: the leaky integrator `LICell` and the
rate unit `RateCell`. `sparx.nn` builds these models from module
attributes and parameters; pure JAX code can use them directly.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from dew.nn.precision import at_least_fp32
from flax import struct
from flax.typing import PrecisionLike
from jax.core import Tracer

from sparx.dynamics.core import NeuronModel, Output, Reset, SynapticInput, fire, jump_after_threshold
from sparx.surrogate import ATan, Surrogate

__all__ = [
    "ACTIVATIONS",
    "ALIFCell",
    "ALIFState",
    "BernoulliCell",
    "BernoulliState",
    "DecayingHebb",
    "Dense",
    "EligibleHebb",
    "FastWeights",
    "HebbianRule",
    "LICell",
    "LIFCell",
    "MembraneState",
    "ModulatedHebb",
    "OjaHebb",
    "PulseCell",
    "RateCell",
    "RateState",
    "RecurrentCell",
    "RecurrentState",
    "RetroactiveHebb",
    "Serial",
    "Sparse",
    "Wiring",
]


def _dimensionless(inputs: SynapticInput) -> jax.Array:
    """The step's input to a dimensionless membrane, its jump; currents, conductances and gap
    junctions are refused."""
    held = inputs.current
    if (inputs.currents or inputs.conductance or inputs.gap is not None
            or not (isinstance(held, float | int) and held == 0)):
        raise ValueError("a dimensionless model takes its input as a jump (SynapticInput.jump), "
                         "not as currents, conductances or gap junctions")
    return jnp.asarray(inputs.jump)


def _jump(inputs: SynapticInput) -> jax.Array:
    """A deterministic model's input, its jump (`_dimensionless`), which fires without noise."""
    if inputs.noise is not None:
        raise ValueError("a deterministic model fires without noise; SynapticInput.noise is for "
                         "a stochastic model (BernoulliCell)")
    return _dimensionless(inputs)


def _at_end(value: jax.Array) -> Output:
    return Output(value, jnp.ones_like(value))


def _never(v: jax.Array) -> jax.Array:
    return jnp.zeros(jnp.shape(v), bool)


def _overwritten(reset: Reset, fired: jax.Array) -> jax.Array | None:
    """Where a reset sets the membrane after this step's threshold test: a zero reset sets it, a
    subtraction or no reset keeps whatever landed before."""
    return fired if reset == "zero" else None


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
    graded = False

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> MembraneState:
        return MembraneState(jnp.zeros(shape, at_least_fp32(dtype)))

    def step(self, state: MembraneState, inputs: SynapticInput, dt: float) -> tuple[MembraneState, Output]:
        x = _jump(inputs)
        v = self.decay ** dt * state.v + x
        v, s = fire(v, self.threshold, self.surrogate, self.reset, detach_reset=self.detach_reset)
        return MembraneState(v), _at_end(s.astype(x.dtype))

    def is_refractory(self, state: MembraneState, dt: float) -> jax.Array:
        return _never(state.v)

    def after_threshold(self, state: MembraneState, jump: jax.Array, fired: jax.Array) -> MembraneState:
        return MembraneState(jump_after_threshold(state.v, jump, _overwritten(self.reset, fired)))


class BernoulliState(NamedTuple):
    v: jax.Array
    """The membrane, after the reset of the step that set it."""
    p: jax.Array
    """The step's firing probability, read before the reset."""


@struct.dataclass
class BernoulliCell:
    """A leaky integrate-and-fire neuron with escape noise: it fires with a probability of its membrane.

        v[t] = decay ** dt * v[t-1] + x[t]
        p[t] = sigmoid(beta * (v[t] - threshold))
        s[t] = 1 where noise[t] < p[t], else 0
        v[t] <- reset(v[t], s[t])

    `noise[t]` is uniform on [0, 1), one draw per neuron per step, which
    the caller passes as `SynapticInput.noise` (`jax.random.uniform` of a
    key), so the cell is a pure function of its inputs and a given noise
    replays one trajectory; noise `1 - s` forces the spikes `s`, since
    `p` lies strictly between 0 and 1. `p` is a probability per step, the
    discrete-time escape noise of Pfister et al. (Neural Computation
    2006), whatever `dt` is; `beta` is the inverse of the noise's
    temperature, and as it grows the cell approaches `LIFCell`. The spike is a
    sample and passes no gradient: `sparx.learn.reinforce` trains a layer
    of these cells from the probability of the spikes they drew.
    """

    decay: jax.Array | float
    threshold: jax.Array | float = 1.0
    beta: jax.Array | float = 1.0
    reset: Reset = struct.field(pytree_node=False, default="subtract")
    graded = False

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> BernoulliState:
        dtype = at_least_fp32(dtype)
        return BernoulliState(jnp.zeros(shape, dtype), jnp.zeros(shape, dtype))

    def step(self, state: BernoulliState, inputs: SynapticInput,
             dt: float) -> tuple[BernoulliState, Output]:
        x = _dimensionless(inputs)
        if inputs.noise is None:
            raise ValueError("a BernoulliCell fires by its noise, one uniform draw on [0, 1) per neuron and "
                             "step: give SynapticInput.noise, or Arrivals.noise through a PointNeuron; a "
                             "sparx.nn layer and a sparx.graph Network draw none")
        v = self.decay ** dt * state.v + x
        p = jax.nn.sigmoid(self.beta * (v - self.threshold))
        s = (inputs.noise < p).astype(v.dtype)
        if self.reset == "zero":
            after = jnp.where(s > 0, 0.0, v)
        elif self.reset == "subtract":
            after = v - s * self.threshold
        elif self.reset == "none":
            after = v
        else:
            raise ValueError(f"reset must be subtract, zero or none, not {self.reset!r}")
        return BernoulliState(after, p), _at_end(s.astype(x.dtype))

    def is_refractory(self, state: BernoulliState, dt: float) -> jax.Array:
        return _never(state.v)

    def after_threshold(self, state: BernoulliState, jump: jax.Array, fired: jax.Array) -> BernoulliState:
        return state._replace(v=jump_after_threshold(state.v, jump, _overwritten(self.reset, fired)))


@struct.dataclass
class LICell:
    """A leaky integrator that never fires; its output is its membrane, a graded value.

    The usual readout of a spiking classifier: the last layer integrates the
    spikes it receives and the loss reads its membrane. As the first model
    of a `Serial` it is a synapse's decaying current.
    """

    decay: jax.Array | float
    graded = True

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> MembraneState:
        return MembraneState(jnp.zeros(shape, at_least_fp32(dtype)))

    def step(self, state: MembraneState, inputs: SynapticInput, dt: float) -> tuple[MembraneState, Output]:
        v = self.decay ** dt * state.v + _jump(inputs)
        return MembraneState(v), _at_end(v)

    def is_refractory(self, state: MembraneState, dt: float) -> jax.Array:
        return _never(state.v)

    def after_threshold(self, state: MembraneState, jump: jax.Array, fired: jax.Array) -> MembraneState:
        # It never fires, so nothing resets the jump away.
        return MembraneState(jump_after_threshold(state.v, jump, None))


class RateState(NamedTuple):
    h: jax.Array
    """The activity, the unit's output."""


ACTIVATIONS = {"tanh": jnp.tanh, "relu": jax.nn.relu, "sigmoid": jax.nn.sigmoid}
"""The activations a `RateCell` takes, by name."""


@struct.dataclass
class RateCell:
    """A leaky rate unit; its output is its activity `h`, a graded value.

        alpha = decay ** dt
        h[t] = alpha h[t-1] + (1 - alpha) f(x[t] + bias)

    FLYNN's leaky integrator (Wang and Chen, arXiv 2607.00025, eq. 1),
    `h_{t+1} = alpha h_t + (1 - alpha) tanh(W h_t + x_t + b)`, trained on
    the whole fly connectome. The recurrent product `W h_t` arrives with
    the external input in the jump `x`: through a `RecurrentCell`
    (`sparx.nn.Recurrent(sparx.nn.Rate())`), or through a projection onto a
    `Delta` receptor of a `sparx.graph.Network` with a delay of one step,
    which delivers the activity of the step before. `decay` is FLYNN's leak
    rate `alpha` per unit of time, `exp(-1 / tau)` for a time constant of
    `tau` (`sparx.dynamics.decay`); they train it directly, one per cell
    class. `activation` is FLYNN's tanh, or ReLU or the logistic sigmoid
    (`ACTIVATIONS`). The activity is a convex combination of its last value
    and the activation, so it stays within the activation's range.
    """

    decay: jax.Array | float
    bias: jax.Array | float = 0.0
    activation: Literal["tanh", "relu", "sigmoid"] = struct.field(pytree_node=False, default="tanh")
    graded = True

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> RateState:
        return RateState(jnp.zeros(shape, at_least_fp32(dtype)))

    def step(self, state: RateState, inputs: SynapticInput, dt: float) -> tuple[RateState, Output]:
        alpha = self.decay ** dt
        drive = ACTIVATIONS[self.activation](_jump(inputs) + self.bias)
        h = (alpha * state.h + (1 - alpha) * drive).astype(state.h.dtype)
        return RateState(h), _at_end(h)

    def is_refractory(self, state: RateState, dt: float) -> jax.Array:
        return _never(state.h)

    def after_threshold(self, state: RateState, jump: jax.Array, fired: jax.Array) -> RateState:
        return RateState(jump_after_threshold(state.h, jump, None))


@struct.dataclass
class PulseCell:
    """RNeuralNet's neuron: each step it outputs a function of the sum of what arrived, and empties the sum.

        s[t] = v[t-1] + x[t]
        output[t] = s[t]                      if s[t] >= threshold
                    exp(s[t] - threshold) - 1   otherwise
        v[t] = 0

    The activation of `Soma_t::ActivationFunction` in Ashish Kumar Singh's
    RNeuralNet-Research (commit d4b7803, with its `A_CONST` of 1): an ELU
    whose exponential branch is shifted by the threshold while its linear
    branch is not, so the output jumps from 0 to `threshold` there, and an
    empty sum gives `exp(-threshold) - 1`, not 0. The output is a graded
    message, sent each step, after which the sum empties, as the original
    neuron resets once it has sent on every outgoing connection. A jump
    that lands after the step (`after_threshold`) waits in `v` for the
    next. The neuron has no time constant, so `dt` changes nothing; a
    threshold of minus infinity passes the sum through unchanged, the
    original's input neurons. `sparx.learn.RNeuralNet` wires these neurons
    as the original does.
    """

    threshold: jax.Array | float = 2.0
    graded = True

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> MembraneState:
        return MembraneState(jnp.zeros(shape, at_least_fp32(dtype)))

    def step(self, state: MembraneState, inputs: SynapticInput, dt: float) -> tuple[MembraneState, Output]:
        s = state.v + _jump(inputs)
        # The minimum keeps the unused branch finite, so its gradient is too.
        below = jnp.exp(jnp.minimum(s - self.threshold, 0.0)) - 1
        output = jnp.where(s >= self.threshold, s, below).astype(state.v.dtype)
        return MembraneState(jnp.zeros_like(state.v)), _at_end(output)

    def is_refractory(self, state: MembraneState, dt: float) -> jax.Array:
        return _never(state.v)

    def after_threshold(self, state: MembraneState, jump: jax.Array, fired: jax.Array) -> MembraneState:
        return MembraneState(jump_after_threshold(state.v, jump, None))


@struct.dataclass
class Serial[First, Second]:
    """Two models in series: each step, the first's `Output.value` is the second's input jump.

    `Serial(LICell(synapse_decay), LIFCell(decay))` is the current-based
    LIF, whose input charges a decaying synaptic current that the membrane
    integrates,

        i[t] = synapse_decay * i[t-1] + x[t]
        v[t] = decay * v[t-1] + i[t]

    snnTorch's `Synaptic` and the CUBA neurons of Zenke and Vogels (2021).
    Longer chains nest. The output is the second model's, in the dtype it
    gives it, and the pair is graded when the second model is. The step's
    noise is the second model's, which fires: `Serial(LICell(...),
    BernoulliCell(...))` is a current-based neuron with escape noise. In
    physical units the counterpart is a `PointNeuron` with an `Exponential`
    synapse, where an arrival shapes the membrane from the next step on.
    """

    first: NeuronModel[First]
    second: NeuronModel[Second]

    @property
    def graded(self) -> bool:
        return self.second.graded

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> tuple[First, Second]:
        return self.first.init_state(shape, dtype), self.second.init_state(shape, dtype)

    def step(self, state: tuple[First, Second], inputs: SynapticInput,
             dt: float) -> tuple[tuple[First, Second], Output]:
        first, between = self.first.step(state[0], dataclasses.replace(inputs, noise=None), dt)
        second, out = self.second.step(state[1], SynapticInput(jump=between.value, noise=inputs.noise), dt)
        return (first, second), out

    def is_refractory(self, state: tuple[First, Second], dt: float) -> jax.Array:
        return self.second.is_refractory(state[1], dt)

    def after_threshold(self, state: tuple[First, Second], jump: jax.Array,
                        fired: jax.Array) -> tuple[First, Second]:
        # The threshold test and the spikes are the second model's, so the jump lands on its membrane.
        return state[0], self.second.after_threshold(state[1], jump, fired)


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
    graded = False

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> ALIFState:
        zeros = jnp.zeros(shape, at_least_fp32(dtype))
        return ALIFState(zeros, zeros, zeros)

    def step(self, state: ALIFState, inputs: SynapticInput, dt: float) -> tuple[ALIFState, Output]:
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

    def is_refractory(self, state: ALIFState, dt: float) -> jax.Array:
        return state.r > 0

    def after_threshold(self, state: ALIFState, jump: jax.Array, fired: jax.Array) -> ALIFState:
        return state._replace(v=jump_after_threshold(state.v, jump, _overwritten(self.reset, fired)))


class Wiring(Protocol):
    """Which units of a recurrent layer reach which, and with what weights: `Dense` or `Sparse`.

    `send(output, fast)` carries a layer's outputs `[..., F]` to every
    unit's input through the wiring's weights, plus `fast`, extra weights
    per example and connection (a plastic trace's), or None. An output takes
    one step to arrive, or as many as its connection's delay, up to
    `longest_delay`, so `send` returns `[longest_delay, ..., F]`, what
    arrives 1, 2, ... steps after the step that sent it; each message is
    weighted when it is sent. A plasticity rule reads values at each
    connection's presynaptic unit (`presynaptic(x)`), at its postsynaptic
    unit (`postsynaptic(x)`), or one per example at every connection
    (`per_example(x)`), so one rule serves every wiring.
    `connections(shape)` is the shape of one value per connection for
    outputs of `shape`, and refuses outputs the wiring does not fit.
    """

    @property
    def longest_delay(self) -> int: ...

    def connections(self, shape: tuple[int, ...]) -> tuple[int, ...]: ...

    def send(self, output: jax.Array, fast: jax.Array | None) -> jax.Array: ...

    def presynaptic(self, x: jax.Array) -> jax.Array: ...

    def postsynaptic(self, x: jax.Array) -> jax.Array: ...

    def per_example(self, x: jax.Array) -> jax.Array: ...


@struct.dataclass
class Dense:
    """Every unit to every unit through `weight`, `[F, F]`, `weight[i, j]` from unit `i` to unit `j`.

    The product multiplies from the right at `precision`, the matrix
    product precision of `jax.lax.dot`. A value per connection is `[..., F,
    F]`.
    """

    weight: jax.Array
    precision: PrecisionLike = struct.field(pytree_node=False, default=None)

    @property
    def longest_delay(self) -> int:
        return 1

    def connections(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if self.weight.shape != (shape[-1], shape[-1]):
            raise ValueError(f"a [{shape[-1]}, {shape[-1]}] weight feeds {shape[-1]} units back, not "
                             f"{list(self.weight.shape)}")
        return (*shape, shape[-1])

    def send(self, output: jax.Array, fast: jax.Array | None) -> jax.Array:
        if fast is None:
            sent = jnp.matmul(output.astype(self.weight.dtype), self.weight, precision=self.precision)
        else:
            weights = self.weight + fast
            sent = jnp.einsum("...i,...ij->...j", output.astype(weights.dtype), weights,
                              precision=self.precision)
        return sent[None]

    def presynaptic(self, x: jax.Array) -> jax.Array:
        return x[..., :, None]

    def postsynaptic(self, x: jax.Array) -> jax.Array:
        return x[..., None, :]

    def per_example(self, x: jax.Array) -> jax.Array:
        return x[..., None, None]


@struct.dataclass
class Sparse:
    """Units wired along the edges `pre[e] -> post[e]` with `weight[e]`, among `size` units.

    For a wiring diagram whose edges are a small part of all pairs, a
    connectome: FLYNN (Wang and Chen, arXiv 2607.00025) trains the whole fly
    brain's 5.3 million edges among 139 thousand neurons. Sending gathers
    the presynaptic outputs and sums them at their postsynaptic units, in
    time and memory proportional to the edges. A value per connection is
    one per edge, `[..., E]`.

    `delay[e]`, whole steps from 1 to `longest_delay`, is how long an output
    takes to cross edge `e`; None is one step for every edge. A message is
    weighted when it is sent, so a weight that changes while it travels
    leaves it as it was. RNeuralNet's connections queue their messages so
    (`sparx.learn.RNeuralNet`).
    """

    pre: jax.Array
    post: jax.Array
    weight: jax.Array
    size: int = struct.field(pytree_node=False)
    delay: jax.Array | None = None
    longest_delay: int = struct.field(pytree_node=False, default=1)

    def connections(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if shape[-1] != self.size:
            raise ValueError(f"the wiring's {self.size} units read inputs of {self.size} features, "
                             f"not {shape[-1]}")
        if self.delay is not None and not isinstance(self.delay, Tracer) and len(self.delay):
            # Out of range, an edge's messages would fall outside the steps a cell keeps.
            delays = np.asarray(self.delay)
            low, high = int(delays.min()), int(delays.max())
            if low < 1 or high > self.longest_delay:
                raise ValueError(f"delays from {low} to {high} steps do not fit 1 to longest_delay "
                                 f"{self.longest_delay}")
        return (*shape[:-1], self.weight.shape[0])

    def send(self, output: jax.Array, fast: jax.Array | None) -> jax.Array:
        weights = self.weight if fast is None else self.weight + fast
        # Units first, so each edge gathers a row of the batch and the gradient scatters rows, not
        # single values: 1.8 times faster forward and back on the continual core's 20,480 edges.
        rows = jnp.moveaxis(output.astype(weights.dtype), -1, 0)[self.pre]
        per_edge = jnp.moveaxis(weights, -1, 0)
        sent = rows * per_edge.reshape(per_edge.shape + (1,) * (rows.ndim - per_edge.ndim))
        slot = self.post if self.delay is None else (self.delay - 1) * self.size + self.post
        arrived = jax.ops.segment_sum(sent, slot, self.longest_delay * self.size)
        return jnp.moveaxis(arrived.reshape(self.longest_delay, self.size, *sent.shape[1:]), 1, -1)

    def presynaptic(self, x: jax.Array) -> jax.Array:
        return _per_edge(x, self.pre)

    def postsynaptic(self, x: jax.Array) -> jax.Array:
        return _per_edge(x, self.post)

    def per_example(self, x: jax.Array) -> jax.Array:
        return x[..., None]


def _per_edge(x: jax.Array, units: jax.Array) -> jax.Array:
    """`x[..., units]`, gathered a row of the batch at a time, so its gradient scatters rows (see `send`)."""
    return jnp.moveaxis(jnp.moveaxis(x, -1, 0)[units], 0, -1)


class HebbianRule[Trace](Protocol):
    """How a recurrent layer's Hebbian trace changes over a step, on any `Wiring`.

    A rule keeps a `Trace` of what it has seen, `init_trace(wiring, shape,
    dtype)` at the start of a sequence for outputs of `shape` `[..., F]`,
    and `hebb(trace)` is the value per connection that `FastWeights.alpha`
    scales. `update(trace, wiring, pre, post, dt)` takes the output fed back
    this step (`pre`, `[..., F]`) and the step's new output (`post`), and
    reads them through the wiring's `presynaptic`, `postsynaptic` and
    `per_example`.
    """

    def init_trace(self, wiring: Wiring, shape: tuple[int, ...], dtype: jnp.dtype) -> Trace: ...

    def hebb(self, trace: Trace) -> jax.Array: ...

    def update(self, trace: Trace, wiring: Wiring, pre: jax.Array, post: jax.Array, dt: float) -> Trace: ...


def _coactivity(wiring: Wiring, pre: jax.Array, post: jax.Array) -> jax.Array:
    """`pre` at each connection's presynaptic unit times `post` at its postsynaptic unit."""
    return wiring.presynaptic(pre) * wiring.postsynaptic(post)


def _connections(wiring: Wiring, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
    """Zeros, one per connection of `wiring` for outputs of `shape`, at least float32."""
    return jnp.zeros(wiring.connections(shape), at_least_fp32(dtype))


def _modulation(post: jax.Array, modulator: jax.Array, bias: jax.Array | float) -> jax.Array:
    """One neuromodulator level per example, `tanh(post . modulator + bias)`, `[...]`."""
    return jnp.tanh(jnp.sum(post * modulator, axis=-1) + bias)


@struct.dataclass
class DecayingHebb:
    """A Hebbian trace that decays toward the latest coactivity at the rate `eta`.

        keep = (1 - eta) ** dt
        hebb[i, j] <- keep hebb[i, j] + (1 - keep) pre[i] post[j]

    Differentiable plasticity's trace (Miconi et al. 2018, eq. 2), at
    `dt = 1` the `hebb = (1 - eta) * hebb + eta * outer(yin, yout)` of their
    `simple/simple.py`, with one `eta` for every connection, learned with
    the weights. A step of `dt` keeps what `dt` unit steps would of the old
    trace, so `eta` is a rate per unit of time and lies in [0, 1).
    """

    eta: jax.Array | float

    def init_trace(self, wiring: Wiring, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        return _connections(wiring, shape, dtype)

    def hebb(self, trace: jax.Array) -> jax.Array:
        return trace

    def update(self, trace: jax.Array, wiring: Wiring, pre: jax.Array, post: jax.Array,
               dt: float) -> jax.Array:
        keep = (1 - self.eta) ** dt
        return keep * trace + (1 - keep) * _coactivity(wiring, pre, post)


@struct.dataclass
class OjaHebb:
    """Oja's rule: a Hebbian trace that each postsynaptic unit's own activity bounds.

        hebb[i, j] <- hebb[i, j] + dt eta post[j] (pre[i] - post[j] hebb[i, j])

    Differentiable plasticity's alternative to the decaying trace (Miconi et
    al. 2018, eq. 3; their `maze/maze.py` with `rule="oja"`), which keeps a
    memory without input instead of letting it decay to zero (Oja 1982).
    """

    eta: jax.Array | float

    def init_trace(self, wiring: Wiring, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        return _connections(wiring, shape, dtype)

    def hebb(self, trace: jax.Array) -> jax.Array:
        return trace

    def update(self, trace: jax.Array, wiring: Wiring, pre: jax.Array, post: jax.Array,
               dt: float) -> jax.Array:
        target = wiring.postsynaptic(post)
        return trace + dt * self.eta * target * (wiring.presynaptic(pre) - target * trace)


@struct.dataclass
class ModulatedHebb:
    """A Hebbian trace whose rate the network's own activity sets through a neuromodulator, clipped.

        m = tanh(sum_j post[j] modulator[j] + modulator_bias)
        eta[j] = m fanout[j] + fanout_bias[j]
        hebb[i, j] <- clip(hebb[i, j] + dt eta[j] pre[i] post[j], -clip, clip)

    Backpropamine's simple neuromodulation (Miconi et al. 2019, eq. 3), with
    the fan-out of their appendix and `simplemaze/maze.py`: one modulator
    level per example, read off the new activity (their `h2mod`, `[F]` and
    a scalar), fanned out to a rate per postsynaptic unit (their
    `modfanout`, `[F]` and `[F]`). The rate can be negative, so the trace
    can grow, shrink or flip sign; the clip, 2 in their code, bounds it.
    """

    modulator: jax.Array
    modulator_bias: jax.Array | float
    fanout: jax.Array | float
    fanout_bias: jax.Array | float
    clip: float = struct.field(pytree_node=False, default=2.0)

    def init_trace(self, wiring: Wiring, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        return _connections(wiring, shape, dtype)

    def hebb(self, trace: jax.Array) -> jax.Array:
        return trace

    def update(self, trace: jax.Array, wiring: Wiring, pre: jax.Array, post: jax.Array,
               dt: float) -> jax.Array:
        level = _modulation(post, self.modulator, self.modulator_bias)
        eta = level[..., None] * self.fanout + self.fanout_bias
        change = dt * wiring.postsynaptic(eta) * _coactivity(wiring, pre, post)
        return jnp.clip(trace + change, -self.clip, self.clip)


class EligibleHebb(NamedTuple):
    hebb: jax.Array
    """The plastic part of the weights, one value per connection, which the modulator writes."""
    eligibility: jax.Array
    """The recent coactivity of each connection."""


@struct.dataclass
class RetroactiveHebb:
    """A Hebbian trace that a neuromodulator writes from an eligibility trace of recent coactivity.

        m = tanh(sum_j post[j] modulator[j] + modulator_bias)
        hebb[i, j] <- clip(hebb[i, j] + dt m eligibility[i, j], -clip, clip)
        keep = (1 - eta) ** dt
        eligibility[i, j] <- keep eligibility[i, j] + (1 - keep) pre[i] post[j]

    Backpropamine's retroactive neuromodulation (Miconi et al. 2019, eqs. 4
    and 5; their `maze/batch.py` with `type="modul"` and the hard clip at
    1, `addpw=3`). The coactivity changes no weight until the modulator
    arrives, so a signal that comes later (a reward) can still credit the
    connections that were active before it, as dopamine gates the plasticity
    recent activity left behind.
    """

    modulator: jax.Array
    modulator_bias: jax.Array | float
    eta: jax.Array | float
    clip: float = struct.field(pytree_node=False, default=1.0)

    def init_trace(self, wiring: Wiring, shape: tuple[int, ...], dtype: jnp.dtype) -> EligibleHebb:
        return EligibleHebb(_connections(wiring, shape, dtype), _connections(wiring, shape, dtype))

    def hebb(self, trace: EligibleHebb) -> jax.Array:
        return trace.hebb

    def update(self, trace: EligibleHebb, wiring: Wiring, pre: jax.Array, post: jax.Array,
               dt: float) -> EligibleHebb:
        level = wiring.per_example(_modulation(post, self.modulator, self.modulator_bias))
        hebb = jnp.clip(trace.hebb + dt * level * trace.eligibility, -self.clip, self.clip)
        keep = (1 - self.eta) ** dt
        return EligibleHebb(hebb, keep * trace.eligibility + (1 - keep) * _coactivity(wiring, pre, post))


@struct.dataclass
class FastWeights[Trace]:
    """Fast weights on a wiring: `alpha` times the Hebbian trace that `rule` keeps, added to each weight.

    Differentiable plasticity (Miconi et al. 2018): `alpha` is how much of
    its trace each connection adds, one per connection (`[F, F]` on a
    `Dense` wiring, `[E]` on a `Sparse` one) or anything that broadcasts to
    it (`[F]`, one per postsynaptic unit on a dense wiring, as in
    Backpropamine's language models), learned by backpropagating through
    the traces. Each example's trace starts at zero, so what the network
    stores in it is what that sequence taught it. The rules are
    `DecayingHebb` and `OjaHebb` (2018), `ModulatedHebb` and
    `RetroactiveHebb` (Backpropamine, 2019), or any `HebbianRule`.

    `connections` makes only some connections of a `Sparse` wiring plastic,
    by their indices in its edge list: the rule keeps a trace for those
    alone, `alpha` is one per plastic connection, and the others keep their
    weights. A network whose few connections adapt fast, as most of a
    brain's synapses do not, then holds and backpropagates through traces
    for those few.
    """

    alpha: jax.Array | float
    rule: HebbianRule[Trace]
    connections: jax.Array | None = None


@struct.dataclass
class _Arrivals:
    """`wiring` as a plasticity rule reads it when its delays differ: each connection's presynaptic value
    is what it delivers this step, its source's output `delay` steps before, from `history`, the last
    outputs newest first, `[longest_delay, ..., F]`."""

    wiring: Sparse
    history: jax.Array

    @property
    def longest_delay(self) -> int:
        return self.wiring.longest_delay

    def connections(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return self.wiring.connections(shape)

    def send(self, output: jax.Array, fast: jax.Array | None) -> jax.Array:
        return self.wiring.send(output, fast)

    def presynaptic(self, x: jax.Array) -> jax.Array:
        """Each connection's delivered value, whatever `x`: the output its source sent `delay` steps ago."""
        delay = jnp.ones_like(self.wiring.pre) if self.wiring.delay is None else self.wiring.delay
        rows = jnp.moveaxis(self.history, -1, 1)  # [longest_delay, F, ...]
        return jnp.moveaxis(rows[delay - 1, self.wiring.pre], 0, -1)

    def postsynaptic(self, x: jax.Array) -> jax.Array:
        return self.wiring.postsynaptic(x)

    def per_example(self, x: jax.Array) -> jax.Array:
        return self.wiring.per_example(x)


class RecurrentState[State, Trace](NamedTuple):
    inner: State
    output: jax.Array
    """The step's output."""
    arriving: jax.Array
    """The outputs sent and on their way, `[longest_delay, ..., F]`: `arriving[k]` reaches the units
    `k + 1` steps after the step."""
    trace: Trace | None = None
    """What the fast weights' rule keeps, one per example; None without fast weights."""
    history: jax.Array | None = None
    """The last outputs, newest first, `[longest_delay, ..., F]`, which fast weights on a wiring with
    longer delays pair with the step's output; None otherwise."""


@struct.dataclass
class RecurrentCell[State, Trace]:
    """Feed a model's output back to its own input through `wiring`, fixed or plastic.

        output[t] = inner.step(x[t] + send(output[t-1], alpha * hebb[t-1]))
        hebb[t] = rule.update(hebb[t-1], output[t-1], output[t], dt)

    The wiring is `Dense`, every unit to every unit, or `Sparse`, along a
    list of edges such as a connectome's, where each edge may take its own
    number of steps (`Sparse.delay`): the output then arrives that many
    steps after it was sent, weighted as the wiring was when it left, and
    the state holds what is on its way. With `fast_weights` (`FastWeights`),
    a Hebbian trace each sequence writes adds fast weights to the wiring's;
    without, the trace stays None and the weights fixed. On a wiring whose
    delays differ, a connection's trace pairs the new output with what the
    connection delivers this step, its source's output `delay` steps
    before, which the state keeps (`RecurrentState.history`), and a message
    carries the fast weights of the step that sent it. Any model runs
    inside: `ALIFCell` gives the recurrent adaptive network (LSNN) of Bellec
    et al. (2020), `RateCell` FLYNN's recurrence (on a `Sparse` wiring, its
    connectome), `RateCell(0.0)` with plasticity Miconi et al.'s networks,
    and a spiking model fast weights between spikes. The trace is part of
    the state, so a stream fed in chunks carries it, and backpropagating
    through a plastic cell holds one per example per step. `cut_gradient`
    stops the gradient at the fed-back output (the weights still receive
    theirs), the gradient e-prop computes online (their `stop_z_gradients`).
    """

    inner: NeuronModel[State]
    wiring: Wiring
    fast_weights: FastWeights[Trace] | None = None
    cut_gradient: bool = struct.field(pytree_node=False, default=False)

    @property
    def graded(self) -> bool:
        return self.inner.graded

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> RecurrentState[State, Trace]:
        self.wiring.connections(shape)
        rule = None if self.fast_weights is None else self.fast_weights.rule
        trace = None if rule is None else rule.init_trace(self._plastic(), shape, dtype)
        output = jnp.zeros(shape, dtype)
        # What is on its way is held in the dtype the wiring sends in, so it reaches the units unrounded.
        sent = jax.eval_shape(self._send, output, trace)
        delayed = rule is not None and self.wiring.longest_delay > 1
        history = jnp.zeros((self.wiring.longest_delay, *shape), dtype) if delayed else None
        return RecurrentState(self.inner.init_state(shape, dtype), output, jnp.zeros(sent.shape, sent.dtype),
                              trace, history)

    def step(self, state: RecurrentState[State, Trace], inputs: SynapticInput,
             dt: float) -> tuple[RecurrentState[State, Trace], Output]:
        fed = dataclasses.replace(inputs, jump=inputs.jump + state.arriving[0])
        inner, out = self.inner.step(state.inner, fed, dt)
        # What arrived promotes the step's input, so the output is cast back
        # to the dtype the carry started with, the input's.
        value = out.value.astype(state.output.dtype)
        trace = self._learned(state, value, dt)
        later = jnp.concatenate([state.arriving[1:], jnp.zeros_like(state.arriving[:1])])
        arriving = later + self._send(value, trace).astype(later.dtype)
        history = None if state.history is None else jnp.concatenate([value[None], state.history[:-1]])
        return RecurrentState(inner, value, arriving, trace, history), Output(value, out.offset)

    def _send(self, output: jax.Array, trace: Trace | None) -> jax.Array:
        """What `output` sends through the wiring and the fast weights of `trace`."""
        sent = jax.lax.stop_gradient(output) if self.cut_gradient else output
        if self.fast_weights is None:
            return self.wiring.send(sent, None)
        assert trace is not None, "init_state gives a cell with fast weights its trace"
        hebb = self.fast_weights.rule.hebb(trace)
        fast = self.fast_weights.alpha * hebb
        if self.fast_weights.connections is None:
            return self.wiring.send(sent.astype(hebb.dtype), fast)
        return self.wiring.send(sent, None) + self._plastic().send(sent.astype(hebb.dtype), fast)

    def _plastic(self) -> Wiring:
        """The connections the fast weights act on: the wiring, or the plastic ones of a sparse wiring, with
        no weight of their own, since the wiring sends that."""
        connections = None if self.fast_weights is None else self.fast_weights.connections
        if connections is None:
            return self.wiring
        if not isinstance(self.wiring, Sparse):
            raise ValueError("plastic connections are edges of a Sparse wiring, picked by index")
        w = self.wiring
        delay = None if w.delay is None else w.delay[connections]
        return Sparse(w.pre[connections], w.post[connections], jnp.zeros_like(w.weight[connections]), w.size,
                      delay, w.longest_delay)

    def _learned(self, state: RecurrentState[State, Trace], value: jax.Array, dt: float) -> Trace | None:
        """The trace after the step that output `value`, in its dtypes; None without fast weights."""
        if self.fast_weights is None:
            return state.trace
        assert state.trace is not None, "init_state gives a cell with fast weights its trace"
        rule = self.fast_weights.rule
        kept = rule.hebb(state.trace).dtype
        wiring = self._plastic()
        if state.history is not None:
            assert isinstance(wiring, Sparse), "only a sparse wiring has delays longer than a step"
            wiring = _Arrivals(wiring, state.history.astype(kept))
        trace = rule.update(state.trace, wiring, state.output.astype(kept), value.astype(kept), dt)
        # The weights promote the update, so each part is cast back to the dtype it started in.
        return jax.tree.map(lambda new, old: new.astype(old.dtype), trace, state.trace)

    def is_refractory(self, state: RecurrentState[State, Trace], dt: float) -> jax.Array:
        return self.inner.is_refractory(state.inner, dt)

    def after_threshold(self, state: RecurrentState[State, Trace], jump: jax.Array,
                        fired: jax.Array) -> RecurrentState[State, Trace]:
        return state._replace(inner=self.inner.after_threshold(state.inner, jump, fired))
