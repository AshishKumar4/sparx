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
whatever the input's dtype (`membrane_dtype`); spikes come back in the
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
from flax import struct
from flax.typing import PrecisionLike

from sparx.dynamics.core import (
    NeuronModel,
    Output,
    Reset,
    SynapticInput,
    fire,
    jump_after_threshold,
    membrane_dtype,
)
from sparx.surrogate import ATan, Surrogate

__all__ = [
    "ACTIVATIONS",
    "ALIFCell",
    "ALIFState",
    "BernoulliCell",
    "BernoulliState",
    "DecayingHebb",
    "EligibleHebb",
    "HebbianRule",
    "LICell",
    "LIFCell",
    "MembraneState",
    "ModulatedHebb",
    "OjaHebb",
    "PlasticRecurrentCell",
    "PlasticState",
    "RateCell",
    "RateState",
    "RecurrentCell",
    "RecurrentState",
    "RetroactiveHebb",
    "Serial",
    "SparseRecurrentCell",
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
        return MembraneState(jnp.zeros(shape, membrane_dtype(dtype)))

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
        dtype = membrane_dtype(dtype)
        return BernoulliState(jnp.zeros(shape, dtype), jnp.zeros(shape, dtype))

    def step(self, state: BernoulliState, inputs: SynapticInput,
             dt: float) -> tuple[BernoulliState, Output]:
        x = _dimensionless(inputs)
        if inputs.noise is None:
            raise ValueError("a BernoulliCell fires by its noise: give SynapticInput.noise, uniform on "
                             "[0, 1)")
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
        return MembraneState(jnp.zeros(shape, membrane_dtype(dtype)))

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
        return RateState(jnp.zeros(shape, membrane_dtype(dtype)))

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
class Serial[First, Second]:
    """Two models in series: each step, the first's `Output.value` is the second's input jump.

    `Serial(LICell(synapse_decay), LIFCell(decay))` is the current-based
    LIF, whose input charges a decaying synaptic current that the membrane
    integrates,

        i[t] = synapse_decay * i[t-1] + x[t]
        v[t] = decay * v[t-1] + i[t]

    snnTorch's `Synaptic` and the CUBA neurons of Zenke and Vogels (2021).
    Longer chains nest. The output is the second model's, in the dtype it
    gives it, and the pair is graded when the second model is. In physical
    units the counterpart is a `PointNeuron` with an `Exponential` synapse,
    where an arrival shapes the membrane from the next step on.
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
        first, between = self.first.step(state[0], inputs, dt)
        second, out = self.second.step(state[1], SynapticInput(jump=between.value), dt)
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
        zeros = jnp.zeros(shape, membrane_dtype(dtype))
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


class RecurrentState[State](NamedTuple):
    inner: State
    output: jax.Array
    """The step's output, fed back in the next."""


@struct.dataclass
class RecurrentCell[State]:
    """Feed a model's output back to its own input jump through `weight`, `[F, F]`.

        output[t] = inner.step(x[t] + output[t-1] @ weight)

    Wrapping `ALIFCell` gives the recurrent adaptive network (LSNN) of Bellec
    et al. (2020); wrapping `RateCell` gives FLYNN's recurrence with a dense
    `weight`, which multiplies from the right (`weight[i, j]` from unit `i`
    to unit `j`). The product runs at `precision`, the matrix product
    precision of `jax.lax.dot`. `cut_gradient` stops the gradient at the
    fed-back output (the weight still receives its gradient), which is the
    gradient e-prop computes online (their `stop_z_gradients`).
    """

    inner: NeuronModel[State]
    weight: jax.Array
    precision: PrecisionLike = struct.field(pytree_node=False, default=None)
    cut_gradient: bool = struct.field(pytree_node=False, default=False)

    @property
    def graded(self) -> bool:
        return self.inner.graded

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> RecurrentState[State]:
        return RecurrentState(self.inner.init_state(shape, dtype), jnp.zeros(shape, dtype))

    def step(self, state: RecurrentState[State], inputs: SynapticInput,
             dt: float) -> tuple[RecurrentState[State], Output]:
        fed_back = jax.lax.stop_gradient(state.output) if self.cut_gradient else state.output
        feedback = jnp.matmul(fed_back.astype(self.weight.dtype), self.weight, precision=self.precision)
        fed = dataclasses.replace(inputs, jump=inputs.jump + feedback)
        inner, out = self.inner.step(state.inner, fed, dt)
        # The feedback promotes the step's input, so the output is cast back
        # to the dtype the carry started with, the input's.
        value = out.value.astype(state.output.dtype)
        return RecurrentState(inner, value), Output(value, out.offset)

    def is_refractory(self, state: RecurrentState[State], dt: float) -> jax.Array:
        return self.inner.is_refractory(state.inner, dt)

    def after_threshold(self, state: RecurrentState[State], jump: jax.Array,
                        fired: jax.Array) -> RecurrentState[State]:
        return state._replace(inner=self.inner.after_threshold(state.inner, jump, fired))


@struct.dataclass
class SparseRecurrentCell[State]:
    """Feed a model's output back to its own input through a list of weighted edges, `pre -> post`.

        output[t] = inner.step(x[t] + sum_e weight[e] output[t-1][pre[e]] at post[e])

    The sparse counterpart of `RecurrentCell`, for a wiring diagram whose
    edges are a small part of all pairs: a connectome, where FLYNN (Wang and
    Chen, arXiv 2607.00025) trains `weight` on the whole fly brain's 5.3
    million edges among 139 thousand neurons. Each step gathers the
    presynaptic outputs and sums them at their postsynaptic neurons, in time
    and memory proportional to the edges. `size` is the number of neurons,
    the last axis of the input; `pre` and `post` index it, and `weight` is
    one value per edge, a leaf that trains like a dense matrix's entries.
    """

    inner: NeuronModel[State]
    pre: jax.Array
    post: jax.Array
    weight: jax.Array
    size: int = struct.field(pytree_node=False)

    @property
    def graded(self) -> bool:
        return self.inner.graded

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> RecurrentState[State]:
        if shape[-1] != self.size:
            raise ValueError(f"the cell's {self.size} neurons read inputs of {self.size} features, "
                             f"not {shape[-1]}")
        return RecurrentState(self.inner.init_state(shape, dtype), jnp.zeros(shape, dtype))

    def step(self, state: RecurrentState[State], inputs: SynapticInput,
             dt: float) -> tuple[RecurrentState[State], Output]:
        sent = state.output.astype(self.weight.dtype)[..., self.pre] * self.weight
        feedback = jnp.moveaxis(jax.ops.segment_sum(jnp.moveaxis(sent, -1, 0), self.post, self.size), 0, -1)
        fed = dataclasses.replace(inputs, jump=inputs.jump + feedback)
        inner, out = self.inner.step(state.inner, fed, dt)
        value = out.value.astype(state.output.dtype)
        return RecurrentState(inner, value), Output(value, out.offset)

    def is_refractory(self, state: RecurrentState[State], dt: float) -> jax.Array:
        return self.inner.is_refractory(state.inner, dt)

    def after_threshold(self, state: RecurrentState[State], jump: jax.Array,
                        fired: jax.Array) -> RecurrentState[State]:
        return state._replace(inner=self.inner.after_threshold(state.inner, jump, fired))



class HebbianRule[Trace](Protocol):
    """How the Hebbian trace of a `PlasticRecurrentCell` changes over a step.

    A rule keeps a `Trace` of what it has seen, `init_trace(shape, dtype)`
    at the start of a sequence for outputs of `shape` `[..., F]`, and
    `hebb(trace)` is the `[..., F, F]` matrix the cell scales by `alpha`.
    `update(trace, pre, post, dt)` takes the output fed back this step
    (`pre`, `[..., F]`) and the step's new output (`post`); the entry
    `[..., i, j]` belongs to the connection from unit `i` to unit `j`.
    """

    def init_trace(self, shape: tuple[int, ...], dtype: jnp.dtype) -> Trace: ...

    def hebb(self, trace: Trace) -> jax.Array: ...

    def update(self, trace: Trace, pre: jax.Array, post: jax.Array, dt: float) -> Trace: ...


def _coactivity(pre: jax.Array, post: jax.Array) -> jax.Array:
    """`pre[..., i] * post[..., j]`, `[..., F, F]`."""
    return pre[..., :, None] * post[..., None, :]


def _connections(shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
    """Zeros, one per connection among outputs of `shape`: `[..., F, F]`, at least float32."""
    return jnp.zeros((*shape, shape[-1]), membrane_dtype(dtype))


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

    def init_trace(self, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        return _connections(shape, dtype)

    def hebb(self, trace: jax.Array) -> jax.Array:
        return trace

    def update(self, trace: jax.Array, pre: jax.Array, post: jax.Array, dt: float) -> jax.Array:
        keep = (1 - self.eta) ** dt
        return keep * trace + (1 - keep) * _coactivity(pre, post)


@struct.dataclass
class OjaHebb:
    """Oja's rule: a Hebbian trace that each postsynaptic unit's own activity bounds.

        hebb[i, j] <- hebb[i, j] + dt eta post[j] (pre[i] - post[j] hebb[i, j])

    Differentiable plasticity's alternative to the decaying trace (Miconi et
    al. 2018, eq. 3; their `maze/maze.py` with `rule="oja"`), which keeps a
    memory without input instead of letting it decay to zero (Oja 1982).
    """

    eta: jax.Array | float

    def init_trace(self, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        return _connections(shape, dtype)

    def hebb(self, trace: jax.Array) -> jax.Array:
        return trace

    def update(self, trace: jax.Array, pre: jax.Array, post: jax.Array, dt: float) -> jax.Array:
        target = post[..., None, :]
        return trace + dt * self.eta * target * (pre[..., :, None] - target * trace)


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

    def init_trace(self, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        return _connections(shape, dtype)

    def hebb(self, trace: jax.Array) -> jax.Array:
        return trace

    def update(self, trace: jax.Array, pre: jax.Array, post: jax.Array, dt: float) -> jax.Array:
        level = _modulation(post, self.modulator, self.modulator_bias)
        eta = level[..., None] * self.fanout + self.fanout_bias
        return jnp.clip(trace + dt * eta[..., None, :] * _coactivity(pre, post), -self.clip, self.clip)


class EligibleHebb(NamedTuple):
    hebb: jax.Array
    """The plastic part of the weights, `[..., F, F]`, which the modulator writes."""
    eligibility: jax.Array
    """The recent coactivity of each connection, `[..., F, F]`."""


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

    def init_trace(self, shape: tuple[int, ...], dtype: jnp.dtype) -> EligibleHebb:
        return EligibleHebb(_connections(shape, dtype), _connections(shape, dtype))

    def hebb(self, trace: EligibleHebb) -> jax.Array:
        return trace.hebb

    def update(self, trace: EligibleHebb, pre: jax.Array, post: jax.Array, dt: float) -> EligibleHebb:
        level = _modulation(post, self.modulator, self.modulator_bias)[..., None, None]
        hebb = jnp.clip(trace.hebb + dt * level * trace.eligibility, -self.clip, self.clip)
        keep = (1 - self.eta) ** dt
        return EligibleHebb(hebb, keep * trace.eligibility + (1 - keep) * _coactivity(pre, post))


class PlasticState[State, Trace](NamedTuple):
    inner: State
    output: jax.Array
    """The step's output, fed back in the next."""
    trace: Trace
    """What the rule keeps, one per example."""


@struct.dataclass
class PlasticRecurrentCell[State, Trace]:
    """Feed a model's output back through fixed weights plus a Hebbian trace that changes as it runs.

        output[t] = inner.step(x[t] + output[t-1] @ (weight + alpha * hebb[t-1]))
        hebb[t] = rule.update(hebb[t-1], output[t-1], output[t], dt)

    Differentiable plasticity (Miconi et al. 2018): `weight` and `alpha`,
    `[F, F]`, are each connection's fixed weight and how much of its trace
    it adds, learned by backpropagating through the traces, and `alpha`
    `[F]` gives one per postsynaptic unit (Backpropamine's language models).
    Each example's trace starts at zero, so what the network stores in it
    is what that sequence taught it. Wrapping `RateCell(0.0)`, a tanh unit
    without leak, gives their networks, with the rule `DecayingHebb` or
    `OjaHebb` (2018), `ModulatedHebb` or `RetroactiveHebb` (Backpropamine,
    2019); a spiking model gives fast weights between spikes. The trace is
    part of the state, so a stream fed in chunks carries it, and
    backpropagating holds one per example per step, `T * B * F * F` values
    for the Hebbian trace alone. `weight[i, j]` runs from unit `i` to unit
    `j`, as in `RecurrentCell`; the product runs at `precision`.
    """

    inner: NeuronModel[State]
    weight: jax.Array
    alpha: jax.Array | float
    rule: HebbianRule[Trace]
    precision: PrecisionLike = struct.field(pytree_node=False, default=None)

    @property
    def graded(self) -> bool:
        return self.inner.graded

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> PlasticState[State, Trace]:
        return PlasticState(self.inner.init_state(shape, dtype), jnp.zeros(shape, dtype),
                            self.rule.init_trace(shape, dtype))

    def step(self, state: PlasticState[State, Trace], inputs: SynapticInput,
             dt: float) -> tuple[PlasticState[State, Trace], Output]:
        hebb = self.rule.hebb(state.trace)
        pre = state.output.astype(hebb.dtype)
        weights = self.weight + self.alpha * hebb
        feedback = jnp.einsum("...i,...ij->...j", pre, weights, precision=self.precision)
        fed = dataclasses.replace(inputs, jump=inputs.jump + feedback)
        inner, out = self.inner.step(state.inner, fed, dt)
        value = out.value.astype(state.output.dtype)
        trace = self.rule.update(state.trace, pre, value.astype(hebb.dtype), dt)
        # The weights promote the trace's update, so each part is cast back to the dtype it started in.
        trace = jax.tree.map(lambda new, old: new.astype(old.dtype), trace, state.trace)
        return PlasticState(inner, value, trace), Output(value, out.offset)

    def is_refractory(self, state: PlasticState[State, Trace], dt: float) -> jax.Array:
        return self.inner.is_refractory(state.inner, dt)

    def after_threshold(self, state: PlasticState[State, Trace], jump: jax.Array,
                        fired: jax.Array) -> PlasticState[State, Trace]:
        return state._replace(inner=self.inner.after_threshold(state.inner, jump, fired))
