"""Spiking neuron layers for Flax linen, over time-major inputs `[T, ...]`.

A neuron layer turns its input `[T, ...]` into spikes `[T, ...]`. It has
no weights of its own beyond optional learnable time constants; the synapses
are ordinary Flax layers. `nn.Dense`, `nn.Conv`, `nn.BatchNorm` and the
pooling functions all treat leading axes as batch axes, so they apply to
every time step at once, as one large matrix product, and only the neurons'
elementwise recurrence runs step by step:

    x = nn.Dense(128)(spikes)       # [T, B, 128], one matmul over T * B rows
    spikes = sparx.nn.LIF(tau=2.0)(x)

A layer builds a neuron model of `sparx.dynamics` and runs it with
`sparx.dynamics.run`. Time constants are in the unit of the layer's `dt`,
so with the default `dt = 1` they count steps.

Two variable collections are opt-in at `apply`:

- `"state"`, when mutable, carries the neurons across calls. Each layer
  starts from the state the collection holds (at rest when it holds none)
  and writes back its final state, so a long or live stream can be fed in
  chunks, down to one step at a time, with the same result as one call.
- `"spike_rates"`, when mutable, receives each spiking layer's firing rate
  per example and neuron, averaged over time (`[B, ...]` for a `[T, B, ...]`
  input), for rate regularizers and monitoring (`sparx.rates`).

Neither is created by `init` or touched when not mutable, so a plain
`apply` costs nothing for them.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

from sparx import dynamics
from sparx.dynamics import (
    ALIFCell,
    LICell,
    LIFCell,
    NeuronModel,
    RecurrentCell,
    Reset,
    Serial,
    SynapticInput,
    decay,
    run,
)
from sparx.registry import neurons
from sparx.surrogate import ATan, Surrogate

__all__ = [
    "ALIF",
    "IF",
    "LI",
    "LIF",
    "RATES",
    "STATE",
    "Dynamics",
    "Izhikevich",
    "Neuron",
    "Recurrent",
    "Synaptic",
    "adopt",
    "history_window",
    "record_rates",
]

STATE = "state"
"""The collection a layer carries its neurons' state in across `apply` calls."""
RATES = "spike_rates"
"""The collection spiking layers sow their per-example, per-neuron firing rates into."""


class Neuron(nn.Module):
    """A population of neurons run over the leading (time) axis of its input.

    A subclass says how to build its model (`sparx.dynamics.NeuronModel`)
    from the input, declaring any parameters there; this base runs it with
    `sparx.dynamics.run`, carries the `"state"` collection and sows
    `"spike_rates"`. The input reaches the model as `inputs(x)` says, a
    jump of the membrane unless a subclass says otherwise. `dt` is the step
    in the unit of the model's time constants. `unroll` is the number of
    time steps one iteration of the compiled loop holds (`jax.lax.scan`).
    """

    spiking: ClassVar[bool] = True
    dt: float = dataclasses.field(default=1.0, kw_only=True)
    unroll: int = dataclasses.field(default=1, kw_only=True)

    def build(self, x: jax.Array) -> NeuronModel:
        """The model for inputs like `x`, `[T, ..., features]`."""
        raise NotImplementedError

    def inputs(self, x: jax.Array) -> SynapticInput:
        """What the model receives from the layer's input `x`: a jump of its membrane."""
        return SynapticInput(jump=x)

    @nn.compact
    def model(self, x: jax.Array) -> NeuronModel:
        """This layer's model, with its parameters, for inputs like `x`."""
        return self.build(x)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run over `x`, `[T, ...]`, and return the outputs `[T, ...]`."""
        return self.run(self.model(x), x)

    def run(self, model: NeuronModel, x: jax.Array) -> jax.Array:
        carrying = self.is_mutable_collection(STATE) and not self.is_initializing()
        state = self.get_variable(STATE, "carry") if carrying else None
        spikes, final = run(model, self.inputs(x), state, dt=self.dt, unroll=self.unroll)
        if carrying:
            self.put_variable(STATE, "carry", final)
        if not self.spiking:
            return spikes.fired
        outputs = spikes.fired.astype(x.dtype)
        record_rates(self, outputs)
        return outputs


def record_rates(module: nn.Module, spikes: jax.Array) -> None:
    """Sow `spikes`' time-averaged rate into `"spike_rates"` when that collection is mutable."""
    if module.is_mutable_collection(RATES) and not module.is_initializing():
        module.sow(RATES, "rate", jnp.mean(spikes, axis=0, dtype=jnp.float32))


def history_window(module: nn.Module, x: jax.Array, held: int) -> jax.Array:
    """`x` `[T, ...]` with the `held` steps before it prepended, `[held + T, ...]`, in `x`'s dtype.

    A causal layer that reads `held` steps back keeps those steps in the
    `"state"` collection when it is mutable, so a stream fed in chunks sees
    the steps the previous chunk ended on; a fresh stream, or a call that
    does not carry state, sees zeros before its first step. The window's
    last `held` steps are stored for the next call.
    """
    carrying = module.is_mutable_collection(STATE) and not module.is_initializing()
    history = module.get_variable(STATE, "carry") if carrying else None
    if history is None:
        history = jnp.zeros((held, *x.shape[1:]), x.dtype)
    window = jnp.concatenate([jnp.asarray(history, x.dtype), x])
    if carrying:
        module.put_variable(STATE, "carry", window[window.shape[0] - held:])
    return window


def adopt(neuron: Neuron, owner: nn.Module, name: str) -> Neuron:
    """`neuron` as `owner`'s child called `name`, so its parameters sit under `owner` wherever it was built.

    A neuron built inside a parent's compact method belongs to that parent,
    and one handed down from a parent's field belongs to the parent; the
    clone makes it `owner`'s own. One built outside any module and given to
    `owner` as its field `name` is already that child.
    """
    if neuron.parent is owner and neuron.name == name:
        return neuron
    return neuron.clone(parent=owner, name=name)


def _decay(module: nn.Module, name: str, tau: float, learn: bool, features: int) -> jax.Array | float:
    """`decay(tau)` per unit of time, or a learnable decay per feature that starts there.

    A learned decay is the sigmoid of its parameter, so no update can move it
    out of (0, 1), where the membrane would grow without bound or flip sign.
    The parametric LIF of Fang et al. (ICCV 2021) also bounds its learned
    decay with a sigmoid, one shared by the layer; here each feature learns
    its own. The parameter stays the decay per unit of time, and the model
    raises it to the power `dt`, so a checkpoint means the same at any step
    and the decay at `dt = 1` is the parameter's sigmoid exactly.
    """
    if not learn:
        return decay(tau)
    start = decay(tau)
    logit = math.log(start / (1 - start))
    return jax.nn.sigmoid(module.param(name, nn.initializers.constant(logit), (features,), jnp.float32))


@neurons("lif")
class LIF(Neuron):
    """Leaky integrate-and-fire (`sparx.dynamics.LIFCell`) with time constant `tau`.

    `learn_tau` learns one decay per feature (last axis), starting at `tau`.
    """

    tau: float = 2.0
    threshold: float = 1.0
    reset: Reset = "subtract"
    surrogate: Surrogate = ATan()
    detach_reset: bool = False
    learn_tau: bool = False

    def build(self, x: jax.Array) -> LIFCell:
        return LIFCell(_decay(self, "decay", self.tau, self.learn_tau, x.shape[-1]), self.threshold,
                       self.reset, self.surrogate, self.detach_reset)


@neurons("if")
class IF(Neuron):
    """Integrate-and-fire: LIF without leak."""

    threshold: float = 1.0
    reset: Reset = "subtract"
    surrogate: Surrogate = ATan()
    detach_reset: bool = False

    def build(self, x: jax.Array) -> LIFCell:
        return LIFCell(1.0, self.threshold, self.reset, self.surrogate, self.detach_reset)


@neurons("li")
class LI(Neuron):
    """A leaky integrator readout (`sparx.dynamics.LICell`); returns its membrane, `[T, ...]`."""

    spiking: ClassVar[bool] = False
    tau: float = 2.0
    learn_tau: bool = False

    def build(self, x: jax.Array) -> LICell:
        return LICell(_decay(self, "decay", self.tau, self.learn_tau, x.shape[-1]))


@neurons("synaptic")
class Synaptic(Neuron):
    """Current-based LIF, `Serial(LICell, LIFCell)`: synaptic time constant `tau_synapse`,
    membrane time constant `tau`."""

    tau: float = 10.0
    tau_synapse: float = 5.0
    threshold: float = 1.0
    reset: Reset = "subtract"
    surrogate: Surrogate = ATan()
    detach_reset: bool = False
    learn_tau: bool = False

    def build(self, x: jax.Array) -> Serial:
        features = x.shape[-1]
        membrane = _decay(self, "decay", self.tau, self.learn_tau, features)
        synapse = _decay(self, "synapse_decay", self.tau_synapse, self.learn_tau, features)
        return Serial(LICell(synapse),
                      LIFCell(membrane, self.threshold, self.reset, self.surrogate, self.detach_reset))


@neurons("alif")
class ALIF(Neuron):
    """Adaptive-threshold LIF (`sparx.dynamics.ALIFCell`).

    Each spike raises the threshold by `beta`, and the rise decays with time
    constant `tau_adapt`. The defaults are Bellec et al.'s (2020) in steps of
    1 ms. `learn_tau` learns both decays per feature. `refractory` is a
    duration in the unit of `dt`.
    """

    tau: float = 20.0
    tau_adapt: float = 200.0
    beta: float = 1.8
    threshold: float = 1.0
    reset: Reset = "subtract"
    surrogate: Surrogate = ATan()
    detach_reset: bool = False
    learn_tau: bool = False
    refractory: float = 0

    def build(self, x: jax.Array) -> ALIFCell:
        features = x.shape[-1]
        return ALIFCell(
            _decay(self, "decay", self.tau, self.learn_tau, features),
            _decay(self, "adapt_decay", self.tau_adapt, self.learn_tau, features),
            self.beta, self.threshold, self.reset, self.surrogate, self.detach_reset, self.refractory)


@neurons("izhikevich")
class Izhikevich(Neuron):
    """Izhikevich's neuron (`sparx.dynamics.Izhikevich`) on input currents; defaults are regular spiking.

    The input is a current in the model's own units, where a constant 10
    drives tonic spiking, and time is in ms, the model's unit, so `dt` is
    the step in ms. The scheme is his published code's, which at the
    default `dt = 1` it reproduces to the last bit. The membrane starts at
    `c`.
    """

    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    surrogate: Surrogate = ATan()

    def inputs(self, x: jax.Array) -> SynapticInput:
        return SynapticInput(current=x)

    def build(self, x: jax.Array) -> dynamics.Izhikevich:
        return dynamics.Izhikevich(a=self.a, b=self.b, c=self.c, d=self.d, v_init=self.c,
                                   surrogate=self.surrogate)


class Dynamics(Neuron):
    """Any neuron model of `sparx.dynamics` as a layer, its fields fixed.

    `Dynamics(neuron=AdEx(), dt=0.1)` runs AdEx in steps of 0.1 ms on input
    currents `[T, ...]` in pA. `drive` names what the input is to the
    model: a `"current"` held over each step, the physical models' input,
    or a `"jump"` of the membrane, the dimensionless family's. A layer that
    learns a model's constants builds the model from its parameters, as
    `LIF` does.
    """

    neuron: NeuronModel = dataclasses.field(kw_only=True)
    drive: Literal["current", "jump"] = "current"

    def inputs(self, x: jax.Array) -> SynapticInput:
        return SynapticInput(current=x) if self.drive == "current" else SynapticInput(jump=x)

    def build(self, x: jax.Array) -> NeuronModel:
        return self.neuron


@neurons("recurrent")
class Recurrent(Neuron):
    """Feed `neuron`'s spikes back into its input through a learned `[F, F]` matrix.

    `Recurrent(ALIF())` is a recurrent adaptive layer (LSNN). The input
    projection stays outside, as an `nn.Dense` before this layer, so it runs
    over all time steps at once; only the feedback product runs inside the
    loop. The wrapped neuron's parameters live under `neuron`, and the
    layer's `dt` must be the neuron's.

    Backpropagation through the feedback multiplies by the recurrent matrix
    at every step, and a heavy-tailed surrogate passes gradient through
    every neuron, even those far from threshold. Training the recurrent
    network of `examples/train_shd.py` with ATan grew the matrix's spectral
    radius from 1 to 5 and the gradient norm past 1e8 within 300 steps;
    with `FastSigmoid(100)` the gradient norm stayed below 10.
    """

    neuron: Neuron = LIF()
    kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    precision: jax.lax.Precision | None = None

    @property
    def spiking(self) -> bool:  # type: ignore[override]
        return self.neuron.spiking

    def inputs(self, x: jax.Array) -> SynapticInput:
        return self.neuron.inputs(x)

    def build(self, x: jax.Array) -> RecurrentCell:
        if self.dt != self.neuron.dt:
            raise ValueError(f"Recurrent steps at dt={self.dt}, its neuron at dt={self.neuron.dt}; "
                             "give both one dt")
        features = x.shape[-1]
        weight = self.param("recurrent", self.kernel_init, (features, features), jnp.float32)
        return RecurrentCell(adopt(self.neuron, self, "neuron").model(x), weight, self.precision)
