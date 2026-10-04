"""Spiking neuron layers for Flax linen, over time-major inputs `[T, ...]`.

A neuron layer turns input currents `[T, ...]` into spikes `[T, ...]`. It has
no weights of its own beyond optional learnable time constants; the synapses
are ordinary Flax layers. `nn.Dense`, `nn.Conv`, `nn.BatchNorm` and the
pooling functions all treat leading axes as batch axes, so they apply to
every time step at once, as one large matrix product, and only the neurons'
elementwise recurrence runs step by step:

    x = nn.Dense(128)(spikes)       # [T, B, 128], one matmul over T * B rows
    spikes = sparx.nn.LIF(tau=2.0)(x)

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
from typing import ClassVar

import flax.linen as nn
import jax
import jax.numpy as jnp

from sparx.cells import (
    ALIFCell,
    Cell,
    IzhikevichCell,
    LICell,
    LIFCell,
    RecurrentCell,
    Reset,
    SynapticCell,
    run,
)
from sparx.surrogate import ATan, Surrogate

__all__ = [
    "ALIF",
    "IF",
    "LI",
    "LIF",
    "RATES",
    "STATE",
    "Izhikevich",
    "Neuron",
    "Recurrent",
    "Synaptic",
    "decay",
    "record_rates",
]

STATE = "state"
"""The collection a layer carries its neurons' state in across `apply` calls."""
RATES = "spike_rates"
"""The collection spiking layers sow their per-example, per-neuron firing rates into."""


class Neuron(nn.Module):
    """A population of neurons run over the leading (time) axis of its input.

    A subclass says how to build its `Cell` from the input, declaring any
    parameters there; this base runs it, carries the `"state"` collection and
    sows `"spike_rates"`. `unroll` is the number of time steps one iteration
    of the compiled loop holds (`jax.lax.scan`).
    """

    spiking: ClassVar[bool] = True
    unroll: int = dataclasses.field(default=1, kw_only=True)

    def build(self, x: jax.Array) -> Cell:
        """The cell for inputs like `x`, `[T, ..., features]`."""
        raise NotImplementedError

    @nn.compact
    def cell(self, x: jax.Array) -> Cell:
        """This layer's cell, with its parameters, for inputs like `x`."""
        return self.build(x)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run over `x`, `[T, ...]`, and return the outputs `[T, ...]`."""
        return self.run(self.cell(x), x)

    def run(self, cell: Cell, x: jax.Array) -> jax.Array:
        carrying = self.is_mutable_collection(STATE) and not self.is_initializing()
        state = self.get_variable(STATE, "carry") if carrying else None
        outputs, final = run(cell, x, state, unroll=self.unroll)
        if carrying:
            self.put_variable(STATE, "carry", final)
        if self.spiking:
            record_rates(self, outputs)
        return outputs


def record_rates(module: nn.Module, spikes: jax.Array) -> None:
    """Sow `spikes`' time-averaged rate into `"spike_rates"` when that collection is mutable."""
    if module.is_mutable_collection(RATES) and not module.is_initializing():
        module.sow(RATES, "rate", jnp.mean(spikes, axis=0, dtype=jnp.float32))


def decay(tau: float) -> float:
    """The per-step decay `exp(-1 / tau)` of a time constant `tau` in steps."""
    if tau <= 0:
        raise ValueError(f"a time constant must be positive, not {tau}")
    return math.exp(-1 / tau)


def _decay(module: nn.Module, name: str, tau: float, learn: bool, features: int) -> jax.Array | float:
    """`decay(tau)`, or a learnable decay per feature that starts there.

    A learned decay is the sigmoid of its parameter, so no update can move it
    out of (0, 1), where the membrane would grow without bound or flip sign.
    The parametric LIF of Fang et al. (ICCV 2021) also bounds its learned
    decay with a sigmoid, one shared by the layer; here each feature learns
    its own.
    """
    if not learn:
        return decay(tau)
    start = decay(tau)
    logit = math.log(start / (1 - start))
    return jax.nn.sigmoid(module.param(name, nn.initializers.constant(logit), (features,), jnp.float32))


class LIF(Neuron):
    """Leaky integrate-and-fire (`sparx.cells.LIFCell`), time constant `tau` steps.

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


class IF(Neuron):
    """Integrate-and-fire: LIF without leak."""

    threshold: float = 1.0
    reset: Reset = "subtract"
    surrogate: Surrogate = ATan()
    detach_reset: bool = False

    def build(self, x: jax.Array) -> LIFCell:
        return LIFCell(1.0, self.threshold, self.reset, self.surrogate, self.detach_reset)


class LI(Neuron):
    """A leaky integrator readout (`sparx.cells.LICell`); returns its membrane, `[T, ...]`."""

    spiking: ClassVar[bool] = False
    tau: float = 2.0
    learn_tau: bool = False

    def build(self, x: jax.Array) -> LICell:
        return LICell(_decay(self, "decay", self.tau, self.learn_tau, x.shape[-1]))


class Synaptic(Neuron):
    """Current-based LIF (`sparx.cells.SynapticCell`): synaptic time constant
    `tau_synapse`, membrane time constant `tau`."""

    tau: float = 10.0
    tau_synapse: float = 5.0
    threshold: float = 1.0
    reset: Reset = "subtract"
    surrogate: Surrogate = ATan()
    detach_reset: bool = False
    learn_tau: bool = False

    def build(self, x: jax.Array) -> SynapticCell:
        features = x.shape[-1]
        return SynapticCell(
            _decay(self, "decay", self.tau, self.learn_tau, features),
            _decay(self, "synapse_decay", self.tau_synapse, self.learn_tau, features),
            self.threshold, self.reset, self.surrogate, self.detach_reset)


class ALIF(Neuron):
    """Adaptive-threshold LIF (`sparx.cells.ALIFCell`).

    Each spike raises the threshold by `beta`, and the rise decays with time
    constant `tau_adapt`. The defaults are Bellec et al.'s (2020) in steps of
    1 ms. `learn_tau` learns both decays per feature.
    """

    tau: float = 20.0
    tau_adapt: float = 200.0
    beta: float = 1.8
    threshold: float = 1.0
    reset: Reset = "subtract"
    surrogate: Surrogate = ATan()
    detach_reset: bool = False
    learn_tau: bool = False

    def build(self, x: jax.Array) -> ALIFCell:
        features = x.shape[-1]
        return ALIFCell(
            _decay(self, "decay", self.tau, self.learn_tau, features),
            _decay(self, "adapt_decay", self.tau_adapt, self.learn_tau, features),
            self.beta, self.threshold, self.reset, self.surrogate, self.detach_reset)


class Izhikevich(Neuron):
    """Izhikevich's neuron (`sparx.cells.IzhikevichCell`); defaults are regular spiking."""

    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    dt: float = 0.5
    surrogate: Surrogate = ATan()

    def build(self, x: jax.Array) -> IzhikevichCell:
        return IzhikevichCell(self.a, self.b, self.c, self.d, self.dt, self.surrogate)


class Recurrent(Neuron):
    """Feed `neuron`'s spikes back into its input through a learned `[F, F]` matrix.

    `Recurrent(ALIF())` is a recurrent adaptive layer (LSNN). The input
    projection stays outside, as an `nn.Dense` before this layer, so it runs
    over all time steps at once; only the feedback product runs inside the
    loop. The wrapped neuron's parameters live under `neuron`.

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

    def build(self, x: jax.Array) -> RecurrentCell:
        features = x.shape[-1]
        weight = self.param("recurrent", self.kernel_init, (features, features), jnp.float32)
        # A neuron built inside a parent's compact method belongs to that
        # parent; the clone makes it this layer's own, so its parameters sit
        # under this layer wherever the caller constructed it. One built
        # outside is already this layer's child, under its field name.
        neuron = (self.neuron if self.neuron.parent is self
                  else self.neuron.clone(parent=self, name="neuron"))
        return RecurrentCell(neuron.cell(x), weight, self.precision)
