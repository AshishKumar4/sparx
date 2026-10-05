"""Synapse kinetics, and a neuron together with the synapses onto it.

A linear synapse model's state is aggregated on the postsynaptic neuron, one
per receptor: linear synapses sum, so a neuron with `K` inputs onto one
receptor carries one state, not `K` (design.md section 4.3). Spikes arrive
weighted, in the receptor's unit: pA of peak current for a current receptor,
nS of peak conductance for a conductance receptor, mV for a delta synapse.

    synapse.init_state(shape, dtype) -> state
    synapse.output(state) -> tuple[Term, ...]   # the waveform over the coming step
    synapse.step(state, arriving, dt) -> state  # decay over the step, then add the arrivals
    synapse.lands                               # where the arrivals reach the neuron (`Landing`)

Arrivals land at the end of the step (`sparx.dynamics.core`), so a spike
shapes the membrane from the next step on, as in NEST and Brian2. Each
kinetic is integrated exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from flax import struct

from sparx.dynamics.core import NeuronModel, Spikes, SynapticInput, Term, membrane_dtype

__all__ = [
    "Alpha",
    "Arrivals",
    "BiExponential",
    "Delta",
    "Exponential",
    "Landing",
    "PointNeuron",
    "PointNeuronState",
    "Receptor",
    "SynapseModel",
]

type Landing = Literal["synapse", "before_threshold", "after_threshold"]
"""Where a synapse's arrivals reach the neuron. `synapse`: into the synapse's
state at the end of the step, whose output shapes the membrane from the next
step on. `before_threshold`: as a voltage jump at the end of the step, before
the threshold test (NEST's delta synapse). `after_threshold`: as a voltage
jump after the test and before the reset (Brian2's `on_pre="v += w"`)."""


class SynapseModel[State](Protocol):
    @property
    def lands(self) -> Landing:
        """Where the arrivals reach the neuron; a synapse that is a voltage jump has no state or output."""
        ...

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> State: ...

    def output(self, state: State) -> tuple[Term, ...]: ...

    def step(self, state: State, arriving: jax.Array | float, dt: float) -> State: ...


@struct.dataclass
class Delta:
    """A voltage jump of the arriving weight (mV), with no kinetics.

    By default the jump lands before the step's threshold test, as NEST's
    `iaf_psc_delta` input does, so a jump due at the end of a step can fire
    the neuron in that step. `after_threshold=True` lands it after the
    test and before the reset, as Brian2's `on_pre="v += w"` does: it takes
    effect from the next step, decaying over it first, and is lost if the
    neuron fired.
    """

    after_threshold: bool = struct.field(pytree_node=False, default=False)

    @property
    def lands(self) -> Landing:
        return "after_threshold" if self.after_threshold else "before_threshold"

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> tuple[()]:
        return ()

    def output(self, state: tuple[()]) -> tuple[Term, ...]:
        return ()

    def step(self, state: tuple[()], arriving: jax.Array | float, dt: float) -> tuple[()]:
        return ()


@struct.dataclass
class Exponential:
    """Jumps by the arriving weight and decays with `tau` (ms): NEST's `iaf_psc_exp` and `iaf_cond_exp`."""

    tau: jax.Array | float = 5.0

    @property
    def lands(self) -> Landing:
        return "synapse"

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        return jnp.zeros(shape, membrane_dtype(dtype))

    def output(self, state: jax.Array) -> tuple[Term, ...]:
        return (Term(state, jnp.zeros_like(state), self.tau),)

    def step(self, state: jax.Array, arriving: jax.Array | float, dt: float) -> jax.Array:
        return state * jnp.exp(-dt / self.tau) + arriving


class AlphaState(NamedTuple):
    value: jax.Array
    slope: jax.Array


@struct.dataclass
class Alpha:
    """`w e / tau * s * exp(-s / tau)`, peaking at the weight `w` at `s = tau`: NEST's `iaf_psc_alpha`.

    The state is the waveform `(value + slope * s) exp(-s / tau)` from the
    start of the step, which a step carries forward exactly.
    """

    tau: jax.Array | float = 2.0

    @property
    def lands(self) -> Landing:
        return "synapse"

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> AlphaState:
        zeros = jnp.zeros(shape, membrane_dtype(dtype))
        return AlphaState(zeros, zeros)

    def output(self, state: AlphaState) -> tuple[Term, ...]:
        return (Term(state.value, state.slope, self.tau),)

    def step(self, state: AlphaState, arriving: jax.Array | float, dt: float) -> AlphaState:
        decay = jnp.exp(-dt / self.tau)
        return AlphaState((state.value + state.slope * dt) * decay,
                          state.slope * decay + arriving * math.e / self.tau)


class BiExponentialState(NamedTuple):
    decay: jax.Array
    rise: jax.Array


@struct.dataclass
class BiExponential:
    """`w f (exp(-s / tau_decay) - exp(-s / tau_rise))`, normalized by `f` to peak at the weight `w`.

    NEST's `iaf_cond_beta` and the common dual-exponential AMPA and NMDA
    kinetics. Needs `tau_rise < tau_decay`; equal time constants are `Alpha`.
    """

    tau_rise: float = struct.field(pytree_node=False, default=0.5)
    tau_decay: float = struct.field(pytree_node=False, default=5.0)

    @property
    def lands(self) -> Landing:
        return "synapse"

    def __post_init__(self):
        if not 0 < self.tau_rise < self.tau_decay:
            raise ValueError(f"BiExponential needs 0 < tau_rise < tau_decay, got {self.tau_rise}, "
                             f"{self.tau_decay}; equal time constants are Alpha")

    @property
    def peak_factor(self) -> float:
        rise, decay = self.tau_rise, self.tau_decay
        peak = rise * decay / (decay - rise) * math.log(decay / rise)
        return 1 / (math.exp(-peak / decay) - math.exp(-peak / rise))

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> BiExponentialState:
        zeros = jnp.zeros(shape, membrane_dtype(dtype))
        return BiExponentialState(zeros, zeros)

    def output(self, state: BiExponentialState) -> tuple[Term, ...]:
        return (Term(state.decay, jnp.zeros_like(state.decay), self.tau_decay),
                Term(-state.rise, jnp.zeros_like(state.rise), self.tau_rise))

    def step(self, state: BiExponentialState, arriving: jax.Array | float, dt: float) -> BiExponentialState:
        added = arriving * self.peak_factor
        return BiExponentialState(state.decay * jnp.exp(-dt / self.tau_decay) + added,
                                  state.rise * jnp.exp(-dt / self.tau_rise) + added)


@struct.dataclass
class Receptor:
    """A synapse model and how its output reaches the membrane.

    `kind="current"` adds the waveform as a current (pA);
    `kind="conductance"` reads its value at the start of the step as a
    conductance (nS) against the neuron's reversal potential for this
    receptor's name. A synapse that lands as a voltage jump (`Delta`) is one either way.
    """

    synapse: SynapseModel = struct.field(default_factory=Exponential)
    kind: Literal["current", "conductance"] = struct.field(pytree_node=False, default="current")


class Arrivals(NamedTuple):
    """One step's input to a `PointNeuron`: a held current (pA), and the weighted spikes arriving
    at the end of the step on each receptor, by name."""

    current: jax.Array | float = 0.0
    spikes: Mapping[str, jax.Array] = {}


class PointNeuronState[State](NamedTuple):
    """A neuron's state and its synapses' by receptor name.

    Each receptor's synapse model has its own state type, and Python has no
    way to type a mapping whose values differ by key, so the synapses'
    states are left untyped.
    """

    neuron: State
    synapses: Mapping[str, object]


@struct.dataclass
class PointNeuron[State]:
    """A neuron model and the synapses onto it, by receptor, stepped together.

    `PointNeuron(LIF(...), {"ex": Receptor(Exponential(2.0))})` is NEST's
    `iaf_psc_exp` for one excitatory receptor; conductance receptors are
    named for their reversal potentials (`"ampa"`, `"gaba_a"`, ...).

    A conductance changes within a step while the neuron holds it, and
    `hold` picks the value held. `"mean"`, the default, is its exact average
    over the step, which makes the leak's decay over the step exact and the
    scheme second order. `"start"` is its value at the start of the step,
    Brian2's `exponential_euler`, first order.

    Two options reproduce models that tie their synapses to the neuron's
    spike, as Shiu et al.'s (2024) whole-brain model does in Brian2:
    `reset_synapses` clears a neuron's synaptic state when it fires (after
    the step's arrivals), and `freeze_synapses` holds it while the neuron
    is refractory: it does not decay, and arrivals are discarded, which is
    what Brian2's `(unless refractory)` flag on a synaptic variable does (a
    conditional write that stops synaptic updates too). Neither is
    physiology, where a synaptic current outlives the spike and input
    during refractoriness is not lost; both are off by default.
    """

    neuron: NeuronModel[State]
    receptors: Mapping[str, Receptor]
    hold: Literal["mean", "start"] = struct.field(pytree_node=False, default="mean")
    reset_synapses: bool = struct.field(pytree_node=False, default=False)
    freeze_synapses: bool = struct.field(pytree_node=False, default=False)

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> PointNeuronState[State]:
        synapses = {name: r.synapse.init_state(shape, dtype) for name, r in self.receptors.items()}
        return PointNeuronState(self.neuron.init_state(shape, dtype), synapses)

    def landing(self, where: Landing) -> list[str]:
        """The receptors whose arrivals land `where`."""
        return [name for name, r in self.receptors.items() if r.synapse.lands == where]

    def advance(self, state: PointNeuronState[State], current: jax.Array | float, jump: jax.Array | float,
                dt: float) -> tuple[PointNeuronState[State], Spikes]:
        """Move the membrane over a step on the synapses' output, with `jump` (mV) landing at its end."""
        currents: list[Term] = []
        conductance: dict[str, jax.Array] = {}
        for name in self.landing("synapse"):
            receptor = self.receptors[name]
            terms = receptor.synapse.output(state.synapses[name])
            if receptor.kind == "current":
                currents.extend(terms)
            else:
                held = (term.amplitude if self.hold == "start" else term.mean(dt) for term in terms)
                conductance[name] = sum(held, jnp.zeros(()))
        received = SynapticInput(current, tuple(currents), conductance, jump)
        cell, spikes = self.neuron.step(state.neuron, received, dt)
        return PointNeuronState(cell, state.synapses), spikes

    def receive(self, state: PointNeuronState[State], arriving: Mapping[str, jax.Array], dt: float,
                fired: jax.Array, frozen: jax.Array) -> PointNeuronState[State]:
        """Decay the synapses over the step and add the weights due at its end; jumps after the
        threshold land on the neuron.

        `fired` is the step's spikes, and `frozen` where the neuron was
        refractory during the step, as `frozen` read it before the step;
        `reset_synapses` and `freeze_synapses` act on them.
        """
        synapses = {}
        # Brian2 judges a step's arrivals by the refractoriness it set at the
        # step's start, or by the spike that ended it.
        dropped = frozen | (fired > 0)
        for name, r in self.receptors.items():
            incoming = arriving.get(name, 0.0)
            old = state.synapses[name]
            new = r.synapse.step(old, incoming, dt)
            if self.freeze_synapses:
                # Linear synapses: a step is the decay plus the arrivals' own response.
                decayed = r.synapse.step(old, 0.0, dt)
                added = jax.tree.map(lambda n, d: n - d, new, decayed)
                kept = jax.tree.map(lambda o, d: jnp.where(frozen, o, d), old, decayed)
                new = jax.tree.map(lambda k, a: k + jnp.where(dropped, 0.0, a), kept, added)
            if self.reset_synapses:
                new = jax.tree.map(lambda n: jnp.where(fired > 0, jnp.zeros_like(n), n), new)
            synapses[name] = new
        neuron = state.neuron
        late = [name for name in self.landing("after_threshold") if name in arriving]
        if late:
            jump = sum((arriving[name] for name in late), jnp.zeros(()))
            neuron = self.neuron.after_threshold(neuron, jump, fired)
        return PointNeuronState(neuron, synapses)

    def frozen(self, state: PointNeuronState[State], dt: float) -> jax.Array:
        """Where the neuron is refractory for the coming step, for `freeze_synapses`."""
        return self.neuron.is_refractory(state.neuron, dt)

    def delta(self, arriving: Mapping[str, jax.Array]) -> jax.Array:
        """The voltage jump (mV) of the weights arriving on receptors that land before the threshold."""
        jumps = (arriving.get(name, 0.0) for name in self.landing("before_threshold"))
        return jnp.asarray(sum(jumps, jnp.zeros(())))

    def step(self, state: PointNeuronState[State], inputs: Arrivals,
             dt: float) -> tuple[PointNeuronState[State], Spikes]:
        frozen = self.frozen(state, dt)
        state, spikes = self.advance(state, inputs.current, self.delta(inputs.spikes), dt)
        return self.receive(state, inputs.spikes, dt, spikes.fired, frozen), spikes
