"""Synapse kinetics, and a neuron together with the synapses onto it.

A linear synapse model's state is aggregated on the postsynaptic neuron, one
per receptor: linear synapses sum, so a neuron with `K` inputs onto one
receptor carries one state, not `K` (design.md section 4.3). Spikes arrive
weighted, in the receptor's unit: pA of peak current for a current receptor,
nS of peak conductance for a conductance receptor, mV for a delta synapse.

    synapse.init_state(shape, dtype) -> state
    synapse.output(state) -> tuple[Term, ...]   # the waveform over the coming step
    synapse.step(state, arriving, dt) -> state  # decay over the step, then add the arrivals

Arrivals land at the end of the step (`sparx.dynamics.core`), so a spike
shapes the membrane from the next step on, as in NEST and Brian2. Each
kinetic is integrated exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from flax import struct

from sparx.dynamics.core import NeuronModel, Spikes, SynapticInput, Term, membrane_dtype

__all__ = ["Alpha", "Arrivals", "BiExponential", "Delta", "Exponential", "PointNeuron", "Receptor",
           "SynapseModel"]


class SynapseModel[State](Protocol):
    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> State: ...

    def output(self, state: State) -> tuple[Term, ...]: ...

    def step(self, state: State, arriving: jax.Array, dt: float) -> State: ...


@struct.dataclass
class Delta:
    """A voltage jump of the arriving weight (mV), with no kinetics: NEST's `iaf_psc_delta` input."""

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> tuple[()]:
        return ()

    def output(self, state: tuple[()]) -> tuple[Term, ...]:
        return ()

    def step(self, state: tuple[()], arriving: jax.Array, dt: float) -> tuple[()]:
        return ()


@struct.dataclass
class Exponential:
    """Jumps by the arriving weight and decays with `tau` (ms): NEST's `iaf_psc_exp` and `iaf_cond_exp`."""

    tau: jax.Array | float = 5.0

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        return jnp.zeros(shape, membrane_dtype(dtype))

    def output(self, state: jax.Array) -> tuple[Term, ...]:
        return (Term(state, jnp.zeros_like(state), self.tau),)

    def step(self, state: jax.Array, arriving: jax.Array, dt: float) -> jax.Array:
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

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> AlphaState:
        zeros = jnp.zeros(shape, membrane_dtype(dtype))
        return AlphaState(zeros, zeros)

    def output(self, state: AlphaState) -> tuple[Term, ...]:
        return (Term(state.value, state.slope, self.tau),)

    def step(self, state: AlphaState, arriving: jax.Array, dt: float) -> AlphaState:
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

    def step(self, state: BiExponentialState, arriving: jax.Array, dt: float) -> BiExponentialState:
        added = arriving * self.peak_factor
        return BiExponentialState(state.decay * jnp.exp(-dt / self.tau_decay) + added,
                                  state.rise * jnp.exp(-dt / self.tau_rise) + added)


@struct.dataclass
class Receptor:
    """A synapse model and how its output reaches the membrane.

    `kind="current"` adds the waveform as a current (pA);
    `kind="conductance"` reads its value at the start of the step as a
    conductance (nS) against the neuron's reversal potential for this
    receptor's name. A `Delta` synapse is a voltage jump either way.
    """

    synapse: Any = struct.field(default_factory=Exponential)
    kind: Literal["current", "conductance"] = struct.field(pytree_node=False, default="current")


class Arrivals(NamedTuple):
    """One step's input to a `PointNeuron`: a held current (pA), and the weighted spikes arriving
    at the end of the step on each receptor, by name."""

    current: jax.Array | float = 0.0
    spikes: Mapping[str, jax.Array] = {}


class PointNeuronState(NamedTuple):
    neuron: Any
    synapses: Mapping[str, Any]


@struct.dataclass
class PointNeuron:
    """A neuron model and the synapses onto it, by receptor, stepped together.

    `PointNeuron(LIF(...), {"ex": Receptor(Exponential(2.0))})` is NEST's
    `iaf_psc_exp` for one excitatory receptor; conductance receptors are
    named for their reversal potentials (`"ampa"`, `"gaba_a"`, ...).

    A conductance changes within a step while the neuron holds it, and
    `hold` picks the value held. `"mean"`, the default, is its exact average
    over the step, which makes the leak's decay over the step exact and the
    scheme second order. `"start"` is its value at the start of the step,
    Brian2's `exponential_euler`, first order.
    """

    neuron: Any
    receptors: Mapping[str, Receptor]
    hold: Literal["mean", "start"] = struct.field(pytree_node=False, default="mean")

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> PointNeuronState:
        synapses = {name: r.synapse.init_state(shape, dtype) for name, r in self.receptors.items()}
        return PointNeuronState(self.neuron.init_state(shape, dtype), synapses)

    def step(self, state: PointNeuronState, inputs: Arrivals, dt: float) -> tuple[PointNeuronState, Spikes]:
        neuron: NeuronModel = self.neuron
        currents: list[Term] = []
        conductance: dict[str, jax.Array] = {}
        jump = jnp.zeros(())
        for name, receptor in self.receptors.items():
            if isinstance(receptor.synapse, Delta):
                jump = jump + inputs.spikes.get(name, 0.0)
            elif receptor.kind == "current":
                currents.extend(receptor.synapse.output(state.synapses[name]))
            else:
                terms = receptor.synapse.output(state.synapses[name])
                held = (term.amplitude if self.hold == "start" else term.mean(dt) for term in terms)
                conductance[name] = sum(held, jnp.zeros(()))
        received = SynapticInput(inputs.current, tuple(currents), conductance, jump)
        cell, spikes = neuron.step(state.neuron, received, dt)
        synapses = {name: r.synapse.step(state.synapses[name], inputs.spikes.get(name, 0.0), dt)
                    for name, r in self.receptors.items()}
        return PointNeuronState(cell, synapses), spikes
