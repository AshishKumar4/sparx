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

Inputs are held constant over a step. Where the subthreshold dynamics are
linear in the voltage for constant inputs (LIF, current- or
conductance-based), the update is their exact solution over the step, as
NEST and Brian2 integrate them; nonlinear models name their scheme.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from flax import struct

__all__ = [
    "NeuronModel",
    "Spikes",
    "SynapticInput",
    "crossing",
    "exact_linear",
    "integrate",
    "membrane_dtype",
]


def membrane_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """The dtype state integrates in for inputs of `dtype`: float32 or wider."""
    return jnp.promote_types(dtype, jnp.float32)


@struct.dataclass
class SynapticInput:
    """What a population receives in one step.

    `current` (pA) adds to the membrane equation as it is. `conductance`
    (nS) holds one total conductance per receptor, keyed by the receptor's
    name, which the neuron model pairs with its reversal potential
    (`reversal`), so an excitatory and an inhibitory conductance pull the
    voltage toward different targets.
    """

    current: jax.Array | float = 0.0
    conductance: Mapping[str, jax.Array] = struct.field(default_factory=dict)


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


def crossing(before: jax.Array, after: jax.Array, threshold: jax.Array | float) -> jax.Array:
    """Where in the step a membrane going from `before` to `after` crossed `threshold`, as a fraction.

    Linear interpolation of the voltage across the step; 0 when it was already
    at or above threshold at the start, clipped to [0, 1].
    """
    rise = after - before
    fraction = (threshold - before) / jnp.where(rise > 0, rise, 1)
    return jnp.clip(jnp.where(rise > 0, fraction, 0), 0, 1)


def integrate[State](model: NeuronModel[State], inputs: SynapticInput, dt: float,
                     state: State | None = None) -> tuple[Spikes, State]:
    """Scan `model` over time-major `inputs` (every leaf `[T, ...]`, or a scalar held over time).

    Returns the spikes `[T, ...]` and the final state, which a later call
    continues from. The population's shape is the per-step shape of the
    current, or of the first conductance when the current is a scalar.
    """
    leaves = [jnp.asarray(inputs.current), *map(jnp.asarray, inputs.conductance.values())]
    timed = [leaf for leaf in leaves if leaf.ndim > 0]
    if not timed:
        raise ValueError("integrate needs at least one input with a leading time axis")
    steps, shape, dtype = timed[0].shape[0], timed[0].shape[1:], timed[0].dtype
    if state is None:
        state = model.init_state(shape, dtype)
    held = SynapticInput(jnp.broadcast_to(jnp.asarray(inputs.current, dtype), (steps, *shape)),
                         {name: jnp.broadcast_to(jnp.asarray(g, dtype), (steps, *shape))
                          for name, g in inputs.conductance.items()})

    def step(state, inputs):
        state, spikes = model.step(state, inputs, dt)
        return state, spikes

    state, spikes = jax.lax.scan(step, state, held)
    return spikes, state
