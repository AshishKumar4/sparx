"""Homeostasis: slow processes that hold a neuron's firing rate near a target.

Two act on a running network. Intrinsic plasticity moves each neuron's
threshold: `IntrinsicPlasticity` wraps any spiking model with a threshold
field and lets that threshold drift by the neuron's own spikes. Synaptic
scaling (`sparx.dynamics.SynapticScaling`, a plasticity rule) scales a
neuron's incoming weights. A network trained by gradients can hold its rates
in a band by its loss instead: `sparx.rate_penalty`, or `RateBand` in a
`sparx.objectives` objective.

Rates are spikes per unit of time: per step for the dimensionless models
at `dt = 1`, per millisecond for the physical ones.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from sparx.dynamics.core import NeuronModel, Output, Reversing, SynapticInput

__all__ = ["IntrinsicPlasticity", "IntrinsicState"]


class IntrinsicState[State](NamedTuple):
    inner: State
    """The wrapped model's state."""
    shift: jax.Array
    """How far each neuron's threshold has moved from the wrapped model's."""


@struct.dataclass
class IntrinsicPlasticity[State]:
    """A spiking model whose threshold drifts to hold each neuron at `target` spikes per unit of time.

    After each step a neuron that fired raises its threshold and one that
    did not lowers it:

        theta[t + 1] = theta[t] + eta (s[t] - target dt)

    the intrinsic plasticity of Lazar, Pipa and Triesch's SORN (Frontiers in
    Computational Neuroscience 2009, equation 7), where `s` is 0 or 1 and
    `target dt` is the rate's share of a step. A neuron firing faster than
    `target` grows harder to fire, one firing slower easier, until on
    average it fires at `target`. The threshold is the wrapped model's
    field `field`, which the shift adds to: `threshold` for the
    dimensionless models, `v_th` for the physical LIF and `v_t` for AdEx.
    Izhikevich's `v_th` and Hodgkin-Huxley's `v_spike` mark the peak of a
    spike already under way rather than a threshold. The shift follows the
    spikes as they are, so gradients through a run pass through it too.
    """

    inner: NeuronModel[State]
    target: jax.Array | float
    eta: jax.Array | float = 0.001
    field: str = struct.field(pytree_node=False, default="threshold")

    @property
    def graded(self) -> bool:
        return False

    @property
    def reversal(self) -> Mapping[str, float]:
        """The wrapped model's reversal potentials (`Reversing`), so a `Network` gives this model the
        conductances it would give the wrapped one."""
        if not isinstance(self.inner, Reversing):
            raise AttributeError(f"{type(self.inner).__name__} reads no conductances")
        return self.inner.reversal

    def shifted(self, state: IntrinsicState[State]) -> NeuronModel[State]:
        """The wrapped model with each neuron's threshold where homeostasis has moved it."""
        inner = self.inner
        if not dataclasses.is_dataclass(inner) or isinstance(inner, type):
            raise TypeError(f"{type(inner).__name__} is not a dataclass whose threshold can be moved")
        return dataclasses.replace(inner, **{self.field: getattr(inner, self.field) + state.shift})

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> IntrinsicState[State]:
        if self.inner.graded:
            raise ValueError("intrinsic plasticity holds a firing rate, and a graded model does not fire")
        if not hasattr(self.inner, self.field):
            raise ValueError(f"{type(self.inner).__name__} has no field {self.field!r} to shift")
        inner = self.inner.init_state(shape, dtype)
        return IntrinsicState(inner, jnp.zeros(shape, jax.tree.leaves(inner)[0].dtype))

    def step(self, state: IntrinsicState[State], inputs: SynapticInput,
             dt: float) -> tuple[IntrinsicState[State], Output]:
        inner, out = self.shifted(state).step(state.inner, inputs, dt)
        shift = state.shift + self.eta * (out.value.astype(state.shift.dtype) - self.target * dt)
        return IntrinsicState(inner, shift.astype(state.shift.dtype)), out

    def is_refractory(self, state: IntrinsicState[State], dt: float) -> jax.Array:
        return self.shifted(state).is_refractory(state.inner, dt)

    def after_threshold(self, state: IntrinsicState[State], jump: jax.Array,
                        fired: jax.Array) -> IntrinsicState[State]:
        return state._replace(inner=self.shifted(state).after_threshold(state.inner, jump, fired))
