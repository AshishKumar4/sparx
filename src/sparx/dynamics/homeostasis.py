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

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from sparx.dynamics.core import NeuronModel, Output, SynapticInput

__all__ = ["IntrinsicPlasticity", "IntrinsicState"]


class IntrinsicState(NamedTuple):
    inner: Any
    """The wrapped model's state."""
    shift: jax.Array
    """How far each neuron's threshold has moved from the wrapped model's."""


@struct.dataclass
class IntrinsicPlasticity:
    """A spiking model whose threshold drifts to hold each neuron at `target` spikes per unit of time.

    After each step a neuron that fired raises its threshold and one that
    did not lowers it:

        theta[t + 1] = theta[t] + eta (s[t] - target dt)

    the intrinsic plasticity of Lazar, Pipa and Triesch's SORN (Frontiers in
    Computational Neuroscience 2009, equation 7), where `s` is 0 or 1 and
    `target dt` is the rate's share of a step. A neuron firing faster than
    `target` grows harder to fire, one firing slower easier, until on
    average it fires at `target`. The threshold is the wrapped model's
    field `field` (`threshold` for the dimensionless models, `v_th` for the
    physical ones), which the shift adds to. The shift follows the spikes
    as they are, so gradients through a run pass through it too.
    """

    inner: NeuronModel
    target: jax.Array | float
    eta: jax.Array | float = 0.001
    field: str = struct.field(pytree_node=False, default="threshold")

    @property
    def graded(self) -> bool:
        return False

    def shifted(self, state: IntrinsicState) -> NeuronModel:
        """The wrapped model with each neuron's threshold where homeostasis has moved it."""
        return self.inner.replace(**{self.field: getattr(self.inner, self.field) + state.shift})

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> IntrinsicState:
        if self.inner.graded:
            raise ValueError("intrinsic plasticity holds a firing rate, and a graded model does not fire")
        if not hasattr(self.inner, self.field):
            raise ValueError(f"{type(self.inner).__name__} has no field {self.field!r} to shift")
        inner = self.inner.init_state(shape, dtype)
        return IntrinsicState(inner, jnp.zeros(shape, jax.tree.leaves(inner)[0].dtype))

    def step(self, state: IntrinsicState, inputs: SynapticInput, dt: float) -> tuple[IntrinsicState, Output]:
        inner, out = self.shifted(state).step(state.inner, inputs, dt)
        shift = state.shift + self.eta * (out.value.astype(state.shift.dtype) - self.target * dt)
        return IntrinsicState(inner, shift), out

    def is_refractory(self, state: IntrinsicState, dt: float) -> jax.Array:
        return self.shifted(state).is_refractory(state.inner, dt)

    def after_threshold(self, state: IntrinsicState, jump: jax.Array, fired: jax.Array) -> IntrinsicState:
        return state._replace(inner=self.shifted(state).after_threshold(state.inner, jump, fired))
