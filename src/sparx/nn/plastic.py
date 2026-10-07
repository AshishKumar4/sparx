"""Plastic recurrent layers: fast weights that a Hebbian trace writes as each sequence runs.

`Plastic` feeds a neuron layer's output back through a learned matrix plus
`alpha * hebb`, where the Hebbian trace `hebb` starts at zero for every
sequence and follows a rule of the units' own activity
(`sparx.dynamics.PlasticRecurrentCell`). Gradient descent learns the fixed
weights, each connection's plasticity `alpha` and the rule's parameters,
through the traces: differentiable plasticity (Miconi et al. 2018) and its
neuromodulated form, Backpropamine (Miconi et al. 2019).

The rule is a module, as the neuron is, so it declares its own learned
parameters:

    layer = sparx.nn.Plastic(sparx.nn.Rate(tau=0), rule=sparx.nn.ModulatedTrace())

A `HebbianTrace` builds a `sparx.dynamics.HebbianRule`, and a new rule is a
dataclass with `init_trace`, `hebb` and `update` and a `HebbianTrace` whose
`build` declares its parameters and returns it.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike

from sparx.dynamics import (
    DecayingHebb,
    HebbianRule,
    ModulatedHebb,
    OjaHebb,
    PlasticRecurrentCell,
    RetroactiveHebb,
    SynapticInput,
)

from .neurons import LIF, Neuron, adopt

__all__ = ["DecayingTrace", "HebbianTrace", "ModulatedTrace", "OjaTrace", "Plastic", "RetroactiveTrace"]


def _linear_init(fan_in: int) -> nn.initializers.Initializer:
    """`torch.nn.Linear`'s default for its weight and its bias, uniform on `+-1 / sqrt(fan_in)`.

    Miconi et al. read their neuromodulator through `torch.nn.Linear`
    layers, so their training starts from these draws.
    """
    bound = fan_in ** -0.5

    def init(key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> jax.Array:
        return jax.random.uniform(key, shape, dtype, -bound, bound)

    return init


class HebbianTrace(nn.Module):
    """The Hebbian trace of a `Plastic` layer, which builds its rule with the rule's learned parameters.

    A subclass declares the parameters in `build(features)`, for `features`
    units, and returns the rule, a `sparx.dynamics.HebbianRule`. `Plastic`
    calls it as its child `rule`, so they sit under `rule`.
    """

    def build(self, features: int) -> HebbianRule:
        raise NotImplementedError

    @nn.compact
    def __call__(self, features: int) -> HebbianRule:
        return self.build(features)


class DecayingTrace(HebbianTrace):
    """`sparx.dynamics.DecayingHebb` with a learned rate starting at `eta`, Miconi et al.'s 0.01."""

    eta: float = 0.01

    def build(self, features: int) -> DecayingHebb:
        return DecayingHebb(self.param("eta", nn.initializers.constant(self.eta), (), jnp.float32))


class OjaTrace(HebbianTrace):
    """`sparx.dynamics.OjaHebb`, Oja's rule, with a learned rate starting at `eta`."""

    eta: float = 0.01

    def build(self, features: int) -> OjaHebb:
        return OjaHebb(self.param("eta", nn.initializers.constant(self.eta), (), jnp.float32))


class ModulatedTrace(HebbianTrace):
    """`sparx.dynamics.ModulatedHebb`: a learned neuromodulator sets each unit's rate, clipped at `clip`.

    The modulator reads the units (`[F]` and a bias) and fans out to them
    (`[F]` and `[F]`), Backpropamine's `h2mod` and `modfanout`, which start
    as `torch.nn.Linear` does; `clip` is their code's 2.
    """

    clip: float = 2.0

    def build(self, features: int) -> ModulatedHebb:
        modulator = self.param("modulator", _linear_init(features), (features,), jnp.float32)
        modulator_bias = self.param("modulator_bias", _linear_init(features), (), jnp.float32)
        fanout = self.param("fanout", _linear_init(1), (features,), jnp.float32)
        fanout_bias = self.param("fanout_bias", _linear_init(1), (features,), jnp.float32)
        return ModulatedHebb(modulator, modulator_bias, fanout, fanout_bias, self.clip)


class RetroactiveTrace(HebbianTrace):
    """`sparx.dynamics.RetroactiveHebb`: a learned neuromodulator writes recent coactivity into the weights.

    The eligibility decays at a learned rate starting at `eta`, their 0.01.
    The modulator reads the units (`[F]` and a bias), Backpropamine's
    `h2DA`, which starts as `torch.nn.Linear` does; `clip` is their code's
    1.
    """

    eta: float = 0.01
    clip: float = 1.0

    def build(self, features: int) -> RetroactiveHebb:
        modulator = self.param("modulator", _linear_init(features), (features,), jnp.float32)
        modulator_bias = self.param("modulator_bias", _linear_init(features), (), jnp.float32)
        eta = self.param("eta", nn.initializers.constant(self.eta), (), jnp.float32)
        return RetroactiveHebb(modulator, modulator_bias, eta, self.clip)


class Plastic(Neuron):
    """Feed `neuron`'s output back through a learned matrix plus fast weights that each sequence writes.

    The feedback runs through `recurrent + alpha * hebb`, both `[F, F]` and
    learned, where `rule` keeps `hebb` (`HebbianTrace`): `DecayingTrace`
    and `OjaTrace` (Miconi et al. 2018), `ModulatedTrace` and
    `RetroactiveTrace` (Backpropamine, Miconi et al. 2019).
    `Plastic(Rate(tau=0))` is their tanh network, and a spiking neuron
    learns fast weights between its spikes. `alpha` starts at `alpha_init`,
    small and random as theirs does (`.01 * randn` in their
    `simple/simple.py`); `recurrent` starts orthogonal, as `Recurrent`'s
    does, where theirs is `kernel_init=nn.initializers.normal(0.01)`. As in
    `Recurrent`, the input projection stays outside, the neuron's
    parameters live under `neuron`, and the layer's `dt` must be the
    neuron's. Backpropagating holds a `[B, F, F]` trace per step.
    """

    neuron: Neuron = LIF()
    rule: HebbianTrace = DecayingTrace()
    kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    alpha_init: nn.initializers.Initializer = nn.initializers.normal(0.01)
    precision: PrecisionLike = None

    def inputs(self, x: jax.Array) -> SynapticInput:
        return self.neuron.inputs(x)

    def build(self, x: jax.Array) -> PlasticRecurrentCell:
        if self.dt != self.neuron.dt:
            raise ValueError(f"Plastic steps at dt={self.dt}, its neuron at dt={self.neuron.dt}; "
                             "give both one dt")
        features = x.shape[-1]
        weight = self.param("recurrent", self.kernel_init, (features, features), jnp.float32)
        alpha = self.param("alpha", self.alpha_init, (features, features), jnp.float32)
        return PlasticRecurrentCell(adopt(self.neuron, self, "neuron").model(x), weight, alpha,
                                    adopt(self.rule, self, "rule")(features), self.precision)
