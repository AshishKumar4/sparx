"""The Hebbian traces of plastic recurrent layers, as modules that declare their learned parameters.

A recurrent layer given a `rule` (`sparx.nn.Recurrent(neuron, rule=...)`,
`sparx.graph.connectome.FLYNN(..., rule=...)`) adds fast weights to its
wiring: `alpha * hebb`, where the Hebbian trace `hebb` starts at zero for
every sequence and follows the rule from the units' own activity
(`sparx.dynamics.FastWeights`). Gradient descent learns the fixed weights,
each connection's `alpha` and the rule's parameters through the traces:
differentiable plasticity (Miconi et al. 2018) and its neuromodulated form,
Backpropamine (Miconi et al. 2019).

    layer = sparx.nn.Recurrent(sparx.nn.Rate(tau=0), rule=sparx.nn.ModulatedTrace())

A `HebbianTrace` builds a `sparx.dynamics.HebbianRule`, which runs on a
dense wiring and on a connectome's sparse one alike. A new rule is a
dataclass with `init_trace`, `hebb` and `update` and a `HebbianTrace` whose
`build` declares its parameters and returns it.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from sparx.dynamics import DecayingHebb, HebbianRule, ModulatedHebb, OjaHebb, RetroactiveHebb

__all__ = ["DecayingTrace", "HebbianTrace", "ModulatedTrace", "OjaTrace", "RetroactiveTrace"]


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
    """The Hebbian trace of a `Recurrent` layer, which builds its rule with the rule's learned parameters.

    A subclass declares the parameters in `build(features)`, for `features`
    units, and returns the rule, a `sparx.dynamics.HebbianRule`. A layer
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

