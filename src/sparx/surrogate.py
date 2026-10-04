"""The spike nonlinearity and the gradients it trains with.

A spike is the Heaviside step of `x = v - threshold`: 1 where the membrane
reaches the threshold, 0 below it. Its true derivative is zero almost
everywhere, so training replaces it with the derivative of a smooth step, the
surrogate, while the forward pass stays exactly binary.

`spike` is a `jax.custom_jvp`, so the surrogate serves forward mode
(`jax.jvp`, forward gradients) and reverse mode (`jax.grad`, which JAX gets by
transposing the linear tangent rule) from one definition, and it batches under
`vmap`. A surrogate is a frozen dataclass: hashable, so it is a static
argument of the rule and a field of a Flax module.

Shapes follow the input; the spike has the input's dtype, which holds 0 and 1
exactly in every float format.
"""

from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "ATan",
    "FastSigmoid",
    "Gaussian",
    "Rectangle",
    "Sigmoid",
    "StraightThrough",
    "Surrogate",
    "Triangle",
    "spike",
]


class Surrogate(ABC):
    """The derivative that stands in for the Heaviside step's in a backward pass.

    `derivative(x)` is evaluated at `x = v - threshold`. Calling a surrogate
    spikes: `ATan()(x)` is `spike(x, ATan())`.
    """

    @abstractmethod
    def derivative(self, x: jax.Array) -> jax.Array:
        """The surrogate's d spike / d x at `x`, in `x`'s dtype."""

    def __call__(self, x: jax.Array) -> jax.Array:
        return spike(x, self)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def spike(x: jax.Array, surrogate: Surrogate) -> jax.Array:
    """The Heaviside step of `x`, 1 where `x >= 0`, differentiated through `surrogate`."""
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError(f"spike takes a floating membrane, not {x.dtype}")
    return (x >= 0).astype(x.dtype)


@spike.defjvp
def _spike_jvp(surrogate: Surrogate, primals: tuple[jax.Array], tangents: tuple[jax.Array]):
    (x,), (dx,) = primals, tangents
    return spike(x, surrogate), surrogate.derivative(x).astype(x.dtype) * dx


@dataclass(frozen=True)
class ATan(Surrogate):
    """The arctangent step's derivative, `alpha / 2 / (1 + (pi / 2 * alpha * x)^2)`.

    Fang et al., "Incorporating Learnable Membrane Time Constant to Enhance
    Learning of Spiking Neural Networks" (ICCV 2021), and SpikingJelly's
    `surrogate.ATan` and snnTorch's `atan` with the same `alpha`. It
    integrates to 1 and its tails fall as `1 / x^2`, so a neuron far from
    threshold still receives gradient.
    """

    alpha: float = 2.0

    def derivative(self, x: jax.Array) -> jax.Array:
        return self.alpha / 2 / (1 + (math.pi / 2 * self.alpha * x) ** 2)


@dataclass(frozen=True)
class Sigmoid(Surrogate):
    """The logistic step's derivative, `alpha * sigmoid(alpha x) * (1 - sigmoid(alpha x))`.

    SpikingJelly's `surrogate.Sigmoid` (default `alpha=4`); snnTorch's
    `sigmoid` names `alpha` its slope (default 25). It integrates to 1.
    """

    alpha: float = 4.0

    def derivative(self, x: jax.Array) -> jax.Array:
        s = jax.nn.sigmoid(self.alpha * x)
        return self.alpha * s * (1 - s)


@dataclass(frozen=True)
class FastSigmoid(Surrogate):
    """SuperSpike's derivative, `1 / (slope * |x| + 1)^2`.

    Zenke and Ganguli, "SuperSpike: Supervised Learning in Multilayer Spiking
    Neural Networks" (Neural Computation 2018), and snnTorch's `fast_sigmoid`
    (default `slope=25`); Zenke's SHD tutorials use `slope=100`. Its peak is
    1 at threshold whatever the slope, so it integrates to `2 / slope`, not 1.
    """

    slope: float = 25.0

    def derivative(self, x: jax.Array) -> jax.Array:
        return 1 / (self.slope * jnp.abs(x) + 1) ** 2


@dataclass(frozen=True)
class Triangle(Surrogate):
    """A piecewise linear bump, `scale * max(0, 1 - |x| / width)`.

    Bellec et al., "Long short-term memory and learning-to-learn in networks
    of spiking neurons" (NeurIPS 2018), use `scale=0.3` on a membrane already
    divided by its threshold. Zero beyond `width`, so neurons far from
    threshold pass no gradient.
    """

    width: float = 1.0
    scale: float = 1.0

    def derivative(self, x: jax.Array) -> jax.Array:
        return self.scale * jnp.maximum(0, 1 - jnp.abs(x) / self.width)


@dataclass(frozen=True)
class Rectangle(Surrogate):
    """A box of height `1 / width` over `|x| < width / 2`, which integrates to 1.

    The rectangular window of Wu et al., "Spatio-Temporal Backpropagation for
    Training High-performance Spiking Neural Networks" (Frontiers 2018).
    """

    width: float = 1.0

    def derivative(self, x: jax.Array) -> jax.Array:
        return (jnp.abs(x) < self.width / 2) / self.width


@dataclass(frozen=True)
class Gaussian(Surrogate):
    """The normal density with standard deviation `sigma`, which integrates to 1.

    The Gaussian window of Wu et al. (Frontiers 2018).
    """

    sigma: float = 0.5

    def derivative(self, x: jax.Array) -> jax.Array:
        return jnp.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * math.sqrt(2 * math.pi))


@dataclass(frozen=True)
class StraightThrough(Surrogate):
    """The identity's derivative, 1 everywhere: the straight-through estimator."""

    def derivative(self, x: jax.Array) -> jax.Array:
        return jnp.ones_like(x)
