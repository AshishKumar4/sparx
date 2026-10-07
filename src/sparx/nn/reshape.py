"""Shape layers for Flax linen, over time-major inputs `[T, B, ...]`, and a wrapper for batch-major ones."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import flax.linen as nn
import jax
import jax.numpy as jnp

__all__ = ["BatchMajor", "Flatten", "Flattens"]


@runtime_checkable
class Flattens(Protocol):
    """A layer that flattens each example's trailing axes into one feature axis in PyTorch's order,
    channels first, and changes nothing else: `flattened_axes()` is how many trailing axes.

    A consumer that maps layers to another library's (NIR's `Flatten`) or
    passes them through a conversion asks a layer for this, not for its class.
    """

    def flattened_axes(self) -> int: ...

    def __call__(self, x: jax.Array) -> jax.Array: ...


class Flatten(nn.Module):
    """Flatten each example's trailing `ndim` axes into one feature axis, channels first.

    Flax lays images out channels last, `[..., H, W, C]`, and a plain
    reshape would order their features H, W, C. PyTorch and NIR lay images
    out channels first and flatten them C, H, W. This layer moves the
    channel axis (the last) in front of the other flattened axes before the
    reshape, so feature `c * H * W + h * W + w` holds channel `c` at
    `(h, w)`, the order of `torch.nn.Flatten`. A dense layer after it then
    takes PyTorch's or NIR's weights as they are, at the cost of one
    transpose. The axes before the last `ndim` (time and batch) are kept.
    """

    ndim: int = 3

    def flattened_axes(self) -> int:
        """`ndim`, the trailing axes this layer flattens (`Flattens`)."""
        return self.ndim

    def __call__(self, x: jax.Array) -> jax.Array:
        if not 1 <= self.ndim <= x.ndim:
            raise ValueError(f"cannot flatten the last {self.ndim} axes of a {x.ndim}-d input")
        x = jnp.moveaxis(x, -1, -self.ndim)
        return x.reshape(*x.shape[:x.ndim - self.ndim], -1)


class BatchMajor(nn.Module):
    """Run a time-major `layer` (a sparx layer or a stack of them) on batch-major input `[B, T, ...]`.

    dew batches records along the first axis, so a model that dew's
    `Supervised` objective trains reads `[B, T, ...]` and its loss reads
    one row per example. This wrapper hands `layer` the input time-major,
    `[T, B, ...]`, and returns its output batch-major again. The layer's
    parameters sit under `layer`.
    """

    layer: nn.Module

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.swapaxes(self.layer(jnp.swapaxes(x, 0, 1)), 0, 1)
