"""Shape layers for Flax linen, over time-major inputs `[T, B, ...]`."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

__all__ = ["Flatten"]


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

    def __call__(self, x: jax.Array) -> jax.Array:
        if not 1 <= self.ndim <= x.ndim:
            raise ValueError(f"cannot flatten the last {self.ndim} axes of a {x.ndim}-d input")
        x = jnp.moveaxis(x, -1, -self.ndim)
        return x.reshape(*x.shape[:x.ndim - self.ndim], -1)
