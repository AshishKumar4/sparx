"""Turn static or analog data into time-major spike trains `[T, ...]`.

Each encoder adds a leading time axis of `steps`: an image batch `[B, H, W, C]`
becomes `[steps, B, H, W, C]`, ready for a network of `sparx.nn` layers.
Values that mean intensities are expected in [0, 1]. Spikes are float32
unless `dtype` says otherwise.

Direct encoding (`repeat`) feeds the analog values as the input current at
every step and lets the first layer do the encoding, as DIET-SNN (Rathi and
Roy, IEEE TNNLS 2021) and SpikingJelly's static-image examples do.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

__all__ = ["delta", "latency", "rate", "repeat"]


def repeat(x: ArrayLike, steps: int) -> jax.Array:
    """`x` at every one of `steps` steps: `[steps, *x.shape]`, as a view-like broadcast."""
    x = jnp.asarray(x)
    return jnp.broadcast_to(x, (steps, *x.shape))


def rate(key: jax.Array, x: ArrayLike, steps: int, dtype: DTypeLike = jnp.float32) -> jax.Array:
    """Bernoulli spikes that fire with probability `x` at each step, independently.

    `x` is clipped to [0, 1]. The spike count over `steps` is binomial with
    mean `steps * x`. Sampling has no gradient with respect to `x`.
    """
    p = jnp.clip(jnp.asarray(x, jnp.float32), 0, 1)
    return jax.random.bernoulli(key, p, (steps, *p.shape)).astype(dtype)


def latency(x: ArrayLike, steps: int, threshold: float = 0.01, dtype: DTypeLike = jnp.float32) -> jax.Array:
    """One spike per value, earlier for larger values: time-to-first-spike coding.

    A value `x` in [0, 1] fires once, at step `round((1 - x) * (steps - 1))`,
    so 1 fires at the first step and `threshold` near the last; values below
    `threshold` never fire. This is snnTorch's `spikegen.latency` with
    `linear=True, normalize=True, clip=True`.
    """
    x = jnp.clip(jnp.asarray(x, jnp.float32), 0, 1)
    when = jnp.round((1 - x) * (steps - 1)).astype(jnp.int32)
    times = jnp.arange(steps).reshape((steps,) + (1,) * x.ndim)
    return ((times == when) & (x >= threshold)).astype(dtype)


def delta(xs: ArrayLike, threshold: float, off_spikes: bool = False, dtype: DTypeLike = jnp.float32) -> jax.Array:
    """Spike where a time-major signal `[T, ...]` rises by at least `threshold` from the step before.

    The step before the first is zero, so a signal that starts at or above
    `threshold` fires at step 0. With `off_spikes`, a fall of at least
    `threshold` emits -1. This is snnTorch's `spikegen.delta` without padding.
    """
    xs = jnp.asarray(xs)
    change = jnp.diff(xs, axis=0, prepend=jnp.zeros_like(xs[:1]))
    out = (change >= threshold).astype(dtype)
    if off_spikes:
        out = out - (change <= -threshold).astype(dtype)
    return out
