"""Turn a batch field into the time-major input `[T, B, ...]` of a spiking network.

An encoder is a frozen dataclass called as `encoder(key, x)` on a batch
field `[B, ...]`. The same object encodes in a plain JAX loop and inside
`sparx.objectives.SpikingClassifierObjective`, and it is registered
(`sparx.registry.spike_encoders`), so a run's record holds it as
`{"name": "rate", "fields": {"steps": 8}}` and rebuilds it in another
process.

Encoders of static data (`Direct`, `Rate`, `Latency`) add a leading time
axis of `steps`: an image batch `[B, H, W, C]` becomes `[steps, B, H, W, C]`.
Encoders of data that already runs over time (`Delta`, `Events`) move each
record's time axis to the front: `[B, T, F]` becomes `[T, B, F]`.

The encoders that read values as intensities (`Direct`, `Rate`, `Latency`,
`Delta`) read a uint8 field as `x / 255` and expect anything else in
[0, 1], so raw image bytes and normalized arrays encode alike. `Events`
reads spike counts or currents, which it passes on unscaled. Every encoder
returns float32.

Direct encoding feeds the analog values as the input current at every step
and lets the first layer do the encoding, as DIET-SNN (Rathi and Roy, IEEE
TNNLS 2021) and SpikingJelly's static-image examples do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from sparx.registry import spike_encoders

__all__ = ["Delta", "Direct", "Events", "Latency", "Rate", "SpikeEncoder"]


class SpikeEncoder(ABC):
    """Turns one batch field `[B, ...]` into the network's time-major input `[T, B, ...]`.

    `key` drives the random encoders (`Rate`); the others ignore it, so
    every encoder is called the same way. It is a JAX PRNG key, as
    `jax.random`'s functions take, never an int seed: an encoder runs inside
    jitted training steps, where the caller splits one key per step. The
    entry points called from outside JAX (dew's `Trainer`,
    `SpikingClassification.logits`) take an int seed, as dew's do, and
    make the key. A subclass implements `encode`.
    """

    def __call__(self, key: jax.Array, x: ArrayLike) -> jax.Array:
        if not _is_key(key):
            given = f"{jnp.asarray(key).dtype}{list(jnp.shape(key))}"
            raise TypeError(f"{type(self).__name__} is called as encoder(key, x) with a JAX PRNG key such as "
                            f"jax.random.key(0); its first argument was {given}")
        return self.encode(key, x)

    @abstractmethod
    def encode(self, key: jax.Array, x: ArrayLike) -> jax.Array: ...


def _is_key(key: ArrayLike) -> bool:
    """A typed key (`jax.random.key`) or a raw one (`jax.random.PRNGKey`, uint32 `[..., 2]`)."""
    if not isinstance(key, jax.Array | np.ndarray):
        return False
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return True
    return key.dtype == jnp.uint32 and key.ndim >= 1 and key.shape[-1] == 2


def _intensities(x: ArrayLike) -> jax.Array:
    """uint8 pixels as [0, 1]; other values unchanged, as float32."""
    x = jnp.asarray(x)
    if x.dtype == jnp.uint8:
        return x.astype(jnp.float32) / 255
    return x.astype(jnp.float32)


@spike_encoders("direct")
@dataclass(frozen=True)
class Direct(SpikeEncoder):
    """The values themselves as the input current at each of `steps` steps, as a broadcast."""

    steps: int

    def encode(self, key: jax.Array, x: ArrayLike) -> jax.Array:
        x = _intensities(x)
        return jnp.broadcast_to(x, (self.steps, *x.shape))


@spike_encoders("rate")
@dataclass(frozen=True)
class Rate(SpikeEncoder):
    """Bernoulli spikes that fire with probability `x` at each of `steps` steps, independently.

    `x` is clipped to [0, 1]. The spike count over `steps` is binomial with
    mean `steps * x`. Each key gives a fresh draw, so a training loop that
    passes its step's key encodes every step anew. Sampling has no gradient
    with respect to `x`.
    """

    steps: int

    def encode(self, key: jax.Array, x: ArrayLike) -> jax.Array:
        p = jnp.clip(_intensities(x), 0, 1)
        return jax.random.bernoulli(key, p, (self.steps, *p.shape)).astype(jnp.float32)


@spike_encoders("latency")
@dataclass(frozen=True)
class Latency(SpikeEncoder):
    """One spike per value over `steps` steps, earlier for larger values: time-to-first-spike coding.

    A value `x` in [0, 1] fires once, at step `round((1 - x) * (steps - 1))`,
    so 1 fires at the first step and `threshold` near the last; values below
    `threshold` never fire. This is snnTorch's `spikegen.latency` with
    `linear=True, normalize=True, clip=True`.
    """

    steps: int
    threshold: float = 0.01

    def encode(self, key: jax.Array, x: ArrayLike) -> jax.Array:
        x = jnp.clip(_intensities(x), 0, 1)
        when = jnp.round((1 - x) * (self.steps - 1)).astype(jnp.int32)
        times = jnp.arange(self.steps).reshape((self.steps,) + (1,) * x.ndim)
        return ((times == when) & (x >= self.threshold)).astype(jnp.float32)


def _time_major(x: jax.Array, time_axis: int) -> jax.Array:
    """A batch `[B, ...]` of records with time on `time_axis`, time moved to the front."""
    return jnp.moveaxis(x, time_axis + 1, 0)


@spike_encoders("delta")
@dataclass(frozen=True)
class Delta(SpikeEncoder):
    """Spike where a signal rises by at least `threshold` from the step before.

    Each record holds the signal over time on axis `time_axis`. The step
    before the first is zero, so a signal that starts at or above
    `threshold` fires at step 0. With `off_spikes`, a fall of at least
    `threshold` emits -1. This is snnTorch's `spikegen.delta` without padding.
    """

    threshold: float
    off_spikes: bool = False
    time_axis: int = 0

    def encode(self, key: jax.Array, x: ArrayLike) -> jax.Array:
        xs = _time_major(_intensities(x), self.time_axis)
        change = jnp.diff(xs, axis=0, prepend=jnp.zeros_like(xs[:1]))
        out = (change >= self.threshold).astype(jnp.float32)
        if self.off_spikes:
            out = out - (change <= -self.threshold).astype(jnp.float32)
        return out


@spike_encoders("events")
@dataclass(frozen=True)
class Events(SpikeEncoder):
    """Data that already holds spikes or currents over time, on axis `time_axis` of each record.

    A record `[T, F]` arrives batched as `[B, T, F]`; the default moves its
    time axis to the front. The values pass unscaled, as float32: a uint8
    field here counts spikes.
    """

    time_axis: int = 0

    def encode(self, key: jax.Array, x: ArrayLike) -> jax.Array:
        return _time_major(jnp.asarray(x), self.time_axis).astype(jnp.float32)
