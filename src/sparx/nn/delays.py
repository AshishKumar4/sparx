"""Synapses with learnable transmission delays.

Each synapse from input `i` to output `j` has a weight and a delay: input
spikes reach the output `d_ij` steps after they were emitted. Hammouamri et
al., "Learning Delays in Spiking Neural Networks using Dilated Convolutions
with Learnable Spacings" (ICLR 2024), learn the delays by spreading each
synapse over a temporal kernel as a Gaussian centered at its delay, so the
loss is differentiable in the delay, and narrowing the Gaussian during
training until each synapse reads a single step.

`DelayedDense` computes

    y[t, j] = bias[j] + sum_i sum_k weight[i, j] * g_ij[k] * x[t - k, i]

over time-major `x` `[T, ..., in]`, with `g_ij` the normalized Gaussian of
width `sigma` at `delay[i, j]`, over `k = 0 .. max_delay`. At `sigma=0` each
synapse reads exactly `round(delay)` steps back; that is the network to
deploy. The layer is causal, and the `"state"` collection carries the last
`max_delay` inputs across calls, so a stream can be fed in chunks.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from sparx.cells import membrane_dtype
from sparx.nn.neurons import STATE

__all__ = ["DelayedDense", "delay_kernel"]


def delay_kernel(delay: jax.Array, max_delay: int, sigma: float | jax.Array) -> jax.Array:
    """Each synapse's weights over its `max_delay + 1` lags, `[max_delay + 1, *delay.shape]`.

    A normalized Gaussian of width `sigma` centered at the delay, clipped to
    `[0, max_delay]`; at `sigma` 0 (a Python number), the one-hot of the
    rounded delay, with no gradient to the delay.
    """
    lags = jnp.arange(max_delay + 1, dtype=jnp.float32).reshape((-1,) + (1,) * delay.ndim)
    center = jnp.clip(delay, 0, max_delay)
    if isinstance(sigma, (int, float)) and sigma == 0:
        return (lags == jnp.round(jax.lax.stop_gradient(center))).astype(jnp.float32)
    density = jnp.exp(-0.5 * ((lags - center) / sigma) ** 2)
    return density / jnp.sum(density, axis=0, keepdims=True)


def _uniform(maximum: float) -> nn.initializers.Initializer:
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, 0, maximum)
    return init


class DelayedDense(nn.Module):
    """A dense layer whose every synapse has a learnable delay of 0 to `max_delay` steps.

    Delays start uniform over `[0, max_delay]`. `sigma`, given at each call,
    is the Gaussian's width in steps: Hammouamri et al. start it near
    `max_delay / 2` and decrease it to 0 over training. It may be traced, as a
    schedule's value, as long as it stays positive; pass the Python number 0
    for the deployed network. The cost is that of a dense layer applied
    `max_delay + 1` times.
    """

    features: int
    max_delay: int
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    precision: jax.lax.Precision | None = None

    @nn.compact
    def __call__(self, x: jax.Array, sigma: float | jax.Array) -> jax.Array:
        if self.max_delay < 0:
            raise ValueError(f"max_delay must be at least 0, not {self.max_delay}")
        inputs = x.shape[-1]
        weight = self.param("kernel", self.kernel_init, (inputs, self.features), jnp.float32)
        delay = self.param("delay", _uniform(float(self.max_delay)), (inputs, self.features), jnp.float32)
        dtype = membrane_dtype(x.dtype)
        kernel = (delay_kernel(delay, self.max_delay, sigma) * weight).astype(dtype)  # [K, in, out]

        held = self.max_delay
        carrying = self.is_mutable_collection(STATE) and not self.is_initializing()
        history = self.get_variable(STATE, "carry") if carrying else None
        if history is None:
            history = jnp.zeros((held, *x.shape[1:]), x.dtype)
        window = jnp.concatenate([jnp.asarray(history, x.dtype), x]).astype(dtype)
        if carrying:
            self.put_variable(STATE, "carry", window[window.shape[0] - held:].astype(x.dtype))

        steps = x.shape[0]
        # Lag k reads the window from offset held - k: output step t sees x[t - k].
        y = jnp.matmul(window[held:held + steps], kernel[0], precision=self.precision)
        for k in range(1, held + 1):
            y = y + jnp.matmul(window[held - k:held - k + steps], kernel[k], precision=self.precision)
        if self.use_bias:
            y = y + self.param("bias", nn.initializers.zeros, (self.features,), jnp.float32).astype(dtype)
        return y
