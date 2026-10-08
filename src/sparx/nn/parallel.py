"""Parallel spiking neurons: membranes that mix time steps with a learned
matrix instead of a recurrence, so a layer is one matrix product over time.

Fang et al., "Parallel Spiking Neurons with High Efficiency and Long-term
Dependencies Learning Ability" (NeurIPS 2023). Removing the reset makes the
LIF membrane a fixed linear combination of past inputs; the PSN learns that
combination,

    H = W X + b,    S = H(H)    over time-major X `[T, ...]`,

and fires where `H >= 0`. With no loop over time, every step is computed by
one `[T, T] x [T, N]` product, which suits matrix units. The three variants
differ in which entries of `W` they learn:

- `PSN`: all `T x T`, non-causal, for a fixed `T`.
- `MaskedPSN`: the `k` most recent steps, `W[t, t-k+1 .. t]`, for a fixed `T`.
- `SlidingPSN`: `k` weights shared across time, for any `T`, and streamable.

Parameter layout, initialization and operation order follow SpikingJelly's
`activation_based.neuron.psn` (commit c6cb8e46); `tests/test_reference.py`
checks spikes and gradients against it. The bias `b` starts at -1, a
threshold of 1. SpikingJelly's `lambda_` for MaskedPSN's progressive masking
is the `masking` argument of the call here.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from dew.nn.precision import at_least_fp32

from sparx.nn.neurons import STATE, history_window, record_rates
from sparx.surrogate import ATan, Surrogate, spike

__all__ = ["PSN", "MaskedPSN", "SlidingPSN", "band_mask"]

# PyTorch's kaiming_uniform_ with a = sqrt(5), SpikingJelly's choice: uniform
# in +-1 / sqrt(fan_in), with fan_in the size of the input (column) axis.
_kaiming_uniform = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform", in_axis=-1, out_axis=-2)


def band_mask(steps: int, k: int) -> jax.Array:
    """`M[i, j] = 1` where `j <= i <= j + k - 1`: each step sees itself and the `k - 1` before it."""
    if k < 1:
        raise ValueError(f"the order k must be at least 1, not {k}")
    i, j = jnp.arange(steps)[:, None], jnp.arange(steps)[None, :]
    return ((j <= i) & (i <= j + k - 1)).astype(jnp.float32)


def _mix(weight: jax.Array, bias: jax.Array, x: jax.Array, precision: jax.lax.Precision | None) -> jax.Array:
    """`weight @ x + bias` over the time axis of `x`, `[T, ...]`, in membrane precision."""
    dtype = at_least_fp32(x.dtype)
    h = jnp.tensordot(weight.astype(dtype), x.astype(dtype), axes=(1, 0), precision=precision)
    return h + bias.reshape(bias.shape + (1,) * (x.ndim - 1)).astype(dtype)


def _fire(module: nn.Module, h: jax.Array, surrogate: Surrogate, dtype: jnp.dtype) -> jax.Array:
    spikes = spike(h, surrogate).astype(dtype)
    record_rates(module, spikes)
    return spikes


def _refuse_streaming(module: nn.Module) -> None:
    if module.is_mutable_collection(STATE) and not module.is_initializing():
        raise ValueError(
            f"{type(module).__name__} mixes every step of its fixed T, so it cannot "
            "continue across calls; stream with SlidingPSN or a recurrent neuron")


class PSN(nn.Module):
    """The parallel spiking neuron over all `T x T` step pairs, for a fixed `T`.

    `T` is the input's leading axis at `init`. Each output step reads every
    input step, earlier and later, so the layer suits classification of a
    whole sequence, not causal or streaming use.
    """

    surrogate: Surrogate = ATan()
    precision: jax.lax.Precision | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        _refuse_streaming(self)
        steps = x.shape[0]
        weight = self.param("weight", _kaiming_uniform, (steps, steps), jnp.float32)
        bias = self.param("bias", nn.initializers.constant(-1.0), (steps,), jnp.float32)
        return _fire(self, _mix(weight, bias, x, self.precision), self.surrogate, x.dtype)


class MaskedPSN(nn.Module):
    """The PSN restricted to the `k` most recent steps, for a fixed `T`.

    `masking` blends the band mask with all ones, `masking * M + (1 -
    masking)`: 0 is the unmasked PSN, 1 the causal order-`k` neuron.
    Fang et al. raise it from 0 to 1 during training (SpikingJelly's
    `lambda_`); pass the schedule's value at each call. It may be traced.
    """

    k: int
    surrogate: Surrogate = ATan()
    precision: jax.lax.Precision | None = None

    @nn.compact
    def __call__(self, x: jax.Array, masking: float | jax.Array = 1.0) -> jax.Array:
        _refuse_streaming(self)
        steps = x.shape[0]
        weight = self.param("weight", _kaiming_uniform, (steps, steps), jnp.float32)
        bias = self.param("bias", nn.initializers.constant(-1.0), (steps,), jnp.float32)
        mask = band_mask(steps, self.k)
        weight = (masking * mask + (1 - masking)) * weight
        return _fire(self, _mix(weight, bias, x, self.precision), self.surrogate, x.dtype)


def _exponential(key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """`(..., 1/4, 1/2, 1)`: SpikingJelly's `exp_init`, an LIF-like decay over the window."""
    del key
    (k,) = shape
    return 0.5 ** jnp.arange(k - 1, -1, -1, dtype=dtype)


def _kaiming_row(key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """SpikingJelly's other SlidingPSN initialization: kaiming uniform on a `[1, k]` matrix."""
    return _kaiming_uniform(key, (1, *shape), dtype)[0]


class SlidingPSN(nn.Module):
    """`k` weights slid over time: `H[t] = sum_i weight[i] * X[t - k + 1 + i] + bias`.

    Works for any `T`, and is causal, so the `"state"` collection carries the
    last `k - 1` inputs across calls and a stream can be fed in chunks.
    `weight[k - 1]` multiplies the current step; the first steps of a fresh
    sequence see zeros before it. `exponential_init` starts the weights at
    `(..., 1/4, 1/2, 1)`, otherwise kaiming uniform (SpikingJelly's
    `exp_init`). The membrane is `k` weighted slices of the window, `O(T k)`,
    in membrane precision; SpikingJelly builds the `[T, T]` Toeplitz matrix
    of the weights instead.
    """

    k: int
    exponential_init: bool = True
    surrogate: Surrogate = ATan()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.k < 1:
            raise ValueError(f"the order k must be at least 1, not {self.k}")
        init = _exponential if self.exponential_init else _kaiming_row
        weight = self.param("weight", init, (self.k,), jnp.float32)
        bias = self.param("bias", nn.initializers.constant(-1.0), (), jnp.float32)
        steps, dtype = x.shape[0], at_least_fp32(x.dtype)
        window = history_window(self, x, self.k - 1).astype(dtype)
        weight = weight.astype(dtype)
        # Step t reads the window's steps t .. t + k - 1, the last of them x[t].
        h = sum((weight[i] * window[i:i + steps] for i in range(self.k)), bias.astype(dtype))
        return _fire(self, h, self.surrogate, x.dtype)
