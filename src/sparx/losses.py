"""Differentiable losses over a network's time-major outputs `[T, B, ...]`.

The classification losses take outputs `[T, B, C]` and integer labels;
`van_rossum` compares spikes with target spike trains, the loss of fitting a
network to recordings. The statistics that only measure spike trains after a
run, without a gradient, are `sparx.spiketrains`.

The classification losses return one loss per example, `[B]`, so a caller
chooses the reduction (dew's objectives sum it over a `Ratio`). For a loss on
one readout of the outputs, reduce time first and use optax: the maximum membrane of a leaky
integrator readout (`jnp.max(v, axis=0)`, Cramer et al., IEEE TNNLS 2020),
its mean, or the spike count (`jnp.sum(spikes, axis=0)`), each as logits for
`optax.softmax_cross_entropy_with_integer_labels`. `softmax_sum_cross_entropy`
is the readout of Hammouamri et al.'s SNN-delays, which takes the softmax
at every step before summing over time.

`Readout` names those choices, and `readout_logits` and `readout_losses`
apply one, so a classifier's objective, its trained task and a recipe's
command line share one list.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import optax

__all__ = ["READOUTS", "Readout", "per_step_cross_entropy", "rate_mse", "readout_logits", "readout_losses",
           "softmax_sum", "softmax_sum_cross_entropy", "van_rossum"]

type Readout = Literal["mean", "max", "sum", "softmax_sum", "per_step"]
"""How the outputs `[T, B, C]` are scored against the labels.

- `mean`: cross entropy of the time-averaged outputs, a firing rate for
  spikes or the mean membrane of a leaky integrator readout.
- `max`: cross entropy of each class's largest output over time, the readout
  Cramer et al. (IEEE TNNLS 2020) use on a leaky integrator for SHD.
- `sum`: cross entropy of the summed outputs, the spike count.
- `softmax_sum`: the softmax of every step summed over time, scored as
  logits (`softmax_sum_cross_entropy`), SNN-delays' `loss='sum'`.
- `per_step`: cross entropy at every step, averaged (`per_step_cross_entropy`).
  Predictions use the mean.
"""

READOUTS: tuple[Readout, ...] = ("mean", "max", "sum", "softmax_sum", "per_step")
"""Every `Readout`, for a value read from a record or a command line."""


def per_step_cross_entropy(outputs: jax.Array, labels: jax.Array) -> jax.Array:
    """Cross entropy of the outputs at every step against `labels`, averaged over time.

    The temporal term of Deng et al., "Temporal Efficient Training of Spiking
    Neural Network via Gradient Re-weighting" (ICLR 2022), and snnTorch's
    `ce_rate_loss`. Each step is asked to classify alone, which they find
    generalizes better than the loss of the time-averaged output. Computed in
    float32.
    """
    steps = outputs.shape[0]
    logits = outputs.astype(jnp.float32)
    labels = jnp.broadcast_to(labels, logits.shape[:-1])
    per_step = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.sum(per_step, axis=0) / steps


def softmax_sum(outputs: jax.Array) -> jax.Array:
    """The class probabilities of every step summed over time, `[T, B, C] -> [B, C]`, in float32.

    Each step votes with a distribution that sums to 1, so no single step's
    large membrane outweighs the rest, as it does in a sum or maximum of the
    raw outputs. The argmax is SNN-delays' prediction.
    """
    return jnp.sum(jax.nn.softmax(outputs.astype(jnp.float32), axis=-1), axis=0)


def softmax_sum_cross_entropy(outputs: jax.Array, labels: jax.Array) -> jax.Array:
    """Cross entropy of `softmax_sum(outputs)` taken as logits, SNN-delays' `loss='sum'`.

    Their `calc_loss` passes the summed probabilities to torch's
    `CrossEntropyLoss`, which applies a log-softmax to them again; this
    keeps that, so the loss is theirs. The summed probabilities lie in
    `[0, T]`, so the loss cannot fall below `log(1 + (C - 1) exp(-T))`.
    """
    return optax.softmax_cross_entropy_with_integer_labels(softmax_sum(outputs), labels)


def rate_mse(spikes: jax.Array, labels: jax.Array, correct: float = 0.8, incorrect: float = 0.2) -> jax.Array:
    """Mean squared error between each output neuron's firing rate and its target rate.

    The labelled class is asked to fire at `correct` of the steps and every
    other class at `incorrect`, so no neuron is pushed to silence or to
    saturation. It is snnTorch's `mse_count_loss` divided by `T` (which
    returns the squared error of spike counts over `T`), so it does not grow
    with `T`, where `T * correct` and `T * incorrect` are whole numbers
    (snnTorch rounds its target counts down).
    """
    rates = jnp.mean(spikes.astype(jnp.float32), axis=0)
    target = jnp.where(jax.nn.one_hot(labels, rates.shape[-1], dtype=bool), correct, incorrect)
    return jnp.mean((rates - target) ** 2, axis=-1)


def van_rossum(spikes: jax.Array, target: jax.Array, tau: float, dt: float = 1.0) -> jax.Array:
    """Van Rossum's (Neural Computation 2001) distance between spike trains, squared; differentiable.

        D^2 = (1 / tau) * integral of (f(t) - g(t))^2 dt,   f = sum_k exp(-(t - t_k) / tau) H(t - t_k)

    for time-major spike counts `[T, ...]` on a grid of `dt`, compared
    elementwise (one train per trailing index) and summed over trains.
    Between grid points the difference of the filtered trains decays
    exponentially, so the integral is exact for spikes on the grid, its tail
    after the last step included: `h_n^2 (1 - exp(-2 dt / tau)) / 2` per
    step and `h^2 / 2` for the tail, `h` the filtered difference. Elephant's
    `van_rossum_distance` is `sqrt(2)` times this distance's square root.
    Spikes through a surrogate train against recorded ones by it.
    """
    decay = jnp.exp(-dt / tau)

    def step(h, difference):
        h = decay * h + difference
        return h, h

    _, h = jax.lax.scan(step, jnp.zeros_like(spikes[0]), spikes - target)
    within = 0.5 * (1 - decay ** 2) * jnp.sum(h[:-1] ** 2)
    return within + 0.5 * jnp.sum(h[-1] ** 2)


def readout_logits(readout: Readout, outputs: jax.Array) -> jax.Array:
    """The class scores `[B, C]` a `readout` predicts from outputs `[T, B, C]`, in float32."""
    outputs = outputs.astype(jnp.float32)
    if readout == "max":
        return jnp.max(outputs, axis=0)
    if readout == "sum":
        return jnp.sum(outputs, axis=0)
    if readout == "softmax_sum":
        return softmax_sum(outputs)
    return jnp.mean(outputs, axis=0)


def readout_losses(readout: Readout, outputs: jax.Array, labels: jax.Array) -> jax.Array:
    """Each example's loss `[B]` under `readout`, for outputs `[T, B, C]` and integer `labels`."""
    if readout == "per_step":
        return per_step_cross_entropy(outputs, labels)
    if readout == "softmax_sum":
        return softmax_sum_cross_entropy(outputs, labels)
    return optax.softmax_cross_entropy_with_integer_labels(readout_logits(readout, outputs), labels)
