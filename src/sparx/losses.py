"""Classification losses over a network's time-major outputs `[T, B, C]`.

Each returns one loss per example, `[B]`, so a caller chooses the reduction
(dew's objectives sum it over a `Ratio`). For a loss on one readout of the
outputs, reduce time first and use optax: the maximum membrane of a leaky
integrator readout (`jnp.max(v, axis=0)`, Cramer et al., IEEE TNNLS 2020),
its mean, or the spike count (`jnp.sum(spikes, axis=0)`), each as logits for
`optax.softmax_cross_entropy_with_integer_labels`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

__all__ = ["per_step_cross_entropy", "rate_mse"]


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
    per_step = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.broadcast_to(labels, logits.shape[:-1]))
    return jnp.sum(per_step, axis=0) / steps


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
