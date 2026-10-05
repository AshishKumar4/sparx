"""Converting a trained ReLU network to a spiking one (Rueckauer et al., Frontiers in Neuroscience 2017).

    layers = normalize(relu_layers, calibration_inputs)      # threshold balancing
    rates = run_converted(layers, inputs, steps=200)           # IF neurons, rate-coded

A ReLU activation becomes the firing rate of an integrate-and-fire neuron
driven by the previous layer's spikes. With reset by subtraction, a neuron
under a constant drive `a` in units of its threshold fires at rate `a` per
step, to within `1 / T` after `T` steps, for `0 <= a <= 1`; a drive above
the threshold saturates at one spike per step. So each layer is scaled so
its activations stay below threshold: by the `percentile`-th percentile of
the layer's activations on calibration data (their robust normalization;
the maximum, Diehl et al. 2015's, is `percentile=100`), and a layer's
weights by the previous layer's scale over its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["DenseLayer", "normalize", "relu_forward", "run_converted"]


class DenseLayer(NamedTuple):
    """`x @ weight + bias`, followed by ReLU in the network being converted (except the last layer)."""

    weight: jax.Array
    bias: jax.Array


def relu_forward(layers: Sequence[DenseLayer], x: jax.Array) -> list[jax.Array]:
    """Every layer's activation: ReLU after each layer but the last, which is left linear."""
    out = []
    for k, layer in enumerate(layers):
        x = x @ layer.weight + layer.bias
        if k < len(layers) - 1:
            x = jax.nn.relu(x)
        out.append(x)
    return out


def normalize(layers: Sequence[DenseLayer], inputs: jax.Array, percentile: float = 99.9) -> list[DenseLayer]:
    """Scale weights and biases so each layer's activations on `inputs` stay at or below the threshold 1.

    Layer `l` with scale `lambda_l`, the `percentile`-th percentile of its
    positive activations, becomes `W lambda_{l-1} / lambda_l` and
    `b / lambda_l` (`lambda_0 = 1` for inputs in [0, 1]). The last layer is
    scaled the same way, which keeps the argmax.
    """
    activations = relu_forward(layers, inputs)
    previous, out = 1.0, []
    for layer, activation in zip(layers, activations, strict=True):
        positive = np.asarray(activation)[np.asarray(activation) > 0]
        scale = float(np.percentile(positive, percentile)) if positive.size else 1.0
        out.append(DenseLayer(layer.weight * previous / scale, layer.bias / scale))
        previous = scale
    return out


def run_converted(layers: Sequence[DenseLayer], inputs: jax.Array, steps: int) -> list[jax.Array]:
    """Run the converted network for `steps` steps on constant `inputs` `[B, in]` (analog input
    current, as Rueckauer et al. drive the first layer); returns each layer's firing rate, spikes per
    step, `[B, n_l]`. Neurons integrate and fire at threshold 1 with reset by subtraction."""

    def step(state, _):
        v, counts = state
        x, new_v, new_counts = inputs, [], []
        for k, layer in enumerate(layers):
            u = v[k] + x @ layer.weight + layer.bias
            s = (u >= 1.0).astype(u.dtype)
            new_v.append(u - s)
            new_counts.append(counts[k] + s)
            x = s
        return (new_v, new_counts), None

    zeros = [jnp.zeros((inputs.shape[0], layer.weight.shape[1]), inputs.dtype) for layer in layers]
    (_, counts), _ = jax.lax.scan(step, (zeros, zeros), None, length=steps)
    return [c / steps for c in counts]
