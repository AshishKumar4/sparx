"""Converting a trained ReLU network to a spiking one (Rueckauer et al., Frontiers in Neuroscience 2017).

    layers = [fold_batch_norm(conv, *bn), MaxPool(), Flatten(), DenseLayer(w, b)]
    layers = normalize(layers, calibration_inputs)           # threshold balancing
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

Dense and convolutional layers carry the neurons. Average pooling and
flattening are linear, so they act on the spikes directly and pass the
averaged or reshaped spikes on to the next layer. Max pooling has no
linear equivalent on spikes; it lets through the spikes of the input
that has fired most so far (their section 2.2.6). Batch normalization is
folded into the layer before it (their section 2.2.3), so it needs no
layer of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "AvgPool",
    "ConvLayer",
    "DenseLayer",
    "Flatten",
    "Layer",
    "MaxPool",
    "fold_batch_norm",
    "normalize",
    "relu_forward",
    "run_converted",
]


class DenseLayer(NamedTuple):
    """`x @ weight + bias`, followed by ReLU in the network being converted (except the last layer)."""

    weight: jax.Array
    bias: jax.Array


class ConvLayer(NamedTuple):
    """A 2D convolution on NHWC inputs with an HWIO `weight`, as `flax.linen.Conv` computes it, plus
    `bias`; followed by ReLU like `DenseLayer`."""

    weight: jax.Array
    bias: jax.Array
    strides: tuple[int, int] = (1, 1)
    padding: Literal["SAME", "VALID"] = "SAME"


class AvgPool(NamedTuple):
    """Mean over `window` patches of an NHWC input with no padding, as `flax.linen.avg_pool`."""

    window: tuple[int, int] = (2, 2)
    strides: tuple[int, int] = (2, 2)


class MaxPool(NamedTuple):
    """Max over `window` patches of an NHWC input with no padding, as `flax.linen.max_pool`."""

    window: tuple[int, int] = (2, 2)
    strides: tuple[int, int] = (2, 2)


class Flatten(NamedTuple):
    """`[B, ...]` to `[B, features]` in row-major order, as `x.reshape(len(x), -1)` in flax models."""


type Layer = DenseLayer | ConvLayer | AvgPool | MaxPool | Flatten


def fold_batch_norm(
    layer: DenseLayer | ConvLayer,
    mean: jax.Array,
    var: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
    epsilon: float = 1e-5,
) -> DenseLayer | ConvLayer:
    """The layer followed by inference-mode batch norm, as one layer with the same outputs.

    Batch norm computes `scale (z - mean) / sigma + offset` per output
    channel with `sigma = sqrt(var + epsilon)`; applied to `z = x W + b` it
    is again affine in `x`, with `W scale / sigma` and
    `scale (b - mean) / sigma + offset` (their section 2.2.3). Folding keeps
    batch norm out of the spiking network, which has no use for it once
    the statistics are fixed. `epsilon` defaults to flax's.
    """
    gain = scale / jnp.sqrt(var + epsilon)
    return layer._replace(weight=layer.weight * gain, bias=gain * (layer.bias - mean) + offset)


def _windows(x: jax.Array, window: tuple[int, int], strides: tuple[int, int]) -> jax.Array:
    """`[B, H, W, C]` to `[B, H', W', C, wh * ww]`: every pooling patch, unpadded, on the last axis."""
    (wh, ww), (sh, sw) = window, strides
    h, w = (x.shape[1] - wh) // sh + 1, (x.shape[2] - ww) // sw + 1
    patches = [x[:, i : i + sh * (h - 1) + 1 : sh, j : j + sw * (w - 1) + 1 : sw, :]
               for i in range(wh) for j in range(ww)]
    return jnp.stack(patches, axis=-1)


def _apply(layer: Layer, x: jax.Array) -> jax.Array:
    """The layer's linear map, or pooling, without the ReLU."""
    match layer:
        case DenseLayer(weight, bias):
            return x @ weight + bias
        case ConvLayer(weight, bias, strides, padding):
            return jax.lax.conv_general_dilated(
                x, weight, strides, padding, dimension_numbers=("NHWC", "HWIO", "NHWC")) + bias
        case AvgPool(window, strides):
            return _windows(x, window, strides).mean(-1)
        case MaxPool(window, strides):
            return _windows(x, window, strides).max(-1)
        case Flatten():
            return x.reshape(x.shape[0], -1)


def relu_forward(layers: Sequence[Layer], x: jax.Array) -> list[jax.Array]:
    """Every layer's activation: ReLU after each dense or convolutional layer but the last layer,
    which is left linear."""
    out = []
    for k, layer in enumerate(layers):
        x = _apply(layer, x)
        if isinstance(layer, DenseLayer | ConvLayer) and k < len(layers) - 1:
            x = jax.nn.relu(x)
        out.append(x)
    return out


def normalize(layers: Sequence[Layer], inputs: jax.Array, percentile: float = 99.9) -> list[Layer]:
    """Scale weights and biases so each layer's activations on `inputs` stay at or below the threshold 1.

    Dense or convolutional layer `l` with scale `lambda_l`, the
    `percentile`-th percentile of its positive activations, becomes
    `W lambda_{l-1} / lambda_l` and `b / lambda_l` (`lambda_0 = 1` for
    inputs in [0, 1]). Pooling and flattening commute with a positive
    scale, so they keep the scale of the layer before them. The last layer
    is scaled the same way, which keeps the argmax.
    """
    activations = relu_forward(layers, inputs)
    previous, out = 1.0, []
    for layer, activation in zip(layers, activations, strict=True):
        if not isinstance(layer, DenseLayer | ConvLayer):
            out.append(layer)
            continue
        positive = np.asarray(activation)[np.asarray(activation) > 0]
        scale = float(np.percentile(positive, percentile)) if positive.size else 1.0
        out.append(layer._replace(weight=layer.weight * previous / scale, bias=layer.bias / scale))
        previous = scale
    return out


def _gate(layer: MaxPool, spikes: jax.Array, seen: jax.Array) -> jax.Array:
    """Each patch's spike from the input with the most spikes before this step.

    Rueckauer et al. (section 2.2.6) gate a spiking max-pool unit so it
    passes only the spikes of its most active input, judged by an estimate
    of the input rates, for example their online average. Here the estimate
    is the spike count so far, the online average times the step count,
    which ranks inputs the same way. Counting before this step
    keeps an input from winning a tie by the spike it is firing now, which
    would let the unit fire faster than any of its inputs.
    """
    winner = jnp.argmax(_windows(seen, layer.window, layer.strides), axis=-1)
    return jnp.take_along_axis(_windows(spikes, layer.window, layer.strides), winner[..., None], -1)[..., 0]


def run_converted(layers: Sequence[Layer], inputs: jax.Array, steps: int) -> list[jax.Array]:
    """Run the converted network for `steps` steps on constant `inputs` (`[B, in]` or `[B, H, W, C]`,
    analog input current, as Rueckauer et al. drive the first layer); returns each layer's output per
    step, `[B, ...]`: the firing rate of dense and convolutional layers and gated max pools, and the
    averaged or reshaped rate for average pools and flattening. Neurons integrate and fire at
    threshold 1 with reset by subtraction."""
    shapes = jax.eval_shape(lambda x: relu_forward(layers, x), inputs)
    input_shapes = [inputs.shape, *(s.shape for s in shapes[:-1])]

    def step(state, _):
        v, seen, counts = state
        x, new_v, new_seen, new_counts = inputs, [], [], []
        for k, layer in enumerate(layers):
            new_v.append(v[k])
            new_seen.append(seen[k])
            if isinstance(layer, MaxPool):
                out = _gate(layer, x, seen[k])
                new_seen[k] = seen[k] + x
            elif isinstance(layer, DenseLayer | ConvLayer):
                u = v[k] + _apply(layer, x)
                out = (u >= 1.0).astype(u.dtype)
                new_v[k] = u - out
            else:
                out = _apply(layer, x)
            new_counts.append(counts[k] + out)
            x = out
        return (new_v, new_seen, new_counts), None

    def zeros(shape: tuple[int, ...]) -> jax.Array:
        return jnp.zeros(shape, inputs.dtype)

    v = [zeros(s.shape if isinstance(layer, DenseLayer | ConvLayer) else ())
         for layer, s in zip(layers, shapes, strict=True)]
    seen = [zeros(shape if isinstance(layer, MaxPool) else ()) for layer, shape in zip(layers, input_shapes,
                                                                                       strict=True)]
    (_, _, counts), _ = jax.lax.scan(step, (v, seen, [zeros(s.shape) for s in shapes]), None, length=steps)
    return [c / steps for c in counts]
