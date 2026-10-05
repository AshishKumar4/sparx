"""Converting a trained ReLU network to a spiking one (Rueckauer et al., Frontiers in Neuroscience 2017).

    model, variables = fold_batch_norm(model, variables)     # a flax nn.Sequential and its variables
    variables = normalize(model, variables, calibration)      # threshold balancing
    snn, snn_variables = convert(model, variables)            # IF neurons, rate-coded
    rates = run_converted(snn, snn_variables, inputs, steps=200)

A ReLU activation becomes the firing rate of an integrate-and-fire neuron
driven by the previous layer's spikes. With reset by subtraction, a neuron
under a constant drive `a` in units of its threshold fires at rate `a` per
step, to within `1 / T` after `T` steps, for `0 <= a <= 1`; a drive above
the threshold saturates at one spike per step. So each layer is scaled so
its activations stay below threshold: by the `percentile`-th percentile of
the layer's activations on calibration data (their robust normalization;
the maximum, Diehl et al. 2015's, is `percentile=100`), and a layer's
weights by the previous layer's scale over its own.

The network is a flax `nn.Sequential`, as `sparx.nir` takes, of

- `nn.Dense` and `nn.Conv`, each followed by `nn.relu` except the last;
- `nn.BatchNorm` right after one of them, which `fold_batch_norm` folds
  into it (their section 2.2.3), so it needs no layer of its own;
- unpadded pooling, `functools.partial(nn.avg_pool, ...)` or
  `functools.partial(nn.max_pool, ...)`;
- `sparx.nn.Flatten`.

`sparx.nn.Flatten` orders an image's features channel first, C, H, W,
as PyTorch and NIR do. A network trained with a plain reshape
`x.reshape(len(x), -1)`, as flax examples write it, or in Keras, orders
them H, W, C; for the dense layer after it, its kernel rows go in
Flatten's order with `kernel.reshape(h, w, c, -1).transpose(2, 0, 1, 3)
.reshape(h * w * c, -1)`. On a 1 x 1 map the two orders are the same.

`convert` keeps the dense and convolutional layers with their (normalized)
parameters and puts a `sparx.nn.IF` neuron (threshold 1) in place of each
ReLU and after the last layer. Average pooling and flattening are linear,
so they act on the spikes directly. Max pooling has no linear equivalent
on spikes; `SpikingMaxPool` lets through the spikes of the input that has
fired most so far (their section 2.2.6). The result is a `sparx.nn` stack
over time-major inputs `[T, B, ...]` like any other: it carries its state
across calls, records rates, trains with surrogate gradients, and a
network of dense layers and hard-reset neurons (`reset="zero"`) exports
to NIR.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from sparx.dynamics import Reset, Spikes, SynapticInput
from sparx.dynamics.core import membrane_dtype
from sparx.nn import IF, STATE, Flatten, Neuron

__all__ = ["SpikingMaxPool", "convert", "fold_batch_norm", "normalize", "run_converted"]

type Params = Mapping[str, jax.Array | np.ndarray]
"""One layer's parameters or statistics: name to array."""
type Variables = Mapping[str, Mapping[str, Params]]
"""A sequential stack's variables, `{"params": {"layers_k": {...}}, ...}`."""
type Pair = tuple[int, int]


def _windows(x: jax.Array, window: Pair, strides: Pair) -> jax.Array:
    """`[..., H, W, C]` to `[..., H', W', C, wh * ww]`: every pooling patch, unpadded, on the last axis."""
    (wh, ww), (sh, sw) = window, strides
    h, w = (x.shape[-3] - wh) // sh + 1, (x.shape[-2] - ww) // sw + 1
    patches = [x[..., i : i + sh * (h - 1) + 1 : sh, j : j + sw * (w - 1) + 1 : sw, :]
               for i in range(wh) for j in range(ww)]
    return jnp.stack(patches, axis=-1)


@struct.dataclass
class _Gate:
    """A max pool on spikes: each patch passes the spike of the input with the most spikes before this step.

    Its state is every input's spike count so far, `[..., H, W, C]`.
    """

    window: Pair = struct.field(pytree_node=False)
    strides: Pair = struct.field(pytree_node=False)

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        return jnp.zeros(shape, membrane_dtype(dtype))

    def step(self, state: jax.Array, inputs: SynapticInput, dt: float) -> tuple[jax.Array, Spikes]:
        spikes = jnp.asarray(inputs.jump)
        winner = jnp.argmax(_windows(state, self.window, self.strides), axis=-1)
        out = jnp.take_along_axis(_windows(spikes, self.window, self.strides), winner[..., None], -1)[..., 0]
        return state + spikes, Spikes(out, jnp.ones_like(out))

    def is_refractory(self, state: jax.Array, dt: float) -> jax.Array:
        return jnp.zeros(state.shape, bool)

    def after_threshold(self, state: jax.Array, jump: jax.Array, fired: jax.Array) -> jax.Array:
        # A gate holds counts, no membrane, so a jump after the threshold has nothing to move.
        return state


class SpikingMaxPool(Neuron):
    """Max pooling of spikes `[T, ..., H, W, C]` over unpadded `window` patches, gated by spike counts.

    Rueckauer et al. (section 2.2.6) gate a spiking max-pool unit so it
    passes only the spikes of its most active input, judged by an estimate
    of the input rates, for example their online average. Here the
    estimate is the spike count so far, the online average times the step
    count, which ranks inputs the same way. Counting before this step
    keeps an input from winning a tie by the spike it is firing now, which
    would let the unit fire faster than any of its inputs. The counts are
    the layer's state, carried across calls like a neuron's.
    """

    window: Pair = (2, 2)
    strides: Pair = (2, 2)

    def build(self, x: jax.Array) -> _Gate:
        return _Gate(self.window, self.strides)


def _pool(layer: object) -> tuple[Literal["max", "avg"], Pair, Pair] | None:
    """The kind, window and strides of a `functools.partial` of flax's 2-d max or average pooling."""
    if not isinstance(layer, functools.partial) or layer.func not in (nn.max_pool, nn.avg_pool):
        return None
    bound = inspect.signature(layer.func).bind(None, *layer.args, **layer.keywords)
    bound.apply_defaults()
    window = tuple(bound.arguments["window_shape"])
    strides = tuple(bound.arguments["strides"] or (1,) * len(window))
    if len(window) != 2 or bound.arguments["padding"] != "VALID":
        raise NotImplementedError("only unpadded 2-d pooling converts")
    kind = "max" if layer.func is nn.max_pool else "avg"
    return kind, (window[0], window[1]), (strides[0], strides[1])


def _call(layer: Callable[..., jax.Array], params: Params | None, x: jax.Array) -> jax.Array:
    """One layer of a sequential stack on `x`, with its parameters."""
    if not isinstance(layer, nn.Module):
        return layer(x)
    out = layer.apply({"params": params} if params else {}, x)
    # `mutable` is unset, so apply returns the outputs alone, not a pair.
    assert not isinstance(out, tuple)
    return out


def fold_batch_norm(model: nn.Sequential,
                    variables: Variables) -> tuple[nn.Sequential, dict[str, dict[str, Params]]]:
    """The stack with each `nn.BatchNorm` folded into the dense or convolutional layer before it.

    Inference-mode batch norm computes `scale (z - mean) / sigma + offset`
    per output channel with `sigma = sqrt(var + epsilon)`; applied to
    `z = x W + b` it is again affine in `x`, with `W scale / sigma` and
    `scale (b - mean) / sigma + offset` (their section 2.2.3). The running
    statistics come from `variables["batch_stats"]` and `epsilon` from the
    layer. Returns the stack without its batch norms, and its variables.
    """
    params, stats = variables["params"], variables.get("batch_stats", {})
    layers: list[Callable[..., jax.Array]] = []
    folded: dict[str, Params] = {}
    for k, layer in enumerate(model.layers):
        name = f"layers_{k}"
        if not isinstance(layer, nn.BatchNorm):
            if name in params:
                folded[f"layers_{len(layers)}"] = params[name]
            layers.append(layer)
            continue
        before = layers[-1] if layers else None
        if not isinstance(before, nn.Dense | nn.Conv) or layer.axis != -1:
            raise NotImplementedError("a BatchNorm over the last axis folds into the Dense or Conv layer "
                                      "right before it")
        own = params.get(name, {})
        gain = own.get("scale", 1.0) / jnp.sqrt(stats[name]["var"] + layer.epsilon)
        previous = folded[f"layers_{len(layers) - 1}"]
        folded[f"layers_{len(layers) - 1}"] = {
            "kernel": previous["kernel"] * gain,
            "bias": gain * (previous.get("bias", 0.0) - stats[name]["mean"]) + own.get("bias", 0.0)}
        layers[-1] = before.clone(use_bias=True)
    return nn.Sequential(layers), {"params": folded}


def normalize(model: nn.Sequential, variables: Variables, inputs: jax.Array,
              percentile: float = 99.9) -> dict[str, dict[str, Params]]:
    """Scale weights and biases so each layer's activations on `inputs` stay at or below the threshold 1.

    Dense or convolutional layer `l` with scale `lambda_l`, the
    `percentile`-th percentile of its positive activations, becomes
    `W lambda_{l-1} / lambda_l` and `b / lambda_l` (`lambda_0 = 1` for
    inputs in [0, 1]). Pooling and flattening commute with a positive
    scale, so they keep the scale of the layer before them. The last layer
    is scaled the same way, which keeps the argmax. Fold batch norm first.
    """
    params = variables["params"]
    x, previous, scaled = jnp.asarray(inputs), 1.0, dict(params)
    for k, layer in enumerate(model.layers):
        if isinstance(layer, nn.BatchNorm):
            raise ValueError("fold batch norm into the layer before it (fold_batch_norm) before normalizing")
        name = f"layers_{k}"
        x = _call(layer, params.get(name), x)
        if not isinstance(layer, nn.Dense | nn.Conv):
            continue
        activation = np.asarray(x)
        positive = activation[activation > 0]
        scale = float(np.percentile(positive, percentile)) if positive.size else 1.0
        scaled[name] = {key: value * previous / scale if key == "kernel" else value / scale
                        for key, value in params[name].items()}
        previous = scale
    return {"params": scaled}


def _spiking(layers: list[Callable[..., jax.Array]], k: int, last: int,
             reset: Reset) -> list[Callable[..., jax.Array]]:
    """What layer `k` of a ReLU stack whose last weight layer is `last` becomes in its spiking network."""
    layer = layers[k]
    if isinstance(layer, nn.Dense | nn.Conv):
        relu = k + 1 < len(layers) and layers[k + 1] is nn.relu
        if not relu and k != last:
            raise NotImplementedError("every Dense or Conv layer but the last is followed by nn.relu")
        return [layer] if relu else [layer, IF(reset=reset)]
    if layer is nn.relu:
        if k == 0 or not isinstance(layers[k - 1], nn.Dense | nn.Conv):
            raise NotImplementedError("nn.relu converts right after a Dense or Conv layer")
        return [IF(reset=reset)]
    pool = _pool(layer)
    if pool is not None:
        kind, window, strides = pool
        return [SpikingMaxPool(window=window, strides=strides) if kind == "max" else layer]
    if isinstance(layer, Flatten):
        return [layer]
    if isinstance(layer, nn.BatchNorm):
        raise ValueError("fold batch norm into the layer before it (fold_batch_norm) before converting")
    raise NotImplementedError(f"cannot convert {layer!r}")


def convert(model: nn.Sequential, variables: Variables, *,
            reset: Reset = "subtract") -> tuple[nn.Sequential, dict[str, dict[str, Params]]]:
    """The spiking network of a ReLU stack: the same dense and convolutional layers with `sparx.nn.IF`
    neurons, threshold 1, in place of the ReLUs and after the last layer, and `SpikingMaxPool` for max
    pooling.

    `reset="subtract"` is Rueckauer et al.'s reset by subtraction, which
    keeps the charge above threshold; `"zero"` is their reset to zero,
    which loses it but is NIR's reset. Fold batch norm and normalize first.
    Returns the stack and its variables.
    """
    layers = list(model.layers)
    weights = [k for k, layer in enumerate(layers) if isinstance(layer, nn.Dense | nn.Conv)]
    if not weights:
        raise ValueError("a network to convert needs a Dense or Conv layer")
    spiking: list[Callable[..., jax.Array]] = []
    converted: dict[str, Params] = {}
    for k in range(len(layers)):
        if k in weights:
            converted[f"layers_{len(spiking)}"] = variables["params"][f"layers_{k}"]
        spiking += _spiking(layers, k, weights[-1], reset)
    return nn.Sequential(spiking), {"params": converted}


def run_converted(model: nn.Sequential, variables: Variables, inputs: jax.Array, steps: int, *,
                  chunk: int = 50) -> jax.Array:
    """The converted network's output firing rates `[B, ...]` over `steps` steps of constant `inputs`.

    The input is analog, the same every step, as Rueckauer et al. drive
    the first layer. The steps run `chunk` at a time, the neurons' state
    carried between calls in the `"state"` collection, so memory holds
    `chunk` steps of every layer's activity whatever `steps` is; the
    result equals one call over all steps.
    """
    state: Mapping[str, Params] = {}
    total = jnp.zeros(())
    for start in range(0, steps, chunk):
        xs = jnp.broadcast_to(inputs, (min(chunk, steps - start), *inputs.shape))
        spikes, updated = model.apply({**variables, STATE: state}, xs, mutable=[STATE])
        state = updated[STATE]
        total = total + spikes.sum(0)
    return total / steps
