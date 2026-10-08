"""Spiking network architectures built from `sparx.nn` layers.

`SEWResNet` is the spike-element-wise residual network of Fang et al., "Deep
Residual Learning in Spiking Neural Networks" (NeurIPS 2021). A plain spiking
ResNet passes the sum of a block's output and its shortcut through another
neuron, which loses the identity map; a SEW block instead combines two spike
trains elementwise, so a block whose residual branch is silent passes its
input through unchanged. The layout follows SpikingJelly's
`model.sew_resnet` (commit c6cb8e46): a 7x7 stem with max pooling, four
stages of basic blocks at 64, 128, 256 and 512 channels, global average
pooling and a linear head, all over time-major `[T, B, H, W, C]` inputs.

`SpikingMLP` is the dense network for event data such as SHD: stacked dense
or delayed synapses, optionally with batch norm, optionally recurrent spiking
layers, and a leaky integrator readout. With every synapse delayed it is the
network of Hammouamri et al.'s SNN-delays (ICLR 2024).

`neuron` is the template every neuron layer of a model copies, such as
`sparx.nn.LIF(tau=2.0, detach_reset=True)`. A run's record holds it as dew
records any class, `{"class": "sparx.nn.neurons:LIF", "fields": {...}}`, and
rebuilds the model from it. Each copy (`sparx.nn.adopt`) belongs to the
block that uses it, so its parameters (a learned time constant) are that
block's own.

Both models take `train` as dew's objectives pass it, and a run names them
by import path (`--model sparx.models:SpikingMLP`).

Their synapses follow dew's precision fields, as dew's models do: `dtype`
is the dtype the convolutions, dense and delayed synapses and batch norms
compute in (None infers it from the input and the parameters),
`param_dtype` the dtype their parameters are stored in, and `precision`
their matmuls' precision. So a run's `--model.dtype bfloat16` reaches them.
Neuron membranes and their learned time constants stay float32 whatever
these are (`sparx.dynamics.core.membrane_dtype`), and spikes come out in
the synapses' dtype, which holds 0 and 1 exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype, PrecisionLike

from sparx.nn.delays import DelayedDense
from sparx.nn.neurons import LI, LIF, Neuron, Recurrent, adopt

__all__ = ["SEWBlock", "SEWResNet", "SpikingMLP", "sew_resnet18", "sew_resnet34"]


type Connect = Literal["add", "and", "iand"]
"""How a SEW block combines its shortcut `s` with its residual output `r`:
`add` is `s + r`, `and` is `s * r`, `iand` is `s * (1 - r)` (SpikingJelly's
`ADD`, `AND` and `IAND`). Only `add` can emit values above 1."""

# kaiming_normal_(mode="fan_out", nonlinearity="relu"), SpikingJelly's and
# torchvision's convolution initialization.
_conv_init = nn.initializers.variance_scaling(2.0, "fan_out", "normal")
# kaiming_uniform_(nonlinearity="relu"): uniform within sqrt(6 / fan_in).
_kaiming_uniform = nn.initializers.variance_scaling(2.0, "fan_in", "uniform")


def connect(shortcut: jax.Array, residual: jax.Array, how: Connect) -> jax.Array:
    if how == "add":
        return shortcut + residual
    if how == "and":
        return shortcut * residual
    if how == "iand":
        return shortcut * (1 - residual)
    raise ValueError(f"connect must be add, and or iand, not {how!r}")


class SEWBlock(nn.Module):
    """A basic SEW block: two 3x3 conv-BN-neuron layers, joined to the shortcut by `connect`.

    With `strides` above 1 or a change of width, the shortcut is a 1x1
    conv-BN-neuron that downsamples to match, as in SpikingJelly.
    """

    features: int
    strides: int = 1
    connect: Connect = "add"
    neuron: Neuron = LIF()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        def conv_bn(x: jax.Array, features: int, kernel: int, strides: int, name: str) -> jax.Array:
            x = nn.Conv(features, (kernel, kernel), (strides, strides), padding=kernel // 2, use_bias=False,
                        kernel_init=_conv_init, dtype=self.dtype, param_dtype=self.param_dtype,
                        precision=self.precision, name=f"{name}_conv")(x)
            return nn.BatchNorm(use_running_average=not train, momentum=0.9, dtype=self.dtype,
                                param_dtype=self.param_dtype, name=f"{name}_bn")(x)

        residual = adopt(self.neuron, self, "sn1")(conv_bn(x, self.features, 3, self.strides, "first"))
        residual = adopt(self.neuron, self, "sn2")(conv_bn(residual, self.features, 3, 1, "second"))
        shortcut = x
        if self.strides != 1 or x.shape[-1] != self.features:
            shortcut = conv_bn(x, self.features, 1, self.strides, "downsample")
            shortcut = adopt(self.neuron, self, "downsample_sn")(shortcut)
        return connect(shortcut, residual, self.connect)


class SEWResNet(nn.Module):
    """A SEW ResNet over `[T, B, H, W, C]` inputs, returning per-step logits `[T, B, classes]`.

    `stages` gives the number of blocks in each of the four stages, `(2, 2, 2,
    2)` for SEW-ResNet-18. `width` scales every stage's channels (64 in the
    paper), which small inputs and quick experiments can shrink. `stem` is
    the paper's 7x7 stride-2 convolution and 3x3 max pooling for ImageNet;
    `"small"` is a single 3x3 stride-1 convolution, the usual stem for
    32x32 inputs such as CIFAR-10. Average the logits over time for a
    rate readout, or pass them to `sparx.losses.per_step_cross_entropy`.
    """

    stages: Sequence[int]
    classes: int
    width: int = 64
    connect: Connect = "add"
    stem: Literal["imagenet", "small"] = "imagenet"
    neuron: Neuron = LIF()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        numerics = {"dtype": self.dtype, "param_dtype": self.param_dtype, "precision": self.precision}
        if self.stem == "imagenet":
            x = nn.Conv(self.width, (7, 7), (2, 2), padding=3, use_bias=False, kernel_init=_conv_init,
                        **numerics)(x)
        else:
            x = nn.Conv(self.width, (3, 3), padding=1, use_bias=False, kernel_init=_conv_init, **numerics)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, dtype=self.dtype,
                         param_dtype=self.param_dtype)(x)
        x = adopt(self.neuron, self, "stem_sn")(x)
        if self.stem == "imagenet":
            # torch's MaxPool2d(3, 2, padding=1) pads with -inf, which never wins the max.
            pad = [(0, 0)] * (x.ndim - 3) + [(1, 1), (1, 1), (0, 0)]
            x = nn.max_pool(jnp.pad(x, pad, constant_values=-jnp.inf), (3, 3), (2, 2))
        for stage, blocks in enumerate(self.stages):
            for block in range(blocks):
                strides = 2 if stage > 0 and block == 0 else 1
                x = SEWBlock(self.width * 2 ** stage, strides, self.connect, self.neuron, **numerics,
                             name=f"stage{stage + 1}_block{block + 1}")(x, train)
        x = jnp.mean(x, axis=(-3, -2))
        return nn.Dense(self.classes, **numerics)(x)


def sew_resnet18(classes: int, **kwargs) -> SEWResNet:
    """SEW-ResNet-18: stages of `(2, 2, 2, 2)` basic blocks."""
    return SEWResNet((2, 2, 2, 2), classes, **kwargs)


def sew_resnet34(classes: int, **kwargs) -> SEWResNet:
    """SEW-ResNet-34: stages of `(3, 4, 6, 3)` basic blocks."""
    return SEWResNet((3, 4, 6, 3), classes, **kwargs)


class SpikingMLP(nn.Module):
    """Dense spiking layers over `[T, B, ...]` with a leaky integrator readout `[T, B, classes]`.

    Each width in `hidden` is a synapse followed by a copy of `neuron`, and
    the readout is a synapse followed by an `LI` integrator. `recurrent`
    feeds each hidden layer's spikes back to itself (`sparx.nn.Recurrent`).
    Every layer steps at the neuron's `dt`, so `readout_tau` is in the unit
    of the neuron's time constants. Trailing input axes are flattened.

    `delays` makes synapses `sparx.nn.DelayedDense`, whose Gaussian width is
    the call's `sigma` (0, the rounded delays, by default). An integer above
    0 delays the first synapse by up to that many steps. A sequence gives
    every synapse's largest delay, hidden layers then the readout, with 0
    for a plain dense synapse: Hammouamri et al.'s SNN-delays delays them
    all. `extend` appends `max_delay // 2` steps of zeros to each delayed
    synapse's input, as SNN-delays pads it on the right, so spikes delayed
    past the input's end still arrive and each delayed synapse lengthens
    the sequence by that much; it changes the output's length, so a stream
    fed in chunks needs it off.

    `batch_norm` normalizes each hidden synapse's output over time and batch
    (`nn.BatchNorm`, torch's momentum of 0.1) before the neuron, as
    SNN-delays does; the readout is not normalized. `use_bias` gives
    synapses a bias. `weight_init` is the weights' initializer: flax's
    `lecun_normal`, or torch's `kaiming_uniform_(nonlinearity="relu")`,
    uniform within `sqrt(6 / fan_in)`, SNN-delays' choice.

    `dropout` acts on hidden spikes in training. `dropout_mask="step"` draws
    a mask per step; `"sequence"` draws one per sequence and holds it over
    time, as SpikingJelly's multi-step `Dropout` does, so a dropped neuron
    is silent for the whole recording.
    """

    hidden: Sequence[int]
    classes: int
    neuron: Neuron = LIF()
    recurrent: bool = False
    delays: int | Sequence[int] = 0
    extend: bool = False
    batch_norm: bool = False
    use_bias: bool = True
    weight_init: Literal["lecun_normal", "kaiming_uniform"] = "lecun_normal"
    dropout: float = 0.0
    dropout_mask: Literal["step", "sequence"] = "step"
    readout_tau: float = 2.0
    learn_readout_tau: bool = False
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None

    def max_delays(self) -> tuple[int, ...]:
        """Each synapse's largest delay, hidden layers then the readout; 0 for a dense synapse."""
        layers = len(self.hidden) + 1
        if isinstance(self.delays, int):
            return (self.delays,) + (0,) * (layers - 1)
        delays = tuple(int(d) for d in self.delays)
        if len(delays) != layers:
            raise ValueError(f"delays gives {len(delays)} synapses' delays; this network has {layers} "
                             f"synapses, {len(self.hidden)} hidden and the readout")
        return delays

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False, sigma: float | jax.Array = 0) -> jax.Array:
        if self.weight_init not in ("lecun_normal", "kaiming_uniform"):
            raise ValueError(f"weight_init must be lecun_normal or kaiming_uniform, not {self.weight_init!r}")
        if self.dropout_mask not in ("step", "sequence"):
            raise ValueError(f"dropout_mask must be step or sequence, not {self.dropout_mask!r}")
        kernel_init = (_kaiming_uniform if self.weight_init == "kaiming_uniform"
                       else nn.initializers.lecun_normal())
        delays = self.max_delays()
        # A sequence's mask is drawn for one step and broadcast over time.
        broadcast = (0,) if self.dropout_mask == "sequence" else ()
        # Differentiated through its Gaussian, a delayed synapse would keep every lag's product for
        # the backward pass: 2.6 of the 3.1 GB a step of SNN-delays' SHD network takes at batch 256.
        # Recomputing it in the backward pass brings the step to 1.6 GB. The rounded delays (a
        # Python 0) read one lag each and stay as they are, since the recomputation would trace the 0.
        rounded = isinstance(sigma, (int, float)) and sigma == 0
        delayed = DelayedDense if rounded else nn.remat(DelayedDense)

        numerics = {"dtype": self.dtype, "param_dtype": self.param_dtype, "precision": self.precision}

        def synapse(x: jax.Array, features: int, max_delay: int, name: str) -> jax.Array:
            if not max_delay:
                return nn.Dense(features, use_bias=self.use_bias, kernel_init=kernel_init, **numerics,
                                name=name)(x)
            if self.extend:
                x = jnp.concatenate([x, jnp.zeros((max_delay // 2, *x.shape[1:]), x.dtype)])
            return delayed(features, max_delay, use_bias=self.use_bias, kernel_init=kernel_init, **numerics,
                           name=name)(x, sigma)

        x = x.reshape(*x.shape[:2], -1)
        for layer, width in enumerate(self.hidden):
            x = synapse(x, width, delays[layer], f"delayed_{layer}" if delays[layer] else f"dense_{layer}")
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=self.dtype,
                                 param_dtype=self.param_dtype, name=f"norm_{layer}")(x)
            if self.recurrent:
                x = Recurrent(neuron=self.neuron, precision=self.precision, dt=self.neuron.dt,
                              name=f"recurrent_{layer}")(x)
            else:
                x = adopt(self.neuron, self, f"neuron_{layer}")(x)
            x = nn.Dropout(self.dropout, broadcast_dims=broadcast, deterministic=not train)(x)
        x = synapse(x, self.classes, delays[-1], "readout")
        return LI(tau=self.readout_tau, learn_tau=self.learn_readout_tau, dt=self.neuron.dt,
                  name="integrator")(x)
