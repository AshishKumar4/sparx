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
(or delayed) synapses, optionally recurrent spiking layers, and a leaky
integrator readout.

`neuron` is the template every neuron layer of a model copies, such as
`sparx.nn.LIF(tau=2.0, detach_reset=True)`. It is a registered value, so a
run's record holds it as `{"name": "lif", "fields": {...}}` and rebuilds the model. Each
copy belongs to the block that uses it, so its parameters (a learned time
constant) are that block's own.

Both models are registered in dew's model registry, `sew_resnet` and
`spiking_mlp`, and take `train` as dew's objectives pass it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from dew.registry import models

from sparx.nn.delays import DelayedDense
from sparx.nn.neurons import LI, LIF, Neuron, Recurrent

__all__ = ["SEWBlock", "SEWResNet", "SpikingMLP", "copy_neuron", "sew_resnet18", "sew_resnet34"]


def copy_neuron(template: Neuron, owner: nn.Module, name: str) -> Neuron:
    """A copy of `template` that is `owner`'s child called `name`, wherever the template was built."""
    return template.clone(parent=owner, name=name)


def _template(neuron: Neuron) -> Neuron:
    """`neuron` unbound, to hand to a submodule as its own template."""
    return neuron.clone(parent=None)

type Connect = Literal["add", "and", "iand"]
"""How a SEW block combines its shortcut `s` with its residual output `r`:
`add` is `s + r`, `and` is `s * r`, `iand` is `s * (1 - r)` (SpikingJelly's
`ADD`, `AND` and `IAND`). Only `add` can emit values above 1."""

# kaiming_normal_(mode="fan_out", nonlinearity="relu"), SpikingJelly's and
# torchvision's convolution initialization.
_conv_init = nn.initializers.variance_scaling(2.0, "fan_out", "normal")


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

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        def conv_bn(x, features, kernel, strides, name):
            x = nn.Conv(features, (kernel, kernel), (strides, strides), padding=kernel // 2, use_bias=False,
                        kernel_init=_conv_init, name=f"{name}_conv")(x)
            return nn.BatchNorm(use_running_average=not train, momentum=0.9, name=f"{name}_bn")(x)

        residual = copy_neuron(self.neuron, self, "sn1")(conv_bn(x, self.features, 3, self.strides, "first"))
        residual = copy_neuron(self.neuron, self, "sn2")(conv_bn(residual, self.features, 3, 1, "second"))
        shortcut = x
        if self.strides != 1 or x.shape[-1] != self.features:
            shortcut = conv_bn(x, self.features, 1, self.strides, "downsample")
            shortcut = copy_neuron(self.neuron, self, "downsample_sn")(shortcut)
        return connect(shortcut, residual, self.connect)


@models("sew_resnet")
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

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        if self.stem == "imagenet":
            x = nn.Conv(self.width, (7, 7), (2, 2), padding=3, use_bias=False, kernel_init=_conv_init)(x)
        else:
            x = nn.Conv(self.width, (3, 3), padding=1, use_bias=False, kernel_init=_conv_init)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9)(x)
        x = copy_neuron(self.neuron, self, "stem_sn")(x)
        if self.stem == "imagenet":
            # torch's MaxPool2d(3, 2, padding=1) pads with -inf, which never wins the max.
            pad = [(0, 0)] * (x.ndim - 3) + [(1, 1), (1, 1), (0, 0)]
            x = nn.max_pool(jnp.pad(x, pad, constant_values=-jnp.inf), (3, 3), (2, 2))
        for stage, blocks in enumerate(self.stages):
            for block in range(blocks):
                strides = 2 if stage > 0 and block == 0 else 1
                x = SEWBlock(self.width * 2 ** stage, strides, self.connect, _template(self.neuron),
                             name=f"stage{stage + 1}_block{block + 1}")(x, train)
        x = jnp.mean(x, axis=(-3, -2))
        return nn.Dense(self.classes)(x)


def sew_resnet18(classes: int, **kwargs) -> SEWResNet:
    """SEW-ResNet-18: stages of `(2, 2, 2, 2)` basic blocks."""
    return SEWResNet((2, 2, 2, 2), classes, **kwargs)


def sew_resnet34(classes: int, **kwargs) -> SEWResNet:
    """SEW-ResNet-34: stages of `(3, 4, 6, 3)` basic blocks."""
    return SEWResNet((3, 4, 6, 3), classes, **kwargs)


@models("spiking_mlp")
class SpikingMLP(nn.Module):
    """Dense spiking layers over `[T, B, ...]` with a leaky integrator readout `[T, B, classes]`.

    Each width in `hidden` is a dense synapse followed by a copy of `neuron`;
    `recurrent` feeds each hidden layer's spikes back to itself
    (`sparx.nn.Recurrent`). `delays` above 0 makes the first synapse a
    `sparx.nn.DelayedDense` with delays of up to that many steps, whose
    Gaussian width is the call's `sigma` (0, the rounded delays, by default).
    Trailing input axes are flattened. `dropout` acts on hidden spikes in
    training.
    """

    hidden: Sequence[int]
    classes: int
    neuron: Neuron = LIF()
    recurrent: bool = False
    delays: int = 0
    dropout: float = 0.0
    readout_tau: float = 2.0
    learn_readout_tau: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False, sigma: float | jax.Array = 0) -> jax.Array:
        x = x.reshape(*x.shape[:2], -1)
        for layer, width in enumerate(self.hidden):
            if layer == 0 and self.delays:
                x = DelayedDense(width, self.delays, name="delayed_0")(x, sigma)
            else:
                x = nn.Dense(width, name=f"dense_{layer}")(x)
            if self.recurrent:
                x = Recurrent(neuron=_template(self.neuron), name=f"recurrent_{layer}")(x)
            else:
                x = copy_neuron(self.neuron, self, f"neuron_{layer}")(x)
            x = nn.Dropout(self.dropout, deterministic=not train)(x)
        x = nn.Dense(self.classes, name="readout")(x)
        return LI(tau=self.readout_tau, learn_tau=self.learn_readout_tau, name="integrator")(x)
