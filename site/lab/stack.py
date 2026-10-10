"""The networks the site's trained models share, stepped one step at a time, and their JSON form.

A stack is `nn.Sequential([Dense, LIF, ..., Dense, LIF, Dense, LI])` of sparx layers: LIF layers that reset
to zero, as NIR describes them, and a leaky-integrator readout whose membranes are the outputs. `Vision`
puts two strided convolutions over a camera's image in front of the same kind of stack. The browser steps
them from the JSON (site/src/engines/stack.ts).
"""

from __future__ import annotations

import base64
import itertools
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import sparx
from sparx.nn import LI, LIF


def stack(widths: tuple[int, ...], tau: float, readout_tau: float) -> nn.Sequential:
    """Dense and LIF layers of `widths[:-1]` neurons, then a Dense layer into `widths[-1]` leaky
    integrators."""
    layers: list[nn.Module] = []
    for width in widths[:-1]:
        layers += [nn.Dense(width), LIF(tau=tau, reset="zero")]
    return nn.Sequential([*layers, nn.Dense(widths[-1]), LI(tau=readout_tau)])


def at_rest(net: nn.Sequential, params: dict, batch: int, inputs: int) -> dict:
    """The network's `state` collection with every neuron at rest, for a batch of `batch`."""
    shapes = jax.eval_shape(
        lambda: net.apply({"params": params}, jnp.zeros((1, batch, inputs)), mutable=["state"])[1]["state"]
    )
    return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes)


def act(net: nn.Module, params: dict, carried: dict, x: jax.Array):
    """One step, carried in the `state` collection: the readout membranes, the new state, and every unit's
    output this step, spikes or (for `Vision(neuron="relu")`) whether it is active, `[B, units]`."""
    variables = {"params": params, "state": carried}
    u, mutated = net.apply(variables, x[None], mutable=["state", "spike_rates"])
    leaves = jax.tree.leaves(mutated["spike_rates"])
    spikes = jnp.concatenate([leaf.reshape(leaf.shape[0], -1) for leaf in leaves], axis=-1)
    return u[0], mutated["state"], spikes


class Graded(nn.Module):
    """The non-spiking counterpart of `LIF`: the same leaky membrane, read out through a ReLU instead of a
    threshold and reset. It records the share of its units that are active, as `LIF` records spikes."""

    tau: float

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h = nn.relu(LI(tau=self.tau)(x))
        if self.is_mutable_collection("spike_rates") and not self.is_initializing():
            self.sow("spike_rates", "rate", jnp.mean(h > 0, axis=0, dtype=jnp.float32))
        return h


class Vision(nn.Module):
    """A camera's image, `(rows, columns, channels)` flattened at the front of each input, through two
    convolutions of stride 2, each into a population of units, then a dense layer of `hidden` units with
    the input's remaining features, and a dense layer into `outputs` leaky integrators. The units are LIF
    neurons that reset to zero, or `Graded` units for the non-spiking counterpart."""

    shape: tuple[int, int, int]
    features: tuple[int, ...] = (16, 32)
    kernels: tuple[int, ...] = (5, 3)
    hidden: int = 128
    outputs: int = 2
    neuron: str = "lif"
    tau: float = 3.0
    readout_tau: float = 4.0

    def unit(self) -> nn.Module:
        return LIF(tau=self.tau, reset="zero") if self.neuron == "lif" else Graded(tau=self.tau)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        rows, columns, channels = self.shape
        pixels = rows * columns * channels
        h = jnp.moveaxis(x[..., :pixels].reshape(*x.shape[:-1], channels, rows, columns), -3, -1)
        for features, kernel in zip(self.features, self.kernels, strict=True):
            h = self.unit()(nn.Conv(features, (kernel, kernel), (2, 2), padding="SAME")(h))
        h = jnp.concatenate([h.reshape(*h.shape[:-3], -1), x[..., pixels:]], -1)
        h = self.unit()(nn.Dense(self.hidden)(h))
        return LI(tau=self.readout_tau)(nn.Dense(self.outputs)(h))

    def layers(self) -> list[tuple[int, int, int]]:
        """Each layer of connections: its inputs, its outputs, and how many outputs one input reaches."""
        rows, columns, channels = self.shape
        h, w, c = rows, columns, channels
        out = []
        for features, kernel in zip(self.features, self.kernels, strict=True):
            # A stride of 2 sends each input to about a quarter of the kernel's positions in each output map.
            out.append((h * w * c, -(-h // 2) * -(-w // 2) * features, kernel * kernel * features // 4))
            h, w, c = -(-h // 2), -(-w // 2), features
        return [*out, (h * w * c, self.hidden, self.hidden), (self.hidden, self.outputs, self.outputs)]

    def operations(self, inputs: jax.Array, units: jax.Array, extra: int) -> tuple[int, jax.Array]:
        """Multiply-adds per step: every connection's, a constant, and the count triggered, `[...]`, those
        from inputs and units that are not zero, given the image's non-zero `inputs` `[...]` and every
        unit's output `units` `[..., units]`, as `act` returns them. The `extra` inputs of the dense layer
        are never zero."""
        layers = self.layers()
        dense = sum(fan_in * fan for fan_in, _, fan in layers) + extra * self.hidden
        sources, at = [inputs], 0
        for _, outputs, _ in layers[:-1]:
            sources.append(jnp.sum(units[..., at:at + outputs] != 0, -1))
            at += outputs
        triggered = sum(n * fan for n, (_, _, fan) in zip(sources, layers, strict=True)) + extra * self.hidden
        return dense, triggered


def to_layers(net: nn.Sequential, params: dict) -> list[dict]:
    """Every layer for the browser, weights as base64 float32."""
    layers = []
    for k, layer in enumerate(net.layers):
        if isinstance(layer, nn.Dense):
            p = params[f"layers_{k}"]
            layers.append(
                {
                    "kind": "dense",
                    "kernel": b64(p["kernel"]),
                    "bias": b64(p["bias"]),
                    "inputs": int(p["kernel"].shape[0]),
                    "outputs": int(p["kernel"].shape[1]),
                }
            )
        elif isinstance(layer, LIF):
            layers.append(
                {
                    "kind": "lif",
                    "decay": sparx.dynamics.decay(layer.tau),
                    "threshold": layer.threshold,
                    "reset": layer.reset,
                }
            )
        elif isinstance(layer, LI):
            layers.append({"kind": "li", "decay": sparx.dynamics.decay(layer.tau)})
    return layers


def from_layers(layers: list[dict]) -> tuple[nn.Sequential, dict]:
    taus = [-1 / math.log(layer["decay"]) for layer in layers if layer["kind"] != "dense"]
    widths = tuple(layer["outputs"] for layer in layers if layer["kind"] == "dense")
    params = {}
    for k, layer in enumerate(layers):
        if layer["kind"] == "dense":
            params[f"layers_{k}"] = {
                "kernel": unb64(layer["kernel"]).reshape(layer["inputs"], layer["outputs"]),
                "bias": unb64(layer["bias"]),
            }
    return stack(widths, taus[0], taus[-1]), params


def b64(x: jax.Array) -> str:
    return base64.b64encode(np.asarray(x, np.float32).tobytes()).decode()


def unb64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), np.float32).copy()


def vision_json(net: Vision, params: dict) -> dict:
    """A `Vision` network for the browser: its configuration and every parameter as base64 float32 by
    path, such as `Conv_0/kernel` `[kernel, kernel, in, out]`."""
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    return {
        "config": {**{k: getattr(net, k) for k in ("features", "kernels", "hidden", "outputs", "neuron")},
                   "shape": list(net.shape), "decay": sparx.dynamics.decay(net.tau),
                   "readout_decay": sparx.dynamics.decay(net.readout_tau)},
        "params": {"/".join(key.key for key in path): {"shape": list(leaf.shape), "data": b64(leaf)}
                   for path, leaf in flat},
    }


def vision_from_json(model: dict) -> tuple[Vision, dict]:
    config = model["config"]
    net = Vision(shape=tuple(config["shape"]), features=tuple(config["features"]),
                 kernels=tuple(config["kernels"]), hidden=config["hidden"], outputs=config["outputs"],
                 neuron=config["neuron"],
                 tau=-1 / math.log(config["decay"]), readout_tau=-1 / math.log(config["readout_decay"]))
    params: dict = {}
    for path, leaf in model["params"].items():
        *parents, name = path.split("/")
        node = params
        for parent in parents:
            node = node.setdefault(parent, {})
        node[name] = unb64(leaf["data"]).reshape(leaf["shape"])
    return net, params


def stack_operations(net: nn.Sequential, total: int, inputs: jax.Array, units: jax.Array,
                     extra: int) -> tuple[int, jax.Array]:
    """`Vision.operations` for a stack of `total` inputs: every connection's multiply-adds per step, and those
    triggered by the non-zero `inputs` (beside `extra` inputs never zero) and units."""
    widths = [layer.features for layer in net.layers if isinstance(layer, nn.Dense)]
    dense = total * widths[0] + sum(a * b for a, b in itertools.pairwise(widths))
    triggered, at = (inputs + extra) * widths[0], 0
    for fan_in, width in itertools.pairwise(widths):
        triggered = triggered + jnp.sum(units[..., at:at + fan_in] != 0, -1) * width
        at += fan_in
    return dense, triggered
