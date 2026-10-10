"""The networks the site's trained models share, stepped one step at a time, and their JSON form.

A stack is `nn.Sequential([Dense, LIF, ..., Dense, LIF, Dense, LI])` of sparx layers: LIF layers that reset
to zero, as NIR describes them, and a leaky-integrator readout whose membranes are the outputs. `Vision`
puts two strided convolutions over a camera's image in front of the same kind of stack, with units that
send binary spikes, spikes of a few bits, real values, or changes of real values. The browser steps them
from the JSON (site/src/engines/stack.ts).
"""

from __future__ import annotations

import base64
import itertools
import math
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

import sparx
from sparx.dynamics import Output, SynapticInput, decay, run
from sparx.dynamics.core import fire
from sparx.nn import LI, LIF, STATE, Neuron, record_rates
from sparx.surrogate import ATan, Surrogate, spike


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
    threshold and reset. It records which of its units are active, as `LIF` records spikes, for the count
    of multiply-adds; the record has no gradient, so the racer's band on firing rates does not train it."""

    tau: float

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h = nn.relu(LI(tau=self.tau)(x))
        if self.is_mutable_collection("spike_rates") and not self.is_initializing():
            self.sow("spike_rates", "rate", jnp.mean(h > 0, axis=0, dtype=jnp.float32))
        return h


class FewBitState(NamedTuple):
    v: jax.Array
    first: jax.Array
    """Whether the step's membrane reached the first threshold, differentiated through the surrogate."""


@struct.dataclass
class FewBitCell:
    """LIF neurons that reset to zero, whose spike carries how many thresholds (multiples of 1) the
    membrane reached, from 1 to `levels`: Loihi 2's graded spikes, of log2(levels + 1) bits. One level
    is `LIFCell(decay, reset="zero")`; each level's step is differentiated through `surrogate`."""

    decay: jax.Array | float
    levels: int = struct.field(pytree_node=False, default=1)
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())
    graded = True

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> FewBitState:
        zeros = jnp.zeros(shape, jnp.promote_types(dtype, jnp.float32))
        return FewBitState(zeros, zeros)

    def step(self, state: FewBitState, inputs: SynapticInput, dt: float) -> tuple[FewBitState, Output]:
        v = self.decay ** dt * state.v + inputs.jump
        after, first = fire(v, 1.0, self.surrogate, "zero")
        payload = first + sum(spike(v - k, self.surrogate) for k in range(2, self.levels + 1))
        payload = payload.astype(inputs.jump.dtype)
        return FewBitState(after, first), Output(payload, jnp.ones_like(payload))


class FewBit(Neuron):
    """`FewBitCell` as a layer: spikes of `bits` bits. It records which of its neurons fire, as `LIF`
    records spikes, through the first level's surrogate, so the racer's band on firing rates trains it as
    it trains `LIF`."""

    tau: float = 3.0
    bits: int = 2

    def build(self, x: jax.Array) -> FewBitCell:
        return FewBitCell(decay(self.tau), 2 ** self.bits - 1)

    def __call__(self, x: jax.Array) -> jax.Array:
        payload, first = run_recorded(self, self.model(x), x, lambda state: state.first)
        record_rates(self, first)
        return payload


class SigmaDeltaState(NamedTuple):
    v: jax.Array
    sent: jax.Array
    """The activation as last sent, which the receivers hold: the sum of the changes sent."""
    fired: jax.Array
    """Whether the step sent a change."""


@struct.dataclass
class SigmaDeltaCell:
    """`Graded`'s unit, a leaky membrane read through a ReLU, that sends the change in its activation
    since it last sent once that change reaches `threshold`, and receivers that add up the changes:
    sigma-delta coding (O'Connor and Welling, arXiv 1611.02024) as Lava's `SigmaDelta` neurons do it on
    Loihi 2. Unlike Lava's, the unit's membrane leaks, as `Graded`'s does, the change is sent unrounded, and
    the cell returns what the receivers hold, the activation as last sent, within `threshold` of the
    activation, rather than the stream of changes. Its gradient is the activation's. At `threshold = 0` it
    is `Graded`, sending every change."""

    decay: jax.Array | float
    threshold: jax.Array | float = 0.1
    graded = True

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> SigmaDeltaState:
        zeros = jnp.zeros(shape, jnp.promote_types(dtype, jnp.float32))
        return SigmaDeltaState(zeros, zeros, jnp.zeros(shape, bool))

    def step(self, state: SigmaDeltaState, inputs: SynapticInput, dt: float,
             ) -> tuple[SigmaDeltaState, Output]:
        v = self.decay ** dt * state.v + inputs.jump
        activation = jax.nn.relu(v)
        change = activation - state.sent
        fired = (jnp.abs(change) >= self.threshold) & (change != 0)
        sent = jnp.where(fired, activation, state.sent)
        value = activation + jax.lax.stop_gradient(sent - activation)
        return SigmaDeltaState(v, sent, fired), Output(value, jnp.ones_like(value))


class SigmaDelta(Neuron):
    """`SigmaDeltaCell` as a layer. It records which of its units send a change each step, for the count
    of multiply-adds; as `Graded`'s, the record has no gradient."""

    tau: float = 3.0
    threshold: float = 0.1

    def build(self, x: jax.Array) -> SigmaDeltaCell:
        return SigmaDeltaCell(decay(self.tau), self.threshold)

    def __call__(self, x: jax.Array) -> jax.Array:
        value, fired = run_recorded(self, self.model(x), x, lambda state: state.fired)
        record_rates(self, fired)
        return value


def run_recorded(layer: Neuron, model, x: jax.Array, record):
    """`layer`'s run of `model` over `x` `[T, ...]`, carrying the `state` collection as `Neuron.run` does,
    and `record(state)` after each step: the outputs and the records, `[T, ...]` each."""
    carrying = layer.is_mutable_collection(STATE) and not layer.is_initializing()
    state = layer.get_variable(STATE, "carry") if carrying else None
    (out, recorded), final = run(model, layer.inputs(x), state, dt=layer.dt, record=record,
                                 unroll=layer.unroll)
    if carrying:
        layer.put_variable(STATE, "carry", final)
    return out.value, recorded


class Dendrites(nn.Module):
    """`units` neurons of `branches` active dendrites each: Poirazi, Brannon and Mel's (2003) two-layer
    neuron, spiking. Each input reaches one branch of each neuron, drawn at random once (`seed`), through
    one weight, so the weights number as a dense layer's. A branch is a LIF membrane that resets to zero,
    and its spike, a dendritic spike, reaches the soma through a learned coupling; the soma is a LIF
    neuron. At the start, two branches spiking together fire the soma and one alone does not. Current
    runs one way, branch to soma: nothing flows back."""

    units: int
    branches: int = 4
    tau: float = 3.0
    seed: int = 0

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        inputs = x.shape[-1]
        rng = np.random.default_rng(self.seed)
        branch = np.stack([rng.permutation(np.arange(inputs) % self.branches) for _ in range(self.units)], 1)
        route = jax.nn.one_hot(branch, self.branches, dtype=x.dtype)  # [inputs, units, branches]
        # Each branch's current varies as a dense layer's unit's would, from a quarter of the inputs.
        init = nn.initializers.variance_scaling(self.branches, "fan_in", "truncated_normal")
        kernel = self.param("kernel", init, (inputs, self.units))
        bias = self.param("bias", nn.initializers.zeros, (self.units, self.branches))
        coupling = self.param("coupling", nn.initializers.constant(0.5), (self.units, self.branches))
        dendritic = LIF(tau=self.tau, reset="zero")(jnp.einsum("...i,iu,iub->...ub", x, kernel, route) + bias)
        return LIF(tau=self.tau, reset="zero")(jnp.sum(dendritic * coupling, -1))


class Vision(nn.Module):
    """A camera's image, `(rows, columns, channels)` flattened at the front of each input, through two
    convolutions of stride 2, each into a population of units, then a dense layer of `hidden` units with
    the input's remaining features, and a dense layer into `outputs` leaky integrators. The units are LIF
    neurons that reset to zero (`neuron="lif"`), sending spikes of `bits` bits (`FewBit`) when `bits` is
    above 1; `Graded` units for the non-spiking counterpart (`"relu"`); or `SigmaDelta` units that send
    their changes of at least `delta` (`"sigma-delta"`). With `"dendritic"`, the units are LIF neurons
    and the dense layer's are `Dendrites` of `branches` branches."""

    shape: tuple[int, int, int]
    features: tuple[int, ...] = (16, 32)
    kernels: tuple[int, ...] = (5, 3)
    hidden: int = 128
    outputs: int = 2
    neuron: str = "lif"
    tau: float = 3.0
    readout_tau: float = 4.0
    bits: int = 1
    delta: float = 0.1
    branches: int = 4

    def unit(self, name: str) -> nn.Module:
        """A population of units, named so that `act` lists the populations in order."""
        if self.neuron == "relu":
            return Graded(tau=self.tau, name=name)
        if self.neuron == "sigma-delta":
            return SigmaDelta(tau=self.tau, threshold=self.delta, name=name)
        if self.bits > 1:
            return FewBit(tau=self.tau, bits=self.bits, name=name)
        return LIF(tau=self.tau, reset="zero", name=name)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        rows, columns, channels = self.shape
        pixels = rows * columns * channels
        h = jnp.moveaxis(x[..., :pixels].reshape(*x.shape[:-1], channels, rows, columns), -3, -1)
        for k, (features, kernel) in enumerate(zip(self.features, self.kernels, strict=True)):
            h = self.unit(f"units_{k}")(nn.Conv(features, (kernel, kernel), (2, 2), padding="SAME")(h))
        h = jnp.concatenate([h.reshape(*h.shape[:-3], -1), x[..., pixels:]], -1)
        last = f"units_{len(self.features)}"
        if self.neuron == "dendritic":
            h = Dendrites(self.hidden, self.branches, self.tau, name=last)(h)
        else:
            h = self.unit(last)(nn.Dense(self.hidden)(h))
        return LI(tau=self.readout_tau)(nn.Dense(self.outputs)(h))

    def fanouts(self) -> tuple[np.ndarray, np.ndarray, int]:
        """How many connections each input and unit sends on: the multiply-adds it triggers when it is not
        zero. For the image's inputs `[pixels]` in the input's order, for the units `[units]` in `act`'s
        order, and for each input past the image, which reaches every hidden neuron (one branch of each,
        for `Dendrites`). A convolution's input reaches the outputs whose windows cover it, fewer at the
        edges."""
        rows, columns, channels = self.shape
        h, w, c = rows, columns, channels
        image, units = np.zeros(0, int), []
        for k, (features, kernel) in enumerate(zip(self.features, self.kernels, strict=True)):
            spread = np.outer(reach(h, kernel), reach(w, kernel)).ravel() * features
            if k == 0:
                image = np.tile(spread, c)  # each channel's image in turn
            else:
                units.append(np.repeat(spread, c))  # the previous population's units, channels innermost
            h, w, c = -(-h // 2), -(-w // 2), features
        units.append(np.full(h * w * c, self.hidden))
        if self.neuron == "dendritic":
            units.append(np.ones(self.hidden * self.branches, int))  # each branch reaches its soma
        units.append(np.full(self.hidden, self.outputs))
        return image, np.concatenate(units), self.hidden


def reach(size: int, kernel: int) -> np.ndarray:
    """How many outputs of a stride-2 convolution with `SAME` padding each of `size` inputs along one axis
    reaches: those whose `kernel` taps, starting `2 i - pad` for output `i`, cover it."""
    outputs = -(-size // 2)
    pad = max((outputs - 1) * 2 + kernel - size, 0) // 2
    return np.array([sum(0 <= r + pad - 2 * i < kernel for i in range(outputs)) for r in range(size)])


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
        "config": {**{k: getattr(net, k)
                      for k in ("features", "kernels", "hidden", "outputs", "neuron", "bits", "delta",
                                "branches")},
                   "shape": list(net.shape), "decay": sparx.dynamics.decay(net.tau),
                   "readout_decay": sparx.dynamics.decay(net.readout_tau)},
        "params": {"/".join(key.key for key in path): {"shape": list(leaf.shape), "data": b64(leaf)}
                   for path, leaf in flat},
    }


def vision_from_json(model: dict) -> tuple[Vision, dict]:
    config = model["config"]
    net = Vision(shape=tuple(config["shape"]), features=tuple(config["features"]),
                 kernels=tuple(config["kernels"]), hidden=config["hidden"], outputs=config["outputs"],
                 neuron=config["neuron"], bits=config.get("bits", 1), delta=config.get("delta", 0.1),
                 branches=config.get("branches", 4),
                 tau=-1 / math.log(config["decay"]), readout_tau=-1 / math.log(config["readout_decay"]))
    params: dict = {}
    for path, leaf in model["params"].items():
        *parents, name = path.split("/")
        node = params
        for parent in parents:
            node = node.setdefault(parent, {})
        node[name] = unb64(leaf["data"]).reshape(leaf["shape"])
    return net, params


def stack_fanouts(net: nn.Sequential, image: int) -> tuple[np.ndarray, np.ndarray, int]:
    """`Vision.fanouts` for a stack whose first `image` inputs are the image."""
    widths = [layer.features for layer in net.layers if isinstance(layer, nn.Dense)]
    units = [np.full(width, after) for width, after in itertools.pairwise(widths)]
    return np.full(image, widths[0]), np.concatenate(units), widths[0]
