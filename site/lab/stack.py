"""The spiking stack the site's trained networks share, stepped one step at a time, and its JSON form.

A stack is `nn.Sequential([Dense, LIF, ..., Dense, LIF, Dense, LI])` of sparx layers: LIF layers that reset
to zero, as NIR describes them, and a leaky-integrator readout whose membranes are the outputs. The browser
steps the same stack from the JSON (site/src/engines/stack.ts).
"""

from __future__ import annotations

import base64
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


def act(net: nn.Sequential, params: dict, carried: dict, x: jax.Array):
    """One step, carried in the `state` collection: the readout membranes, the new state, all spikes."""
    variables = {"params": params, "state": carried}
    u, mutated = net.apply(variables, x[None], mutable=["state", "spike_rates"])
    spikes = jnp.concatenate(jax.tree.leaves(mutated["spike_rates"]), axis=-1)
    return u[0], mutated["state"], spikes


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
