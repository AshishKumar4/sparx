"""Exchanging networks through NIR, the Neuromorphic Intermediate Representation (Pedersen et al. 2024).

    graph = to_nir(model, variables, dt=1e-3)                 # a flax nn.Sequential of Dense and LIF layers
    model, variables = from_nir(graph, dt=1e-4, discretization="euler")

NIR describes neurons in continuous time, `tau dv/dt = (v_leak - v) + r I`
with a hard reset to `v_reset`, and leaves the step to each library. A
sparx `LIF` is the discrete `v <- decay v + x`, so a conversion names how
time was discretized:

- `"exact"` (sparx's own): the leak's exact solution over a step, the
  input held over it, so `tau = -dt / ln(decay)` and `r = 1 / (1 - decay)`.
- `"euler"`: forward Euler, `decay = 1 - dt / tau` and `r = tau / dt`,
  which snnTorch's NIR import and export assume at `dt = 1e-4` s.

`r` is folded into the preceding linear layer on import. NIR times are in
seconds. Supported: `nn.Sequential` stacks of `flax.linen.Dense` and
`sparx.nn.LIF` with `reset="zero"` (NIR's reset), one threshold per layer;
anything else raises.
"""

from __future__ import annotations

import itertools
import math
from typing import Literal

import flax.linen as nn
import numpy as np

from sparx.nn import LIF

__all__ = ["from_nir", "to_nir"]

Discretization = Literal["exact", "euler"]


def _continuous(decay: float, dt: float, discretization: Discretization) -> tuple[float, float]:
    """`(tau, r)` of the continuous LIF whose `discretization` at `dt` is `v <- decay v + x`."""
    if discretization == "exact":
        return -dt / math.log(decay), 1 / (1 - decay)
    tau = dt / (1 - decay)
    return tau, tau / dt


def _discrete(tau: float, r: float, dt: float, discretization: Discretization) -> tuple[float, float]:
    """`(decay, input scale)` of a continuous LIF's `discretization` at `dt`."""
    if discretization == "exact":
        decay = math.exp(-dt / tau)
        return decay, r * (1 - decay)
    return 1 - dt / tau, r * dt / tau


def to_nir(model: nn.Sequential, variables, *, dt: float, discretization: Discretization = "exact"):
    """The NIR graph of a sequential stack of `nn.Dense` and `sparx.nn.LIF` layers, at step `dt` seconds."""
    import nir

    params = variables["params"]
    nodes, names, size = {}, [], None
    for k, layer in enumerate(model.layers):
        name = str(k)
        if isinstance(layer, nn.Dense):
            p = params[f"layers_{k}"]
            kernel = np.asarray(p["kernel"])
            bias = np.asarray(p["bias"]) if "bias" in p else np.zeros(kernel.shape[1])
            nodes[name] = nir.Affine(weight=kernel.T, bias=bias)
            size = kernel.shape[1]
        elif isinstance(layer, LIF):
            if layer.reset != "zero" or layer.learn_tau:
                raise NotImplementedError("NIR's LIF resets to v_reset: "
                                          "export LIF(reset='zero') with a fixed tau")
            if size is None:
                raise NotImplementedError("a LIF layer must follow a Dense layer to be exported")
            tau, r = _continuous(math.exp(-1 / layer.tau), dt, discretization)
            ones = np.ones(size)
            nodes[name] = nir.LIF(tau=tau * ones, r=r * ones, v_leak=0 * ones,
                                  v_threshold=layer.threshold * ones, v_reset=0 * ones)
        else:
            raise NotImplementedError(f"cannot export {type(layer).__name__} to NIR")
        names.append(name)
    first = nodes[names[0]]
    if not isinstance(first, nir.Affine):
        raise NotImplementedError("the stack must start with a Dense layer")
    nodes["input"] = nir.Input(input_type={"input": np.array([first.weight.shape[1]])})
    nodes["output"] = nir.Output(output_type={"output": np.array([size])})
    order = ["input", *names, "output"]
    return nir.NIRGraph(nodes=nodes, edges=list(itertools.pairwise(order)),
                        metadata={"dt": dt, "discretization": discretization, "exporter": "sparx"})


def from_nir(graph, *, dt: float, discretization: Discretization = "exact"):
    """A sequential stack and its variables from a NIR graph of a chain of `Affine`/`Linear` and `LIF`
    nodes, read with `discretization` at `dt` seconds."""
    import nir

    successors = dict(graph.edges)
    if len(successors) != len(graph.edges):
        raise NotImplementedError("only chains of nodes import: a node has more than one successor")
    node, chain = "input", []
    while node in successors:
        node = successors[node]
        if node != "output":
            chain.append(graph.nodes[node])
    layers, params = [], {}
    for node in chain:
        k = len(layers)
        if isinstance(node, (nir.Affine, nir.Linear)):
            weight = np.asarray(node.weight)
            bias = np.asarray(node.bias) if isinstance(node, nir.Affine) else np.zeros(weight.shape[0])
            layers.append(nn.Dense(weight.shape[0]))
            params[f"layers_{k}"] = {"kernel": weight.T.copy(), "bias": bias.copy()}
        elif isinstance(node, nir.LIF):
            tau, r = np.asarray(node.tau, float), np.asarray(node.r, float)
            reset = 0.0 if node.v_reset is None else node.v_reset
            if not np.allclose(node.v_leak, 0) or not np.allclose(reset, 0):
                raise NotImplementedError("only LIF nodes with v_leak = v_reset = 0 import")
            if np.unique(tau).size != 1 or np.unique(node.v_threshold).size != 1:
                raise NotImplementedError("a LIF node imports with one tau and one threshold")
            decay, scale = _discrete(float(tau.ravel()[0]), 1.0, dt, discretization)
            scales = r * scale
            previous = params.get(f"layers_{k - 1}")
            if previous is None:
                raise NotImplementedError("a LIF node must follow an Affine or Linear node to import")
            previous["kernel"] = previous["kernel"] * scales  # NIR's r scales the input current
            previous["bias"] = previous["bias"] * scales
            layers.append(LIF(tau=-1 / math.log(decay), threshold=float(np.ravel(node.v_threshold)[0]),
                              reset="zero"))
        else:
            raise NotImplementedError(f"cannot import NIR node {type(node).__name__}")
    return nn.Sequential(layers), {"params": params}
