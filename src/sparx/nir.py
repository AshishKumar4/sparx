"""Exchanging networks through NIR, the Neuromorphic Intermediate Representation (Pedersen et al. 2024).

    graph = to_nir(model, variables, dt=1e-3)                 # a flax nn.Sequential, see below
    model, variables = from_nir(graph, dt=1e-4, discretization="euler")

NIR describes neurons in continuous time, `tau dv/dt = (v_leak - v) + r I`
with a hard reset to `v_reset`, and leaves the step to each library. A
sparx `LIF` is the discrete `v <- decay v + x`, so a conversion names how
time was discretized:

- `"exact"` (sparx's own): the leak's exact solution over a step, the
  input held over it, so `tau = -dt / ln(decay)` and `r = 1 / (1 - decay)`.
- `"euler"`: forward Euler, `decay = 1 - dt / tau` and `r = tau / dt`,
  which snnTorch's NIR import and export assume at `dt = 1e-4` s.

`r` is folded into the preceding weight layer on import. NIR times are in
seconds. Supported: `nn.Sequential` stacks of these layers, with anything
else refused:

- `flax.linen.Dense`, as NIR `Affine` (or `Linear` on import);
- `flax.linen.Conv` with a 2-d kernel, as `Conv2d`;
- `sparx.nn.Flatten` over each example's whole shape, as `Flatten`;
- `sparx.nn.LIF` with `reset="zero"` (NIR's reset) and one threshold, as `LIF`;
- `sparx.nn.IF` with `reset="zero"`, as `IF`, whose `r = 1 / dt` makes a
  step add its input to the membrane, under either discretization;
- `sparx.nn.LI`, a classifier's readout, as `LI`, the same leak without a
  threshold;
- `sparx.nn.Recurrent(LIF(...))` on a flat input, as a `LIF` node with a
  `Linear` edge from its output back to its input.

Images. NIR follows PyTorch: one example of a 2-d convolution's input is
`[C, H, W]`, and its weight is `[C_out, C_in / groups, kH, kW]`. Flax is
channels last: the input is `[..., H, W, C]` and the kernel
`[kH, kW, C_in / groups, C_out]`. The weights are transposed both ways, and
a sparx tensor is a NIR tensor with its channel axis moved last; per-neuron
parameters of a `LIF` node after a convolution (`tau`, `r`, `v_threshold`)
are `[C, H, W]` in NIR. A NIR `Flatten` orders the features C, H, W, as
PyTorch does, and so does `sparx.nn.Flatten`, which moves the channel axis
first before it reshapes; a plain flax reshape would order them H, W, C
and scramble the dense layer after it. NIR's convolution pads both sides
equally, so a flax padding that pads one side more (`"SAME"` with an even
kernel or a stride) is refused. NIR 1.0.8 types a `Conv2d` from its
kernel's height alone and as if it were ungrouped, so its graph check
rejects a non-square or grouped kernel. `from_nir` reads both from a
graph built without the check, and `to_nir` raises NIR's type error.
`to_nir` needs `input_shape`, one example's shape in sparx's layout
(`(H, W, C)`), when the stack does not start with a `Dense` layer.

Recurrence. sparx's `Recurrent` adds `s[t-1] @ W` to the input of step `t`.
The feedback is the previous step's spikes, delayed by one step, since a
spike cannot reach its own neuron within the step that fired it. snnTorch's
`RLeaky` does the same, and exports as a `LIF` node `"k.lif"` and an
`Affine` node `"k.w_rec"` with edges both ways between them, inside the
chain; this module reads that cycle as one `Recurrent` layer and writes it
the same way, with node names to match. The NIR weight is `[out, in]`, so
`W` is its transpose. sparx's recurrent matrix has no bias: an `Affine`
feedback's bias is a constant input every step, so it is added to the bias
of the layer before the `LIF` node on import, and the feedback exports as a
`Linear` node. A recurrent subgraph nested as a `NIRGraph` node, with its
own `Input` and `Output`, imports too.

snnTorch 1.0's `import_nir` cannot read a recurrent graph (its subgraph
step fails on the graph it writes), so `to_nir`'s recurrent graphs are
checked against sparx's own import and snnTorch's export only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from sparx.dynamics import Dense, LICell, LIFCell, NeuronModel, RecurrentCell, decay
from sparx.nn import IF, LI, LIF, Flatten, Flattens, Modelled, Recurrent

if TYPE_CHECKING:
    import nir

__all__ = ["from_nir", "to_nir"]

Discretization = Literal["exact", "euler"]
type LayerParams = Mapping[str, jax.Array | np.ndarray]
"""One layer's parameters: name to array."""
type StackVariables = Mapping[str, Mapping[str, LayerParams]]
"""A sequential stack's variables, `{"params": {"layers_k": {...}}}`."""
type Pair = tuple[int, int]
type Edges = list[tuple[str, str]]


def _continuous(kept: float, dt: float, discretization: Discretization) -> tuple[float, float]:
    """`(tau, r)` of the continuous LIF whose `discretization` at `dt` is `v <- kept * v + x`."""
    if discretization == "exact":
        return -dt / math.log(kept), 1 / (1 - kept)
    tau = dt / (1 - kept)
    return tau, tau / dt


def _discrete(tau: float, dt: float, discretization: Discretization) -> tuple[float, float]:
    """`(decay, input scale)` of a continuous LIF's `discretization` at `dt`, for an input scale `r` of 1."""
    if discretization == "exact":
        kept = decay(tau, dt)
        return kept, 1 - kept
    return 1 - dt / tau, dt / tau


def _to_nir_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """NIR's shape of one example whose sparx shape is `shape`: the channel axis moves first."""
    return tuple(shape) if len(shape) < 2 else (shape[-1], *shape[:-1])


def _to_sparx_layout(values: np.ndarray) -> np.ndarray:
    """A NIR per-neuron array in sparx's layout: the channel axis moves last."""
    return values if values.ndim < 2 else np.moveaxis(values, 0, -1)


def _pair(value: int | Sequence[int] | np.ndarray | None, default: int = 1) -> Pair:
    """An int or a pair of ints, as a pair of Python ints."""
    if value is None:
        return default, default
    if isinstance(value, int | np.integer):
        return int(value), int(value)
    first, second = (int(v) for v in value)
    return first, second


# Export


def _output_shape(layer: nn.Module, params: LayerParams, shape: tuple[int, ...]) -> tuple[int, ...]:
    """The shape of one example out of `layer`, for one example of `shape` in."""
    x = jax.ShapeDtypeStruct((1, 1, *shape), jnp.float32)
    out = jax.eval_shape(lambda x: layer.apply({"params": params}, x), x)
    return tuple(out.shape[2:])


def _export_neuron(cell: NeuronModel, step: float, shape: tuple[int, ...], dt: float,
                   discretization: Discretization) -> nir.LIF | nir.IF | nir.LI:
    """The NIR node of a layer's model stepped at `step` in the unit of its decay: `LI` for an `LICell`,
    and for a `LIFCell`, `IF` when it does not leak, else `LIF`."""
    import nir

    ones = np.ones(_to_nir_shape(shape))
    if isinstance(cell, LICell):
        if np.ndim(cell.decay):
            raise NotImplementedError("NIR's LI holds one time constant: export a fixed tau")
        tau, r = _continuous(float(cell.decay) ** step, dt, discretization)
        return nir.LI(tau=tau * ones, r=r * ones, v_leak=0 * ones)
    if not isinstance(cell, LIFCell):
        raise NotImplementedError(f"NIR has no neuron for a {type(cell).__name__}: LIF and IF export")
    if cell.reset != "zero" or np.ndim(cell.decay) or np.ndim(cell.threshold):
        raise NotImplementedError("NIR's LIF and IF reset to v_reset and hold one time constant and "
                                  "threshold: export reset='zero' with a fixed tau")
    threshold = float(cell.threshold) * ones
    if cell.decay == 1.0:
        return nir.IF(r=ones / dt, v_threshold=threshold, v_reset=0 * ones)
    tau, r = _continuous(float(cell.decay) ** step, dt, discretization)
    return nir.LIF(tau=tau * ones, r=r * ones, v_leak=0 * ones, v_threshold=threshold, v_reset=0 * ones)


def _export_dense(params: LayerParams, shape: tuple[int, ...]) -> nir.Affine:
    import nir

    if len(shape) != 1:
        raise NotImplementedError("a Dense layer exports on flat inputs: put a Flatten before it")
    kernel = np.asarray(params["kernel"])
    bias = np.asarray(params["bias"]) if "bias" in params else np.zeros(kernel.shape[1])
    return nir.Affine(weight=kernel.T, bias=bias)


def _symmetric_padding(layer: nn.Conv, spatial: Pair, strides: Pair, dilation: Pair) -> Pair:
    """The padding on each side of each spatial axis, which NIR states once for both sides."""
    padding = layer.padding
    kernel = _pair(layer.kernel_size)
    if isinstance(padding, str):
        if padding not in ("SAME", "VALID"):
            raise NotImplementedError(f"NIR has no {padding} padding")
        dilated = [(k - 1) * d + 1 for k, d in zip(kernel, dilation, strict=True)]
        pads = jax.lax.padtype_to_pads(spatial, dilated, strides, padding)
    elif isinstance(padding, int):
        pads = [(padding, padding)] * 2
    else:
        pads = [(p, p) if isinstance(p, int) else tuple(p) for p in padding]
    if len(pads) != 2 or any(len(p) != 2 or p[0] != p[1] for p in pads):
        raise NotImplementedError(f"NIR pads both sides of an axis equally, which {padding!r} does not "
                                  f"at input {spatial}")
    return int(pads[0][0]), int(pads[1][0])


def _export_conv(layer: nn.Conv, params: LayerParams, shape: tuple[int, ...]) -> nir.Conv2d:
    import nir

    if isinstance(layer.kernel_size, int) or len(layer.kernel_size) != 2 or len(shape) != 3:
        raise NotImplementedError("only 2-d convolutions export, on inputs [H, W, C]")
    if _pair(layer.input_dilation) != (1, 1) or layer.mask is not None:
        raise NotImplementedError("NIR's Conv2d has no input dilation or kernel mask")
    kernel = np.asarray(params["kernel"])
    bias = np.asarray(params["bias"]) if "bias" in params else np.zeros(kernel.shape[-1])
    spatial = (shape[0], shape[1])
    strides, dilation = _pair(layer.strides), _pair(layer.kernel_dilation)
    return nir.Conv2d(input_shape=spatial, weight=kernel.transpose(3, 2, 0, 1), stride=strides,
                      padding=_symmetric_padding(layer, spatial, strides, dilation), dilation=dilation,
                      groups=layer.feature_group_count, bias=bias)


def _export_flatten(layer: Flattens, shape: tuple[int, ...]) -> nir.Flatten:
    import nir

    if layer.flattened_axes() != len(shape):
        raise NotImplementedError("a Flatten exports when it flattens each example's whole shape")
    return nir.Flatten(input_type={"input": np.array(_to_nir_shape(shape))}, start_dim=0, end_dim=-1)


def _export_layer(layer: nn.Module, params: LayerParams, name: str, shape: tuple[int, ...], dt: float,
                  discretization: Discretization) -> tuple[dict[str, nir.NIRNode], Edges, str]:
    """The NIR nodes of one layer, the edges among them, and the node its input and output use."""
    import nir

    if isinstance(layer, Modelled):
        x = jnp.zeros((1, 1, *shape), jnp.float32)
        cell = layer.apply({"params": params}, x, method="model")
        # `mutable` is unset, so apply returns the model alone, not a pair.
        assert not isinstance(cell, tuple)
        if not isinstance(cell, RecurrentCell):
            return {name: _export_neuron(cell, layer.dt, shape, dt, discretization)}, [], name
        if len(shape) != 1:
            raise NotImplementedError("a Recurrent layer exports on flat inputs")
        if not isinstance(cell.wiring, Dense) or cell.fast_weights is not None:
            raise NotImplementedError("NIR holds a fixed dense recurrence; a sparse or plastic one has no "
                                      "node")
        lif, w_rec = f"{name}.lif", f"{name}.w_rec"
        nodes: dict[str, nir.NIRNode] = {
            lif: _export_neuron(cell.inner, layer.dt, shape, dt, discretization),
            w_rec: nir.Linear(weight=np.asarray(cell.wiring.weight).T),
        }
        return nodes, [(lif, w_rec), (w_rec, lif)], lif
    if isinstance(layer, nn.Dense):
        node: nir.NIRNode = _export_dense(params, shape)
    elif isinstance(layer, nn.Conv):
        node = _export_conv(layer, params, shape)
    elif isinstance(layer, Flattens):
        node = _export_flatten(layer, shape)
    else:
        raise NotImplementedError(f"cannot export {type(layer).__name__} to NIR")
    return {name: node}, [], name


def _first_shape(model: nn.Sequential, params: Mapping[str, LayerParams],
                 input_shape: Sequence[int] | None) -> tuple[int, ...]:
    if input_shape is not None:
        return tuple(int(n) for n in input_shape)
    if model.layers and isinstance(model.layers[0], nn.Dense):
        return (int(np.shape(params["layers_0"]["kernel"])[0]),)
    raise ValueError("pass input_shape, one example's shape (H, W, C for images), "
                     "for a stack that does not start with a Dense layer")


def to_nir(model: nn.Sequential, variables: StackVariables, *, dt: float,
           discretization: Discretization = "exact",
           input_shape: Sequence[int] | None = None) -> nir.NIRGraph:
    """The NIR graph of a sequential stack of the layers above, at step `dt` seconds.

    `input_shape` is one example's input shape in sparx's layout, `(H, W, C)`
    for images; a stack that starts with a `Dense` layer has it from the
    kernel.
    """
    import nir

    params = variables["params"]
    shape = _first_shape(model, params, input_shape)
    nodes: dict[str, nir.NIRNode] = {"input": nir.Input(input_type={"input": np.array(_to_nir_shape(shape))})}
    edges: Edges = []
    previous = "input"
    for k, layer in enumerate(model.layers):
        if not isinstance(layer, nn.Module):
            raise NotImplementedError(f"cannot export the function {layer!r} to NIR")
        layer_params = params.get(f"layers_{k}", {})
        new_nodes, new_edges, port = _export_layer(layer, layer_params, str(k), shape, dt, discretization)
        nodes.update(new_nodes)
        edges += [(previous, port), *new_edges]
        previous = port
        shape = _output_shape(layer, layer_params, shape)
    nodes["output"] = nir.Output(output_type={"output": np.array(_to_nir_shape(shape))})
    edges.append((previous, "output"))
    return nir.NIRGraph(nodes=nodes, edges=edges,
                        metadata={"dt": dt, "discretization": discretization, "exporter": "sparx"})


# Import


def _inline(graph: nir.NIRGraph) -> tuple[dict[str, nir.NIRNode], Edges]:
    """`graph`'s nodes and edges with each nested `NIRGraph` node replaced by its contents.

    An inner node `n` of subgraph `g` becomes `g.n`; an edge into `g` goes to
    whatever `g`'s `Input` feeds, and an edge out of `g` leaves from whatever
    feeds `g`'s `Output`.
    """
    import nir

    nodes: dict[str, nir.NIRNode] = dict(graph.nodes)
    edges: Edges = [(a, b) for a, b in graph.edges]
    for key, node in graph.nodes.items():
        if not isinstance(node, nir.NIRGraph):
            continue
        inner, inner_edges = _inline(node)
        ends = [k for k, n in inner.items() if isinstance(n, nir.Input | nir.Output)]
        heads = [b for a, b in inner_edges if isinstance(inner[a], nir.Input)]
        tails = [a for a, b in inner_edges if isinstance(inner[b], nir.Output)]
        del nodes[key]
        nodes.update({f"{key}.{k}": n for k, n in inner.items() if k not in ends})
        edges = ([(a, b) for a, b in edges if key not in (a, b)]
                 + [(a, f"{key}.{h}") for a, b in edges if b == key for h in heads]
                 + [(f"{key}.{t}", b) for a, b in edges if a == key for t in tails]
                 + [(f"{key}.{a}", f"{key}.{b}") for a, b in inner_edges if a not in ends and b not in ends])
    return nodes, edges


def _feedback(nodes: dict[str, nir.NIRNode], edges: Edges) -> dict[str, str]:
    """The recurrent edges: each `LIF` node fed back by a weight node that only it feeds and is fed by."""
    import nir

    feedback = {}
    for a, b in edges:
        if not (isinstance(nodes[a], nir.LIF) and isinstance(nodes[b], nir.Affine | nir.Linear)):
            continue
        into, out_of = [x for x, y in edges if y == b], [y for x, y in edges if x == b]
        if into == [a] and out_of == [a]:
            feedback[a] = b
    return feedback


def _chain(graph: nir.NIRGraph) -> list[tuple[nir.NIRNode, nir.NIRNode | None]]:
    """The nodes from input to output, each with its recurrent weight node, if any."""
    import nir

    nodes, edges = _inline(graph)
    feedback = _feedback(nodes, edges)
    edges = [(a, b) for a, b in edges if a not in feedback.values() and b not in feedback.values()]
    successors = dict(edges)
    if len(successors) != len(edges):
        raise NotImplementedError("only chains of nodes import: a node has more than one successor")
    node = next(k for k, n in nodes.items() if isinstance(n, nir.Input))
    chain: list[tuple[nir.NIRNode, nir.NIRNode | None]] = []
    while node in successors:
        node = successors[node]
        if not isinstance(nodes[node], nir.Output):
            chain.append((nodes[node], nodes[feedback[node]] if node in feedback else None))
    if len(chain) + len(feedback) + 2 != len(nodes):
        raise NotImplementedError("only chains of nodes import: some nodes are off the input's path")
    return chain


def _import_affine(node: nir.Affine | nir.Linear) -> tuple[nn.Module, dict[str, np.ndarray]]:
    import nir

    weight = np.asarray(node.weight)
    bias = np.asarray(node.bias) if isinstance(node, nir.Affine) else np.zeros(weight.shape[0])
    return nn.Dense(weight.shape[0]), {"kernel": weight.T.copy(), "bias": bias.copy()}


def _import_padding(node: nir.Conv2d, kernel: Pair, dilation: Pair) -> list[Pair]:
    padding = node.padding
    if isinstance(padding, str):
        if padding == "valid":
            return [(0, 0), (0, 0)]
        # PyTorch's "same" puts the odd pixel of padding after the input.
        totals = [d * (k - 1) for k, d in zip(kernel, dilation, strict=True)]
        return [(t // 2, t - t // 2) for t in totals]
    return [(p, p) for p in _pair(padding)]


def _import_conv(node: nir.Conv2d) -> tuple[nn.Module, dict[str, np.ndarray]]:
    weight = np.asarray(node.weight)
    if weight.ndim != 4:
        raise NotImplementedError("a Conv2d weight is [C_out, C_in / groups, kH, kW]")
    kernel, dilation = (weight.shape[2], weight.shape[3]), _pair(node.dilation)
    layer = nn.Conv(weight.shape[0], kernel, strides=_pair(node.stride),
                    padding=_import_padding(node, kernel, dilation), kernel_dilation=dilation,
                    feature_group_count=int(node.groups))
    return layer, {"kernel": weight.transpose(2, 3, 1, 0).copy(), "bias": np.asarray(node.bias).copy()}


def _import_flatten(node: nir.Flatten) -> Flatten:
    shape = node.input_type.get("input") if node.input_type else None
    if shape is None:
        raise NotImplementedError("a Flatten node imports with its input type")
    ndim = len(shape)
    if node.start_dim % ndim != 0 or node.end_dim % ndim != ndim - 1:
        raise NotImplementedError("a Flatten node imports when it flattens each example's whole shape")
    return Flatten(ndim=ndim)


def _per_channel(values: np.ndarray) -> np.ndarray:
    """A neuron node's per-neuron `r` as one value per output channel (feature), which a weight layer
    absorbs."""
    values = _to_sparx_layout(np.asarray(values, float))
    rows = values.reshape(-1, values.shape[-1]) if values.ndim else values.reshape(1, 1)
    if not np.all(rows == rows[0]):
        raise NotImplementedError("a LIF node imports with one r per channel")
    return rows[0]


def _fold_input_scale(previous: dict[str, np.ndarray] | None, scales: np.ndarray, bias: np.ndarray,
                      kind: str) -> None:
    """Fold a neuron node's input scale, and a constant input `bias`, into the weight layer before it."""
    if previous is not None and "kernel" in previous:
        # NIR's r scales the input current, which a constant feedback bias joins.
        kernel = previous["kernel"]
        previous["kernel"] = (kernel * scales).astype(kernel.dtype)
        previous["bias"] = ((previous["bias"] + bias) * scales).astype(kernel.dtype)
    elif not np.allclose(scales, 1) or np.any(bias != 0):
        raise NotImplementedError(f"a {kind} node must follow an Affine, Linear or Conv2d node to import")


def _import_if(node: nir.IF, previous: dict[str, np.ndarray] | None, dt: float) -> IF:
    """The layer of an `IF` node, with `r dt` folded into `previous`."""
    if not np.allclose(0.0 if node.v_reset is None else node.v_reset, 0):
        raise NotImplementedError("only IF nodes with v_reset = 0 import")
    if np.unique(node.v_threshold).size != 1:
        raise NotImplementedError("an IF node imports with one threshold")
    _fold_input_scale(previous, _per_channel(np.asarray(node.r)) * dt, np.zeros(()), "IF")
    return IF(threshold=float(np.ravel(node.v_threshold)[0]), reset="zero")


def _import_lif(node: nir.LIF, w_rec: nir.NIRNode | None, previous: dict[str, np.ndarray] | None,
                dt: float, discretization: Discretization) -> tuple[nn.Module, dict[str, np.ndarray]]:
    """The layer of a `LIF` node, its feedback through `w_rec` if any, with `r` folded into `previous`."""
    import nir

    tau = np.asarray(node.tau, float)
    reset = 0.0 if node.v_reset is None else node.v_reset
    if not np.allclose(node.v_leak, 0) or not np.allclose(reset, 0):
        raise NotImplementedError("only LIF nodes with v_leak = v_reset = 0 import")
    if np.unique(tau).size != 1 or np.unique(node.v_threshold).size != 1:
        raise NotImplementedError("a LIF node imports with one tau and one threshold")
    kept, scale = _discrete(float(tau.ravel()[0]), dt, discretization)
    scales = _per_channel(np.asarray(node.r)) * scale
    bias = np.asarray(w_rec.bias) if isinstance(w_rec, nir.Affine) else np.zeros(())
    _fold_input_scale(previous, scales, bias, "LIF")
    lif = LIF(tau=-1 / math.log(kept), threshold=float(np.ravel(node.v_threshold)[0]), reset="zero")
    if w_rec is None:
        return lif, {}
    if not isinstance(w_rec, nir.Affine | nir.Linear):
        raise NotImplementedError(f"cannot import a {type(w_rec).__name__} feedback")
    weight = np.asarray(w_rec.weight)
    return Recurrent(neuron=lif), {"recurrent": (weight.T * scales).astype(weight.dtype)}


def _import_li(node: nir.LI, previous: dict[str, np.ndarray] | None, dt: float,
               discretization: Discretization) -> LI:
    """The layer of an `LI` node, with `r` folded into `previous`."""
    tau = np.asarray(node.tau, float)
    if not np.allclose(node.v_leak, 0):
        raise NotImplementedError("only LI nodes with v_leak = 0 import")
    if np.unique(tau).size != 1:
        raise NotImplementedError("an LI node imports with one tau")
    kept, scale = _discrete(float(tau.ravel()[0]), dt, discretization)
    _fold_input_scale(previous, _per_channel(np.asarray(node.r)) * scale, np.zeros(()), "LI")
    return LI(tau=-1 / math.log(kept))


def from_nir(graph: nir.NIRGraph, *, dt: float, discretization: Discretization = "exact"
             ) -> tuple[nn.Sequential, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """A sequential stack and its variables from a NIR graph of a chain of the nodes above, read with
    `discretization` at `dt` seconds. Image layers take inputs `[T, B, H, W, C]`."""
    import nir

    layers: list[nn.Module] = []
    params: dict[str, dict[str, np.ndarray]] = {}
    for node, w_rec in _chain(graph):
        k = len(layers)
        if w_rec is not None and not isinstance(node, nir.LIF):
            raise NotImplementedError("only LIF nodes import with feedback")
        if isinstance(node, nir.Affine | nir.Linear):
            layer, layer_params = _import_affine(node)
        elif isinstance(node, nir.Conv2d):
            layer, layer_params = _import_conv(node)
        elif isinstance(node, nir.Flatten):
            layer, layer_params = _import_flatten(node), {}
        elif isinstance(node, nir.LIF):
            layer, layer_params = _import_lif(node, w_rec, params.get(f"layers_{k - 1}"), dt, discretization)
        elif isinstance(node, nir.IF):
            layer, layer_params = _import_if(node, params.get(f"layers_{k - 1}"), dt), {}
        elif isinstance(node, nir.LI):
            layer, layer_params = _import_li(node, params.get(f"layers_{k - 1}"), dt, discretization), {}
        else:
            raise NotImplementedError(f"cannot import NIR node {type(node).__name__}")
        layers.append(layer)
        if layer_params:
            params[f"layers_{k}"] = layer_params
    return nn.Sequential(layers), {"params": params}

