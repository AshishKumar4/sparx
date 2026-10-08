"""Event delivery: a projection's spikes sent over the out-edges of the neurons that fired.

    e = event_edges(edges, weight, delays, pre=population.size)        # the connectome's storage
    arrived = deliver_events(e, weight, sent, out, width=16, release=None, key=None)

A step delivers its spiking neurons in passes of up to `width`, as many
passes as it has spikes for (none in a silent step), so its cost follows
the activity, not the edge count, and no spike is left out. Each pass takes
the next spiking neurons with `jax.lax.top_k`. With out-degrees near even or
few (`EVENT_PADDING`, `EVENT_ROWS`) each neuron's out-edges are one padded
row, and a pass gathers its neurons' rows. Otherwise their out-edges,
contiguous when edges are sorted by presynaptic neuron, are laid end to end
by a prefix sum of their out-degrees and walked in blocks of `EVENT_BLOCK`
slots; each slot finds its neuron by binary search. With stochastic
release, each pass, and each block of a pass, draws from `key` folded with
its index, the key of the step that sends.

The functions read arrays and the two settings of a projection that event
delivery has, its pass width and its release; `sparx.graph.network` decides
which projections deliver by events and where what they send lands.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

from sparx.dynamics.synapses import StochasticRelease
from sparx.graph.connectivity import EdgeList

__all__ = ["EVENT_BLOCK", "EVENT_PADDING", "EVENT_ROWS", "deliver_events", "event_edges", "sends_ahead"]

EVENT_BLOCK = 4096
EVENT_PADDING = 2.0
"""An event projection whose out-degrees are this close to even, the widest at most this many times the
mean, keeps each neuron's out-edges as one padded row, read without a search; so does one whose padded
rows hold at most `EVENT_ROWS` entries, whatever the spread."""
EVENT_ROWS = 2 ** 20

type Edges = Mapping[str, jax.Array]
"""An event projection's stored edges (`event_edges`), on the device."""


def event_edges(edges: EdgeList, weight: np.ndarray, delays: np.ndarray, pre: int) -> dict[str, np.ndarray]:
    """An event projection's edges sorted by presynaptic neuron, with where each neuron's run starts and
    how long it is, its delay or each edge's, and its padded rows when the out-degrees are near even or
    few (`EVENT_PADDING`)."""
    order = np.lexsort((edges.post, edges.pre))
    counts = np.bincount(edges.pre, minlength=pre)
    built = {"delay": delays[order] if np.ndim(delays) else np.asarray(delays, np.int32),
             "by_pre": edges.post[order],
             "start": np.append(np.cumsum(counts) - counts, 0).astype(np.int32),
             "count": np.append(counts, 0).astype(np.int32),
             "weight": weight[order]}
    if len(edges) and counts.max() * pre <= max(EVENT_PADDING * len(edges), EVENT_ROWS):
        built["rows"] = _padded(counts, len(edges))
    return built


def _padded(counts: np.ndarray, edges: int) -> np.ndarray:
    """Each presynaptic neuron's out-edges, by their index in presynaptic order, as one row padded with
    `edges`, past the last edge; a last row of padding stands for no neuron."""
    widest = int(counts.max())
    starts = np.cumsum(counts) - counts
    column = np.arange(widest)
    rows = np.where(column < counts[:, None], starts[:, None] + column, edges)
    return np.concatenate([rows, np.full((1, widest), edges)]).astype(np.int32)


def sends_ahead(e: Mapping[str, np.ndarray | jax.Array]) -> bool:
    """Whether a projection's built edges deliver by events over delays of their own, sending each spike's
    weights ahead to the steps they are due in."""
    return "by_pre" in e and np.ndim(e["delay"]) == 1


def deliver_events(e: Edges, weight: jax.Array, sent: jax.Array, out: jax.Array, *, width: int,
                   release: StochasticRelease | None, key: jax.Array | None,
                   t: jax.Array | None = None) -> jax.Array:
    """What the neurons that sent `sent` deliver over the edges `e` of weights `weight`, added to `out`.

    `out` is a value per postsynaptic neuron, or, with `t`, for a
    projection whose edges have their own delays, the rows of what is on
    its way, `[rows, post]`: an edge sending in step `t` adds to row
    `(t + delay) % rows`. `width` is the most neurons one pass delivers,
    and `release`, when given, transmits each edge's spike by a draw from
    `key`.

    A `while_loop` has no reverse mode, so without stochastic release the
    delivery is differentiated as its edge list (`_as_edges`).
    """
    if weight.shape[0] == 0:
        return out
    passes = functools.partial(_passes, width=width, release=release, key=key)
    if release is not None:
        return passes(weight, sent, out, e, t)
    return _as_edges(passes, weight, sent, out, e, t)


def _passes(weight: jax.Array, sent: jax.Array, out: jax.Array, e: Edges, t: jax.Array | None, *, width: int,
            release: StochasticRelease | None, key: jax.Array | None) -> jax.Array:
    """`deliver_events`' passes over the spiking neurons."""
    size = sent.shape[0]
    width = min(width, size)
    deliver = _rows if "rows" in e else _slots

    def busy(carry: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        return jnp.any(carry[1] != 0)

    def one_pass(carry: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        out, left, n = carry
        value, index = jax.lax.top_k(left, width)  # what a neuron sends is never below 0
        active = jnp.where(value != 0, index, size)
        passed = None if key is None else jax.random.fold_in(key, n)
        out = deliver(e, weight, sent, release, passed, active, out, t)
        return out, left.at[active].set(0, mode="drop"), n + 1

    return jax.lax.while_loop(busy, one_pass, (out, sent, jnp.zeros((), jnp.int32)))[0]


def _rows(e: Edges, weight: jax.Array, sent: jax.Array, release: StochasticRelease | None,
          key: jax.Array | None, active: jax.Array, out: jax.Array, t: jax.Array | None) -> jax.Array:
    """The `active` neurons' padded rows of out-edges, gathered and summed onto their targets; an
    index past the last neuron stands for none."""
    # A padding slot, or a neuron that stands for none, is an edge index past the last edge.
    edge = e["rows"][active]
    valid = edge < weight.shape[0]
    spiked = sent.at[active].get(mode="fill", fill_value=0)[:, None]
    weights = weight.at[edge].get(mode="fill", fill_value=0)
    if release is None:
        value = jnp.where(valid, weights * spiked, 0)
    else:
        assert key is not None
        value = jnp.where(valid, release.transmit(key, weights, spiked), 0)
    return out.at[_landing(e, edge, out, t)].add(value, mode="drop")


def _slots(e: Edges, weight: jax.Array, sent: jax.Array, release: StochasticRelease | None,
           key: jax.Array | None, active: jax.Array, out: jax.Array, t: jax.Array | None) -> jax.Array:
    """The `active` neurons' out-edges, laid end to end and walked in blocks of slots."""
    degree = e["count"][active]
    ends = jnp.cumsum(degree)
    total = ends[-1]

    def block(i, out):
        slot = i * EVENT_BLOCK + jnp.arange(EVENT_BLOCK)
        owner = jnp.minimum(jnp.searchsorted(ends, slot, side="right"), len(active) - 1)
        edge = e["start"][active[owner]] + slot - (ends[owner] - degree[owner])
        valid = slot < total
        edge = jnp.where(valid, edge, 0)
        spiked = sent.at[active[owner]].get(mode="fill", fill_value=0)
        if release is None:
            value = jnp.where(valid, weight[edge] * spiked, 0)
        else:
            assert key is not None
            drawn = release.transmit(jax.random.fold_in(key, i), weight[edge], spiked)
            value = jnp.where(valid, drawn, 0)
        return out.at[_landing(e, edge, out, t)].add(value, mode="drop")

    blocks = (total + EVENT_BLOCK - 1) // EVENT_BLOCK
    return jax.lax.fori_loop(0, blocks, block, out)


def _landing(e: Edges, edge: jax.Array, out: jax.Array, t: jax.Array | None) -> tuple[jax.Array, ...]:
    """Where in `out` each event edge's value lands: its postsynaptic neuron, or with `t`, that neuron in the
    row its delay reaches from step `t`. An index past the last edge lands past the end, to be dropped."""
    post = e["by_pre"].at[edge].get(mode="fill", fill_value=out.shape[-1])
    if t is None:
        return (post,)
    lag = e["delay"].at[edge].get(mode="fill", fill_value=0)
    return ((t + lag) % out.shape[0], post)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _as_edges(deliver: Callable[..., jax.Array], weight: jax.Array, sent: jax.Array, out: jax.Array,
              e: Edges, t: jax.Array | None) -> jax.Array:
    """`deliver(weight, sent, out, e, t)`, an event projection's delivery, differentiated as its edge list.

    Each edge adds `weight * sent[pre]` where it lands in `out`, a map
    linear in what the neurons send, so its tangent is the same sum over
    every edge, which autodiff transposes for the gradient. A membrane near
    threshold has a surrogate tangent without a spike, so the tangent visits
    every edge, as differentiating `format="edges"` does. With `t`, an
    edge's value lands in row `(t + delay) % rows`: it is scattered by delay
    and then rolled by `t`, so the indices the gradient keeps are the same
    every step.
    """
    return deliver(weight, sent, out, e, t)


def _edge_tangent(deliver: Callable[..., jax.Array], primals: tuple, tangents: tuple
                  ) -> tuple[jax.Array, jax.Array]:
    """`_as_edges`' delivery and its tangent; a tangent that is zero by construction is skipped."""
    weight, sent, out, e, t = primals
    d_weight, d_sent, d_out = tangents[:3]
    edges, size = weight.shape[0], sent.shape[0]
    pre = jnp.repeat(jnp.arange(size), e["count"][:-1], total_repeat_length=edges)
    value = jnp.zeros(edges, out.dtype)
    if not isinstance(d_sent, SymbolicZero):
        value = value + weight * d_sent[pre]
    if not isinstance(d_weight, SymbolicZero):
        value = value + d_weight * sent[pre]
    if t is None:
        arrived = jnp.zeros_like(out).at[e["by_pre"]].add(value)
    else:
        arrived = jnp.roll(jnp.zeros_like(out).at[e["delay"], e["by_pre"]].add(value), t, axis=0)
    return deliver(weight, sent, out, e, t), arrived if isinstance(d_out, SymbolicZero) else d_out + arrived


_as_edges.defjvp(_edge_tangent, symbolic_zeros=True)
