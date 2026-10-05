"""Which pairs of neurons a projection connects.

A connectivity rule draws an edge list once, when a network is built, on
the host: edge counts of random rules are random, and edges are data, not
traced values. Rules take a NumPy generator seeded from the network's key,
so a network rebuilt from the same key has the same edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

__all__ = ["AllToAll", "Connectivity", "EdgeList", "FixedInDegree", "FixedOutDegree", "FixedProbability",
           "FromEdges", "OneToOne"]


class EdgeList(NamedTuple):
    """Edges `pre[i] -> post[i]`, int32, sorted by postsynaptic neuron and then presynaptic.

    `order[i]` is the edge's position in the order the rule produced it,
    which per-edge values given with a `FromEdges` table follow.
    """

    pre: np.ndarray
    post: np.ndarray
    order: np.ndarray

    @classmethod
    def of(cls, pre, post) -> EdgeList:
        pre, post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
        order = np.lexsort((pre, post))
        return cls(pre[order].astype(np.int32), post[order].astype(np.int32), order)

    def __len__(self) -> int:
        return len(self.pre)


class Connectivity(Protocol):
    def edges(self, rng: np.random.Generator, pre: int, post: int, same: bool) -> EdgeList:
        """Draw edges between populations of `pre` and `post` neurons; `same` when they are one population."""
        ...


def _drop_autapses(pre: np.ndarray, post: np.ndarray, same: bool, autapses: bool):
    if same and not autapses:
        keep = pre != post
        return pre[keep], post[keep]
    return pre, post


@dataclass(frozen=True)
class AllToAll:
    """Every presynaptic neuron to every postsynaptic one (without self-connections unless `autapses`)."""

    autapses: bool = False

    def edges(self, rng: np.random.Generator, pre: int, post: int, same: bool) -> EdgeList:
        sources, targets = np.meshgrid(np.arange(pre), np.arange(post), indexing="ij")
        return EdgeList.of(*_drop_autapses(sources.ravel(), targets.ravel(), same, self.autapses))


@dataclass(frozen=True)
class OneToOne:
    """Neuron `i` to neuron `i`; the populations must be the same size."""

    def edges(self, rng: np.random.Generator, pre: int, post: int, same: bool) -> EdgeList:
        if pre != post:
            raise ValueError(f"OneToOne needs populations of one size, got {pre} and {post}")
        return EdgeList.of(np.arange(pre), np.arange(post))


@dataclass(frozen=True)
class FixedProbability:
    """Each pair independently with probability `p` (Erdős-Rényi), NEST's `pairwise_bernoulli`."""

    p: float
    autapses: bool = False

    def edges(self, rng: np.random.Generator, pre: int, post: int, same: bool) -> EdgeList:
        # Draw the count per postsynaptic neuron, then distinct sources:
        # memory is per edge, not per pair.
        pool = pre - (1 if same and not self.autapses else 0)
        counts = rng.binomial(pool, self.p, size=post)
        sources, targets = [], []
        for target, count in enumerate(counts):
            chosen = rng.choice(pool, size=count, replace=False)
            if same and not self.autapses:
                chosen = chosen + (chosen >= target)  # skip the neuron itself
            sources.append(chosen)
            targets.append(np.full(count, target))
        if not sources:
            return EdgeList.of(np.zeros(0), np.zeros(0))
        return EdgeList.of(np.concatenate(sources), np.concatenate(targets))


@dataclass(frozen=True)
class FixedInDegree:
    """Each postsynaptic neuron receives exactly `k` inputs, drawn without replacement unless `multapses`.

    Brunel (2000) connects this way: every neuron receives `C_E` excitatory
    and `C_I` inhibitory inputs. NEST's `fixed_indegree`.
    """

    k: int
    autapses: bool = False
    multapses: bool = False

    def edges(self, rng: np.random.Generator, pre: int, post: int, same: bool) -> EdgeList:
        pool = pre - (1 if same and not self.autapses else 0)
        if not self.multapses and self.k > pool:
            raise ValueError(f"FixedInDegree({self.k}) exceeds the {pool} possible sources")
        if self.multapses:
            chosen = rng.integers(0, pool, size=(post, self.k))
        else:
            chosen = np.stack([rng.choice(pool, size=self.k, replace=False) for _ in range(post)]).reshape(
                post, self.k)
        targets = np.repeat(np.arange(post), self.k).reshape(post, self.k)
        if same and not self.autapses:
            chosen = chosen + (chosen >= targets)
        return EdgeList.of(chosen.ravel(), targets.ravel())


@dataclass(frozen=True)
class FixedOutDegree:
    """Each presynaptic neuron projects to exactly `k` targets. NEST's `fixed_outdegree`."""

    k: int
    autapses: bool = False
    multapses: bool = False

    def edges(self, rng: np.random.Generator, pre: int, post: int, same: bool) -> EdgeList:
        transposed = FixedInDegree(self.k, self.autapses, self.multapses).edges(rng, post, pre, same)
        return EdgeList.of(transposed.post, transposed.pre)


@dataclass(frozen=True)
class FromEdges:
    """Given edges, as from a connectome table. Pairs may repeat (multapses)."""

    pre: np.ndarray
    post: np.ndarray

    def edges(self, rng: np.random.Generator, pre: int, post: int, same: bool) -> EdgeList:
        sources, targets = np.asarray(self.pre), np.asarray(self.post)
        if len(sources) != len(targets):
            raise ValueError("FromEdges needs as many presynaptic as postsynaptic indices")
        if len(sources) and (sources.min() < 0 or sources.max() >= pre or targets.min() < 0
                             or targets.max() >= post):
            raise ValueError(f"FromEdges indices fall outside populations of {pre} and {post} neurons")
        return EdgeList.of(sources, targets)
