"""Read and regularize the firing rates spiking layers sow.

Applying a network with the `"spike_rates"` collection mutable collects
every spiking layer's rate per example and neuron, averaged over time:

    outputs, sown = net.apply(params, spikes, mutable=["spike_rates"])
    sown["spike_rates"]  # {"LIF_0": {"rate": (rates [B, 128],)}, ...}

`firing_rates` summarizes them per layer for logging; `rate_penalty`
keeps each neuron's rate in a band as a differentiable loss term.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

__all__ = ["firing_rates", "rate_penalty"]

type Sown = Mapping[str, object]
"""The `"spike_rates"` collection, or the whole mutated-variables dict holding it."""


def _rates(sown: Sown) -> dict[str, jax.Array]:
    """Each sown rate array by the path of the layer that sowed it."""
    tree = sown.get("spike_rates", sown)
    found: dict[str, jax.Array] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        keys = [str(entry.key) for entry in path if isinstance(entry, jax.tree_util.DictKey)]
        if keys and keys[-1] == "rate":
            keys = keys[:-1]
        name = "/".join(keys)
        call = path[-1]
        # A layer called more than once sows once per call.
        if isinstance(call, jax.tree_util.SequenceKey) and call.idx > 0:
            name = f"{name}#{call.idx}"
        found[name] = leaf
    return found


def firing_rates(sown: Sown) -> dict[str, jax.Array]:
    """The mean firing rate of each spiking layer, in spikes per step, keyed by layer path."""
    return {name: jnp.mean(rate) for name, rate in _rates(sown).items()}


def rate_penalty(sown: Sown, lower: float = 0.0, upper: float = 1.0,
                 rows: jax.Array | None = None) -> jax.Array:
    """The squared distance of each neuron's rate outside `[lower, upper]`, averaged over neurons.

    A neuron's rate is its time-averaged spikes averaged over the batch (the
    leading axis of each sown array), in spikes per step as `firing_rates`
    reports it, and so are `lower` and `upper`. Each example is weighed by
    `rows` when given, `[B]`, so a batch's repeated rows can weigh nothing. Silent
    neurons below `lower` receive gradient to fire and saturated ones above
    `upper` to stop, the role of the activity regularizers of Zenke and
    Vogels (Neural Computation 2021). Every neuron of every layer weighs the
    same.
    """
    total, count = jnp.zeros((), jnp.float32), 0
    for rate in _rates(sown).values():
        if rows is None:
            per_neuron = jnp.mean(rate, axis=0)
        else:
            weights = jnp.reshape(rows, (-1, *(1,) * (rate.ndim - 1)))
            per_neuron = jnp.sum(rate * weights, axis=0) / jnp.sum(rows)
        total = total + jnp.sum(jax.nn.relu(per_neuron - upper) ** 2 + jax.nn.relu(lower - per_neuron) ** 2)
        count += per_neuron.size
    if count == 0:
        raise ValueError("no firing rates were sown; apply the network with mutable=['spike_rates']")
    return total / count
