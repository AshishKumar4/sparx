"""Running a network for a long time: compiled chunks, state carried between them, records on the host.

    result = simulate(network, variables, duration=1000.0, key=key,
                      monitors=(SpikeRaster("e"), PopulationRate("e")), chunk=100.0)
    result.records[0]          # [steps, neurons] spikes of "e", a NumPy array
    result.variables           # the variables with the state after the run, to continue from

Simulation without gradients is a function, not a runner beside dew's
`Trainer` (design.md section 6.4). It compiles one chunk of the time loop
once, runs it as many times as the duration needs, and moves each chunk's
monitor records to the host, so device memory holds one chunk of records
however long the run. The step counter lives in the state, so noise drawn
per step is the same whatever the chunk length, and a run continued from
`result.variables` is the run it would have been without the break.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sparx.graph.network import Monitor, Network

__all__ = ["Simulation", "simulate"]


@dataclass(frozen=True)
class Simulation:
    """What `simulate` returns: each monitor's records over the run, on the host, and the final variables."""

    records: tuple[Any, ...]
    variables: Mapping[str, Any]
    dt: float

    @property
    def times(self) -> np.ndarray:
        """The end of each recorded step, in ms from the start of this run."""
        steps = len(jax.tree.leaves(self.records)[0]) if jax.tree.leaves(self.records) else 0
        return (np.arange(steps) + 1) * self.dt


def simulate(network: Network, variables: Mapping[str, Any], *, duration: float,
             key: jax.Array | None = None, drive: Mapping[str, Any] | None = None,
             monitors: Sequence[Monitor] = (), chunk: float = 100.0, trials: int | None = None,
             mesh: Mesh | None = None) -> Simulation:
    """Run `network` from `variables` for `duration` ms in compiled chunks of `chunk` ms.

    `drive` maps a `CurrentInput`'s name to its values over the whole run,
    `[steps, ...]` (`[trials, steps, ...]` with `trials`); `key` seeds
    Poisson inputs. The last chunk may be shorter, which compiles it once
    more.

    `trials` runs that many independent trials from the same variables, each
    with its own noise (`fold_in(key, trial)`): records and state gain a
    leading trial axis. `mesh` spreads the run over devices along its first
    axis: the trials when there are trials, else the neurons, whose state
    each device keeps for its share of every population (the step's
    exchange of spikes the compiler partitions). Either gives one device's
    results.
    """
    steps = round(duration / network.dt)
    per_chunk = max(1, min(steps, round(chunk / network.dt)))
    key = jax.random.key(0) if key is None else key
    monitors = tuple(monitors)
    time_axis = 0 if trials is None else 1
    drive = _checked(drive, steps, time_axis)
    fixed = {name: value for name, value in variables.items() if name != "state"}
    state = variables["state"]

    def run(fixed, state, drive, key, length):
        records, updates = network.apply({**fixed, "state": state}, drive, steps=length, monitors=monitors,
                                         rngs={"noise": key}, mutable=["state"])
        return records, updates["state"]

    if trials is not None:
        keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(trials))
        state = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (trials, *jnp.shape(leaf))), state)
        run = jax.vmap(run, in_axes=(None, 0, 0, 0, None))
    else:
        keys = key
    run = jax.jit(run, static_argnames="length")
    if mesh is not None:
        state, keys = _place(network, mesh, state, keys, trials)
    chunks = []
    for start in range(0, steps, per_chunk):
        stop = min(start + per_chunk, steps)
        window = (slice(None),) * time_axis + (slice(start, stop),)
        records, state = run(fixed, state, {name: value[window] for name, value in drive.items()}, keys,
                             stop - start)
        if mesh is not None:
            state, _ = _place(network, mesh, state, keys, trials)
        _check_capacity(state, stop * network.dt)
        chunks.append(jax.device_get(records))
    return Simulation(_join(monitors, chunks, time_axis), {**fixed, "state": state}, network.dt)


def _checked(drive: Mapping[str, Any] | None, steps: int, time_axis: int) -> dict[str, np.ndarray]:
    drive = {name: np.asarray(value) for name, value in (drive or {}).items()}
    for name, value in drive.items():
        if value.ndim <= time_axis or value.shape[time_axis] != steps:
            raise ValueError(f"drive {name!r} needs a time axis of {steps} steps at {time_axis}, "
                             f"got shape {value.shape}")
    return drive


def _check_capacity(state, until: float) -> None:
    overflow = jax.device_get(state["network"]["overflow"])
    over = {name: int(np.max(count)) for name, count in overflow.items() if np.max(count)}
    if over:
        raise RuntimeError(f"event projections exceeded their capacity in {over} steps (by projection) "
                           f"before {until} ms; raise Projection.capacity")


def _join(monitors: tuple[Monitor, ...], chunks: list, time_axis: int) -> tuple[Any, ...]:
    """Each monitor's records over the chunks: concatenated over time, or summed if it accumulates."""
    if not chunks:
        return ()

    def join(monitor, parts):
        if monitor.accumulate:
            return sum(parts[1:], parts[0])
        return jax.tree.map(lambda *xs: np.concatenate(xs, axis=time_axis), *parts)

    return tuple(join(m, parts) for m, parts in zip(monitors, zip(*chunks, strict=True), strict=True))


def _place(network: Network, mesh: Mesh, state, keys, trials: int | None):
    """Shard the state over the mesh's first axis: along trials, or along each population's neurons."""
    axis = mesh.axis_names[0]
    devices = mesh.shape[axis]
    if trials is not None:
        if trials % devices:
            raise ValueError(f"{trials} trials do not divide over {devices} devices")
        sharding = NamedSharding(mesh, PartitionSpec(axis))
        return jax.device_put(state, sharding), jax.device_put(keys, sharding)
    sizes = {p.size for p in network.populations}
    for size in sizes:
        if size % devices:
            raise ValueError(f"a population of {size} neurons does not divide over {devices} devices")

    def place(leaf):
        shape = jnp.shape(leaf)
        if shape and shape[-1] in sizes:
            spec = PartitionSpec(*([None] * (len(shape) - 1)), axis)
            return jax.device_put(leaf, NamedSharding(mesh, spec))
        return jax.device_put(leaf, NamedSharding(mesh, PartitionSpec()))

    return jax.tree.map(place, state), keys
