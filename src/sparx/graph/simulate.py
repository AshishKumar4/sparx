"""Running a network for a long time: compiled chunks, state carried between them, records on the host.

    result = simulate(network, variables, duration=1000.0, key=key,
                      monitors=(Spikes("e"), PopulationRate("e")), chunk=100.0)
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
import numpy as np

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
             monitors: Sequence[Monitor] = (), chunk: float = 100.0) -> Simulation:
    """Run `network` from `variables` for `duration` ms in compiled chunks of `chunk` ms.

    `drive` maps a `CurrentInput`'s name to its values over the whole run,
    `[steps, ...]`; `key` seeds Poisson inputs. The last chunk may be
    shorter, which compiles it once more.
    """
    steps = round(duration / network.dt)
    per_chunk = max(1, min(steps, round(chunk / network.dt)))
    key = jax.random.key(0) if key is None else key
    monitors = tuple(monitors)
    drive = {name: np.asarray(value) for name, value in (drive or {}).items()}
    for name, value in drive.items():
        if value.ndim == 0 or value.shape[0] != steps:
            raise ValueError(f"drive {name!r} needs a leading axis of {steps} steps, got shape {value.shape}")
    fixed = {name: value for name, value in variables.items() if name != "state"}

    def run(fixed, state, drive, length):
        records, updates = network.apply({**fixed, "state": state}, drive, steps=length, monitors=monitors,
                                         rngs={"noise": key}, mutable=["state"])
        return records, updates["state"]

    run = jax.jit(run, static_argnames="length")
    state = variables["state"]
    chunks = []
    for start in range(0, steps, per_chunk):
        stop = min(start + per_chunk, steps)
        records, state = run(fixed, state, {name: value[start:stop] for name, value in drive.items()},
                             stop - start)
        over = {k: int(v) for k, v in jax.device_get(state["network"]["overflow"]).items() if int(v)}
        if over:
            raise RuntimeError(f"event projections exceeded their capacity in {over} steps (by projection) "
                               f"before {stop * network.dt} ms; raise Projection.capacity")
        chunks.append(jax.device_get(records))
    records = jax.tree.map(lambda *parts: np.concatenate(parts), *chunks) if chunks else ()
    return Simulation(tuple(records), {**fixed, "state": state}, network.dt)
