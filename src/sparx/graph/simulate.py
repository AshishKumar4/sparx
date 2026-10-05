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

It takes dew's pieces for placement and persistence: a `MeshSpec` builds
the device mesh as dew's `Trainer` builds it, a `Layout`'s rules map the
logical axes of the state (`neurons`, `trials`) onto it, and `Checkpoints`
keeps the state of a long run so it resumes after an interruption.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from dew.checkpoints import Checkpoints
from dew.nn.sharding import DATA_AXIS, FSDP_AXIS, LogicalAxes, LogicalAxisRules, logical_spec
from dew.objectives.base import Variables
from dew.training.distributed import Layout, MeshSpec
from dew.training.state import TrainState
from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding

from sparx.graph.network import Drive, Monitor, Network

__all__ = ["LAYOUT", "RULES", "Simulation", "simulate"]

RULES: LogicalAxisRules = (
    ("trials", DATA_AXIS),
    ("neurons", FSDP_AXIS),
    ("neurons", DATA_AXIS),
)
"""Where the logical axes of a simulation's state go on dew's mesh (design.md section 6.3).

Trials split over the data axis. Each population's neurons split over the
fsdp axis, or over the data axis when fsdp has one device and no trial axis
takes it, so `MeshSpec()` partitions one network's neurons over every
device and `MeshSpec(fsdp=n)` runs trials over the rest. An axis that does
not divide a dimension leaves it whole, as `dew.nn.sharding.logical_spec`
places any array."""

LAYOUT = Layout(rules=RULES)
"""The layout `simulate` places state with unless given another."""


@dataclass(frozen=True)
class Simulation:
    """What `simulate` returns: each monitor's records over the steps this call ran, on the host, and the
    final variables."""

    records: tuple[np.ndarray, ...]
    variables: Variables
    dt: float
    start: float = 0.0
    """When the records begin, in ms from the start of the run: later than 0 for a run resumed from a
    checkpoint."""

    @property
    def times(self) -> np.ndarray:
        """The end of each recorded step, in ms from the start of the run."""
        steps = len(self.records[0]) if self.records else 0
        return self.start + (np.arange(steps) + 1) * self.dt


def simulate(network: Network, variables: Variables, *, duration: float,
             key: jax.Array | None = None, drive: Drive | None = None,
             monitors: Sequence[Monitor] = (), chunk: float = 100.0, trials: int | None = None,
             mesh: MeshSpec | None = None, layout: Layout = LAYOUT,
             checkpoints: Checkpoints | None = None) -> Simulation:
    """Run `network` from `variables` for `duration` ms in compiled chunks of `chunk` ms.

    `drive` maps a `CurrentInput`'s name to its values over the whole run,
    `[steps, ...]` (`[trials, steps, ...]` with `trials`); `key` seeds
    Poisson inputs. The last chunk may be shorter, which compiles it once
    more.

    `trials` runs that many independent trials from the same variables, each
    with its own noise (`fold_in(key, trial)`): records and state gain a
    leading trial axis. `mesh` builds a device mesh as dew's `Trainer` does
    and spreads the run over it, the trials and the neurons, whose state
    each device keeps for its share of every population (the step's
    exchange of spikes the compiler partitions), as `layout`'s rules place
    the axes `trials` and `neurons` (`RULES`). Either gives one device's
    results.

    `checkpoints` writes the state after every chunk, and a run whose
    directory holds a checkpoint starts from it: its records begin there
    (`Simulation.start`), and an accumulating monitor sums from there. A
    run resumes with the same network, `variables` (its `connectome` and
    `params`, which are not saved), `key`, `dt` and `trials` as the run
    that wrote it, and is then the run it would have been without the
    break.
    """
    steps = round(duration / network.dt)
    per_chunk = max(1, min(steps, round(chunk / network.dt)))
    key = jax.random.key(0) if key is None else key
    monitors = tuple(monitors)
    time_axis = 0 if trials is None else 1
    drive = _checked(drive, steps, time_axis)
    fixed = {name: value for name, value in variables.items() if name != "state"}
    state = variables["state"]

    def run(fixed: Variables, state: Variables, drive: Mapping[str, jax.Array], key: jax.Array,
            length: int) -> tuple[tuple[jax.Array, ...], Variables]:
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
    place = _placement(network, mesh.build(), layout, trials) if mesh is not None else None
    if place is not None:
        state, keys = jax.device_put(state, place(state)), jax.device_put(keys, place(keys))
    done = 0
    if checkpoints is not None:
        done, state = _resumed(checkpoints, network, state, key, trials, steps, place)
    chunks = []
    for start in range(done, steps, per_chunk):
        stop = min(start + per_chunk, steps)
        window = (slice(None),) * time_axis + (slice(start, stop),)
        records, state = run(fixed, state, {name: value[window] for name, value in drive.items()}, keys,
                             stop - start)
        if place is not None:
            state = jax.device_put(state, place(state))
        _check_capacity(state, stop * network.dt)
        if checkpoints is not None:
            _save(checkpoints, stop, state, key, network.dt, trials)
        chunks.append(jax.device_get(records))
    if checkpoints is not None:
        checkpoints.wait()
    return Simulation(_join(monitors, chunks, time_axis), {**fixed, "state": state}, network.dt,
                      done * network.dt)


def _checked(drive: Drive | None, steps: int, time_axis: int) -> dict[str, np.ndarray]:
    checked = {name: np.asarray(value) for name, value in (drive or {}).items()}
    for name, value in checked.items():
        if value.ndim <= time_axis or value.shape[time_axis] != steps:
            raise ValueError(f"drive {name!r} needs a time axis of {steps} steps at {time_axis}, "
                             f"got shape {value.shape}")
    return checked


def _check_capacity(state: Variables, until: float) -> None:
    overflow = jax.device_get(state["network"]["overflow"])
    over = {name: int(np.max(count)) for name, count in overflow.items() if np.max(count)}
    if over:
        raise RuntimeError(f"event projections exceeded their capacity in {over} steps (by projection) "
                           f"before {until} ms; raise Projection.capacity")


def _join(monitors: tuple[Monitor, ...], chunks: list[tuple[np.ndarray, ...]],
          time_axis: int) -> tuple[np.ndarray, ...]:
    """Each monitor's records over the chunks: concatenated over time, or summed if it accumulates."""
    if not chunks:
        return ()

    def join(monitor: Monitor, parts: tuple[np.ndarray, ...]) -> np.ndarray:
        if not monitor.accumulate:
            return np.concatenate(parts, axis=time_axis)
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total

    return tuple(join(m, parts) for m, parts in zip(monitors, zip(*chunks, strict=True), strict=True))


type Placement = Callable[[Variables | jax.Array], Variables | jax.Array]


def _placement(network: Network, mesh: Mesh, layout: Layout, trials: int | None) -> Placement:
    """A tree's shardings on `mesh`: the trial axis leading every leaf with trials, and each population's
    neurons along a leaf's last axis.

    A leaf ends in a population's neurons when its last dimension is a
    population's size: membranes, synapses, spike buffers and the traces of
    plasticity rules. A plastic projection's weights are per edge, and stay
    whole whatever their count.
    """
    sizes = {p.size for p in network.populations}
    lead: LogicalAxes = () if trials is None else ("trials",)

    def sharding(path: tuple, leaf: jax.Array) -> NamedSharding:
        shape = jnp.shape(leaf)
        inner = shape[len(lead):]
        names = [getattr(entry, "key", None) for entry in path]
        per_edge = "plastic" in names and names[-1] == "weight"
        on_neurons = bool(inner) and inner[-1] in sizes and not per_edge
        axes = (*lead, *(None,) * (len(inner) - on_neurons), *(("neurons",) if on_neurons else ()))
        return NamedSharding(mesh, logical_spec(axes, shape, rules=layout.axis_rules, mesh=mesh))

    def place(tree: Variables | jax.Array) -> Variables | jax.Array:
        return jax.tree_util.tree_map_with_path(sharding, tree)

    return place


def _save(checkpoints: Checkpoints, step: int, state: Variables, key: jax.Array, dt: float,
          trials: int | None) -> None:
    """Write the state after `step` steps as a dew checkpoint.

    Dew's checkpoints hold a `TrainState`; a simulation's has its state as
    the variables, its step count as the step and its key, and no
    optimizer, average or loss scale.
    """
    zero = jnp.zeros((), jnp.int32)
    held = TrainState(step=jnp.asarray(step, jnp.int32), microstep=zero, updates=zero,
                      variables={"state": state}, opt_state=(), ema=None, key=key, scale=None,
                      window_size=jnp.ones((), jnp.int32))
    checkpoints.save(step, held, None, control={"dt": dt, "trials": trials})


def _resumed(checkpoints: Checkpoints, network: Network, state: Variables, key: jax.Array,
             trials: int | None, steps: int, place: Placement | None) -> tuple[int, Variables]:
    """The step the latest checkpoint is at and its state, placed as `place` does; 0 and `state` when the
    directory holds none."""
    latest = checkpoints.latest
    if latest is None:
        return 0, state
    written = checkpoints.control(latest)
    ran = {"dt": network.dt, "trials": trials}
    if written != ran:
        raise ValueError(f"the checkpoint at {checkpoints.path(latest)} was written by a run with {written}, "
                         f"and this run has {ran}")
    if latest > steps:
        raise ValueError(f"the checkpoint at {checkpoints.path(latest)} is {latest} steps into its run, "
                         f"past this run's {steps}")
    device = SingleDeviceSharding(jax.devices()[0])
    shardings = place(state) if place is not None else jax.tree.map(lambda _: device, state)
    template = {"variables": {"state": jax.tree.map(
                    lambda leaf, where: jax.ShapeDtypeStruct(jnp.shape(leaf), leaf.dtype, sharding=where),
                    state, shardings)},
                "key": jax.ShapeDtypeStruct(key.shape, key.dtype, sharding=device)}
    restored, _ = checkpoints.restore(template, latest)
    if not np.array_equal(jax.random.key_data(restored["key"]), jax.random.key_data(key)):
        raise ValueError(f"the checkpoint at {checkpoints.path(latest)} was written by a run with another "
                         f"key; resume it with the key it ran with")
    return latest, restored["variables"]["state"]
