"""Networks of populations joined by projections, stepped on one clock.

    network = Network(
        populations=(Population("e", 3200, LIF(...), {"ex": Receptor(Exponential(5.0), "conductance"), ...}),
                     Population("i", 800, ...)),
        projections=(Projection("e", "e", FixedProbability(0.02), weight=6.0, delay=0.1, receptor="ex"), ...),
        dt=0.1,
    )
    variables = network.init(key)
    outputs, updates = network.apply(variables, steps=10_000, monitors=(Spikes("e"),),
                                     rngs={"noise": key}, mutable=["state"])

A `Network` is a Flax module, so dew trains, shards and checkpoints it. Its
variables are split by role (design.md section 5.1):

| Collection | Holds |
| --- | --- |
| `connectome` | each projection's edges, delays, and its weights unless trainable |
| `params` | the weights of trainable projections |
| `state` | per population, neuron and synapse states and a ring buffer of recent spikes; per projection, |
|         | plasticity traces, plastic weights and short-term release; the step count |

One step covers `(t, t + dt]` and runs in NEST's order, which the
single-neuron tests pin against NEST and Brian2 (`sparx.dynamics.core`):

1. Delta synapses deliver the spikes due at the end of the step as
   voltage jumps.
2. Each population advances its membranes on its synapses' output and
   detects spikes.
3. The spikes enter each population's ring buffer.
4. Every other synapse receives the spikes due at the end of the step,
   which shape the membrane from the next step on.
5. Plasticity updates traces and weights.
6. Monitors record.

A spike sent in step `m` over a delay of `D` steps (`round(delay / dt)`)
is due at the end of step `m + D`: NEST's and Brian2's timing for a
delay of `D dt` (they stamp the spike differently, NEST at the end of its
step and Brian2 at the start, but deliver it alike). Kinetic synapses take
`D >= 0`, Brian2's default being 0; delta synapses need `D >= 1`, since a
jump due in the step that sent it would feed back into that step's
threshold test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from sparx.dynamics.plasticity import PairSTDP, TripletSTDP, TsodyksMarkram
from sparx.dynamics.synapses import PointNeuron, PointNeuronState, Receptor, jumps_first
from sparx.graph.connectivity import Connectivity, EdgeList

__all__ = ["ArrivalInput", "CurrentInput", "Monitor", "Network", "PoissonInput", "Population",
           "PopulationRate", "Projection", "Spikes", "StateMonitor"]

DENSE_LIMIT = 2 ** 25
DENSE_DENSITY = 0.02

PerEdge = float | np.ndarray | Callable[[np.random.Generator, int], np.ndarray]
"""A value for every edge: one for all, an array in the connectivity's order, or `f(rng, count)`."""


@dataclass(frozen=True)
class Population:
    """`size` neurons of one model, with the synapses onto them by receptor name.

    `initial(rng, state)` may replace the resting state each neuron starts
    from (random voltages, say); it gets a NumPy generator and the
    `PointNeuronState` at rest, with `[size]` leaves. `reset_synapses` and
    `freeze_synapses` are `PointNeuron`'s options.
    """

    name: str
    size: int
    neuron: Any
    receptors: Mapping[str, Receptor] = field(default_factory=dict)
    hold: Literal["mean", "start"] = "mean"
    initial: Callable[[np.random.Generator, PointNeuronState], PointNeuronState] | None = None
    reset_synapses: bool = False
    freeze_synapses: bool = False

    @property
    def cell(self) -> PointNeuron:
        return PointNeuron(self.neuron, self.receptors, hold=self.hold, reset_synapses=self.reset_synapses,
                           freeze_synapses=self.freeze_synapses)


@dataclass(frozen=True)
class Projection:
    """Synapses from population `pre` onto receptor `receptor` of population `post`.

    `weight` is in the receptor's unit (pA, nS or mV) and `delay` in ms,
    rounded to whole steps; either may be per edge. `plasticity` (pair or
    triplet STDP) makes the weights state that evolves with the spikes;
    `short_term` scales each spike by its presynaptic neuron's release.
    `trainable` puts fixed weights in `params` for gradient training.
    """

    pre: str
    post: str
    connectivity: Connectivity
    weight: PerEdge = 1.0
    delay: PerEdge = 1.0
    receptor: str = "ex"
    plasticity: PairSTDP | TripletSTDP | None = None
    short_term: TsodyksMarkram | None = None
    trainable: bool = False
    name: str | None = None
    format: Literal["auto", "edges", "dense", "events"] = "auto"
    """How the projection is stored and delivered: an edge list gathered and summed per postsynaptic neuron,
    or a dense `[pre, post]` matrix multiplied by the spikes. `"auto"` takes the matrix for a projection with
    one delay and fixed weights when it has at most `DENSE_LIMIT` entries and a density of at least
    `DENSE_DENSITY`, where a matrix product costs less than the gather on CPUs and accelerators.
    `"events"` visits only the edges of neurons that spiked, for large graphs with sparse activity
    (connectomes): its cost per step is the spiking neurons' out-degree, not the edge count."""
    capacity: tuple[int, int] = (1024, 262144)
    """For `format="events"`: the most presynaptic neurons spiking in one step, and the most edges they
    reach, that a step delivers. A step over capacity is counted, and `simulate` raises."""

    @property
    def key(self) -> str:
        return self.name or f"{self.pre}->{self.post}:{self.receptor}"


@dataclass(frozen=True)
class PoissonInput:
    """`count` independent Poisson sources of `rate` Hz onto each neuron of `target`, through `receptor`.

    Brunel's (2000) external drive: their sum is one Poisson process of
    `count * rate` Hz, sampled per neuron and step. `neurons` limits the
    input to those indices of `target`, as an optogenetic stimulus does.
    """

    target: str
    rate: float
    weight: float
    receptor: str = "ex"
    count: int = 1
    neurons: tuple[int, ...] | None = None


@dataclass(frozen=True)
class CurrentInput:
    """A current (pA) into `target`, from the `drive` passed to the network under `name` (`[T, size]` or
    `[T]` per step, or a constant)."""

    target: str
    name: str


@dataclass(frozen=True)
class ArrivalInput:
    """Weights arriving on `receptor` of `target` at the end of each step, from the `drive` passed under
    `name` (`[T, size]`): recorded spike trains replayed into a network, in the receptor's unit."""

    target: str
    name: str
    receptor: str = "ex"


class Monitor:
    """Something recorded every step: `record(spikes, states)` with both keyed by population."""

    def record(self, spikes: Mapping[str, jax.Array],
               states: Mapping[str, PointNeuronState]) -> Any:  # noqa: ANN401 - a monitor's record is any pytree
        raise NotImplementedError


@dataclass(frozen=True)
class Spikes(Monitor):
    """Which neurons of `population` fired, as booleans."""

    population: str

    def record(self, spikes, states):
        return spikes[self.population] > 0


@dataclass(frozen=True)
class PopulationRate(Monitor):
    """The fraction of `population` that fired in the step; divide by `dt` for a rate."""

    population: str

    def record(self, spikes, states):
        return jnp.mean(spikes[self.population])


@dataclass(frozen=True)
class StateMonitor(Monitor):
    """`read(state)` of `population` each step; `read` defaults to the membrane voltage."""

    population: str
    read: Callable[[PointNeuronState], jax.Array] = lambda state: state.neuron.v

    def record(self, spikes, states):
        return self.read(states[self.population])


def _per_edge(value: PerEdge, rng: np.random.Generator, edges: EdgeList, what: str) -> np.ndarray:
    if callable(value):
        out = np.asarray(value(rng, len(edges)), np.float64)
    elif np.ndim(value) == 0:
        return np.full(len(edges), float(value))  # type: ignore[arg-type]
    else:
        out = np.asarray(value, np.float64)
        if out.shape != (len(edges),):
            raise ValueError(f"{what} has shape {out.shape}, the projection has {len(edges)} edges")
        out = out[edges.order]
    if out.shape != (len(edges),):
        raise ValueError(f"{what} gave shape {out.shape} for {len(edges)} edges")
    return out


def _dense(p: Projection, delays: np.ndarray, edges: int, pre: int, post: int) -> bool:
    if p.format != "auto":
        if p.format == "dense" and (np.ndim(delays) or p.plasticity is not None or p.trainable):
            raise ValueError(f"{p.key}: a dense projection needs one delay and fixed weights")
        return p.format == "dense"
    small = pre * post <= DENSE_LIMIT and edges >= DENSE_DENSITY * pre * post
    return small and not np.ndim(delays) and p.plasticity is None and not p.trainable


def _poisson_table(mean: float) -> np.ndarray:
    """The Poisson CDF at 0, 1, ... until its tail is below 1e-16: a count is how many entries a uniform
    draw exceeds."""
    cdf, term, k = [], np.exp(-mean), 0
    total = term
    while 1 - total > 1e-16 and k < 10 * mean + 50:
        cdf.append(total)
        k += 1
        term *= mean / k
        total += term
    cdf.append(total)
    return np.asarray(cdf)


def _seed(key: jax.Array) -> np.random.Generator:
    return np.random.default_rng(np.asarray(jax.random.key_data(key)).ravel().astype(np.uint64))


class Network(nn.Module):
    """Populations and projections stepped together; see the module docstring."""

    populations: Sequence[Population]
    projections: Sequence[Projection] = ()
    inputs: Sequence[PoissonInput | CurrentInput | ArrivalInput] = ()
    dt: float = 0.1
    dtype: Any = jnp.float32
    """The dtype of the state: membranes, synapses, traces and spike buffers."""

    def _check(self) -> dict[str, Population]:
        populations = {p.name: p for p in self.populations}
        if len(populations) != len(self.populations):
            raise ValueError("population names must be unique")
        keys = [p.key for p in self.projections]
        if len(set(keys)) != len(keys):
            raise ValueError(f"projection names must be unique, got {keys}; name repeated ones")
        for p in self.projections:
            for side in (p.pre, p.post):
                if side not in populations:
                    raise ValueError(f"projection {p.key} names unknown population {side!r}")
            if p.receptor not in populations[p.post].receptors:
                raise ValueError(f"projection {p.key} targets receptor {p.receptor!r}, "
                                 f"which {p.post!r} lacks")
        for source in self.inputs:
            if source.target not in populations:
                raise ValueError(f"input onto unknown population {source.target!r}")
            receptor = getattr(source, "receptor", None)
            if receptor is not None and receptor not in populations[source.target].receptors:
                raise ValueError(f"input onto receptor {receptor!r}, which {source.target!r} lacks")
        return populations

    def _build(self, populations: Mapping[str, Population]) -> dict[str, dict[str, np.ndarray]]:
        """Draw every projection's edges, delays (whole steps) and weights from one key."""
        rng = _seed(self.make_rng("params"))
        built = {}
        for p in self.projections:
            pre, post = populations[p.pre], populations[p.post]
            edges = p.connectivity.edges(rng, pre.size, post.size, p.pre == p.post)
            delays = np.rint(_per_edge(p.delay, rng, edges, f"{p.key} delay") / self.dt).astype(np.int32)
            minimum = 1 if jumps_first(post.receptors[p.receptor].synapse) else 0
            if len(delays) and delays.min() < minimum:
                raise ValueError(f"{p.key}: delays must be at least {minimum} step(s) of {self.dt} ms "
                                 f"onto a {'delta' if minimum else 'kinetic'} synapse")
            if p.plasticity is not None and np.ndim(delays) and len(np.unique(delays)) > 1:
                raise ValueError(f"{p.key}: a plastic projection needs one delay for all its edges")
            if len(delays) and np.all(delays == delays[0]):
                delays = np.asarray(delays[0], np.int32)  # one delay: read one row of the ring per step
            weight = _per_edge(p.weight, rng, edges, f"{p.key} weight").astype(np.dtype(self.dtype))
            if p.format == "events":
                if np.ndim(delays) or p.plasticity is not None or p.trainable:
                    raise ValueError(f"{p.key}: an event projection needs one delay and fixed weights")
                order = np.lexsort((edges.post, edges.pre))
                counts = np.bincount(edges.pre, minlength=pre.size)
                built[p.key] = {"delay": np.asarray(np.ravel(delays)[0] if np.size(delays) else 0, np.int32),
                                "by_pre": edges.post[order],
                                "start": np.append(np.cumsum(counts) - counts, 0).astype(np.int32),
                                "count": np.append(counts, 0).astype(np.int32),
                                "weight": weight[order]}
            elif _dense(p, delays, len(edges), pre.size, post.size):
                matrix = np.zeros((pre.size, post.size), weight.dtype)
                np.add.at(matrix, (edges.pre, edges.post), weight)  # repeated pairs sum
                built[p.key] = {"delay": np.asarray(np.ravel(delays)[0] if np.size(delays) else 0, np.int32),
                                "weight": matrix}
            else:
                built[p.key] = {"pre": edges.pre, "post": edges.post, "delay": delays, "weight": weight}
        return built

    def _lags(self, edges) -> dict[str, int]:
        """How many steps of spikes each population's ring buffer keeps."""
        lags = {p.name: 1 for p in self.populations}
        for p in self.projections:
            if np.size(edges[p.key]["delay"]):
                longest = int(np.max(np.asarray(edges[p.key]["delay"]))) + 1
                lags[p.pre] = max(lags[p.pre], longest)
                if p.plasticity is not None:
                    lags[p.post] = max(lags[p.post], longest)
        return lags

    def _rest(self, populations: Mapping[str, Population], weights, lags) -> dict[str, Any]:
        """The state a run starts from: resting neurons (or `initial`), empty buffers, fresh traces."""
        rng = _seed(self.make_rng("params"))
        out = {}
        for name, pop in populations.items():
            cell = pop.cell.init_state((pop.size,), self.dtype)
            if pop.initial is not None:
                cell = pop.initial(rng, cell)
            out[name] = {"cell": cell, "buffer": jnp.zeros((lags[name], pop.size), self.dtype)}
        plastic = {}
        for p in self.projections:
            pre, post = populations[p.pre].size, populations[p.post].size
            entry = {}
            if p.plasticity is not None:
                entry["traces"] = p.plasticity.init_traces(pre, post, self.dtype)
                entry["weight"] = weights[p.key]
            if p.short_term is not None:
                entry["release"] = p.short_term.rest((pre,), self.dtype)
                entry["buffer"] = jnp.zeros((lags[p.pre], pre), self.dtype)
            if entry:
                plastic[p.key] = entry
        overflow = {p.key: jnp.zeros((), jnp.int32) for p in self.projections if p.format == "events"}
        return {"populations": out, "projections": plastic, "overflow": overflow,
                "t": jnp.zeros((), jnp.int32)}

    @nn.compact
    def __call__(self, drive: Mapping[str, Any] | None = None, *, steps: int | None = None,
                 monitors: Sequence[Monitor] = ()) -> tuple[Any, ...]:
        """Advance `steps` steps (or as many as `drive` has); returns each monitor's records over time."""
        populations = self._check()
        built: dict[str, dict[str, np.ndarray]] = {}

        def build():
            if not built:  # edges and their weights come from one draw
                built.update(self._build(populations))
            return built

        def structure():
            return {k: {n: jnp.asarray(a) for n, a in v.items() if n != "weight"} for k, v in build().items()}

        edges = self.variable("connectome", "edges", structure).value
        weights = {}
        for p in self.projections:
            if p.trainable:
                weights[p.key] = self.param(f"weight:{p.key}",
                                            lambda _, k=p.key: jnp.asarray(build()[k]["weight"]))
            else:
                weights[p.key] = self.variable("connectome", f"weight:{p.key}",
                                               lambda k=p.key: jnp.asarray(build()[k]["weight"])).value
        state = self.variable("state", "network", lambda: self._rest(populations, weights, self._lags(edges)))
        if self.is_initializing():
            return ()
        drive = dict(drive or {})
        if steps is None:
            timed = [np.shape(v)[0] for v in jax.tree.leaves(drive) if np.ndim(v) > 0]
            if not timed:
                raise ValueError("pass `steps` or a time-major `drive`")
            steps = timed[0]
        key = self.make_rng("noise") if any(isinstance(s, PoissonInput) for s in self.inputs) else None
        stepper = _Stepper(self, populations, edges, weights, tuple(monitors), key)
        held = {name: jnp.broadcast_to(jnp.asarray(value, self.dtype), (steps, *np.shape(value)[1:]))
                if np.ndim(value) > 0 else jnp.full((steps,), value, self.dtype)
                for name, value in drive.items()}
        state.value, records = jax.lax.scan(stepper, state.value, held, length=steps)
        return records


class _Stepper:
    """A network's step, closed over its structure and fixed variables, in the documented order."""

    def __init__(self, network: Network, populations: Mapping[str, Population], edges, weights, monitors,
                 key):
        self.network, self.populations, self.edges, self.weights = network, populations, edges, weights
        self.monitors, self.key, self.dt = monitors, key, network.dt
        self.into: dict[str, list[Projection]] = {name: [] for name in populations}
        for p in network.projections:
            self.into[p.post].append(p)

    def is_delta(self, population: str, receptor: str) -> bool:
        return jumps_first(self.populations[population].receptors[receptor].synapse)

    def deliver(self, t, p: Projection, weight, ring) -> jax.Array:
        """Weighted spikes of `p` due at the end of step `t`, summed per postsynaptic neuron."""
        e = self.edges[p.key]
        if "by_pre" in e:
            return self.events(p, e, weight, ring[(t - e["delay"]) % ring.shape[0]])
        if "pre" not in e:
            return ring[(t - e["delay"]) % ring.shape[0]] @ weight
        if e["delay"].ndim == 0:
            sent = ring[(t - e["delay"]) % ring.shape[0]][e["pre"]]
        else:
            sent = ring[(t - e["delay"]) % ring.shape[0], e["pre"]]
        return jax.ops.segment_sum(weight * sent, e["post"], num_segments=self.populations[p.post].size,
                                   indices_are_sorted=True)

    def events(self, p: Projection, e, weight, sent) -> jax.Array:
        """Delivery that visits only the edges of neurons that spiked, up to `p.capacity`.

        The spiking neurons' out-edges, contiguous when edges are sorted by
        presynaptic neuron, are laid end to end into a fixed number of slots
        by a prefix sum of their out-degrees; each slot finds its neuron by
        binary search. Over capacity, the step is counted in the state.
        """
        neurons, slots = p.capacity
        size = sent.shape[0]
        active = jnp.nonzero(sent, size=neurons, fill_value=size)[0]
        degree = e["count"][active]
        ends = jnp.cumsum(degree)
        slot = jnp.arange(slots)
        owner = jnp.minimum(jnp.searchsorted(ends, slot, side="right"), neurons - 1)
        edge = e["start"][active[owner]] + slot - (ends[owner] - degree[owner])
        valid = slot < ends[-1]
        edge = jnp.where(valid, edge, 0)
        value = jnp.where(valid, weight[edge] * sent.at[active[owner]].get(mode="fill", fill_value=0), 0)
        self.overflowed[p.key] = (jnp.sum(sent != 0) > neurons) | (ends[-1] > slots)
        return jax.ops.segment_sum(value, e["by_pre"][edge], num_segments=self.populations[p.post].size)

    def external(self, t, drive_t):
        """This step's Poisson arrivals by population and receptor, and injected currents by population."""
        arrivals: dict[str, dict[str, jax.Array]] = {name: {} for name in self.populations}
        currents = {name: jnp.zeros((), self.network.dtype) for name in self.populations}
        for i, source in enumerate(self.network.inputs):
            if isinstance(source, CurrentInput):
                currents[source.target] = currents[source.target] + drive_t[source.name]
                continue
            if isinstance(source, ArrivalInput):
                incoming = arrivals[source.target].get(source.receptor, 0.0)
                arrivals[source.target][source.receptor] = incoming + drive_t[source.name]
                continue
            assert self.key is not None
            # A static mean: invert its CDF, exact to 1e-16 and vectorized, where
            # jax.random.poisson loops per draw.
            table = jnp.asarray(_poisson_table(source.count * source.rate * self.dt / 1000.0), jnp.float32)
            key = jax.random.fold_in(jax.random.fold_in(self.key, t), i)
            size = self.populations[source.target].size
            if source.neurons is None:
                draws = jnp.sum(jax.random.uniform(key, (size, 1)) > table, axis=1)
            else:
                chosen = jnp.asarray(source.neurons, jnp.int32)
                counts = jnp.sum(jax.random.uniform(key, (len(source.neurons), 1)) > table, axis=1)
                draws = jnp.zeros(size, counts.dtype).at[chosen].add(counts)
            incoming = arrivals[source.target].get(source.receptor, 0.0)
            draws = draws.astype(self.network.dtype)
            arrivals[source.target][source.receptor] = incoming + source.weight * draws
        return arrivals, currents

    def gather(self, name, t, external, rings, weights, delta: bool) -> dict[str, jax.Array]:
        """Everything due at the end of step `t` on `name`'s delta (or kinetic) receptors."""
        arrivals = {k: v for k, v in external.items() if self.is_delta(name, k) == delta}
        for p in self.into[name]:
            if self.is_delta(name, p.receptor) == delta:
                due = self.deliver(t, p, weights[p.key], rings[p.key])
                arrivals[p.receptor] = arrivals.get(p.receptor, 0.0) + due
        return arrivals

    def plasticity(self, t, plastic, fired, buffers):
        """STDP: presynaptic spikes as sent, postsynaptic ones after the projection's (dendritic) delay."""
        for p in self.network.projections:
            if p.plasticity is None:
                continue
            e = self.edges[p.key]
            delay = e["delay"] if e["delay"].ndim == 0 else 0  # plastic projections have one delay
            ring = buffers[p.post]
            arrived = ring[(t - delay) % ring.shape[0]]
            traces, weight = p.plasticity.step(plastic[p.key]["traces"], plastic[p.key]["weight"],
                                               fired[p.pre], arrived, e["pre"], e["post"], self.dt)
            plastic[p.key] = {**plastic[p.key], "traces": traces, "weight": weight}

    def __call__(self, state, drive_t):
        t, pops, plastic = state["t"], state["populations"], dict(state["projections"])
        self.overflowed: dict[str, jax.Array] = {}
        projections = self.network.projections
        weights = {p.key: plastic[p.key]["weight"] if p.plasticity is not None else self.weights[p.key]
                   for p in projections}
        external, currents = self.external(t, drive_t)

        def rings(buffers):
            return {p.key: plastic[p.key]["buffer"] if p.short_term is not None else buffers[p.pre]
                    for p in projections}

        # 1-2. Delta jumps due now, then the membranes.
        cells, fired = {}, {}
        frozen = {name: pop.cell.frozen(pops[name]["cell"], self.dt)
                  for name, pop in self.populations.items()}
        before = rings({name: pops[name]["buffer"] for name in self.populations})
        for name, pop in self.populations.items():
            jumps = pop.cell.delta(self.gather(name, t, external[name], before, weights, delta=True))
            cells[name], spikes = pop.cell.advance(pops[name]["cell"], currents[name], jumps, self.dt)
            fired[name] = spikes.fired

        # 3. Spikes into the ring buffers; short-term release scales them as they are sent.
        buffers = {}
        for name in self.populations:
            ring = pops[name]["buffer"]
            buffers[name] = ring.at[t % ring.shape[0]].set(fired[name].astype(ring.dtype))
        for p in projections:
            if p.short_term is not None:
                release, efficacy = p.short_term.step(plastic[p.key]["release"], fired[p.pre], self.dt)
                ring = plastic[p.key]["buffer"]
                plastic[p.key] = {**plastic[p.key], "release": release,
                                  "buffer": ring.at[t % ring.shape[0]].set(efficacy.astype(ring.dtype))}

        # 4. Kinetic synapses receive what is due at the end of the step.
        after = rings(buffers)
        new_pops = {}
        for name, pop in self.populations.items():
            due = self.gather(name, t, external[name], after, weights, delta=False)
            cell = pop.cell.receive(cells[name], due, self.dt, fired[name], frozen[name])
            new_pops[name] = {"cell": cell, "buffer": buffers[name]}

        # 5-6. Plasticity, then monitors.
        self.plasticity(t, plastic, fired, buffers)
        records = tuple(m.record(fired, {n: v["cell"] for n, v in new_pops.items()}) for m in self.monitors)
        overflow = {k: v + self.overflowed.get(k, jnp.zeros((), bool)).astype(jnp.int32)
                    for k, v in state["overflow"].items()}
        return {"populations": new_pops, "projections": plastic, "overflow": overflow, "t": t + 1}, records
