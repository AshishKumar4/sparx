"""Networks of populations joined by projections, stepped on one clock.

    receptors = {"ampa": Receptor(Exponential(5.0), "conductance"),
                 "gaba_a": Receptor(Exponential(10.0), "conductance")}
    network = Network(
        populations=(Population("e", 3200, LIF(...), receptors), Population("i", 800, LIF(...), receptors)),
        projections=(Projection("e", "e", FixedProbability(0.02), weight=6.0, delay=0.1, receptor="ampa"),
                     ...),
        dt=0.1,
    )
    variables = network.init(key)
    records, updates = network.apply(variables, steps=10_000, monitors={"spikes": SpikeRaster("e")},
                                     rngs={"noise": key}, mutable=["state"])
    records["spikes"]                       # [steps, 3200] booleans

A `Network` is a Flax module, so dew trains, shards and checkpoints it. Its
variables are split by role (design.md section 5.1):

| Collection | Holds |
| --- | --- |
| `connectome` | each projection's edges, delays, and its weights unless trainable |
| `params` | the weights of trainable projections |
| `state` | per population, neuron and synapse states and a ring buffer of recent outputs; per plastic |
|         | projection, traces and weights; per depressing projection, release; per receptor fed by |
|         | events over edges of their own delays, what is on its way; each modulator's concentration; |
|         | the step count |

`Network.connections(variables)` reads every projection's edges, weights
and delays back out of them, whichever collection holds the weights.

The network is checked when it is constructed. Every receptor a projection
or input names exists on its target, every conductance receptor has a
reversal potential in its neuron model, and each neuron model accepts what
its receptors and inputs deliver, so a dimensionless model given a kinetic
synapse is refused before any step runs.

A population sends what its neuron model outputs (`sparx.dynamics.Output`):
spikes, or a graded value every step for a graded model (a
`GradedPotential`'s release, a `RateCell`'s activity). A projection from a
graded population delivers the weighted values through dense or edge
delivery onto a `Graded` synapse, or onto a `Delta` receptor of a
dimensionless model as its jump; event delivery, which skips the neurons
that did not fire, is for spikes only. Gap junctions (`GapJunction`) couple
the membranes of two populations, and modulators (`Modulator`) turn a
population's spikes into a concentration that plasticity reads.

One step covers `(t, t + dt]` and runs in NEST's order, which the
single-neuron tests pin against NEST and Brian2 (`sparx.dynamics.core`):

1. Delta synapses deliver the spikes due at the end of the step as
   voltage jumps.
2. Each population advances its membranes on its synapses' output and its
   gap junctions (`GapJunction`), and emits its output (spikes, or graded
   values).
3. The outputs enter each population's ring buffer, and each modulator
   takes up the spikes of its source. An event projection whose edges
   have their own delays sends its spikes now: each edge adds its weight to
   what its target's receptor receives at the end of step `t + delay`, as
   NEST queues a spike in each target's ring buffer.
4. Every other synapse receives what is due at the end of the step, which
   shapes the membrane from the next step on.
5. Plasticity updates traces and weights, reading the modulators.
6. Monitors record.

A spike sent in step `m` over a delay of `D` steps (`delay / dt`, which
must be whole) is due at the end of step `m + D`: NEST's and Brian2's
timing for a delay of `D dt` (they stamp the spike differently, NEST at the
end of its step and Brian2 at the start, but deliver it alike). Kinetic
synapses take `D >= 0`, Brian2's default being 0; delta synapses need
`D >= 1`, since a jump due in the step that sent it would feed back into
that step's threshold test.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Protocol, TypedDict, runtime_checkable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from dew.objectives.base import Variables
from jax.custom_derivatives import SymbolicZero

from sparx.dynamics.core import Gap, NeuronModel, Output, Term
from sparx.dynamics.plasticity import Plasticity, TsodyksMarkram, TsodyksMarkramState
from sparx.dynamics.synapses import Graded, PointNeuron, PointNeuronState, Receptor, StochasticRelease
from sparx.graph.connectivity import Connectivity, EdgeList

__all__ = ["ArrivalInput", "Connections", "CurrentInput", "Drive", "GapJunction", "Modulator",
           "ModulatorTrace", "Monitor", "Network", "NetworkState", "OutputTrace", "PerEdge", "PerNeuron",
           "PlasticState", "PoissonInput", "Population", "PopulationRate", "PopulationState", "Projection",
           "ReleaseState", "Reversing", "SpikeCounts", "SpikeRaster", "SpikeTimes", "StateMonitor",
           "whole_steps"]

DENSE_LIMIT = 2 ** 25
EVENT_BLOCK = 4096
EVENT_PADDING = 2.0
"""An event projection whose out-degrees are this close to even, the widest at most this many times the
mean, keeps each neuron's out-edges as one padded row, read without a search; so does one whose padded
rows hold at most `EVENT_ROWS` entries, whatever the spread."""
EVENT_ROWS = 2 ** 20
DENSE_DENSITY = 0.02
GRID_TOLERANCE = 1e-3
"""How far from a whole number of steps, in steps, a time may be and still count as on the grid: wide
enough for float32 delays of a few hundred ms, and far narrower than any time meant to fall between
steps."""

type PerEdge = float | np.ndarray | Callable[[np.random.Generator, int], np.ndarray]
"""A value for every edge: one for all, an array in the connectivity's order, or `f(rng, count)`."""

type PerNeuron = float | np.ndarray | Callable[[np.random.Generator, int], np.ndarray]
"""A value for every neuron of a population: one for all, an array `[size]`, or `f(rng, size)`."""

type Drive = Mapping[str, jax.Array | np.ndarray | float]
"""External input by the name of the `CurrentInput` or `ArrivalInput` that reads it: `[T, ...]` per step,
or a constant."""


def whole_steps(time: float | np.ndarray, dt: float, what: str) -> np.ndarray:
    """`time` (ms) as a whole number of steps of `dt`; raises if it falls between steps.

    Rounding a time onto the grid moves it by up to half a step, and which
    way a time near half a step goes depends on floating-point error in
    `time / dt` (0.15 ms at a 0.1 ms step is 1 step or 2). Delays decide
    when spikes land and durations how many steps run, so a time off the
    grid is refused and the caller picks the step it meant.
    """
    exact = np.asarray(time, np.float64) / dt
    steps = np.rint(exact)
    off = np.abs(exact - steps) > GRID_TOLERANCE
    if np.any(off):
        shown = np.unique(np.asarray(time)[off])[:3].tolist()
        raise ValueError(f"{what} must be a whole number of steps of {dt} ms; {shown} is not. Round it to "
                         f"the grid, as `np.round(x / dt) * dt` does")
    return steps.astype(np.int64)


@runtime_checkable
class Reversing(Protocol):
    """A neuron model that reads conductances, against the reversal potential (mV) of each receptor by
    name: the physical models of `sparx.dynamics.neurons`."""

    @property
    def reversal(self) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class Population:
    """`size` neurons of one model, with the synapses onto them by receptor name.

    `neuron` is any neuron model of `sparx.dynamics`, physical or
    dimensionless; a dimensionless one (`ALIFCell`, say) takes its input
    through delta receptors, as voltage jumps. A conductance receptor is
    named for a reversal potential of the neuron model (`LIF`'s defaults are
    `ampa`, `nmda`, `gaba_a` and `gaba_b`).

    `initial` sets where each neuron starts, by name: a field of the neuron
    model's state (`"v"`) or a receptor, whose synapse state it sets, each
    to one value for all, an array `[size]` or `f(rng, size)` drawn from a
    NumPy generator seeded by the network's key. Everything else starts at
    rest. `reset_synapses` and `freeze_synapses` are `PointNeuron`'s options.
    """

    name: str
    size: int
    neuron: NeuronModel
    receptors: Mapping[str, Receptor] = field(default_factory=dict)
    hold: Literal["mean", "start"] = "mean"
    initial: Mapping[str, PerNeuron] = field(default_factory=dict)
    reset_synapses: bool = False
    freeze_synapses: bool = False

    @property
    def point_neuron(self) -> PointNeuron:
        """The population's neuron model and its synapses, stepped together."""
        return PointNeuron(self.neuron, self.receptors, hold=self.hold, reset_synapses=self.reset_synapses,
                           freeze_synapses=self.freeze_synapses)

    @property
    def graded(self) -> bool:
        """Whether the population sends a graded value every step, as its neuron model says."""
        return self.neuron.graded


@dataclass(frozen=True)
class Projection:
    """Synapses from population `pre` onto receptor `receptor` of population `post`.

    `receptor` has no default: it sets the unit of `weight` (pA, nS or mV)
    and, through the reversal potential, the sign of a conductance, so a
    weight means nothing without it. `delay` is in ms and must be a whole
    number of steps (`whole_steps`); either may be per edge. `plasticity`
    (a `Plasticity` rule, pair or triplet STDP) makes the weights state that
    evolves with the spikes; `short_term` scales each spike by its
    presynaptic neuron's release; `release` makes each edge transmit a spike
    only with a probability, drawn from the network's `noise` key.
    `trainable` puts fixed weights in `params` for gradient training. A
    projection from a graded population transmits `weight` times the
    presynaptic value every step, and takes none of the spike-driven
    options.
    """

    pre: str
    post: str
    connectivity: Connectivity
    weight: PerEdge = 1.0
    delay: PerEdge = 1.0
    receptor: str = field(kw_only=True)
    plasticity: Plasticity | None = None
    short_term: TsodyksMarkram | None = None
    release: StochasticRelease | None = None
    trainable: bool = False
    name: str | None = None
    format: Literal["auto", "edges", "dense", "events"] = "auto"
    """How the projection is stored and delivered. `"events"` visits only the edges of neurons that spiked,
    so a step costs the spiking neurons' out-degree, not the edge count; `"edges"` gathers and sums an edge
    list per postsynaptic neuron; `"dense"` multiplies the spikes by a `[pre, post]` matrix. `"auto"` takes
    events whenever it can, from a spiking population with fixed weights. Otherwise it takes
    the matrix when it has at most `DENSE_LIMIT` entries and a density of at least `DENSE_DENSITY`, where a
    matrix product costs less than the gather, and the edge list beyond. A graded population has no silent
    neurons to skip, so its projections are edges or dense; one with stochastic release draws per edge, so
    it is edges or events. On Brunel's network at 12,500 neurons, events took 11.5 s per simulated second
    and edges 457 s (`benchmarks/bench_networks.py`, a 4-core CPU)."""
    per_pass: int = 16
    """For `format="events"`: how many spiking neurons one pass of a step delivers; a step makes as many
    passes as its spikes need. Each pass finds them with `jax.lax.top_k` over the population. On a 4-core
    CPU, 16 was fastest for CUBA and COBA (2 and 7 spikes a step) and 32 for Brunel's 12,500 neurons (46),
    10% either way (`benchmarks/bench_networks.py`)."""

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
    receptor: str
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
    receptor: str


@dataclass(frozen=True)
class GapJunction:
    """Electrical synapses between populations `a` and `b`, each passing `I = g (v_partner - v)`.

    Each edge of `connectivity` (from a neuron of `a` to one of `b`) is one
    junction of conductance `weight` (nS), symmetric: the current into the
    neuron of `a` is `g (v_b - v_a)`, and the current into the neuron of `b`
    is its negative. `a` and `b` may be one population. Both need a membrane
    voltage `v` in mV, as the physical models have.

    The coupling has no delay, and each step solves it in two passes. Every
    coupled population first advances with its partners' voltages held at
    their values at the start of the step, which predicts their voltages
    at its end. It then advances again from the start, with each partner's
    voltage moving linearly from its start to its predicted end. A neuron
    integrates the coupling against its own voltage as a conductance, and
    its partners' side as a current waveform (`sparx.dynamics.Gap`), which
    a linear membrane solves exactly and stably. The result is second order
    in `dt`. This is one iteration of the waveform relaxation NEST uses
    (Hahne et al., Frontiers in Neuroinformatics 2015), which iterates
    until the interpolated voltages agree to a tolerance and so converges
    to the coupled solution; without it NEST holds each partner at its
    voltage of the step before. `tests/test_signalling.py` compares sparx
    with the analytic solution of two coupled passive cells and with NEST.
    Coupled populations advance twice per step; the others once.
    """

    a: str
    b: str
    connectivity: Connectivity
    weight: PerEdge = 1.0
    name: str | None = None

    @property
    def key(self) -> str:
        return self.name or f"{self.a}<->{self.b}"


@dataclass(frozen=True)
class Modulator:
    """A neuromodulator that the spikes of `source` release and volume transmission spreads, one
    concentration for the whole network.

        dc/dt = -c / tau + release * sum_k delta(t - t_k)

    over the spikes `t_k` of `source`'s neurons, or of the neurons
    `neurons` (indices) when given, such as the dopaminergic neurons of a
    connectome. Each spike adds `release` to the concentration, which decays
    with `tau` (ms) between spikes, exactly on the step grid. The
    concentration is in the network's state (`NetworkState["modulators"]`,
    by `name`), and each `Plasticity` rule reads it every step as a third
    factor.
    """

    name: str
    source: str
    tau: float
    release: float = 1.0
    neurons: tuple[int, ...] | None = None


class Monitor:
    """Something recorded every step: `record(outputs, states, dt, modulators)` with outputs and states
    keyed by population, `dt` the step in ms, and the modulators' concentrations by name.

    A record is one array per step, stacked over the run into `[T, ...]`.
    A monitor with `accumulate = True` is summed over the steps of a run
    instead of stacked, so its memory does not grow with the run. Monitors
    are passed by name (`monitors={"rate": PopulationRate("e")}`), and the
    records come back under the same names.
    """

    accumulate: bool = False

    def record(self, outputs: Mapping[str, jax.Array], states: Mapping[str, PointNeuronState], dt: float,
               modulators: Mapping[str, jax.Array]) -> jax.Array:
        raise NotImplementedError


@dataclass(frozen=True)
class SpikeRaster(Monitor):
    """Which neurons of `population` fired, as booleans."""

    population: str

    def record(self, outputs, states, dt, modulators):
        return outputs[self.population] > 0


@dataclass(frozen=True)
class SpikeCounts(Monitor):
    """Each neuron's spike count over the run, summed as it goes: rates of large populations."""

    population: str
    accumulate = True

    def record(self, outputs, states, dt, modulators):
        return (outputs[self.population] > 0).astype(jnp.int32)


@dataclass(frozen=True)
class SpikeTimes(Monitor):
    """The indices of up to `capacity` neurons of `population` that fired each step, padded with -1:
    a raster of a large population at the cost of `capacity` integers per step."""

    population: str
    capacity: int = 64

    def record(self, outputs, states, dt, modulators):
        fired = outputs[self.population] > 0
        return jnp.nonzero(fired, size=self.capacity, fill_value=-1)[0].astype(jnp.int32)


@dataclass(frozen=True)
class PopulationRate(Monitor):
    """The rate of `population` in each step, in Hz: the fraction of its neurons that fired in the step,
    over the step's length in seconds.

    Hz is the unit of `PoissonInput.rate` and `sparx.spiketrains.rates_hz`,
    so a rate read here compares with them without conversion.
    """

    population: str

    def record(self, outputs, states, dt, modulators):
        return jnp.mean(outputs[self.population]) * (1000.0 / dt)


@dataclass(frozen=True)
class OutputTrace(Monitor):
    """What `population` sends each step, for the neurons `neurons` (indices) or all of them: a graded
    population's values (a release, an activity), or a spiking one's spikes as 0 and 1."""

    population: str
    neurons: tuple[int, ...] | None = None

    def record(self, outputs, states, dt, modulators):
        values = outputs[self.population]
        return values if self.neurons is None else values[np.asarray(self.neurons)]


@dataclass(frozen=True)
class ModulatorTrace(Monitor):
    """The concentration of the modulator named `modulator` after each step."""

    modulator: str

    def record(self, outputs, states, dt, modulators):
        return modulators[self.modulator]


_SPIKE_MONITORS = (SpikeRaster, SpikeCounts, SpikeTimes, PopulationRate)
"""The monitors that read a population's output as spikes, and so refuse a graded population."""


@runtime_checkable
class _Membrane(Protocol):
    """A neuron state with a membrane voltage, as the state of each single sparx neuron model has."""

    @property
    def v(self) -> jax.Array: ...


@runtime_checkable
class _Wrapper(Protocol):
    """A neuron state that holds another model's, as a `RecurrentCell`'s does."""

    @property
    def inner(self) -> _NeuronState: ...


type _NeuronState = _Membrane | _Wrapper | tuple[_NeuronState, ...]
"""What a neuron state holds its voltage in: itself, the model it wraps, or the last of several in series."""


@runtime_checkable
class _Record(Protocol):
    """A neuron state with named fields (a NamedTuple), which `Population.initial` sets by name."""

    @property
    def _fields(self) -> tuple[str, ...]: ...


def _membrane(state: PointNeuronState) -> jax.Array:
    """The voltage `v` of a population's neuron model; for models in series (`Serial`), the last one's,
    whose threshold fires the population, and for a `RecurrentCell`, its inner model's."""
    neuron: object = state.neuron
    while not isinstance(neuron, _Membrane):
        if isinstance(neuron, _Wrapper):
            neuron = neuron.inner
        elif isinstance(neuron, tuple) and neuron:
            neuron = neuron[-1]
        else:
            raise TypeError(f"{type(neuron).__name__} has no voltage `v`; give StateMonitor a `read`")
    return neuron.v


@dataclass(frozen=True)
class StateMonitor(Monitor):
    """`read(state)` of `population` each step, for the neurons `neurons` (indices) or all of them.

    `read` takes the population's `PointNeuronState` and returns one value
    per neuron; it defaults to the membrane voltage. A voltage trace of
    every neuron costs `steps * size` values, 32 MB for 800 neurons over
    1 s at 0.1 ms, so `neurons` keeps the few a figure shows.
    """

    population: str
    read: Callable[[PointNeuronState], jax.Array] = _membrane
    neurons: tuple[int, ...] | None = None

    def record(self, outputs, states, dt, modulators):
        values = self.read(states[self.population])
        return values if self.neurons is None else values[np.asarray(self.neurons)]


class Connections(NamedTuple):
    """A projection's synapses, one entry per edge, sorted by presynaptic then postsynaptic neuron.

    `weight` is in the receptor's unit, with a leading trial axis for a
    plastic projection simulated over trials; `delay` is in ms.
    """

    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    delay: np.ndarray


def _drawn(value: PerEdge, rng: np.random.Generator, count: int, what: str) -> np.ndarray:
    """`count` values: one repeated, an array of that length as given, or `value(rng, count)`."""
    if callable(value):
        out = np.asarray(value(rng, count), np.float64)
    elif np.ndim(value) == 0:
        out = np.full(count, float(value))
    else:
        out = np.asarray(value, np.float64)
    if out.shape != (count,):
        raise ValueError(f"{what} has shape {out.shape}, and needs {count} values")
    return out


def _per_edge(value: PerEdge, rng: np.random.Generator, edges: EdgeList, what: str) -> np.ndarray:
    out = _drawn(value, rng, len(edges), what)
    # A given array follows the connectivity's order; edges are stored in another.
    return out if callable(value) or np.ndim(value) == 0 else out[edges.order]


def _initial(population: Population, state: PointNeuronState, rng: np.random.Generator) -> PointNeuronState:
    """`state` at rest with the fields `population.initial` names set, drawn in the order it names them."""
    neuron, synapses = state.neuron, dict(state.synapses)
    for name, value in population.initial.items():
        drawn = _drawn(value, rng, population.size, f"population {population.name!r} initial {name!r}")
        if name in population.receptors:
            synapses[name] = jnp.asarray(drawn, jnp.result_type(synapses[name]))
        else:
            dtype = jnp.result_type(getattr(neuron, name))
            neuron = neuron._replace(**{name: jnp.asarray(drawn, dtype)})
    return PointNeuronState(neuron, synapses)


def _events(p: Projection, pre: Population) -> bool:
    """Whether `p` delivers by events: as asked, or under `"auto"` whenever it can, from a spiking
    population with fixed weights, since its cost then follows the spikes, not the edges."""
    fixed = p.plasticity is None and not p.trainable
    if p.format == "events" and not fixed:
        raise ValueError(f"{p.key}: an event projection needs fixed weights")
    return p.format == "events" or (p.format == "auto" and fixed and not pre.graded)


def _ahead(e: Mapping[str, np.ndarray | jax.Array]) -> bool:
    """Whether a projection's built edges deliver by events over delays of their own, sending each spike's
    weights ahead to the steps they are due in."""
    return "by_pre" in e and np.ndim(e["delay"]) == 1


def _dense(p: Projection, delays: np.ndarray, edges: int, pre: int, post: int) -> bool:
    if p.format != "auto":
        if p.format == "dense" and (np.ndim(delays) or p.plasticity is not None or p.trainable):
            raise ValueError(f"{p.key}: a dense projection needs one delay and fixed weights")
        if p.format == "dense" and p.release is not None:
            raise ValueError(f"{p.key}: stochastic release draws per edge, so its projection is edges or "
                             f"events")
        return p.format == "dense"
    small = pre * post <= DENSE_LIMIT and edges >= DENSE_DENSITY * pre * post
    return small and not np.ndim(delays) and p.plasticity is None and not p.trainable and p.release is None


def _landing(e: Mapping[str, jax.Array], edge: jax.Array, out: jax.Array, t: jax.Array | None
             ) -> tuple[jax.Array, ...]:
    """Where in `out` each event edge's value lands: its postsynaptic neuron, or with `t`, that neuron in the
    row its delay reaches from step `t`. An index past the last edge lands past the end, to be dropped."""
    post = e["by_pre"].at[edge].get(mode="fill", fill_value=out.shape[-1])
    if t is None:
        return (post,)
    lag = e["delay"].at[edge].get(mode="fill", fill_value=0)
    return ((t + lag) % out.shape[0], post)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _as_edges(deliver: Callable[..., jax.Array], weight: jax.Array, sent: jax.Array, out: jax.Array,
              e: Mapping[str, jax.Array], t: jax.Array | None) -> jax.Array:
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


def _event_edges(edges: EdgeList, weight: np.ndarray, delays: np.ndarray, pre: int) -> dict[str, np.ndarray]:
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
        built["rows"] = _rows(counts, len(edges))
    return built


def _pending(projections: Sequence[Projection], edges: Mapping[str, Mapping[str, np.ndarray | jax.Array]]
             ) -> dict[str, tuple[str, int]]:
    """Each receptor that event projections over delays of their own feed, `population:receptor`, with its
    population and how many steps of arrivals it holds: its longest delay and the step itself."""
    out: dict[str, tuple[str, int]] = {}
    for p in projections:
        if _ahead(edges[p.key]):
            rows = int(np.max(np.asarray(edges[p.key]["delay"]), initial=0)) + 1
            out[_target(p)] = (p.post, max(rows, out.get(_target(p), (p.post, 1))[1]))
    return out


def _target(p: Projection) -> str:
    """The receptor `p` feeds, `population:receptor`."""
    return f"{p.post}:{p.receptor}"


def _rows(counts: np.ndarray, edges: int) -> np.ndarray:
    """Each presynaptic neuron's out-edges, by their index in presynaptic order, as one row padded with
    `edges`, past the last edge; a last row of padding stands for no neuron."""
    widest = int(counts.max())
    starts = np.cumsum(counts) - counts
    column = np.arange(widest)
    rows = np.where(column < counts[:, None], starts[:, None] + column, edges)
    return np.concatenate([rows, np.full((1, widest), edges)]).astype(np.int32)


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


class PopulationState(TypedDict):
    """A population's neurons and synapses, and the ring buffer of its spikes over the longest delay."""

    point_neuron: PointNeuronState
    buffer: jax.Array


class PlasticState(TypedDict):
    """A plastic projection's traces and the weights they change.

    Each `Plasticity` rule has its own traces type, so `traces` is typed as
    any object.
    """

    traces: object
    weight: jax.Array


class ReleaseState(TypedDict):
    """A projection's short-term release per presynaptic neuron, and the ring buffer of the efficacies
    its spikes were sent with."""

    release: TsodyksMarkramState
    buffer: jax.Array


class NetworkState(TypedDict):
    """The `state` collection of a `Network`, carried between calls."""

    populations: dict[str, PopulationState]
    pending: dict[str, jax.Array]
    """By `population:receptor`, the weights event projections over edges of their own delays have sent
    and that are not yet due, `[rows, size]`: row `t % rows` arrives at the end of step `t`."""
    ready: dict[str, jax.Array]
    """By `population:receptor` of a delta receptor in `pending`, what arrives at the end of the next
    step, `[size]`, read once the step that reached it has sent its spikes."""
    plastic: dict[str, PlasticState]
    """By projection key, the projections with plasticity."""
    short_term: dict[str, ReleaseState]
    """By projection key, the projections with short-term release."""
    modulators: dict[str, jax.Array]
    """Each modulator's concentration, by name."""
    t: jax.Array
    """The steps run so far."""


def _check_receptor(population: Population, receptor: str, what: str) -> None:
    if receptor not in population.receptors:
        named = ", ".join(map(repr, population.receptors)) or "none"
        raise ValueError(f"{what} targets receptor {receptor!r}, which population {population.name!r} lacks; "
                         f"it has {named}")


def _check_conductances(population: Population) -> None:
    """Refuse a conductance receptor that the neuron model has no reversal potential for.

    A model reads each conductance against its reversal potential for the
    receptor's name, so a name it lacks would fail at the first step.
    """
    neuron, model = population.neuron, type(population.neuron).__name__
    where = f"population {population.name!r}"
    for name, receptor in population.receptors.items():
        if receptor.kind != "conductance" or receptor.synapse.lands != "synapse":
            continue
        if not isinstance(neuron, Reversing):
            raise ValueError(f"{where}: receptor {name!r} is a conductance, and {model} has no reversal "
                             f"potentials to read it against")
        if name not in neuron.reversal:
            known = ", ".join(map(repr, neuron.reversal))
            raise ValueError(f"{where}: conductance receptor {name!r} has no reversal potential; "
                             f"{model}.reversal has {known}. Name the receptor after one of them, or give "
                             f"{model} a reversal potential for it")


def _check_inputs(population: Population, rest: PointNeuronState, current: bool, dt: float) -> None:
    """Refuse a population whose neuron model will not step on what the network gives it: what its
    kinetic receptors and current input deliver, and no noise.

    Whether a model takes currents is the model's own answer, so one step
    is traced on abstract values as the network will step it. A
    dimensionless model then refuses a kinetic synapse before a run starts,
    with the population named, and a stochastic model (`BernoulliCell`) the
    noise a network does not draw.
    """
    kinetic = [name for name, r in population.receptors.items() if r.synapse.lands == "synapse"]
    held = jnp.zeros((population.size,), jnp.float32) if current else 0.0
    model, where = type(population.neuron).__name__, f"population {population.name!r}"
    try:
        jax.eval_shape(lambda state: population.point_neuron.advance(state, held, jnp.zeros(()), dt), rest)
    except ValueError as error:
        if not kinetic and not current:
            raise ValueError(f"{where}: {model} cannot step in a Network: {error}") from error
        sources = [f"receptor {name!r} ({type(population.receptors[name].synapse).__name__})"
                   for name in kinetic] + (["a CurrentInput"] if current else [])
        raise ValueError(f"{where}: {model} cannot read {' or '.join(sources)}: {error}. Its input has to "
                         f"arrive as voltage jumps, through Delta receptors") from error


def _check_initial(population: Population, rest: PointNeuronState) -> None:
    """Refuse an `initial` entry that names neither a field of the neuron model's state nor a receptor
    whose synapse state is one array."""
    where, model = f"population {population.name!r}", type(population.neuron).__name__
    fields = rest.neuron._fields if isinstance(rest.neuron, _Record) else ()
    for name in population.initial:
        if name in population.receptors and name in fields:
            raise ValueError(f"{where}: initial {name!r} names both a receptor and a field of {model}'s "
                             f"state; rename the receptor")
        if name in population.receptors and not isinstance(rest.synapses[name], jax.ShapeDtypeStruct):
            synapse = type(population.receptors[name].synapse).__name__
            raise ValueError(f"{where}: initial {name!r} sets receptor {name!r}, whose {synapse} state is "
                             f"not one array")
        if name not in population.receptors and name not in fields:
            known = ", ".join(map(repr, (*fields, *population.receptors)))
            raise ValueError(f"{where}: initial {name!r} is neither a field of {model}'s state nor a "
                             f"receptor; it can set {known}")


def _check_graded(p: Projection, pre: Population, post: Population) -> None:
    """Refuse a projection whose delivery does not fit what its presynaptic population sends.

    A graded population sends a value every step. It reaches a `Graded`
    synapse, which follows it, or a delta receptor as a jump; a spiking
    synapse would add the value as if it were a spike, once per step, and
    the spike-driven options (events, plasticity, short-term and stochastic
    release) have no spike to act on. A `Graded` synapse fed spikes would
    hold each one for a step, so it takes only graded input.
    """
    synapse = post.receptors[p.receptor].synapse
    if not pre.graded:
        if isinstance(synapse, Graded):
            raise ValueError(f"{p.key}: receptor {p.receptor!r} is a Graded synapse, which follows a graded "
                             f"value, and population {p.pre!r} sends spikes")
        if p.plasticity is not None and post.graded:
            raise ValueError(f"{p.key}: plasticity pairs spikes, and population {p.post!r} is graded")
        return
    if synapse.lands == "synapse" and not isinstance(synapse, Graded):
        raise ValueError(f"{p.key}: population {p.pre!r} is graded and sends a value every step, which "
                         f"receptor {p.receptor!r} ({type(synapse).__name__}) would take as spikes; give "
                         f"it a Graded synapse, or a Delta receptor for a dimensionless target")
    if p.format == "events":
        raise ValueError(f"{p.key}: population {p.pre!r} is graded, and event delivery visits only neurons "
                         f"that spiked; use format 'edges' or 'dense'")
    for option in ("plasticity", "short_term", "release"):
        if getattr(p, option) is not None:
            raise ValueError(f"{p.key}: {option} acts on spikes, and population {p.pre!r} is graded")


def _check_coupled(population: Population, junction: str, dt: float) -> None:
    """Refuse a gap junction onto a population that has no membrane voltage or whose model cannot take
    the coupling, by tracing one coupled step."""
    model, where = type(population.neuron).__name__, f"gap junction {junction}"
    shape, dtype = (population.size,), jnp.dtype(jnp.float32)
    rest = jax.eval_shape(lambda: population.point_neuron.init_state(shape, dtype))
    try:
        jax.eval_shape(_membrane, rest)
    except TypeError as error:
        raise ValueError(f"{where}: {model} of population {population.name!r} has no membrane voltage to "
                         f"couple") from error
    zeros = jnp.zeros(shape, jnp.float32)
    try:
        gap = Gap(zeros, Term(zeros, zeros, jnp.inf))
        jax.eval_shape(lambda state: population.point_neuron.advance(state, 0.0, jnp.zeros(()), dt, gap),
                       rest)
    except ValueError as error:
        raise ValueError(f"{where}: {model} of population {population.name!r} cannot take a gap junction: "
                         f"{error}") from error


def _check_sources(network: Network, populations: Mapping[str, Population]) -> None:
    """Refuse inputs onto unknown populations or receptors, and Poisson spikes onto a `Graded` synapse."""
    for source in network.inputs:
        if source.target not in populations:
            raise ValueError(f"input onto unknown population {source.target!r}")
        if isinstance(source, CurrentInput):
            continue
        _check_receptor(populations[source.target], source.receptor, f"a {type(source).__name__}")
        if (isinstance(source, PoissonInput)
                and isinstance(populations[source.target].receptors[source.receptor].synapse, Graded)):
            raise ValueError(f"a PoissonInput sends spikes, and receptor {source.receptor!r} of "
                             f"{source.target!r} is a Graded synapse")


def _check_couplings(network: Network, populations: Mapping[str, Population]) -> None:
    """Refuse gap junctions and modulators that name what the network lacks or cannot take."""
    for junction in network.junctions:
        for side in (junction.a, junction.b):
            if side not in populations:
                raise ValueError(f"gap junction {junction.key} names unknown population {side!r}")
            _check_coupled(populations[side], junction.key, network.dt)
    names = [m.name for m in network.modulators]
    if len(set(names)) != len(names):
        raise ValueError(f"modulator names must be unique, got {names}")
    for m in network.modulators:
        if m.source not in populations:
            raise ValueError(f"modulator {m.name!r} names unknown population {m.source!r}")
        if populations[m.source].graded:
            raise ValueError(f"modulator {m.name!r} is released by spikes, and population {m.source!r} is "
                             f"graded")
        if m.tau <= 0:
            raise ValueError(f"modulator {m.name!r} needs a positive time constant, not {m.tau}")
    _check_modulated(network)


def _check_modulated(network: Network) -> None:
    """Refuse plasticity that reads a modulator the network lacks, or assumes another time constant."""
    taus = {m.name: m.tau for m in network.modulators}
    for p in network.projections:
        read = {} if p.plasticity is None else p.plasticity.modulated_by()
        for name, tau in read.items():
            if name not in taus:
                raise ValueError(f"{p.key}: its plasticity reads modulator {name!r}, which the network "
                                 f"lacks; it has {', '.join(map(repr, taus)) or 'none'}")
            if tau is not None and tau != taus[name]:
                raise ValueError(f"{p.key}: its plasticity integrates modulator {name!r} with a time "
                                 f"constant of {tau} ms, and the modulator decays with {taus[name]} ms")


def _check_population(population: Population, current: bool, dt: float) -> None:
    _check_conductances(population)
    # The checks read the state's structure, which is the same in every dtype; float32 traces whether or
    # not float64 is enabled where the network is built.
    shape, dtype = (population.size,), jnp.dtype(jnp.float32)
    rest = jax.eval_shape(lambda: population.point_neuron.init_state(shape, dtype))
    _check_inputs(population, rest, current, dt)
    _check_initial(population, rest)


class Network(nn.Module):
    """Populations and projections stepped together; see the module docstring."""

    populations: Sequence[Population]
    projections: Sequence[Projection] = ()
    inputs: Sequence[PoissonInput | CurrentInput | ArrivalInput] = ()
    dt: float = 0.1
    dtype: jax.typing.DTypeLike = jnp.float32
    """The dtype of the state: membranes, synapses, traces and spike buffers."""
    junctions: Sequence[GapJunction] = ()
    """Gap junctions, electrical coupling between the membranes of populations."""
    modulators: Sequence[Modulator] = ()
    """Neuromodulators released by populations' spikes, which plasticity reads."""

    def __post_init__(self) -> None:
        super().__post_init__()
        populations = {p.name: p for p in self.populations}
        if len(populations) != len(self.populations):
            raise ValueError("population names must be unique")
        keys = [p.key for p in self.projections] + [j.key for j in self.junctions]
        if len(set(keys)) != len(keys):
            raise ValueError(f"projection and gap junction names must be unique, got {keys}; name repeated "
                             f"ones")
        for p in self.projections:
            for side in (p.pre, p.post):
                if side not in populations:
                    raise ValueError(f"projection {p.key} names unknown population {side!r}")
            _check_receptor(populations[p.post], p.receptor, f"projection {p.key}")
            _check_graded(p, populations[p.pre], populations[p.post])
        _check_sources(self, populations)
        _check_couplings(self, populations)
        currents = {source.target for source in self.inputs if isinstance(source, CurrentInput)}
        for population in self.populations:
            _check_population(population, population.name in currents, self.dt)

    def _build(self, populations: Mapping[str, Population]) -> dict[str, dict[str, np.ndarray]]:
        """Draw every projection's edges, delays (whole steps) and weights from one key."""
        rng = _seed(self.make_rng("params"))
        built = {}
        for p in self.projections:
            pre, post = populations[p.pre], populations[p.post]
            edges = p.connectivity.edges(rng, pre.size, post.size, p.pre == p.post)
            delays = whole_steps(_per_edge(p.delay, rng, edges, f"{p.key} delay"), self.dt,
                                 f"{p.key}: a delay").astype(np.int32)
            minimum = 1 if post.receptors[p.receptor].synapse.lands == "before_threshold" else 0
            if len(delays) and delays.min() < minimum:
                raise ValueError(f"{p.key}: delays must be at least {minimum} step(s) of {self.dt} ms "
                                 f"onto a {'delta' if minimum else 'kinetic'} synapse")
            if p.plasticity is not None and np.ndim(delays) and len(np.unique(delays)) > 1:
                raise ValueError(f"{p.key}: a plastic projection needs one delay for all its edges")
            if not len(delays):
                delays = np.asarray(minimum, np.int32)  # no edges: any delay will do
            elif np.all(delays == delays[0]):
                delays = np.asarray(delays[0], np.int32)  # one delay: read one row of the ring per step
            weight = _per_edge(p.weight, rng, edges, f"{p.key} weight").astype(np.dtype(self.dtype))
            if _events(p, pre):
                built[p.key] = _event_edges(edges, weight, delays, pre.size)
            elif _dense(p, delays, len(edges), pre.size, post.size):
                matrix = np.zeros((pre.size, post.size), weight.dtype)
                np.add.at(matrix, (edges.pre, edges.post), weight)  # repeated pairs sum
                built[p.key] = {"delay": np.asarray(np.ravel(delays)[0] if np.size(delays) else 0, np.int32),
                                "weight": matrix}
            else:
                built[p.key] = {"pre": edges.pre, "post": edges.post, "delay": delays, "weight": weight}
        # After the projections, so adding junctions leaves the projections' draws as they were.
        for j in self.junctions:
            a, b = populations[j.a], populations[j.b]
            edges = j.connectivity.edges(rng, a.size, b.size, j.a == j.b)
            weight = _per_edge(j.weight, rng, edges, f"{j.key} weight").astype(np.dtype(self.dtype))
            built[j.key] = {"a": edges.pre, "b": edges.post, "weight": weight}
        return built

    def _lags(self, edges) -> dict[str, int]:
        """How many steps of output each population's ring buffer keeps; a projection that sends its spikes
        ahead reads only the step's."""
        lags = {p.name: 1 for p in self.populations}
        for p in self.projections:
            if np.size(edges[p.key]["delay"]) and not _ahead(edges[p.key]):
                longest = int(np.max(np.asarray(edges[p.key]["delay"]))) + 1
                lags[p.pre] = max(lags[p.pre], longest)
                if p.plasticity is not None:
                    lags[p.post] = max(lags[p.post], longest)
        return lags

    def _rest(self, populations: Mapping[str, Population], weights: Mapping[str, jax.Array],
              edges: Mapping[str, Mapping[str, jax.Array]], lags: Mapping[str, int]) -> NetworkState:
        """The state a run starts from: resting neurons (or `initial`), empty buffers, fresh traces."""
        rng = _seed(self.make_rng("params"))
        dtype = jnp.dtype(self.dtype)
        out: dict[str, PopulationState] = {}
        for name, pop in populations.items():
            point_neuron = pop.point_neuron.init_state((pop.size,), dtype)
            if pop.initial:
                point_neuron = _initial(pop, point_neuron, rng)
            buffer = jnp.zeros((lags[name], pop.size), dtype)
            out[name] = {"point_neuron": point_neuron, "buffer": buffer}
        plastic: dict[str, PlasticState] = {}
        short_term: dict[str, ReleaseState] = {}
        for p in self.projections:
            pre, post = populations[p.pre].size, populations[p.post].size
            if p.plasticity is not None:
                count = int(np.size(edges[p.key]["pre"]))
                plastic[p.key] = {"traces": p.plasticity.init_state(pre, post, count, dtype),
                                  "weight": weights[p.key]}
            if p.short_term is not None:
                short_term[p.key] = {"release": p.short_term.init_state((pre,), dtype),
                                     "buffer": jnp.zeros((lags[p.pre], pre), dtype)}
        held = _pending(self.projections, edges)
        pending = {key: jnp.zeros((rows, populations[post].size), dtype)
                   for key, (post, rows) in held.items()}
        delta = {_target(p) for p in self.projections
                 if populations[p.post].receptors[p.receptor].synapse.lands == "before_threshold"}
        ready = {key: jnp.zeros(populations[post].size, dtype)
                 for key, (post, _) in held.items() if key in delta}
        modulators = {m.name: jnp.zeros((), dtype) for m in self.modulators}
        return {"populations": out, "pending": pending, "ready": ready, "plastic": plastic,
                "short_term": short_term, "modulators": modulators, "t": jnp.zeros((), jnp.int32)}

    def _noise(self) -> jax.Array | None:
        """The key Poisson inputs and stochastic release draw from, None when nothing draws."""
        drawing = [type(s).__name__ for s in self.inputs if isinstance(s, PoissonInput)]
        drawing += [f"stochastic release on {p.key}" for p in self.projections if p.release is not None]
        if not drawing:
            return None
        if not self.has_rng("noise"):
            raise ValueError(f"the network draws from the `noise` key ({', '.join(drawing)}): apply it with "
                             f"rngs={{'noise': key}}")
        # Every step draws from it, and JAX's default Threefry-2x32 compiles to a loop on CPU, where
        # Threefry-4x32 compiles unrolled: 10,000 uniform draws took 208 us against 34 us on a 4-core CPU,
        # and Brunel's network at 12,500 neurons 12.4 s per simulated second against 9.6 s.
        bits = jax.random.bits(self.make_rng("noise"), (4,), jnp.uint32)
        return jax.random.wrap_key_data(bits, impl="threefry4x32")

    def connections(self, variables: Variables) -> dict[str, Connections]:
        """Every projection's synapses after a run, by projection key, as NEST's `GetConnections` reads them.

            pre, post, weight, delay = network.connections(result.variables)["e->e:ampa"]

        Weights come from where the projection keeps them: the state for a
        plastic one (as learned so far), `params` for a trainable one, the
        connectome otherwise. A dense projection stores a matrix and no edge
        list, so its connections are the matrix's nonzero entries, with
        repeated pairs summed.
        """
        sizes = {p.name: p.size for p in self.populations}
        out = {}
        for p in self.projections:
            e = {name: np.asarray(a) for name, a in variables["connectome"]["edges"][p.key].items()}
            if p.plasticity is not None:
                weight = np.asarray(variables["state"]["network"]["plastic"][p.key]["weight"])
            elif p.trainable:
                weight = np.asarray(variables["params"][f"weight:{p.key}"])
            else:
                weight = np.asarray(variables["connectome"][f"weight:{p.key}"])
            if "by_pre" in e:
                pre = np.repeat(np.arange(sizes[p.pre], dtype=np.int32), e["count"][:-1])
                post = e["by_pre"]
            elif "pre" in e:
                pre, post = e["pre"], e["post"]
            else:
                pre, post = np.nonzero(weight)
                weight = weight[pre, post]
            order = np.lexsort((post, pre))
            delay = np.broadcast_to(e["delay"] * self.dt, pre.shape)
            out[p.key] = Connections(pre[order].astype(np.int32), post[order].astype(np.int32),
                                     weight[..., order], delay[order])
        return out

    @nn.compact
    def __call__(self, drive: Drive | None = None, *, steps: int | None = None,
                 monitors: Mapping[str, Monitor] | None = None) -> dict[str, jax.Array]:
        """Advance `steps` steps (or as many as `drive` has); returns each monitor's records over time, under
        the monitor's name."""
        populations = {p.name: p for p in self.populations}
        monitors = dict(monitors or {})
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
            if p.trainable and self.is_initializing():
                weights[p.key] = self.param(f"weight:{p.key}",
                                            lambda _, k=p.key: jnp.asarray(build()[k]["weight"]))
            elif p.trainable:
                # Flax would check the stored weights' shape by tracing their initializer, which draws
                # the edges from the "params" key that only `init` has; the stored weights are read.
                weights[p.key] = self.get_variable("params", f"weight:{p.key}")
            else:
                weights[p.key] = self.variable("connectome", f"weight:{p.key}",
                                               lambda k=p.key: jnp.asarray(build()[k]["weight"])).value
        for j in self.junctions:
            weights[j.key] = self.variable("connectome", f"weight:{j.key}",
                                           lambda k=j.key: jnp.asarray(build()[k]["weight"])).value
        state = self.variable("state", "network",
                              lambda: self._rest(populations, weights, edges, self._lags(edges)))
        if self.is_initializing():
            return {}
        drive = dict(drive or {})
        timed = [int(np.shape(v)[0]) for v in jax.tree.leaves(drive) if np.ndim(v) > 0]
        if steps is None and not timed:
            raise ValueError("pass `steps` or a time-major `drive`")
        length: int = steps if steps is not None else timed[0]
        _check_monitors(monitors, populations)
        stepper = _Stepper(self, populations, edges, weights, monitors, self._noise())
        held = {name: jnp.broadcast_to(jnp.asarray(value, self.dtype), (length, *np.shape(value)[1:]))
                if np.ndim(value) > 0 else jnp.full((length,), value, self.dtype)
                for name, value in drive.items()}
        state.value, records = _scan(stepper, state.value, held, length, monitors)
        return records


def _check_monitors(monitors: Mapping[str, Monitor], populations: Mapping[str, Population]) -> None:
    """Refuse a monitor that reads spikes from a graded population."""
    for name, m in monitors.items():
        if isinstance(m, _SPIKE_MONITORS) and populations[m.population].graded:
            raise ValueError(f"monitor {name!r} ({type(m).__name__}) reads spikes, and population "
                             f"{m.population!r} is graded; record it with OutputTrace")


def _scan(stepper: _Stepper, state: NetworkState, held: Mapping[str, jax.Array], steps: int,
          monitors: Mapping[str, Monitor]) -> tuple[NetworkState, dict[str, jax.Array]]:
    """Run `steps` steps; stack each monitor's records, or sum them for accumulating monitors."""
    summed = [name for name, m in monitors.items() if m.accumulate]
    if not summed:
        return jax.lax.scan(stepper, state, held, length=steps)
    shapes = jax.eval_shape(stepper, state, jax.tree.map(lambda x: x[0], held))[1]
    zeros = {name: jnp.zeros(shapes[name].shape, shapes[name].dtype) for name in summed}

    def step(carry, drive_t):
        current, totals = carry
        current, records = stepper(current, drive_t)
        totals = {name: totals[name] + records[name] for name in summed}
        return (current, totals), {name: r for name, r in records.items() if name not in totals}

    (state, totals), stacked = jax.lax.scan(step, (state, zeros), held, length=steps)
    return state, {name: totals[name] if name in totals else stacked[name] for name in monitors}


class _Stepper:
    """A network's step, closed over its structure and fixed variables, in the documented order."""

    def __init__(self, network: Network, populations: Mapping[str, Population], edges, weights,
                 monitors: Mapping[str, Monitor], key):
        self.network, self.populations, self.edges, self.weights = network, populations, edges, weights
        self.monitors, self.key, self.dt = monitors, key, network.dt
        self.into: dict[str, list[Projection]] = {name: [] for name in populations}
        for p in network.projections:
            self.into[p.post].append(p)
        self.coupled = sorted({side for j in network.junctions for side in (j.a, j.b)})
        # Each stochastic projection draws from its own stream, numbered after the Poisson inputs' streams.
        self.streams = {p.key: len(network.inputs) + i for i, p in enumerate(network.projections)
                        if p.release is not None}
        self.ahead = [p for p in network.projections if _ahead(edges[p.key])]
        self.targets = {_target(p): (p.post, p.receptor) for p in self.ahead}

    def is_delta(self, population: str, receptor: str) -> bool:
        return self.populations[population].receptors[receptor].synapse.lands == "before_threshold"

    def release_key(self, t, p: Projection) -> jax.Array | None:
        """The key of `p`'s release draws in step `t`, or None for a projection that transmits every spike."""
        if p.release is None:
            return None
        assert self.key is not None
        return jax.random.fold_in(jax.random.fold_in(self.key, t), self.streams[p.key])

    def deliver(self, t, p: Projection, weight, ring) -> jax.Array:
        """Weighted outputs of `p` due at the end of step `t`, summed per postsynaptic neuron."""
        e = self.edges[p.key]
        key = self.release_key(t, p)
        if "by_pre" in e:
            out = jnp.zeros(self.populations[p.post].size, weight.dtype)
            return self.events(p, e, weight, ring[(t - e["delay"]) % ring.shape[0]], key, out)
        if "pre" not in e:
            return ring[(t - e["delay"]) % ring.shape[0]] @ weight
        if e["delay"].ndim == 0:
            sent = ring[(t - e["delay"]) % ring.shape[0]][e["pre"]]
        else:
            sent = ring[(t - e["delay"]) % ring.shape[0], e["pre"]]
        if p.release is None:
            transmitted = weight * sent
        else:
            assert key is not None
            transmitted = p.release.transmit(key, weight, sent)
        return jax.ops.segment_sum(transmitted, e["post"], num_segments=self.populations[p.post].size,
                                   indices_are_sorted=True)

    def send(self, t, rings: Mapping[str, jax.Array], weights, pending: Mapping[str, jax.Array]
             ) -> dict[str, jax.Array]:
        """`pending` with the spikes of step `t` sent ahead by the event projections whose edges have their
        own delays, each edge's weight in the row of the step it is due in, after emptying the row each
        receptor read last: a kinetic one's of step `t - 1`, a delta one's of step `t`, read at the end of
        step `t - 1`.

        A row read in a step before the buffer is written would make XLA
        copy the whole buffer to keep the read, every step: 0.36 ms against
        0.012 ms for 51 rows of 15,435 neurons on a 4-core CPU. So a step
        writes first and reads after, and a delta receptor's row for the
        next step is read at the end of this one (`NetworkState.ready`).
        """
        pending = {key: rows.at[(t - (0 if self.is_delta(*self.targets[key]) else 1)) % rows.shape[0]].set(0)
                   for key, rows in pending.items()}
        for p in self.ahead:
            ring = rings[p.key]
            pending[_target(p)] = self.events(p, self.edges[p.key], weights[p.key], ring[t % ring.shape[0]],
                                              self.release_key(t, p), pending[_target(p)], t)
        return pending

    def next_ready(self, pending: Mapping[str, jax.Array], t) -> dict[str, jax.Array]:
        """What each delta receptor in `pending` receives at the end of step `t + 1`, once step `t` sent."""
        return {key: rows[(t + 1) % rows.shape[0]] for key, rows in pending.items()
                if self.is_delta(*self.targets[key])}

    def events(self, p: Projection, e, weight, sent, key: jax.Array | None, out: jax.Array,
               t: jax.Array | None = None) -> jax.Array:
        """Delivery that visits only the edges of neurons that spiked, added to `out`.

        `out` is a value per postsynaptic neuron, or, with `t`, for a
        projection whose edges have their own delays, the rows of what is on
        its way, `[rows, post]`: an edge sending in step `t` adds to row
        `(t + delay) % rows`.

        A step delivers its spiking neurons in passes of up to `p.per_pass`,
        as many passes as it has spikes for (none in a silent step), so its
        cost follows the activity, not the edge count, and no spike is left
        out. Each pass takes the next spiking neurons with `jax.lax.top_k`.
        With out-degrees near even or few (`EVENT_PADDING`, `EVENT_ROWS`)
        each neuron's out-edges are one padded row, and a pass gathers its
        neurons' rows. Otherwise their out-edges, contiguous when edges are
        sorted by presynaptic neuron, are laid end to end by a prefix sum of
        their out-degrees and walked in blocks of `EVENT_BLOCK` slots; each
        slot finds its neuron by binary search. With stochastic release, each
        pass, and each block of a pass, draws from `key` folded with its
        index, the key of the step that sends.

        A `while_loop` has no reverse mode, so without stochastic release the
        delivery is differentiated as its edge list (`_as_edges`).
        """
        if weight.shape[0] == 0:
            return out
        passes = functools.partial(self._passes, p, key=key)
        if p.release is not None:
            return passes(weight, sent, out, e, t)
        return _as_edges(passes, weight, sent, out, e, t)

    def _passes(self, p: Projection, weight, sent, out: jax.Array, e, t: jax.Array | None,
                key: jax.Array | None) -> jax.Array:
        """`events`' passes over the spiking neurons."""
        size = sent.shape[0]
        width = min(p.per_pass, size)
        deliver = self._rows if "rows" in e else self._slots

        def busy(carry: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
            return jnp.any(carry[1] != 0)

        def one_pass(carry: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
            out, left, n = carry
            value, index = jax.lax.top_k(left, width)  # what a neuron sends is never below 0
            active = jnp.where(value != 0, index, size)
            passed = None if key is None else jax.random.fold_in(key, n)
            out = deliver(p, e, weight, sent, passed, active, out, t)
            return out, left.at[active].set(0, mode="drop"), n + 1

        return jax.lax.while_loop(busy, one_pass, (out, sent, jnp.zeros((), jnp.int32)))[0]

    def _rows(self, p: Projection, e, weight, sent, key: jax.Array | None, active: jax.Array,
              out: jax.Array, t: jax.Array | None) -> jax.Array:
        """The `active` neurons' padded rows of out-edges, gathered and summed onto their targets; an
        index past the last neuron stands for none."""
        # A padding slot, or a neuron that stands for none, is an edge index past the last edge.
        edge = e["rows"][active]
        valid = edge < weight.shape[0]
        spiked = sent.at[active].get(mode="fill", fill_value=0)[:, None]
        weights = weight.at[edge].get(mode="fill", fill_value=0)
        if p.release is None:
            value = jnp.where(valid, weights * spiked, 0)
        else:
            assert key is not None
            value = jnp.where(valid, p.release.transmit(key, weights, spiked), 0)
        return out.at[_landing(e, edge, out, t)].add(value, mode="drop")

    def _slots(self, p: Projection, e, weight, sent, key: jax.Array | None, active: jax.Array,
               out: jax.Array, t: jax.Array | None) -> jax.Array:
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
            if p.release is None:
                value = jnp.where(valid, weight[edge] * spiked, 0)
            else:
                assert key is not None
                drawn = p.release.transmit(jax.random.fold_in(key, i), weight[edge], spiked)
                value = jnp.where(valid, drawn, 0)
            return out.at[_landing(e, edge, out, t)].add(value, mode="drop")

        blocks = (total + EVENT_BLOCK - 1) // EVENT_BLOCK
        return jax.lax.fori_loop(0, blocks, block, out)

    def external(self, t, drive_t):
        """This step's Poisson arrivals by population and receptor, and injected currents by population.

        A population no current is injected into has no entry, and its model reads no current.
        """
        arrivals: dict[str, dict[str, jax.Array]] = {name: {} for name in self.populations}
        currents: dict[str, jax.Array] = {}
        for i, source in enumerate(self.network.inputs):
            if isinstance(source, CurrentInput):
                currents[source.target] = (currents.get(source.target, jnp.zeros((), self.network.dtype))
                                           + drive_t[source.name])
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

    def gather(self, name, t, external, rings, weights, sent: Mapping[str, jax.Array],
               delta: bool) -> dict[str, jax.Array]:
        """Everything due at the end of step `t` on `name`'s delta (or kinetic) receptors; `sent` holds
        what was sent ahead to them, the ready row of a delta receptor, the pending rows of a kinetic one."""
        arrivals = {k: v for k, v in external.items() if self.is_delta(name, k) == delta}
        for p in self.into[name]:
            if self.is_delta(name, p.receptor) == delta and not _ahead(self.edges[p.key]):
                due = self.deliver(t, p, weights[p.key], rings[p.key])
                arrivals[p.receptor] = arrivals.get(p.receptor, 0.0) + due
        for receptor in self.populations[name].receptors:
            ahead = sent.get(f"{name}:{receptor}")
            if ahead is not None and self.is_delta(name, receptor) == delta:
                due = ahead if delta else ahead[t % ahead.shape[0]]
                arrivals[receptor] = arrivals.get(receptor, 0.0) + due
        return arrivals

    def gaps(self, start: Mapping[str, jax.Array], end: Mapping[str, jax.Array]) -> dict[str, Gap]:
        """Each coupled population's gap-junction conductance, and the drive of its partners' voltages
        moving linearly from `start` to `end` over the step, by population."""
        conductance: dict[str, jax.Array] = {}
        drive: dict[str, jax.Array] = {}
        slope: dict[str, jax.Array] = {}
        for j in self.network.junctions:
            e, g = self.edges[j.key], self.weights[j.key]
            for here, there, mine, theirs in ((j.a, j.b, e["a"], e["b"]), (j.b, j.a, e["b"], e["a"])):
                size = self.populations[here].size
                conductance[here] = conductance.get(here, 0.0) + jax.ops.segment_sum(g, mine, size)
                before, after = start[there][theirs].astype(g.dtype), end[there][theirs].astype(g.dtype)
                drive[here] = drive.get(here, 0.0) + jax.ops.segment_sum(g * before, mine, size)
                rise = g * (after - before) / self.dt
                slope[here] = slope.get(here, 0.0) + jax.ops.segment_sum(rise, mine, size)
        return {name: Gap(conductance[name], Term(drive[name], slope[name], jnp.inf)) for name in conductance}

    def modulate(self, before: Mapping[str, jax.Array],
                 fired: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        """Each modulator's concentration after the step: decayed over it, then raised by its spikes."""
        out = {}
        for m in self.network.modulators:
            spikes = fired[m.source] if m.neurons is None else fired[m.source][np.asarray(m.neurons)]
            released = m.release * jnp.sum(spikes, dtype=before[m.name].dtype)
            out[m.name] = before[m.name] * math.exp(-self.dt / m.tau) + released
        return out

    def plasticity(self, t: jax.Array, plastic: dict[str, PlasticState], fired: Mapping[str, jax.Array],
                   buffers: Mapping[str, jax.Array], modulators: Mapping[str, jax.Array]) -> None:
        """STDP: presynaptic spikes as sent, postsynaptic ones after the projection's (dendritic) delay;
        the modulators as they are after this step's release."""
        for p in self.network.projections:
            if p.plasticity is None:
                continue
            e = self.edges[p.key]
            delay = e["delay"] if e["delay"].ndim == 0 else 0  # plastic projections have one delay
            ring = buffers[p.post]
            arrived = ring[(t - delay) % ring.shape[0]]
            traces, weight = p.plasticity.step(plastic[p.key]["traces"], plastic[p.key]["weight"],
                                               fired[p.pre], arrived, e["pre"], e["post"], self.dt,
                                               modulators=modulators)
            plastic[p.key] = {"traces": traces, "weight": weight}

    def __call__(self, state: NetworkState,
                 drive_t: Mapping[str, jax.Array]) -> tuple[NetworkState, dict[str, jax.Array]]:
        t, pops = state["t"], state["populations"]
        plastic, short_term = dict(state["plastic"]), dict(state["short_term"])
        projections = self.network.projections
        weights = {p.key: plastic[p.key]["weight"] if p.plasticity is not None else self.weights[p.key]
                   for p in projections}
        external, currents = self.external(t, drive_t)

        def rings(buffers):
            return {p.key: short_term[p.key]["buffer"] if p.short_term is not None else buffers[p.pre]
                    for p in projections}

        # 1-2. Delta jumps due now, then the membranes, coupled by gap junctions.
        moved, outputs = {}, {}
        frozen = {name: pop.point_neuron.frozen(pops[name]["point_neuron"], self.dt)
                  for name, pop in self.populations.items()}
        before = rings({name: pops[name]["buffer"] for name in self.populations})
        jumps = {name: pop.point_neuron.delta(self.gather(name, t, external[name], before, weights,
                                                          state["ready"], delta=True))
                 for name, pop in self.populations.items()}

        def advance(name: str, gap: Gap | None) -> tuple[PointNeuronState, Output]:
            return self.populations[name].point_neuron.advance(
                pops[name]["point_neuron"], currents.get(name, 0.0), jumps[name], self.dt, gap)

        gaps: dict[str, Gap] = {}
        if self.network.junctions:
            # Predict each coupled membrane's end voltage with its partners held, then advance it with
            # them moving linearly to theirs: one iteration of waveform relaxation, second order in dt.
            start = {name: _membrane(pops[name]["point_neuron"]) for name in self.coupled}
            held = self.gaps(start, start)
            end = {name: _membrane(advance(name, held[name])[0]) for name in self.coupled}
            gaps = self.gaps(start, end)
        for name in self.populations:
            moved[name], out = advance(name, gaps.get(name))
            outputs[name] = out.value

        # 3. Outputs into the ring buffers; short-term release scales spikes as they are sent; modulators take
        # up the spikes.
        modulators = self.modulate(state["modulators"], outputs)
        buffers = {}
        for name in self.populations:
            ring = pops[name]["buffer"]
            buffers[name] = ring.at[t % ring.shape[0]].set(outputs[name].astype(ring.dtype))
        for p in projections:
            if p.short_term is not None:
                release, efficacy = p.short_term.step(short_term[p.key]["release"], outputs[p.pre], self.dt)
                ring = short_term[p.key]["buffer"]
                short_term[p.key] = {"release": release,
                                     "buffer": ring.at[t % ring.shape[0]].set(efficacy.astype(ring.dtype))}

        after = rings(buffers)
        pending = self.send(t, after, weights, state["pending"])

        # 4. Every other synapse receives what is due at the end of the step.
        new_pops: dict[str, PopulationState] = {}
        for name, pop in self.populations.items():
            due = self.gather(name, t, external[name], after, weights, pending, delta=False)
            point_neuron = pop.point_neuron.receive(moved[name], due, self.dt, outputs[name], frozen[name])
            new_pops[name] = {"point_neuron": point_neuron, "buffer": buffers[name]}
        ready = self.next_ready(pending, t)

        # 5-6. Plasticity, then monitors.
        self.plasticity(t, plastic, outputs, buffers, modulators)
        states = {n: v["point_neuron"] for n, v in new_pops.items()}
        records = {name: m.record(outputs, states, self.dt, modulators) for name, m in self.monitors.items()}
        return {"populations": new_pops, "pending": pending, "ready": ready, "plastic": plastic,
                "short_term": short_term, "modulators": modulators, "t": t + 1}, records
