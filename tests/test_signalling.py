"""Signalling beyond spikes in networks: graded transmission, stochastic release and gap junctions."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reference

from sparx.dynamics import (
    LIF,
    Arrivals,
    Delta,
    Exponential,
    Graded,
    GradedPotential,
    LICell,
    LIFCell,
    PairSTDP,
    PointNeuron,
    RateCell,
    Receptor,
    StochasticRelease,
    SynapticInput,
    TsodyksMarkram,
    run,
)
from sparx.graph import (
    AllToAll,
    ArrivalInput,
    CurrentInput,
    FromEdges,
    GapJunction,
    Network,
    OneToOne,
    OutputTrace,
    Population,
    Projection,
    SpikeRaster,
    StateMonitor,
)

GAP = np.load(Path(__file__).parent / "fixtures" / "nest_gap.npz")


def forced(size: int, receptor: str = "x") -> Population:
    """A population that fires exactly where its `ArrivalInput` named `receptor` sends 1."""
    return Population("s", size, LIFCell(0.0, threshold=0.5), {receptor: Receptor(Delta())})


# Graded transmission.


@pytest.mark.parametrize("format", ["edges", "dense"])
def test_a_graded_projection_is_the_presynaptic_release_through_the_graded_synapse(format):
    # Three graded neurons release onto two through a conductance; the network equals running the first
    # population alone and replaying its weighted release into the second's synapses.
    steps, dt = 400, 0.1
    rng = np.random.default_rng(0)
    current = rng.uniform(200.0, 1200.0, (steps, 3))
    pre, post = np.array([0, 1, 2, 2]), np.array([0, 0, 1, 0])
    weight = np.array([2.0, 1.5, 3.0, 0.5])  # nS at full release
    receptors = {"ampa": Receptor(Graded(4.0), "conductance")}
    populations = (Population("a", 3, GradedPotential()), Population("b", 2, GradedPotential(), receptors))
    network = Network(populations,
                      (Projection("a", "b", FromEdges(pre, post), weight=weight, delay=0.0, receptor="ampa",
                                  format=format),),
                      (CurrentInput("a", "i"),), dt=dt, dtype=jnp.float64)
    with jax.enable_x64(new_val=True):
        records, _ = network.apply(network.init(jax.random.key(0)), {"i": current},
                                   monitors={"release": OutputTrace("a"), "v": StateMonitor("b")},
                                   mutable=["state"])
        out, _ = run(GradedPotential(), SynapticInput(jnp.asarray(current)), dt=dt)
        matrix = np.zeros((3, 2))
        np.add.at(matrix, (pre, post), weight)
        arrivals = Arrivals(spikes={"ampa": out.value @ jnp.asarray(matrix)})  # due at the end of each step
        (_, v), _ = run(PointNeuron(GradedPotential(), receptors), arrivals, dt=dt,
                        record=lambda state: state.neuron.v)
        records, release, v = jax.tree.map(np.asarray, (records, out.value, v))
    np.testing.assert_allclose(records["release"], release, rtol=0, atol=1e-14)  # observed 0
    np.testing.assert_allclose(records["v"], v, rtol=0, atol=1e-10)  # observed 0
    assert v.max() - v.min() > 5  # the release moved the postsynaptic membranes


@pytest.mark.parametrize("format", ["edges", "dense"])
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_flynn_on_a_sparse_connectome_is_their_recurrence(format, activation):
    # FLYNN's h_{t+1} = alpha h_t + (1 - alpha) f(W h_t + x_t + b) as a network: a sparse projection onto
    # a delta receptor with a delay of one step delivers W h_t.
    n, steps = 40, 60
    rng = np.random.default_rng(1)
    pre, post = rng.integers(0, n, 200), rng.integers(0, n, 200)
    weight = rng.normal(0, 0.5, 200)
    alpha, bias = rng.uniform(0.5, 0.95, n), rng.normal(0, 0.2, n)
    sensory = np.zeros((steps, n))
    sensory[:, :5] = rng.normal(0, 1.0, (steps, 5))  # most of x_t is zero: only sensory neurons receive it
    with jax.enable_x64(new_val=True):
        unit = RateCell(jnp.asarray(alpha), jnp.asarray(bias), activation)  # float64 leaves
        network = Network((Population("brain", n, unit, {"in": Receptor(Delta())}),),
                          (Projection("brain", "brain", FromEdges(pre, post), weight=weight, delay=1.0,
                                      receptor="in", format=format),),
                          (ArrivalInput("brain", "x", "in"),), dt=1.0, dtype=jnp.float64)
        records, _ = network.apply(network.init(jax.random.key(0)), {"x": sensory},
                                   monitors={"h": OutputTrace("brain")}, mutable=["state"])
        h = np.asarray(records["h"])
    matrix = np.zeros((n, n))
    np.add.at(matrix, (pre, post), weight)
    expected = reference.flynn(sensory, matrix, alpha, bias, activation)
    np.testing.assert_allclose(h, expected, rtol=0, atol=1e-12)  # observed 3.3e-16
    assert np.abs(expected).max() > 0.3


def test_a_graded_population_refuses_event_delivery_and_spike_driven_options():
    graded = Population("g", 4, GradedPotential())
    target = Population("t", 4, LIF(), {"syn": Receptor(Graded(2.0)), "exp": Receptor(Exponential(2.0))})

    def build(**options):
        receptor = options.pop("receptor", "syn")
        return Network((graded, target), (Projection("g", "t", AllToAll(), receptor=receptor, **options),))

    with pytest.raises(ValueError, match="event delivery"):
        build(format="events")
    with pytest.raises(ValueError, match="Graded synapse"):
        build(receptor="exp")
    with pytest.raises(ValueError, match="release acts on spikes"):
        build(release=StochasticRelease(0.5))
    with pytest.raises(ValueError, match="short_term acts on spikes"):
        build(short_term=TsodyksMarkram())
    with pytest.raises(ValueError, match="plasticity acts on spikes"):
        build(plasticity=PairSTDP())
    with pytest.raises(ValueError, match="sends spikes"):  # and the other way round
        Network((Population("s", 4, LIF()), target), (Projection("s", "t", AllToAll(), receptor="syn"),))
    network = build(format="edges")
    with pytest.raises(ValueError, match="OutputTrace"):
        network.apply(network.init(jax.random.key(0)), steps=2, monitors={"r": SpikeRaster("g")},
                      mutable=["state"])
    with pytest.raises(ValueError, match="never spikes"):
        PointNeuron(GradedPotential(), {}, reset_synapses=True)


# Stochastic release.


def released(format: str, p: float, quantal: float, *, short_term: TsodyksMarkram | None = None,
             steps: int = 400, senders: int = 200, receivers: int = 50) -> np.ndarray:
    """What each of `receivers` neurons receives per step from `senders` neurons that fire every step,
    all to all with weight 1, through stochastic release: `[steps - 1, receivers]`."""
    network = Network((forced(senders), Population("r", receivers, LICell(0.0), {"in": Receptor(Delta())})),
                      (Projection("s", "r", AllToAll(), weight=1.0, delay=1.0, receptor="in", format=format,
                                  capacity=senders, release=StochasticRelease(p, quantal),
                                  short_term=short_term),),
                      (ArrivalInput("s", "x", "x"),), dt=1.0)
    records, _ = network.apply(network.init(jax.random.key(0)), {"x": np.ones((steps, senders))},
                               monitors={"in": OutputTrace("r")}, rngs={"noise": jax.random.key(1)},
                               mutable=["state"])
    return np.asarray(records["in"], np.float64)[1:]  # a delay of one step: nothing arrives in the first


@pytest.mark.parametrize("format", ["edges", "events"])
def test_stochastic_release_is_binomial_per_edge(format):
    # 200 synapses onto each neuron, each releasing with p = 0.3 a quantum of 2.5 times its weight: the
    # input is 2.5 Binomial(200, 0.3), mean 150 and variance 2.5^2 * 42 = 262.5.
    p, quantal, k = 0.3, 2.5, 200
    received = released(format, p, quantal)
    n = received.size  # 399 steps * 50 neurons
    mean, variance = k * p * quantal, k * p * (1 - p) * quantal ** 2
    # Within 4 standard errors; observed 0.02 and 0.74 for the mean, 2.4 and 0.8 for the variance.
    assert abs(received.mean() - mean) < 4 * np.sqrt(variance / n)
    assert abs(received.var() / variance - 1) < 4 * np.sqrt(2 / n)
    assert np.all(received % quantal == 0)  # whole quanta
    # Each edge draws on its own: neurons sharing every presynaptic spike are uncorrelated.
    correlation = np.corrcoef(received.T)[np.triu_indices(received.shape[1], 1)]
    assert abs(correlation.mean()) < 0.01 and np.abs(correlation).max() < 0.3  # observed 0.002, 0.22


def test_stochastic_release_needs_the_noise_key_and_edges():
    network = Network((forced(2), Population("r", 2, LICell(0.0), {"in": Receptor(Delta())})),
                      (Projection("s", "r", AllToAll(), delay=1.0, receptor="in", format="edges",
                                  release=StochasticRelease(0.5)),),
                      (ArrivalInput("s", "x", "x"),), dt=1.0)
    with pytest.raises(ValueError, match="noise"):
        network.apply(network.init(jax.random.key(0)), steps=2, mutable=["state"])
    with pytest.raises(ValueError, match="edges or events"):
        dense = Projection("s", "r", AllToAll(), delay=1.0, receptor="in", format="dense",
                           release=StochasticRelease(0.5))
        Network(network.populations, (dense,), network.inputs, dt=1.0).init(jax.random.key(0))


@pytest.mark.parametrize("format", ["edges", "events"])
def test_with_short_term_plasticity_a_synapse_releases_with_p_times_the_efficacy(format):
    # Each step every sender fires; Tsodyks-Markram depression sets the efficacy u x of step t, and each
    # synapse releases with probability p u x: the input of a neuron is Binomial(200, p u x).
    p, k, steps = 0.8, 200, 300
    depression = TsodyksMarkram(U=0.4, tau_rec=20.0)
    received = released(format, p, 1.0, short_term=depression, steps=steps)
    state = depression.init_state((1,), jnp.float32)
    efficacy = []
    for _ in range(steps - 1):
        state, e = depression.step(state, jnp.ones(1), 1.0)
        efficacy.append(float(e[0]))
    q = p * np.asarray(efficacy)[:, None]  # the efficacy sent at step t arrives at step t + 1
    z = (received - k * q) / np.sqrt(k * q * (1 - q))
    assert abs(z.mean()) < 4 / np.sqrt(z.size) and abs(z.std() - 1) < 0.05
    assert efficacy[0] > 2 * efficacy[-1]  # the synapses depressed


# Gap junctions.


def two_coupled_cells(g: float, dt: float, steps: int, *, separate: bool = False) -> np.ndarray:
    """NEST's fixture's two passive cells joined by one junction of `g` nS, their voltage after each step."""
    c, g_l, e_l = (float(GAP[f"param/{k}"]) for k in ("C_m", "g_L", "E_L"))
    cell = GradedPotential(tau_m=c / g_l, c_m=c, e_l=e_l)
    current, start = GAP["currents"], GAP["start"]
    if separate:
        populations = (Population("a", 1, cell, initial={"v": start[:1]}),
                       Population("b", 1, cell, initial={"v": start[1:]}))
        junction = GapJunction("a", "b", OneToOne(), weight=g)
        inputs = (CurrentInput("a", "ia"), CurrentInput("b", "ib"))
        drive = {"ia": np.full((steps, 1), current[0]), "ib": np.full((steps, 1), current[1])}
        monitors = {"a": StateMonitor("a"), "b": StateMonitor("b")}
    else:
        populations = (Population("a", 2, cell, initial={"v": start}),)
        junction = GapJunction("a", "a", FromEdges([0], [1]), weight=g)
        inputs, drive = (CurrentInput("a", "i"),), {"i": np.broadcast_to(current, (steps, 2))}
        monitors = {"a": StateMonitor("a")}
    network = Network(populations, inputs=inputs, junctions=(junction,), dt=dt, dtype=jnp.float64)
    with jax.enable_x64(new_val=True):
        records, _ = network.apply(network.init(jax.random.key(0)), drive, monitors=monitors,
                                   mutable=["state"])
        return np.concatenate([np.asarray(records[name]) for name in monitors], axis=1)


def coupled_exactly(g: float, t: np.ndarray) -> np.ndarray:
    """The two cells' voltages at `t`: their mean relaxes with g_L, their difference with g_L + 2 g."""
    c, g_l, e_l = (float(GAP[f"param/{k}"]) for k in ("C_m", "g_L", "E_L"))
    current, u = GAP["currents"], GAP["start"] - e_l
    total, gap = current.sum() / g_l, (current[0] - current[1]) / (g_l + 2 * g)
    s = total + (u.sum() - total) * np.exp(-t * g_l / c)
    d = gap + (u[0] - u[1] - gap) * np.exp(-t * (g_l + 2 * g) / c)
    return e_l + np.stack([s + d, s - d], axis=-1) / 2


@pytest.mark.parametrize(("g", "tolerance"), [(5.0, 5e-4), (50.0, 1e-2)])
def test_gap_junctions_converge_at_second_order_to_two_coupled_cells(g, tolerance):
    steps = 200  # 20 ms, five membrane time constants
    t = (np.arange(steps) + 1) * 0.1
    errors = []
    for k in (1, 2, 4):
        v = two_coupled_cells(g, 0.1 / k, steps * k)[k - 1::k]
        errors.append(np.abs(v - coupled_exactly(g, t)).max())
    # Observed 1.9e-4, 4.8e-5, 1.2e-5 mV at 5 nS; 7.1e-3, 1.8e-3, 4.7e-4 at 50 nS.
    assert errors[0] < tolerance
    assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5  # second order
    coupled = coupled_exactly(g, t)
    assert np.abs(coupled[-1, 0] - coupled[-1, 1]) < 0.9 * np.abs(coupled_exactly(0.0, t)[-1] @ [1, -1])


@pytest.mark.parametrize(("g", "tolerance"), [(5.0, 5e-4), (50.0, 1e-2)])
def test_gap_junctions_agree_with_nests_waveform_relaxation(g, tolerance):
    # NEST iterates the coupled cells to 1e-4 and lands within 1.5e-5 mV of the exact solution, so the
    # difference is sparx's second-order error. Observed 1.9e-4 mV at 5 nS and 7.1e-3 at 50 nS. Without
    # waveform relaxation NEST holds the partner a step behind and is 0.24 and 2.1 mV off.
    nest = GAP[f"g{g:g}/wfr"]
    np.testing.assert_allclose(two_coupled_cells(g, 0.1, len(nest)), nest, rtol=0, atol=tolerance)


def test_a_junction_between_two_populations_is_the_junction_within_one():
    apart = two_coupled_cells(5.0, 0.1, 50, separate=True)
    np.testing.assert_array_equal(apart, two_coupled_cells(5.0, 0.1, 50))


def test_a_gap_junction_refuses_a_model_without_a_physical_membrane():
    with pytest.raises(ValueError, match="cannot take a gap junction"):
        Network((Population("a", 2, LIFCell(0.9)),), junctions=(GapJunction("a", "a", AllToAll()),))
    with pytest.raises(ValueError, match="no membrane voltage"):
        Network((Population("a", 2, RateCell(0.9)),), junctions=(GapJunction("a", "a", AllToAll()),))


