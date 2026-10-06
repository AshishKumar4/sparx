"""Graded transmission in networks."""


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
    LIFCell,
    PairSTDP,
    PointNeuron,
    RateCell,
    Receptor,
    SynapticInput,
    TsodyksMarkram,
    run,
)
from sparx.graph import (
    AllToAll,
    ArrivalInput,
    CurrentInput,
    FromEdges,
    Network,
    OutputTrace,
    Population,
    Projection,
    SpikeRaster,
    StateMonitor,
)


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


