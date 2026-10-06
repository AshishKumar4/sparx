"""Graded populations in networks."""

import jax
import pytest

from sparx.dynamics import LIF, Delta, LICell, LIFCell, PairSTDP, PointNeuron, Receptor, TsodyksMarkram
from sparx.graph import AllToAll, Network, Population, Projection, SpikeRaster


def test_a_graded_population_refuses_event_delivery_and_spike_driven_options():
    graded = Population("g", 4, LICell(0.9))
    target = Population("t", 4, LIFCell(0.9), {"in": Receptor(Delta())})

    def build(**options):
        projection = Projection("g", "t", AllToAll(), receptor="in", **options)
        return Network((graded, target), (projection,), dt=1.0)

    with pytest.raises(ValueError, match="event delivery"):
        build(format="events")
    with pytest.raises(ValueError, match="short_term acts on spikes"):
        build(short_term=TsodyksMarkram())
    with pytest.raises(ValueError, match="plasticity acts on spikes"):
        build(plasticity=PairSTDP())
    network = build(format="edges")
    with pytest.raises(ValueError, match="OutputTrace"):
        network.apply(network.init(jax.random.key(0)), steps=2, monitors={"r": SpikeRaster("g")},
                      mutable=["state"])
    with pytest.raises(ValueError, match="never spikes"):
        PointNeuron(LICell(0.9), {}, reset_synapses=True)
    assert graded.graded and not Population("s", 1, LIF()).graded
