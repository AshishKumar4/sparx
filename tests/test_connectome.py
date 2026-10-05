"""Connectomes: the tables, and Shiu et al.'s whole-brain model against their published runs."""


import jax
import numpy as np
import pytest

from sparx.dynamics import LIF, Exponential, Receptor
from sparx.graph import (
    FixedProbability,
    Network,
    PoissonInput,
    Population,
    Projection,
    SpikeCounts,
    Spikes,
    SpikeTimes,
    simulate,
)
from sparx.graph.connectome import Connectome

DT = 0.1


def test_connectome_indexes_ids_and_silences_neurons():
    brain = Connectome(ids=np.array([50, 10, 40, 30]), pre=np.array([0, 1, 1, 3]),
                       post=np.array([1, 2, 3, 0]), synapses=np.array([4, -2, 7, 1]))
    np.testing.assert_array_equal(brain.index([40, 50, 30]), [2, 0, 3])
    with pytest.raises(KeyError, match="20"):
        brain.index([20])
    quiet = brain.without([1])
    np.testing.assert_array_equal(quiet.pre, [0, 3])
    np.testing.assert_array_equal(quiet.synapses, [4, 1])


def test_spike_counts_and_times_agree_with_full_spike_records():
    network = Network((Population("a", 300, LIF(), {"ex": Receptor(Exponential(5.0))}),),
                      (Projection("a", "a", FixedProbability(0.05), weight=30.0, delay=1.0),),
                      inputs=(PoissonInput("a", rate=1000.0, weight=60.0, count=5),), dt=DT)
    result = simulate(network, network.init(jax.random.key(0)), duration=60.0, key=jax.random.key(1),
                      monitors=(Spikes("a"), SpikeCounts("a"), SpikeTimes("a", capacity=300)), chunk=25.0)
    spikes, counts, times = result.records
    np.testing.assert_array_equal(counts, spikes.sum(0))
    rebuilt = np.zeros_like(spikes)
    steps, slots = np.nonzero(times >= 0)
    rebuilt[steps, times[steps, slots]] = True
    np.testing.assert_array_equal(rebuilt, spikes)
    assert counts.sum() > 300
