"""Connectomes: the tables, and Shiu et al.'s whole-brain model against their published runs."""

import os
from pathlib import Path

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
from sparx.graph.connectome import Connectome, shiu2024

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


SHIU_REPO = Path(os.environ.get("SPARX_SHIU_REPO", Path(__file__).resolve().parents[2] / "ref-shiu"))
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
         720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
         720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]
MN9 = 720575940660219265


@pytest.mark.skipif(not (SHIU_REPO / "2023_03_23_connectivity_630_final.parquet").exists(),
                    reason="needs github.com/philshiu/Drosophila_brain_model at $SPARX_SHIU_REPO "
                           "or ../ref-shiu")
def test_shiu2024_reproduces_their_published_sugar_activation():
    # Their published run: 21 sugar-sensing neurons at 100 Hz on FlyWire
    # v630, 30 trials of 1 s in Brian2. Three trials here (about 90 s on 4
    # CPU cores); ten trials give a rate correlation of 0.9989 and MN9 at
    # 67.1 Hz against their 67.0 +- 6.6 (docs/fidelity.md).
    published = np.load(Path(__file__).parent / "fixtures" / "shiu.npz")
    brain = Connectome.from_shiu(SHIU_REPO / "2023_03_23_completeness_630_final.csv",
                                 SHIU_REPO / "2023_03_23_connectivity_630_final.parquet")
    network = shiu2024(brain, stimuli=[(brain.index(SUGAR), 100.0)])
    variables = network.init(jax.random.key(0))
    trials = 3
    counts = sum(simulate(network, variables, duration=1000.0, key=jax.random.key(trial),
                          monitors=(SpikeCounts("brain"),), chunk=1000.0).records[0]
                 for trial in range(trials))
    rates = counts / trials
    ids, theirs, spread = (published[f"sugarR_100Hz/{k}"] for k in ("ids", "rate", "std"))
    ours = rates[brain.index(ids)]
    assert abs(rates.sum() / theirs.sum() - 1) < 0.05
    active = theirs >= 1.0
    assert np.corrcoef(ours[active], theirs[active])[0, 1] > 0.99
    mn9 = ids == MN9
    assert abs(ours[mn9] - theirs[mn9]) < 3 * spread[mn9] * np.sqrt(1 / 30 + 1 / trials)
