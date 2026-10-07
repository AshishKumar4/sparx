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
    SpikeRaster,
    SpikeTimes,
    from_record,
    simulate,
)
from sparx.graph.connectome import FLYWIRE_630_MEDIAN_INPUTS, Connectome, matched_w_syn, shiu2024

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
                      (Projection("a", "a", FixedProbability(0.05), weight=30.0, delay=1.0, receptor="ex"),),
                      inputs=(PoissonInput("a", rate=1000.0, weight=60.0, receptor="ex", count=5),), dt=DT)
    result = simulate(network, network.init(jax.random.key(0)), duration=60.0, key=jax.random.key(1),
                      monitors={"spikes": SpikeRaster("a"), "counts": SpikeCounts("a"),
                                "times": SpikeTimes("a", capacity=300)},
                      chunk=25.0)
    spikes, counts, times = (result.records[name] for name in ("spikes", "counts", "times"))
    np.testing.assert_array_equal(counts, spikes.sum(0))
    rebuilt = np.zeros_like(spikes)
    steps, slots = np.nonzero(times >= 0)
    rebuilt[steps, times[steps, slots]] = True
    np.testing.assert_array_equal(rebuilt, spikes)
    assert counts.sum() > 300


def test_a_model_on_a_connectome_rebuilds_from_its_record(tmp_path):
    # Shiu et al.'s tables in miniature, named in the record by their reader.
    pa = pytest.importorskip("pyarrow")
    import pyarrow.csv
    import pyarrow.parquet

    pyarrow.csv.write_csv(pa.table({"id": [11, 12, 13, 14]}), tmp_path / "completeness.csv")
    edges = {"Presynaptic_Index": [0, 1, 2, 3], "Postsynaptic_Index": [1, 2, 3, 0],
             "Excitatory x Connectivity": [5, -3, 8, 2]}
    pyarrow.parquet.write_table(pa.table(edges), tmp_path / "connectivity.parquet")
    completeness, connectivity = str(tmp_path / "completeness.csv"), str(tmp_path / "connectivity.parquet")
    reader = {"class": "sparx.graph.connectome:Connectome.from_shiu",
              "fields": {"completeness": completeness, "connectivity": connectivity}}
    record = {"class": "shiu2024",
              "fields": {"connectome": reader, "stimuli": [[[0, 2], 150.0]], "silenced": [3], "w_syn": 0.2}}
    built = from_record(record)
    expected = shiu2024(Connectome.from_shiu(completeness, connectivity), stimuli=[([0, 2], 150.0)],
                        silenced=[3], w_syn=0.2)
    jax.tree.map(np.testing.assert_array_equal, built.init(jax.random.key(0)),
                 expected.init(jax.random.key(0)))
    assert built.inputs == expected.inputs


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
                          monitors={"brain": SpikeCounts("brain")}, chunk=1000.0).records["brain"]
                 for trial in range(trials))
    rates = counts / trials
    ids, theirs, spread = (published[f"sugarR_100Hz/{k}"] for k in ("ids", "rate", "std"))
    ours = rates[brain.index(ids)]
    assert abs(rates.sum() / theirs.sum() - 1) < 0.05
    active = theirs >= 1.0
    assert np.corrcoef(ours[active], theirs[active])[0, 1] > 0.99
    mn9 = ids == MN9
    assert abs(ours[mn9] - theirs[mn9]) < 3 * spread[mn9] * np.sqrt(1 / 30 + 1 / trials)


def test_malecns_reader_keeps_neurons_and_signs_them_by_transmitter(tmp_path):
    pa = pytest.importorskip("pyarrow")
    feather = pytest.importorskip("pyarrow.feather")
    feather.write_feather(pa.table({"bodyId": [30, 10, 20, 40, 50],
                                    "status": ["Traced", "Traced", "Anchor", "Orphan", None]}),
                          tmp_path / "annotations.feather")
    feather.write_feather(pa.table({"body": [10, 20, 30, 40],
                                    "consensus_nt": ["gaba", "unclear", "acetylcholine", "gaba"],
                                    "predicted_nt": ["gaba", "glutamate", "acetylcholine", "gaba"]}),
                          tmp_path / "nt.feather")
    feather.write_feather(pa.table({"body_pre": [10, 20, 30, 40, 10], "body_post": [20, 30, 10, 10, 50],
                                    "weight": [5, 3, 7, 9, 2]}), tmp_path / "weights.feather")
    brain = Connectome.from_malecns(tmp_path / "annotations.feather", tmp_path / "nt.feather",
                                    tmp_path / "weights.feather")
    np.testing.assert_array_equal(brain.ids, [10, 20, 30])  # orphans and unlabelled segments dropped
    edges = sorted(zip(brain.ids[brain.pre].tolist(), brain.ids[brain.post].tolist(), brain.synapses.tolist(),
                       strict=True))
    # GABA inhibits; "unclear" falls back to the prediction (glutamate, inhibitory); ACh excites.
    assert edges == [(10, 20, -5), (20, 30, -3), (30, 10, 7)]


def test_matched_w_syn_scales_the_weight_by_the_median_neurons_input():
    # Neuron 0 gets 4 + 8 = 12 synapses, 1 gets 3, 2 gets 6 (signs ignored): median 6.
    brain = Connectome(np.arange(3), np.array([1, 2, 0, 0]), np.array([0, 0, 1, 2]), np.array([4, -8, 3, -6]))
    np.testing.assert_array_equal(brain.inputs(), [12, 3, 6])
    assert matched_w_syn(brain) == pytest.approx(0.275 * FLYWIRE_630_MEDIAN_INPUTS / 6)
    assert matched_w_syn(brain, w_syn=1.0, reference=6.0) == 1.0


MALECNS = Path(os.environ.get("SPARX_MALECNS", Path(__file__).resolve().parents[2] / "data" / "malecns"))
MALECNS_TABLES = ("body-annotations-male-cns-v0.9-minconf-0.5.feather",
                  "body-neurotransmitters-male-cns-v0.9.feather",
                  "connectome-weights-male-cns-v0.9-minconf-0.5.feather")
MALECNS_MN9 = 10331  # rootSide L; its partner 16949 is rootSide R
SUGAR_TYPES = ("LB3b", "LB3c", "LB3d", "LB4b")  # the cell types of Shiu et al.'s 21 FlyWire sugar neurons


@pytest.mark.skipif(not all((MALECNS / table).exists() for table in MALECNS_TABLES),
                    reason="needs the male CNS v0.9 release tables (gs://flyem-male-cns/v0.9) "
                           "at $SPARX_MALECNS or ../data/malecns")
def test_shiu2024_on_the_male_cns_activates_mn9_from_sugar_neurons_at_the_matched_weight():
    # Shiu et al.'s sugar experiment on the male CNS: the gustatory neurons
    # of their 21 FlyWire sugar neurons' types, on one side, at 100 Hz.
    # FlyWire's run recruits about 400 neurons and drives MN9 at 67 Hz. At
    # the matched weight (0.163 mV) the male CNS recruits about 670 and MN9
    # fires at about 80 Hz; at FlyWire's 0.275 mV it recruits 18,000.
    feather = pytest.importorskip("pyarrow.feather")
    brain = Connectome.from_malecns(*(MALECNS / table for table in MALECNS_TABLES))
    bodies = feather.read_table(MALECNS / MALECNS_TABLES[0], columns=["bodyId", "type", "rootSide"])
    types = np.asarray(bodies.column("type").to_pylist(), object)
    side = np.asarray(bodies.column("rootSide").to_pylist(), object)
    sugar = np.asarray(bodies.column("bodyId").to_numpy())[np.isin(types, SUGAR_TYPES) & (side == "R")]
    w_syn = matched_w_syn(brain)
    assert w_syn == pytest.approx(0.163, abs=0.001)
    network = shiu2024(brain, stimuli=[(brain.index(sugar), 100.0)], w_syn=w_syn)
    rates = simulate(network, network.init(jax.random.key(0)), duration=1000.0, key=jax.random.key(0),
                     monitors={"brain": SpikeCounts("brain")}, chunk=1000.0).records["brain"]
    assert rates[brain.index([MALECNS_MN9])[0]] > 40.0
    assert 200 < np.count_nonzero(rates) < 2000
