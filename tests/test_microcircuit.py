"""Potjans and Diesmann's cortical microcircuit against NEST, on one network and over its ensemble.

`tools/make_microcircuit_fixtures.py` runs NEST 3.10 two ways at a fifth of
the model's neurons and inputs: on the network sparx draws, which sparx in
float64 reproduces spike for spike, and as the reference implementation
(INM-6/microcircuit-PD14-model) draws it at 15 seeds. Over the ensemble a
chaotic network matches in distribution, not spike for spike, so its
check is the reference's own (Dasbach et al. 2021): the Kolmogorov-Smirnov
distance of each population's rates, interspike irregularity and pairwise
correlations, with the distances between NEST's seeds as the scale of
agreement.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import ks_2samp, truncnorm

from sparx.graph import FixedTotalNumber, SpikeRaster, microcircuit, simulate
from sparx.graph.models import MICROCIRCUIT_POPULATIONS
from sparx.spiketrains import cv_isi, rates_hz

NEST = np.load(Path(__file__).parent / "fixtures" / "microcircuit.npz")
SCALE, START, STOP = float(NEST["meta/scale"]), float(NEST["meta/start"]), float(NEST["meta/stop"])
SAMPLE, BIN, SEEDS = int(NEST["meta/sample"]), float(NEST["meta/bin"]), [int(s) for s in NEST["meta/seeds"]]
DT = 0.1


def correlations(spikes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Spike-count correlation coefficients of every pair of `SAMPLE` neurons, in bins of `BIN` ms."""
    sample = rng.choice(spikes.shape[1], SAMPLE, replace=False)
    per = round(BIN / DT)
    counts = spikes[: spikes.shape[0] // per * per, sample].reshape(-1, per, SAMPLE).sum(1).T
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.corrcoef(counts)
    return cc[np.triu_indices(SAMPLE, 1)]


def distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(ks_2samp(a, b).statistic)


@pytest.fixture(scope="module")
def built():
    network = microcircuit(SCALE, SCALE)
    return network, network.init(jax.random.key(0))


@pytest.fixture(scope="module")
def sparx_statistics(built) -> dict[str, dict[str, np.ndarray]]:
    network, variables = built
    monitors = {name: SpikeRaster(name) for name in MICROCIRCUIT_POPULATIONS}
    result = simulate(network, variables, duration=STOP, key=jax.random.key(1), monitors=monitors,
                      chunk=100.0)
    rng = np.random.default_rng(0)
    out = {}
    for name in MICROCIRCUIT_POPULATIONS:
        window = np.asarray(result.records[name][round(START / DT):])
        out[name] = {"rates": rates_hz(window, DT), "cvs": cv_isi(window, minimum=2),
                     "ccs": correlations(window, rng)}
    return out


def test_the_network_is_the_one_the_reference_derives(built):
    # Neurons, synapse counts and currents as the reference derives them at this scale; weights and
    # delays drawn from its distributions.
    network, variables = built
    populations = {p.name: p for p in network.populations}
    np.testing.assert_array_equal([populations[name].size for name in MICROCIRCUIT_POPULATIONS],
                                  NEST["derived/neurons"])
    current = [float(populations[name].neuron.i_e) for name in MICROCIRCUIT_POPULATIONS]
    np.testing.assert_allclose(current, NEST["derived/current"], rtol=1e-12)  # observed 0
    connections = network.connections(variables)
    counted = np.zeros((8, 8), int)
    for p in network.projections:
        target, source = MICROCIRCUIT_POPULATIONS.index(p.post), MICROCIRCUIT_POPULATIONS.index(p.pre)
        assert isinstance(p.connectivity, FixedTotalNumber)
        edges = connections[p.key]
        count = len(edges.pre)
        counted[target, source] = count
        mean = NEST["derived/weight"][target, source]
        assert np.all(np.sign(edges.weight) == np.sign(mean))
        # 10% spread around the mean, drawn again where the sign flips (one in 10^23).
        assert abs(edges.weight.mean() - mean) <= 4 * 0.1 * abs(mean) / np.sqrt(count) + 1e-6 * abs(mean)
        # Normal around 1.5 or 0.75 ms, 50% spread, drawn again below half a step, rounded to steps.
        delay = 1.5 if source % 2 == 0 else 0.75
        expected = truncnorm.mean((DT / 2 - delay) / (delay / 2), np.inf, loc=delay, scale=delay / 2)
        assert abs(edges.delay.mean() - expected) <= 4 * (delay / 2) / np.sqrt(count) + 0.005
        assert edges.delay.min() >= DT and np.allclose(edges.delay / DT, np.round(edges.delay / DT))
    np.testing.assert_array_equal(counted, NEST["derived/synapses"])


def test_poisson_background_input_is_the_one_the_reference_derives():
    # In place of the constant current, 8 Hz from each external input, and the current that makes up what
    # fewer, stronger inputs lose.
    network = microcircuit(SCALE, SCALE, background="poisson")
    current = [float(p.neuron.i_e) for p in network.populations]
    np.testing.assert_allclose(current, NEST["poisson/current"], rtol=1e-12)  # observed 1.4e-15
    assert [source.target for source in network.inputs] == list(MICROCIRCUIT_POPULATIONS)
    assert [source.count for source in network.inputs] == NEST["poisson/external"].tolist()
    assert all(source.rate == 8.0 for source in network.inputs)
    np.testing.assert_allclose([source.weight for source in network.inputs], NEST["poisson/weight_external"],
                               rtol=1e-12)  # observed 0


def test_nest_and_sparx_spike_alike_on_the_network_sparx_draws(built):
    # NEST ran the exported edges, weights, delays, voltages and currents; in float64 every spike agrees.
    exact = float(NEST["meta/exact"])
    with jax.enable_x64(new_val=True):
        network = dataclasses.replace(built[0], dtype=jnp.float64)
        monitors = {name: SpikeRaster(name) for name in MICROCIRCUIT_POPULATIONS}
        result = simulate(network, network.init(jax.random.key(0)), duration=exact, monitors=monitors,
                          chunk=exact)
    total = 0
    for i, name in enumerate(MICROCIRCUIT_POPULATIONS):
        steps, neurons = np.nonzero(np.asarray(result.records[name]))
        ours = sorted(zip(steps.tolist(), neurons.tolist(), strict=True))
        theirs = sorted(zip(NEST[f"exact/{i}/steps"].tolist(), NEST[f"exact/{i}/neurons"].tolist(),
                            strict=True))
        assert ours == theirs, name
        total += len(ours)
    assert total > 10_000


@pytest.mark.parametrize("name", MICROCIRCUIT_POPULATIONS)
def test_each_population_fires_at_nests_rate(sparx_statistics, name):
    nest = np.array([NEST[f"{seed}/{name}/mean"] for seed in SEEDS], np.float64)
    rate = sparx_statistics[name]["rates"].mean()
    assert abs(rate - nest.mean()) <= 4 * nest.std(), (rate, nest.mean(), nest.std())


@pytest.mark.parametrize("statistic", ["rates", "cvs", "ccs"])
@pytest.mark.parametrize("name", MICROCIRCUIT_POPULATIONS)
def test_each_distribution_is_as_close_to_nests_as_nests_seeds_are_to_each_other(sparx_statistics, name,
                                                                               statistic):
    runs = [NEST[f"{seed}/{name}/{statistic}"].astype(np.float64) for seed in SEEDS]
    between = [distance(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
    ours = np.mean([distance(sparx_statistics[name][statistic], run) for run in runs])
    assert ours <= np.mean(between) + 3 * np.std(between), (ours, np.mean(between), np.std(between))
