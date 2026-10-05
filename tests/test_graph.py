"""Networks: structure, timing, and recurrent networks against NEST spike for spike."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.dynamics import LIF, Delta, Exponential, Receptor
from sparx.graph import (
    AllToAll,
    CurrentInput,
    FixedInDegree,
    FixedProbability,
    FromEdges,
    Network,
    OneToOne,
    PoissonInput,
    Population,
    Projection,
    Spikes,
    StateMonitor,
    simulate,
)
from sparx.graph.analysis import cv_isi, firing_rates, population_fano
from sparx.graph.models import brunel

NEST = np.load(Path(__file__).parent / "fixtures" / "nest.npz")
DT = float(NEST["meta/dt"])


def nest_network(model):
    case = {key.split("/", 2)[2]: NEST[key] for key in NEST.files if key.startswith(f"network/{model}/")}
    p = {key[len("param/"):]: float(v) for key, v in case.items() if key.startswith("param/")}
    if model == "iaf_cond_exp":
        neuron = LIF(tau_m=p["C_m"] / p["g_L"], c_m=p["C_m"], e_l=p["E_L"], v_th=p["V_th"],
                     v_reset=p["V_reset"], t_ref=p["t_ref"], reversal={"ex": p["E_ex"], "in": p["E_in"]})
        receptors = {"ex": Receptor(Exponential(p["tau_syn_ex"]), "conductance"),
                     "in": Receptor(Exponential(p["tau_syn_in"]), "conductance")}
    else:
        neuron = LIF(tau_m=p["tau_m"], c_m=p["C_m"], e_l=p["E_L"], v_th=p["V_th"], v_reset=p["V_reset"],
                     t_ref=p["t_ref"])
        if model == "iaf_psc_delta":
            receptors = {"ex": Receptor(Delta()), "in": Receptor(Delta())}
        else:
            receptors = {"ex": Receptor(Exponential(p["tau_syn_ex"])),
                         "in": Receptor(Exponential(p["tau_syn_in"]))}
    projections = []
    for receptor, sign in (("ex", 1), ("in", -1)):
        mine = np.sign(case["weight"]) == sign
        # NEST's conductance models take inhibitory weights negative and use their size.
        weight = np.abs(case["weight"][mine]) if model == "iaf_cond_exp" else case["weight"][mine]
        projections.append(Projection("n", "n", FromEdges(case["pre"][mine], case["post"][mine]),
                                      weight=weight, delay=case["delay"][mine], receptor=receptor))
    network = Network((Population("n", len(case["current"]), neuron, receptors),), tuple(projections),
                      inputs=(CurrentInput("n", "dc"),), dt=DT, dtype=jnp.float64)
    return network, case


@pytest.mark.parametrize("model", ["iaf_psc_exp", "iaf_psc_delta", "iaf_cond_exp"])
def test_recurrent_network_fires_with_nest_spike_for_spike(model):
    # 60 neurons, 10% connectivity, mixed-sign weights, per-edge delays of
    # 1 to 3 ms, per-neuron currents, 500 ms.
    network, case = nest_network(model)
    steps = len(case["spikes"])
    with jax.enable_x64(new_val=True):
        variables = network.init(jax.random.key(0))
        drive = {"dc": np.broadcast_to(case["current"], (steps, len(case["current"])))}
        (fired,), _ = network.apply(variables, drive, monitors=(Spikes("n"),), mutable=["state"])
    fired = np.asarray(fired, np.float64)
    expected = case["spikes"]
    assert expected.sum() > 300
    if model == "iaf_cond_exp":
        # Conductances held at their step average part from NEST's adaptive
        # solution by 1e-3 mV; in a recurrent network that eventually moves
        # a spike, and the change propagates.
        differ = np.flatnonzero((fired != expected).any(axis=1))
        start = differ[0] if len(differ) else len(fired)
        assert start > 500
        np.testing.assert_array_equal(fired[:start], expected[:start])
        # The first difference is one spike moved by one step.
        moved = np.argwhere(fired[start:start + 2] != expected[start:start + 2])
        assert len(moved) == 2 and moved[0, 1] == moved[1, 1]
        assert abs(fired.sum() / expected.sum() - 1) < 0.05
    else:
        np.testing.assert_array_equal(fired, expected)


def test_a_delay_of_d_steps_lands_at_the_end_of_step_m_plus_d():
    # One neuron driven to fire once; its spike reaches a second neuron's
    # exponential synapse D steps later and moves its membrane from the next step.
    neuron = LIF(t_ref=1000.0)
    pops = (Population("a", 1, neuron, {"ex": Receptor(Exponential(2.0))}),
            Population("b", 1, neuron, {"ex": Receptor(Exponential(2.0))}))
    for delay_steps in (0, 1, 7):
        network = Network(pops, (Projection("a", "b", OneToOne(), weight=100.0, delay=delay_steps * DT),),
                          inputs=(CurrentInput("a", "kick"),), dt=DT)
        variables = network.init(jax.random.key(0))
        drive = {"kick": np.where(np.arange(60) < 5, 1e5, 0.0)[:, None]}
        monitors = (Spikes("a"), StateMonitor("b"))
        (a, v), _ = network.apply(variables, drive, monitors=monitors, mutable=["state"])
        sent = int(np.flatnonzero(np.asarray(a[:, 0]))[0])
        moved = int(np.flatnonzero(np.asarray(v[:, 0]) != neuron.e_l)[0])
        assert moved == sent + delay_steps + 1


def test_delta_synapses_refuse_zero_delays():
    network = Network((Population("a", 2, LIF(), {"ex": Receptor(Delta())}),),
                      (Projection("a", "a", AllToAll(), weight=1.0, delay=0.0),), dt=DT)
    with pytest.raises(ValueError, match="at least 1 step"):
        network.init(jax.random.key(0))


def test_connectivity_rules_draw_what_they_promise():
    rng = np.random.default_rng(0)
    edges = FixedInDegree(30).edges(rng, 200, 100, same=False)
    assert np.all(np.bincount(edges.post, minlength=100) == 30)
    assert len(set(zip(edges.pre.tolist(), edges.post.tolist(), strict=True))) == len(edges)
    edges = FixedInDegree(30).edges(rng, 100, 100, same=True)
    assert not np.any(edges.pre == edges.post)
    edges = FixedProbability(0.1).edges(rng, 300, 300, same=True)
    assert abs(len(edges) / (300 * 299) - 0.1) < 0.01 and not np.any(edges.pre == edges.post)
    assert len(AllToAll().edges(rng, 5, 5, same=True)) == 20


def test_the_same_key_builds_the_same_network():
    network = Network((Population("a", 50, LIF(), {"ex": Receptor(Exponential(2.0))}),),
                      (Projection("a", "a", FixedProbability(0.2),
                                  weight=lambda rng, n: rng.normal(size=n)),), dt=DT)
    one, two = network.init(jax.random.key(3)), network.init(jax.random.key(3))
    jax.tree.map(np.testing.assert_array_equal, one["connectome"], two["connectome"])
    other = network.init(jax.random.key(4))
    assert not np.array_equal(one["connectome"]["weight:a->a:ex"], other["connectome"]["weight:a->a:ex"])


BRUNEL = np.load(Path(__file__).parent / "fixtures" / "brunel.npz")
REGIMES = {"sr": (3.0, 2.0), "ai": (5.0, 2.0), "si_fast": (6.0, 4.0), "si_slow": (4.5, 0.9)}


@pytest.mark.parametrize("regime", list(REGIMES))
def test_brunel_regimes_match_nests_statistics(regime):
    # Chaotic networks cannot match spike for spike: the excitatory rate,
    # interspike irregularity and population synchrony agree with NEST's,
    # on the scale of NEST's own spread over seeds.
    g, eta = REGIMES[regime]
    network = brunel(int(BRUNEL["meta/order"]), g=g, eta=eta)
    result = simulate(network, network.init(jax.random.key(0)), duration=float(BRUNEL["meta/duration"]),
                      key=jax.random.key(1), monitors=(Spikes("e"),), chunk=100.0)
    window = result.records[0][round(float(BRUNEL["meta/skip"]) / DT):]
    rate, cv, fano = firing_rates(window, DT).mean(), cv_isi(window).mean(), population_fano(window, DT)
    nest_stats = BRUNEL[f"{regime}/stats"]
    mean, spread = nest_stats.mean(0), nest_stats.std(0)
    assert abs(rate - mean[0]) <= max(0.03 * mean[0], 4 * spread[0])
    assert abs(cv - mean[1]) <= 0.02 + 4 * spread[1]
    assert nest_stats[:, 2].min() / 2 <= fano <= nest_stats[:, 2].max() * 2


def small_network():
    network = Network((Population("a", 200, LIF(), {"ex": Receptor(Exponential(5.0))}),),
                      (Projection("a", "a", FixedProbability(0.1), weight=20.0, delay=1.5),),
                      inputs=(PoissonInput("a", rate=1000.0, weight=60.0, count=5),), dt=DT)
    return network, network.init(jax.random.key(0))


def test_simulate_does_not_depend_on_the_chunk_length():
    network, variables = small_network()
    runs = [simulate(network, variables, duration=50.0, key=jax.random.key(1), monitors=(Spikes("a"),),
                     chunk=chunk).records[0] for chunk in (50.0, 7.0)]
    np.testing.assert_array_equal(*runs)
    assert runs[0].sum() > 100


def test_a_run_continued_from_its_variables_is_the_unbroken_run():
    network, variables = small_network()
    whole = simulate(network, variables, duration=50.0, key=jax.random.key(1), monitors=(Spikes("a"),))
    first = simulate(network, variables, duration=20.0, key=jax.random.key(1), monitors=(Spikes("a"),))
    second = simulate(network, first.variables, duration=30.0, key=jax.random.key(1), monitors=(Spikes("a"),))
    np.testing.assert_array_equal(np.concatenate([first.records[0], second.records[0]]), whole.records[0])


@pytest.mark.parametrize("mean", [0.05, 0.9, 7.5])
def test_poisson_counts_follow_the_poisson_distribution(mean):
    from scipy import stats

    from sparx.graph.network import _poisson_table

    table = _poisson_table(mean)
    draws = np.sum(np.random.default_rng(0).random((200_000, 1)) > table, axis=1)
    np.testing.assert_allclose(table, stats.poisson.cdf(np.arange(len(table)), mean), rtol=1e-12)
    # Bins expected to hold at least 50 draws, the tail lumped into the last.
    last = int(np.max(np.flatnonzero(stats.poisson.pmf(np.arange(len(table)), mean) * len(draws) >= 50)))
    observed = np.bincount(np.minimum(draws, last), minlength=last + 1)
    expected = np.append(stats.poisson.pmf(np.arange(last), mean), stats.poisson.sf(last - 1, mean))
    expected = expected * len(draws)
    assert stats.chisquare(observed, expected).pvalue > 1e-3


def test_dense_and_edge_projections_deliver_the_same_spikes():
    def network(format):
        return Network((Population("a", 300, LIF(), {"ex": Receptor(Exponential(5.0)),
                                                      "in": Receptor(Exponential(10.0))}),),
                       (Projection("a", "a", FixedInDegree(30, multapses=True), weight=25.0, delay=1.5,
                                   format=format),
                        Projection("a", "a", FixedProbability(0.05), weight=-40.0, delay=0.8, receptor="in",
                                   format=format)),
                       inputs=(PoissonInput("a", rate=1000.0, weight=60.0, count=5),), dt=DT,
                       dtype=jnp.float64)

    with jax.enable_x64(new_val=True):
        runs = []
        for format in ("dense", "edges"):
            net = network(format)
            result = simulate(net, net.init(jax.random.key(0)), duration=100.0, key=jax.random.key(1),
                              monitors=(Spikes("a"),))
            runs.append(result.records[0])
    np.testing.assert_array_equal(*runs)
    assert runs[0].sum() > 500
