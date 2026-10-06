"""Networks: structure, timing, and recurrent networks against NEST spike for spike."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dew.checkpoints import Checkpoints

from sparx.dynamics import (
    LIF,
    ALIFCell,
    Delta,
    Exponential,
    LICell,
    LIFCell,
    PairSTDP,
    Receptor,
    Serial,
    decay,
    run,
)
from sparx.graph import (
    AllToAll,
    ArrivalInput,
    CurrentInput,
    FixedInDegree,
    FixedProbability,
    FromEdges,
    Network,
    OneToOne,
    PoissonInput,
    Population,
    PopulationRate,
    Projection,
    SpikeCounts,
    SpikeRaster,
    StateMonitor,
    from_record,
    simulate,
)
from sparx.graph.models import brunel, coba, cuba
from sparx.nn import LIF as LayerLIF
from sparx.spiketrains import cv_isi, population_fano, rates_hz

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
        records, _ = network.apply(variables, drive, monitors={"n": SpikeRaster("n")}, mutable=["state"])
    fired = np.asarray(records["n"], np.float64)
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
        network = Network(pops, (Projection("a", "b", OneToOne(), weight=100.0, delay=delay_steps * DT,
                                            receptor="ex"),),
                          inputs=(CurrentInput("a", "kick"),), dt=DT)
        variables = network.init(jax.random.key(0))
        drive = {"kick": np.where(np.arange(60) < 5, 1e5, 0.0)[:, None]}
        monitors = {"a": SpikeRaster("a"), "v": StateMonitor("b")}
        records, _ = network.apply(variables, drive, monitors=monitors, mutable=["state"])
        a, v = records["a"], records["v"]
        sent = int(np.flatnonzero(np.asarray(a[:, 0]))[0])
        moved = int(np.flatnonzero(np.asarray(v[:, 0]) != neuron.e_l)[0])
        assert moved == sent + delay_steps + 1


def test_delta_synapses_refuse_zero_delays():
    network = Network((Population("a", 2, LIF(), {"ex": Receptor(Delta())}),),
                      (Projection("a", "a", AllToAll(), weight=1.0, delay=0.0, receptor="ex"),), dt=DT)
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
                                  weight=lambda rng, n: rng.normal(size=n), receptor="ex"),), dt=DT)
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
                      key=jax.random.key(1), monitors={"e": SpikeRaster("e")}, chunk=100.0)
    window = result.records["e"][round(float(BRUNEL["meta/skip"]) / DT):]
    rate, cv, fano = rates_hz(window, DT).mean(), cv_isi(window).mean(), population_fano(window, DT)
    nest_stats = BRUNEL[f"{regime}/stats"]
    mean, spread = nest_stats.mean(0), nest_stats.std(0)
    assert abs(rate - mean[0]) <= max(0.03 * mean[0], 4 * spread[0])
    assert abs(cv - mean[1]) <= 0.02 + 4 * spread[1]
    assert nest_stats[:, 2].min() / 2 <= fano <= nest_stats[:, 2].max() * 2


def small_network():
    network = Network((Population("a", 200, LIF(), {"ex": Receptor(Exponential(5.0))}),),
                      (Projection("a", "a", FixedProbability(0.1), weight=20.0, delay=1.5, receptor="ex"),),
                      inputs=(PoissonInput("a", rate=1000.0, weight=60.0, receptor="ex", count=5),), dt=DT)
    return network, network.init(jax.random.key(0))


def test_simulate_does_not_depend_on_the_chunk_length():
    network, variables = small_network()
    runs = [simulate(network, variables, duration=50.0, key=jax.random.key(1),
                     monitors={"a": SpikeRaster("a")}, chunk=chunk).records["a"] for chunk in (50.0, 7.0)]
    np.testing.assert_array_equal(*runs)
    assert runs[0].sum() > 100


def test_a_run_continued_from_its_variables_is_the_unbroken_run():
    network, variables = small_network()
    monitors = {"a": SpikeRaster("a")}
    whole = simulate(network, variables, duration=50.0, key=jax.random.key(1), monitors=monitors)
    first = simulate(network, variables, duration=20.0, key=jax.random.key(1), monitors=monitors)
    second = simulate(network, first.variables, duration=30.0, key=jax.random.key(1), monitors=monitors)
    np.testing.assert_array_equal(np.concatenate([first.records["a"], second.records["a"]]),
                                  whole.records["a"])


def test_a_run_resumed_from_its_checkpoint_is_the_unbroken_run(tmp_path):
    # The first call stops after 20 ms, as a preempted run would; the second
    # asks for the whole 50 ms and continues from the checkpoint.
    network, variables = small_network()
    monitors = {"raster": SpikeRaster("a"), "counts": SpikeCounts("a")}
    whole = simulate(network, variables, duration=50.0, key=jax.random.key(1), monitors=monitors, chunk=10.0)
    directory = str(tmp_path / "run")
    simulate(network, variables, duration=20.0, key=jax.random.key(1), monitors=monitors, chunk=10.0,
             checkpoints=Checkpoints(directory))
    resumed = simulate(network, variables, duration=50.0, key=jax.random.key(1), monitors=monitors,
                       chunk=10.0, checkpoints=Checkpoints(directory))
    skipped = round(20.0 / DT)
    assert resumed.start == pytest.approx(20.0)
    np.testing.assert_array_equal(resumed.records["raster"], whole.records["raster"][skipped:])
    np.testing.assert_array_equal(resumed.records["counts"], whole.records["raster"][skipped:].sum(0))
    np.testing.assert_allclose(resumed.times, whole.times[skipped:])  # observed 2.2e-16 relative
    jax.tree.map(np.testing.assert_array_equal, resumed.variables["state"], whole.variables["state"])


def test_a_checkpoint_refuses_a_run_with_another_key(tmp_path):
    network, variables = small_network()
    directory = str(tmp_path / "run")
    simulate(network, variables, duration=5.0, key=jax.random.key(1), checkpoints=Checkpoints(directory))
    with pytest.raises(ValueError, match="another key"):
        simulate(network, variables, duration=10.0, key=jax.random.key(2), checkpoints=Checkpoints(directory))


def test_a_network_record_rebuilds_its_network():
    record = {"name": "brunel", "fields": {"order": 50, "g": 4.5, "eta": 0.9}}
    built, expected = from_record(record), brunel(50, g=4.5, eta=0.9)
    jax.tree.map(np.testing.assert_array_equal, built.init(jax.random.key(0)),
                 expected.init(jax.random.key(0)))
    assert built.populations == expected.populations and built.inputs == expected.inputs


@pytest.mark.parametrize("mean", [0.05, 0.9, 7.5])
def test_poisson_counts_follow_the_poisson_distribution(mean):
    from scipy import stats

    from sparx.graph.network import _poisson_table

    table = _poisson_table(mean)
    draws = np.sum(np.random.default_rng(0).random((200_000, 1)) > table, axis=1)
    # Observed 1.0e-15 relative.
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
                                   receptor="ex", format=format),
                        Projection("a", "a", FixedProbability(0.05), weight=-40.0, delay=0.8, receptor="in",
                                   format=format)),
                       inputs=(PoissonInput("a", rate=1000.0, weight=60.0, receptor="ex", count=5),), dt=DT,
                       dtype=jnp.float64)

    with jax.enable_x64(new_val=True):
        runs = []
        for format in ("dense", "edges"):
            net = network(format)
            result = simulate(net, net.init(jax.random.key(0)), duration=100.0, key=jax.random.key(1),
                              monitors={"a": SpikeRaster("a")})
            runs.append(result.records["a"])
    np.testing.assert_array_equal(*runs)
    assert runs[0].sum() > 500


BENCHMARKS = np.load(Path(__file__).parent / "fixtures" / "benchmarks.npz")


@pytest.mark.parametrize("name", ["cuba", "coba"])
def test_brette_benchmarks_match_brian2s_statistics(name):
    # Brian2's runs over four seeds set the scale: rates of both populations,
    # interspike irregularity and synchrony of the excitatory one.
    network = {"cuba": cuba, "coba": coba}[name]()
    result = simulate(network, network.init(jax.random.key(0)), duration=float(BENCHMARKS["meta/duration"]),
                      monitors={"e": SpikeRaster("e"), "i": SpikeRaster("i")})
    skip = round(float(BENCHMARKS["meta/skip"]) / DT)
    e, i = result.records["e"][skip:], result.records["i"][skip:]
    got = np.array([rates_hz(e, DT).mean(), rates_hz(i, DT).mean(), cv_isi(e).mean(),
                    population_fano(e, DT)])
    brian2 = BENCHMARKS[f"{name}/stats"]
    mean, spread = brian2.mean(0), brian2.std(0)
    for k in (0, 1):
        assert abs(got[k] - mean[k]) <= max(0.05 * mean[k], 4 * spread[k])
    assert abs(got[2] - mean[2]) <= 0.02 + 4 * spread[2]
    assert brian2[:, 3].min() / 2 <= got[3] <= brian2[:, 3].max() * 2


def test_a_layer_stack_is_the_same_network_through_sparx_nn_and_graph():
    # sparx.nn's LIF with a hard reset is a physical LIF at dt = 1 resting
    # and resetting at 0 with threshold 1, driven through delta synapses;
    # each projection adds its one-step delay.
    rng = np.random.default_rng(0)
    steps, sizes, tau = 60, (12, 20, 6), 2.0
    x = (rng.random((steps, sizes[0])) < 0.3).astype(np.float64)
    w1, w2 = rng.normal(0.6, 0.5, sizes[:2]), rng.normal(0.5, 0.6, sizes[1:])
    with jax.enable_x64(new_val=True):
        layer = LayerLIF(tau=tau, reset="zero")
        h = layer.apply({}, jnp.asarray(x @ w1)[:, None])
        out = np.asarray(layer.apply({}, h @ w2)[:, 0])
        h = np.asarray(h[:, 0])

        neuron = LIF(tau_m=tau, c_m=tau, e_l=0.0, v_th=1.0, v_reset=0.0, t_ref=0.0)
        receptors = {"ex": Receptor(Delta())}
        every = np.indices(sizes[:2]).reshape(2, -1)
        network = Network(
            (Population("in", sizes[0], neuron, receptors), Population("h", sizes[1], neuron, receptors),
             Population("out", sizes[2], neuron, receptors)),
            (Projection("in", "h", FromEdges(*every), weight=w1.ravel(), delay=1.0, receptor="ex"),
             Projection("h", "out", FromEdges(*np.indices(sizes[1:]).reshape(2, -1)), weight=w2.ravel(),
                        delay=1.0, receptor="ex")),
            inputs=(ArrivalInput("in", "spikes", "ex"),), dt=1.0, dtype=jnp.float64)
        records, _ = network.apply(
            network.init(jax.random.key(0)), {"spikes": np.concatenate([x * 10, np.zeros((2, sizes[0]))])},
            monitors={name: SpikeRaster(name) for name in ("in", "h", "out")}, mutable=["state"])
        inputs, hidden, output = records["in"], records["h"], records["out"]
    np.testing.assert_array_equal(np.asarray(inputs[:steps]), x > 0)
    np.testing.assert_array_equal(np.asarray(hidden[1:steps + 1]), h > 0)
    np.testing.assert_array_equal(np.asarray(output[2:]), out > 0)
    assert h.sum() > 50 and out.sum() > 20


def test_a_population_of_a_dimensionless_model_is_that_model_on_its_delta_inputs():
    # An ALIF population hosted by a network, driven through a delta
    # receptor, fires as the model run alone on the same jumps.
    steps, size = 80, 9
    drive = np.random.default_rng(3).normal(0.4, 0.9, (steps, size))
    alif = ALIFCell(decay(20.0), decay(200.0), beta=0.3, refractory=2)
    network = Network((Population("a", size, alif, {"ex": Receptor(Delta())}),),
                      inputs=(ArrivalInput("a", "drive", "ex"),), dt=1.0)
    records, _ = network.apply(network.init(jax.random.key(0)), {"drive": drive},
                               monitors={"a": SpikeRaster("a")}, mutable=["state"])
    fired = records["a"]
    expected = run(alif, jnp.asarray(drive, jnp.float32))[0].value
    np.testing.assert_array_equal(np.asarray(fired), np.asarray(expected) > 0)
    assert 20 < int(expected.sum()) < steps * size // 2  # it fires, and adaptation holds it back


def test_a_dimensionless_population_refuses_a_current():
    with pytest.raises(ValueError, match="jump"):
        Network((Population("a", 3, ALIFCell(0.9, 0.99)),), inputs=(CurrentInput("a", "dc"),), dt=1.0)


BRIAN2 = np.load(Path(__file__).parent / "fixtures" / "brian2.npz")


def shiu_network(format):
    case = {k[len("shiu/"):]: BRIAN2[k] for k in BRIAN2.files if k.startswith("shiu/")}
    size = case["spikes"].shape[1]
    stimulated = np.unique(case["stim_neuron"])
    t_ref = np.full(size, 2.1)  # Brian2's 2.2 ms (it counts one step less)
    t_ref[stimulated] = 0.0
    # g_L = C / tau_m = 1 nS, so a synaptic current in pA is Brian2's g in mV.
    neuron = LIF(tau_m=20.0, c_m=20.0, e_l=-52.0, v_th=-45.0, v_reset=-52.0, t_ref=jnp.asarray(t_ref))
    population = Population("n", size, neuron, {"syn": Receptor(Exponential(5.0)),
                                                "drive": Receptor(Delta(after_threshold=True))},
                            reset_synapses=True, freeze_synapses=True)
    projection = Projection("n", "n", FromEdges(case["pre"], case["post"]), weight=case["counts"] * 0.275,
                            delay=1.8, receptor="syn", format=format, capacity=64)
    network = Network((population,), (projection,), inputs=(ArrivalInput("n", "stim", "drive"),), dt=DT,
                      dtype=jnp.float64)
    drive = np.zeros_like(case["spikes"])
    np.add.at(drive, (case["stim_step"], case["stim_neuron"]), 68.75)
    return network, drive, case["spikes"]


@pytest.mark.parametrize("format", ["edges", "events"])
def test_shius_neuron_model_fires_with_brian2_spike_for_spike(format):
    network, drive, expected = shiu_network(format)
    with jax.enable_x64(new_val=True):
        result = simulate(network, network.init(jax.random.key(0)), duration=len(drive) * DT,
                          drive={"stim": drive}, monitors={"n": SpikeRaster("n")})
    np.testing.assert_array_equal(result.records["n"], expected > 0)
    assert expected[:, 10:].sum() > 40  # the network, not only the stimulated neurons, fires


def test_event_projections_raise_when_over_capacity():
    network, drive, _ = shiu_network("events")
    network = network.clone(projections=(dataclasses.replace(network.projections[0], capacity=2),))
    with jax.enable_x64(new_val=True), pytest.raises(RuntimeError, match="capacity"):
        simulate(network, network.init(jax.random.key(0)), duration=len(drive) * DT, drive={"stim": drive})


def test_population_rate_is_the_raster_mean_in_hz():
    network, variables = small_network()
    result = simulate(network, variables, duration=30.0, key=jax.random.key(1),
                      monitors={"raster": SpikeRaster("a"), "rate": PopulationRate("a")})
    raster, rate = result.records["raster"], result.records["rate"]
    np.testing.assert_allclose(rate, raster.mean(1) * 1000.0 / DT, rtol=1e-6)
    np.testing.assert_allclose(rate.mean(), rates_hz(raster, DT).mean(), rtol=1e-6)
    assert rate.mean() > 1.0


def test_a_conductance_receptor_without_a_reversal_potential_is_refused_when_built():
    receptors = {"ex": Receptor(Exponential(5.0), "conductance")}
    with pytest.raises(ValueError, match=r"population 'a': conductance receptor 'ex'.*'ampa', 'nmda'"):
        Network((Population("a", 3, LIF(), receptors),))
    Network((Population("a", 3, LIF(reversal={"ex": 0.0}), receptors),))  # named, so accepted


def test_a_projection_onto_a_missing_receptor_names_the_ones_there_are():
    population = Population("a", 3, LIF(), {"ampa": Receptor(Exponential(5.0))})
    with pytest.raises(ValueError, match=r"receptor 'ex', which population 'a' lacks; it has 'ampa'"):
        Network((population,), (Projection("a", "a", AllToAll(), receptor="ex"),))
    with pytest.raises(ValueError, match=r"receptor 'gaba_a', which population 'a' lacks"):
        Network((population,), inputs=(PoissonInput("a", rate=10.0, weight=1.0, receptor="gaba_a"),))


@pytest.mark.parametrize("kind", ["current", "conductance"])
def test_a_dimensionless_population_refuses_a_kinetic_receptor_when_built(kind):
    receptors = {"ex": Receptor(Exponential(5.0), kind)}
    with pytest.raises(ValueError, match=r"population 'a'") as refused:
        Network((Population("a", 3, ALIFCell(0.9, 0.99), receptors),), dt=1.0)
    assert "'ex'" in str(refused.value) and "ALIFCell" in str(refused.value)


def test_a_time_off_the_step_grid_is_refused():
    network = Network((Population("a", 2, LIF(), {"ex": Receptor(Exponential(2.0))}),),
                      (Projection("a", "a", AllToAll(), delay=0.15, receptor="ex"),), dt=0.1)
    with pytest.raises(ValueError, match=r"delay must be a whole number of steps of 0.1 ms; \[0.15\]"):
        network.init(jax.random.key(0))
    network, variables = small_network()
    with pytest.raises(ValueError, match="duration must be a whole number"):
        simulate(network, variables, duration=1.05)
    with pytest.raises(ValueError, match="chunk must be a whole number"):
        simulate(network, variables, duration=1.0, chunk=0.25)


def test_poisson_inputs_ask_for_their_key():
    network, variables = small_network()
    with pytest.raises(ValueError, match=r"rngs=.'noise': key."):
        network.apply(variables, steps=3, mutable=["state"])


def test_initial_state_is_drawn_per_field_and_receptor():
    neuron = LIF(tau_m=20.0, e_l=-60.0)
    receptors = {"ampa": Receptor(Exponential(5.0), "conductance")}
    initial = {"v": lambda rng, n: rng.uniform(-60.0, -50.0, n), "ampa": np.arange(4.0)}
    network = Network((Population("a", 4, neuron, receptors, initial=initial),))
    state = network.init(jax.random.key(0))["state"]["network"]["populations"]["a"]["point_neuron"]
    v = np.asarray(state.neuron.v)
    assert v.dtype == np.float32 and np.all((v >= -60.0) & (v < -50.0)) and len(np.unique(v)) == 4
    np.testing.assert_array_equal(state.synapses["ampa"], np.arange(4.0))
    np.testing.assert_array_equal(state.neuron.refractory, 0.0)  # what it does not name stays at rest
    with pytest.raises(ValueError, match=r"initial 'u' is neither.*'v'.*'ampa'"):
        Network((Population("a", 4, neuron, receptors, initial={"u": 0.0}),))


def test_a_state_monitor_records_the_neurons_it_names_and_reads_a_serial_models_voltage():
    network, variables = small_network()
    monitors = {"all": StateMonitor("a"), "some": StateMonitor("a", neurons=(3, 0, 7))}
    result = simulate(network, variables, duration=10.0, key=jax.random.key(1), monitors=monitors)
    assert result.records["some"].shape == (100, 3)
    np.testing.assert_array_equal(result.records["some"], result.records["all"][:, [3, 0, 7]])

    # A Serial model fires on its second model's membrane, which the default reads.
    synaptic = Serial(LICell(decay(5.0)), LIFCell(decay(20.0)))
    serial = Network((Population("a", 2, synaptic, {"ex": Receptor(Delta())}),),
                     inputs=(ArrivalInput("a", "drive", "ex"),), dt=1.0)
    drive = np.full((6, 2), 0.3)
    records, _ = serial.apply(serial.init(jax.random.key(0)), {"drive": drive},
                              monitors={"v": StateMonitor("a")}, mutable=["state"])
    final = run(synaptic, jnp.asarray(drive, jnp.float32))[1]
    np.testing.assert_allclose(records["v"][-1], final[1].v, rtol=1e-6)


def test_connections_read_each_storage_back_as_the_edges_it_was_given():
    pre, post = np.array([2, 0, 1, 0]), np.array([1, 2, 0, 1])
    weight, delay = np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.2, 0.1, 0.3, 0.1])
    order = np.lexsort((post, pre))
    receptors = {"ex": Receptor(Exponential(2.0)), "in": Receptor(Exponential(2.0))}
    for format in ("edges", "dense", "events"):
        per_edge = format == "edges"
        network = Network((Population("a", 3, LIF(), receptors),), (
            Projection("a", "a", FromEdges(pre, post), weight=weight, delay=delay if per_edge else 0.1,
                       receptor="ex", format=format),
            Projection("a", "a", FromEdges(pre, post), weight=-weight, receptor="in", trainable=True)),
            dt=0.1)
        connections = network.connections(network.init(jax.random.key(0)))
        got = connections["a->a:ex"]
        np.testing.assert_array_equal(got.pre, pre[order])
        np.testing.assert_array_equal(got.post, post[order])
        np.testing.assert_array_equal(got.weight, weight[order])
        np.testing.assert_allclose(got.delay, delay[order] if per_edge else 0.1)
        np.testing.assert_array_equal(connections["a->a:in"].weight, -weight[order])


def test_connections_read_plastic_weights_as_learned():
    network = Network((Population("a", 40, LIF(), {"ex": Receptor(Exponential(5.0))}),),
                      (Projection("a", "a", FixedProbability(0.2), weight=20.0, delay=1.0, receptor="ex",
                                  plasticity=PairSTDP(lambda_=0.05)),),
                      inputs=(PoissonInput("a", rate=1000.0, weight=60.0, receptor="ex", count=5),), dt=DT)
    variables = network.init(jax.random.key(0))
    before = network.connections(variables)["a->a:ex"]
    result = simulate(network, variables, duration=50.0, key=jax.random.key(1))
    learned = network.connections(result.variables)["a->a:ex"]
    np.testing.assert_array_equal(learned.pre, before.pre)
    np.testing.assert_array_equal(learned.post, before.post)
    np.testing.assert_array_equal(before.weight, 20.0)
    plastic = result.variables["state"]["network"]["plastic"]["a->a:ex"]["weight"]
    np.testing.assert_array_equal(np.sort(learned.weight), np.sort(np.asarray(plastic)))
    assert not np.allclose(learned.weight, 20.0)
