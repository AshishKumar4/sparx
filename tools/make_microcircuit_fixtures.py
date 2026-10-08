"""Write NEST's runs of the cortical microcircuit, for sparx's comparison, to `tests/fixtures`.

Two checks, in two steps. First, in sparx's environment, `export` writes the
network `sparx.graph.models.microcircuit` draws at `SCALE` from key 0, in
float64: every projection's edges, weights and delays, and each neuron's
initial voltage and current. Then, in an environment with NEST, `nest`

- builds that network in NEST from the exported arrays (`iaf_psc_exp`,
  connected edge by edge) and saves its spikes over `EXACT` ms, which sparx
  in float64 reproduces spike for spike;
- runs the PyNEST implementation of Potjans and Diesmann's (2014) model
  from INM-6/microcircuit-PD14-model, unchanged, at `SCALE` of its neurons
  and of their inputs, with its default constant background current, for
  `SEEDS`, and saves per population the mean rate over `[START, STOP]` ms
  and, as the reference's `helpers.py` computes them, the rates and the CVs
  of interspike intervals (neurons with more than two spikes) of up to
  `KEPT` neurons and the spike-count correlations of `SAMPLE` neurons in
  bins of `BIN` ms;
- saves the parameters the reference derives at `SCALE`: neurons and
  synapses per population, mean weights and each population's current,
  and the currents, external in-degrees and weight with Poisson background
  input in place of the constant current.

The reference verifies an implementation by the Kolmogorov-Smirnov
distance between these distributions, against the distances between its
own seeds (Dasbach et al. 2021). `SCALE` is the reference's own for its
verification data: at a tenth, the currents that make up for the lost
inputs leave every population but L6I below threshold, and the network
falls silent in NEST as in sparx.

    git clone https://github.com/INM-6/microcircuit-PD14-model
    git -C microcircuit-PD14-model checkout f79f8ac
    python tools/make_microcircuit_fixtures.py export /tmp/microcircuit-network.npz
    # with NEST 3.10, matplotlib and scipy:
    python tools/make_microcircuit_fixtures.py nest microcircuit-PD14-model /tmp/microcircuit-network.npz
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "microcircuit.npz"
SCALE, START, STOP, DT, SEEDS = 0.2, 500.0, 1500.0, 0.1, tuple(range(1, 16))
EXACT = 300.0
SAMPLE, KEPT, BIN = 50, 500, 2.0


def export(path: Path) -> None:
    """The network sparx draws at `SCALE` from key 0, in float64, as arrays."""
    import jax

    jax.config.update("jax_enable_x64", val=True)
    import dataclasses

    import jax.numpy as jnp

    from sparx.graph.models import MICROCIRCUIT_POPULATIONS, microcircuit

    network = dataclasses.replace(microcircuit(SCALE, SCALE), dtype=jnp.float64)
    variables = network.init(jax.random.key(0))
    populations = variables["state"]["network"]["populations"]
    out = {}
    for i, population in enumerate(network.populations):
        assert population.name == MICROCIRCUIT_POPULATIONS[i]
        out[f"{i}/v"] = np.asarray(populations[population.name]["point_neuron"].neuron.v)
        out[f"{i}/i_e"] = np.asarray(population.neuron.i_e)
    connections = network.connections(variables)
    for p in network.projections:
        key = f"{MICROCIRCUIT_POPULATIONS.index(p.post)}/{MICROCIRCUIT_POPULATIONS.index(p.pre)}"
        pre, post, weight, delay = connections[p.key]
        out.update({f"{key}/pre": pre, f"{key}/post": post, f"{key}/weight": weight, f"{key}/delay": delay})
    np.savez(path, **out)
    print(f"wrote {path}")


def statistics(times: np.ndarray, senders: np.ndarray, neurons: np.ndarray, rng: np.random.Generator
               ) -> dict[str, np.ndarray]:
    """One population's mean rate (Hz) over `[START, STOP]`, and samples of its rates, ISI CVs and
    pairwise spike-count correlations."""
    within = (times >= START) & (times <= STOP)
    times, senders = times[within], senders[within]
    rates = np.array([np.sum(senders == n) for n in neurons]) / (STOP - START) * 1e3
    cvs = []
    for n in neurons:
        own = np.sort(times[senders == n])
        if len(own) > 2:
            intervals = np.diff(own)
            cvs.append(intervals.std() / intervals.mean())
    sample = rng.choice(neurons, SAMPLE, replace=False)
    edges = np.arange(START, STOP + BIN, BIN)
    counts = np.array([np.histogram(times[senders == n], edges)[0] for n in sample])
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.corrcoef(counts)
    return {"mean": np.asarray(rates.mean()), "rates": rng.permutation(rates)[:KEPT],
            "cvs": rng.permutation(np.asarray(cvs))[:KEPT], "ccs": cc[np.triu_indices(SAMPLE, 1)]}


def reference(path: Path, seed: int) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    """The reference's run at `seed`: its statistics, its build and simulation times, its derived values."""
    import nest

    sys.path.insert(0, str(path / "PyNEST" / "src"))
    from microcircuit import network
    from microcircuit.network_params import default_net_dict
    from microcircuit.sim_params import default_sim_dict
    from microcircuit.stimulus_params import default_stim_dict

    net_dict = {**default_net_dict, "N_scaling": SCALE, "K_scaling": SCALE}
    with tempfile.TemporaryDirectory() as data:
        sim_dict = {**default_sim_dict, "data_path": data + "/", "rng_seed": seed, "print_time": False,
                    "store_metadata": False, "sim_resolution": DT, "local_num_threads": 4}
        start = time.perf_counter()
        net = network.Network(sim_dict, net_dict, dict(default_stim_dict))
        derived = {"neurons": net.num_neurons, "synapses": net.num_synapses, "weight": net.weight_matrix_mean,
                   "weight_external": np.asarray(net.weight_ext), "current": net.DC_amp}
        net.create()
        net.spike_recorders.record_to = "memory"  # the reference writes ASCII files
        net.connect()
        built = time.perf_counter()
        nest.Simulate(STOP)
        simulated = time.perf_counter()
        rng = np.random.default_rng(seed)
        out = {}
        for name, population, recorder in zip(net_dict["populations"], net.pops, net.spike_recorders,
                                               strict=True):
            events = recorder.events
            ids = np.asarray(population.tolist())
            for key, value in statistics(events["times"], events["senders"], ids, rng).items():
                out[f"{name}/{key}"] = value
    return out, {"build": built - start, "simulate": simulated - built}, derived


def poisson(path: Path) -> dict[str, np.ndarray]:
    """What the reference derives at `SCALE` with Poisson background input: currents, in-degrees, weight."""
    sys.path.insert(0, str(path / "PyNEST" / "src"))
    from microcircuit import network
    from microcircuit.network_params import default_net_dict
    from microcircuit.sim_params import default_sim_dict
    from microcircuit.stimulus_params import default_stim_dict

    net_dict = {**default_net_dict, "N_scaling": SCALE, "K_scaling": SCALE, "bg_input_type": "poisson"}
    with tempfile.TemporaryDirectory() as data:
        sim_dict = {**default_sim_dict, "data_path": data + "/", "print_time": False, "store_metadata": False}
        net = network.Network(sim_dict, net_dict, dict(default_stim_dict))
    return {"poisson/current": net.DC_amp, "poisson/external": net.ext_indegrees,
            "poisson/weight_external": np.asarray(net.weight_ext)}


def exact(exported: Path) -> dict[str, np.ndarray]:
    """NEST's spikes over `EXACT` ms on the network sparx drew: each population's steps and neurons."""
    import nest

    network = np.load(exported)
    nest.ResetKernel()
    nest.resolution = DT
    nest.local_num_threads = 4
    params = {"C_m": 250.0, "tau_m": 10.0, "tau_syn_ex": 0.5, "tau_syn_in": 0.5, "E_L": -65.0, "V_th": -50.0,
              "V_reset": -65.0, "t_ref": 2.0}
    count = len([k for k in network.files if k.endswith("/v")])
    populations = []
    for i in range(count):
        population = nest.Create("iaf_psc_exp", len(network[f"{i}/v"]),
                                 params={**params, "I_e": float(network[f"{i}/i_e"])})
        population.V_m = network[f"{i}/v"]
        populations.append(population)
    first = [p[0].global_id for p in populations]
    for target in range(count):
        for source in range(count):
            key = f"{target}/{source}"
            if f"{key}/pre" not in network.files:
                continue
            nest.Connect(network[f"{key}/pre"] + first[source], network[f"{key}/post"] + first[target],
                         conn_spec="one_to_one", syn_spec={"weight": network[f"{key}/weight"],
                                                           "delay": network[f"{key}/delay"]})
    recorders = [nest.Create("spike_recorder") for _ in range(count)]
    for population, recorder in zip(populations, recorders, strict=True):
        nest.Connect(population, recorder)
    nest.Simulate(EXACT)
    out = {}
    for i, recorder in enumerate(recorders):
        events = recorder.events
        # NEST stamps a spike at the end of the step it fired in.
        out[f"exact/{i}/steps"] = np.rint(events["times"] / DT).astype(np.int32) - 1
        out[f"exact/{i}/neurons"] = (events["senders"] - first[i]).astype(np.int32)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export").add_argument("network", type=Path, help="where to write sparx's network")
    run = commands.add_parser("nest")
    run.add_argument("reference", type=Path, help="checkout of INM-6/microcircuit-PD14-model at f79f8ac")
    run.add_argument("network", type=Path, help="the network `export` wrote")
    args = parser.parse_args()
    if args.command == "export":
        export(args.network)
        return
    import nest

    nest.set_verbosity("M_ERROR")
    cases = {"meta/nest": np.array(nest.__version__), "meta/scale": np.array(SCALE),
             "meta/start": np.array(START), "meta/stop": np.array(STOP), "meta/sample": np.array(SAMPLE),
             "meta/bin": np.array(BIN), "meta/seeds": np.array(SEEDS), "meta/exact": np.array(EXACT)}
    cases.update(poisson(args.reference))
    cases.update(exact(args.network))
    print("exact", sum(len(v) for k, v in cases.items() if k.endswith("/steps")), "spikes", flush=True)
    for seed in SEEDS:
        out, seconds, derived = reference(args.reference, seed)
        cases.update({f"derived/{key}": np.asarray(value) for key, value in derived.items()})
        cases.update({f"{seed}/{key}": value.astype(np.float16) for key, value in out.items()})
        cases[f"{seed}/seconds"] = np.array([seconds["build"], seconds["simulate"]])
        means = {key.split("/")[0]: round(float(value), 3) for key, value in out.items()
                 if key.endswith("/mean")}
        print(seed, means, {k: round(v, 1) for k, v in seconds.items()}, flush=True)
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (nest {nest.__version__})")


if __name__ == "__main__":
    main()
