"""Time NEST and Brian2 on the networks `benchmarks/bench_networks.py` times in sparx.

    python benchmarks/bench_reference_simulators.py                       # every network and simulator
    python benchmarks/bench_reference_simulators.py --networks cuba --simulators nest
    python benchmarks/bench_reference_simulators.py --networks microcircuit --simulators nest \
        --microcircuit microcircuit-PD14-model

Brunel's (2000) network at the paper's size, 12,500 neurons in the
asynchronous irregular regime (`g = 5`, `eta = 2`, NEST's
`brunel_delta_nest.py`), and Brette et al.'s (2007) CUBA and COBA on 4,000
neurons, as `sparx.graph.models` builds them, and Potjans and Diesmann's
(2014) cortical microcircuit at a fifth of its neurons and inputs, as the
PyNEST implementation of INM-6/microcircuit-PD14-model at `--microcircuit`
(commit f79f8ac) builds it, all at `dt = 0.1` ms:

- NEST (`iaf_psc_delta`, `iaf_psc_exp`, `iaf_cond_exp`) on `--threads`
  threads. The build is creating and connecting; the time per simulated
  second is `Simulate` of `--seconds` after a first `Simulate` of 100 ms,
  which prepares the kernel. NEST needs a delay of at least one step,
  where CUBA and COBA have none, so they take 0.1 ms.
- Brian2's runtime mode, generated Cython (`cython`), timed as NEST is.
- Brian2's C++ standalone mode (`cpp_standalone`) on `--threads` OpenMP
  threads. The time per simulated second is the sum of its profiled code
  objects other than those that create synapses, over a run of `--seconds`.

Each prints the excitatory population's mean rate after the first 100 ms
(after 500 ms, layer 2/3's, in the microcircuit, whose onset lasts that
long), to show it runs the network sparx runs. The microcircuit runs in
NEST only. Needs NEST 3.10 and Brian2 2.10 (the
reference environment of HANDOFF.md) and a C++ compiler.
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import results

DT = 0.1
WARMUP = 100.0
"""ms, simulated before the timed run; the rate is over the timed run."""


def brunel_parameters():
    order, g, eta, j, delay = 2500, 5.0, 2.0, 0.1, 1.5
    c_e, c_i = order * 4 // 10, order // 10
    nu_threshold = 20.0 / (j * c_e * 20.0)  # 1/ms
    return {"excitatory": 4 * order, "inhibitory": order, "c_e": c_e, "c_i": c_i, "g": g, "j": j,
            "delay": delay, "rate": eta * nu_threshold * 1000.0}  # Hz from each of c_e sources


def nest_network(name, threads, seed):
    import nest

    nest.ResetKernel()
    nest.verbosity = nest.VerbosityLevel.ERROR
    nest.resolution = DT
    nest.local_num_threads = threads
    nest.rng_seed = seed
    rng = np.random.default_rng(seed)
    if name == "brunel":
        p = brunel_parameters()
        params = {"C_m": 250.0, "tau_m": 20.0, "t_ref": 2.0, "E_L": 0.0, "V_reset": 10.0, "V_m": 0.0,
                  "V_th": 20.0}
        e = nest.Create("iaf_psc_delta", p["excitatory"], params=params)
        i = nest.Create("iaf_psc_delta", p["inhibitory"], params=params)
        noise = nest.Create("poisson_generator", params={"rate": p["rate"] * p["c_e"]})
        nest.Connect(noise, e + i, syn_spec={"weight": p["j"], "delay": p["delay"]})
        nest.Connect(e, e + i, {"rule": "fixed_indegree", "indegree": p["c_e"]},
                     {"weight": p["j"], "delay": p["delay"]})
        nest.Connect(i, e + i, {"rule": "fixed_indegree", "indegree": p["c_i"]},
                     {"weight": -p["g"] * p["j"], "delay": p["delay"]})
    else:
        common = {"C_m": 200.0, "tau_m": 20.0, "V_th": -50.0, "V_reset": -60.0, "t_ref": 5.0,
                  "tau_syn_ex": 5.0, "tau_syn_in": 10.0}
        if name == "cuba":
            model, params, weights = "iaf_psc_exp", {**common, "E_L": -49.0}, (16.2, -90.0)
        else:
            model, weights = "iaf_cond_exp", (6.0, -67.0)
            params = {key: value for key, value in common.items() if key != "tau_m"}
            params |= {"E_L": -60.0, "g_L": 10.0, "E_ex": 0.0, "E_in": -80.0}
        e = nest.Create(model, 3200, params=params)
        i = nest.Create(model, 800, params=params)
        neurons = e + i
        neurons.V_m = list(rng.uniform(-60.0, -50.0, 4000))
        if name == "coba":
            neurons.g_ex = list(rng.normal(40.0, 15.0, 4000))
            neurons.g_in = list(rng.normal(200.0, 120.0, 4000))
        for source, weight in ((e, weights[0]), (i, weights[1])):
            nest.Connect(source, neurons, {"rule": "pairwise_bernoulli", "p": 0.02},
                         {"weight": weight, "delay": DT})
    recorder = nest.Create("spike_recorder")
    nest.Connect(e, recorder)
    return nest, e, recorder


def bench_nest(name, threads, seconds):
    start = time.perf_counter()
    nest, e, recorder = nest_network(name, threads, seed=1)
    built = time.perf_counter() - start
    nest.Simulate(WARMUP)
    recorder.n_events = 0
    start = time.perf_counter()
    nest.Simulate(1000.0 * seconds)
    elapsed = time.perf_counter() - start
    rate = recorder.n_events / len(e) / seconds
    return built, elapsed / seconds, rate


def bench_nest_microcircuit(reference, threads, seconds):
    import sys

    import nest

    sys.path.insert(0, os.path.join(reference, "PyNEST", "src"))
    from microcircuit import network
    from microcircuit.network_params import default_net_dict
    from microcircuit.sim_params import default_sim_dict
    from microcircuit.stimulus_params import default_stim_dict

    with tempfile.TemporaryDirectory() as data:
        sim_dict = {**default_sim_dict, "data_path": data + "/", "rng_seed": 1, "print_time": False,
                    "store_metadata": False, "sim_resolution": DT, "local_num_threads": threads}
        start = time.perf_counter()
        net = network.Network(sim_dict, {**default_net_dict, "N_scaling": 0.2, "K_scaling": 0.2},
                              dict(default_stim_dict))
        net.create()
        net.spike_recorders.record_to = "memory"
        net.connect()
        built = time.perf_counter() - start
        nest.Simulate(500.0)
        recorder = net.spike_recorders[0]
        recorder.n_events = 0
        start = time.perf_counter()
        nest.Simulate(1000.0 * seconds)
        elapsed = time.perf_counter() - start
        rate = recorder.n_events / len(net.pops[0]) / seconds
    return built, elapsed / seconds, rate


def brian2_network(name, b2):
    b2.defaultclock.dt = DT * b2.ms
    rng = np.random.default_rng(1)
    if name == "brunel":
        p = brunel_parameters()
        n = p["excitatory"] + p["inhibitory"]
        group = b2.NeuronGroup(n, "dv/dt = -v / (20 * ms) : volt (unless refractory)",
                               threshold="v > 20 * mV", reset="v = 10 * mV", refractory=2 * b2.ms,
                               method="exact")
        group.v = 0 * b2.mV
        synapses = []
        for count, first, size, weight in ((p["c_e"], 0, p["excitatory"], p["j"]),
                                           (p["c_i"], p["excitatory"], p["inhibitory"], -p["g"] * p["j"])):
            # Drawn with replacement, autapses allowed: NEST's fixed_indegree.
            sources = first + rng.integers(0, size, (n, count))
            targets = np.repeat(np.arange(n), count)
            s = b2.Synapses(group, group, on_pre=f"v += {weight} * mV", delay=p["delay"] * b2.ms)
            s.connect(i=sources.ravel(), j=targets)
            synapses.append(s)
        noise = b2.PoissonInput(group, "v", N=p["c_e"], rate=p["rate"] * b2.Hz, weight=p["j"] * b2.mV)
        monitor = b2.SpikeMonitor(group[:p["excitatory"]])
        return b2.Network(group, *synapses, noise, monitor), monitor, p["excitatory"]
    namespace = {"taum": 20 * b2.ms, "taue": 5 * b2.ms, "taui": 10 * b2.ms, "Vt": -50 * b2.mV,
                 "Vr": -60 * b2.mV, "Ee": 0 * b2.mV, "Ei": -80 * b2.mV, "gl": 10 * b2.nS, "Cm": 200 * b2.pF}
    if name == "cuba":
        equations = """dv/dt = (ge + gi - (v - El)) / taum : volt (unless refractory)
                       dge/dt = -ge / taue : volt
                       dgi/dt = -gi / taui : volt"""
        namespace.update(El=-49 * b2.mV, we=60 * 0.27 / 10 * b2.mV, wi=-20 * 4.5 / 10 * b2.mV)
        method = "exact"
    else:
        equations = """dv/dt = (gl * (El - v) + ge * (Ee - v) + gi * (Ei - v)) / Cm : volt (unless refractory)
                       dge/dt = -ge / taue : siemens
                       dgi/dt = -gi / taui : siemens"""
        namespace.update(El=-60 * b2.mV, we=6 * b2.nS, wi=67 * b2.nS)
        method = "exponential_euler"
    group = b2.NeuronGroup(4000, equations, threshold="v > Vt", reset="v = Vr", refractory=5 * b2.ms,
                           method=method, namespace=namespace)
    group.v = "Vr + rand() * (Vt - Vr)"
    if name == "coba":
        group.ge = "(randn() * 1.5 + 4) * 10 * nS"
        group.gi = "(randn() * 12 + 20) * 10 * nS"
    excitatory = b2.Synapses(group, group, on_pre="ge += we", namespace=namespace)
    inhibitory = b2.Synapses(group, group, on_pre="gi += wi", namespace=namespace)
    excitatory.connect("i < 3200", p=0.02)
    inhibitory.connect("i >= 3200", p=0.02)
    monitor = b2.SpikeMonitor(group[:3200])
    return b2.Network(group, excitatory, inhibitory, monitor), monitor, 3200


def bench_brian2_runtime(name, seconds):
    import brian2 as b2

    b2.start_scope()
    b2.prefs.codegen.target = "cython"
    b2.seed(1)
    start = time.perf_counter()
    network, monitor, excitatory = brian2_network(name, b2)
    network.run(0 * b2.ms)  # creates the synapses and compiles the code objects
    built = time.perf_counter() - start
    network.run(WARMUP * b2.ms)
    before = monitor.num_spikes
    start = time.perf_counter()
    network.run(1000.0 * seconds * b2.ms)
    elapsed = time.perf_counter() - start
    return built, elapsed / seconds, (monitor.num_spikes - before) / excitatory / seconds


def bench_brian2_standalone(name, threads, seconds):
    import brian2 as b2

    b2.set_device("cpp_standalone", build_on_run=False)
    b2.prefs.devices.cpp_standalone.openmp_threads = threads
    b2.start_scope()
    b2.seed(1)
    network, monitor, excitatory = brian2_network(name, b2)
    network.run(WARMUP * b2.ms)
    network.run(1000.0 * seconds * b2.ms, profile=True)
    start = time.perf_counter()
    b2.device.build(directory=tempfile.mkdtemp(prefix=f"brian2-{name}-"), run=True)
    total = time.perf_counter() - start
    profile = network.profiling_info
    simulated = sum(float(t) for code, t in profile if "synapses_create" not in code)
    times = np.asarray(monitor.t / b2.ms)
    rate = np.sum(times >= WARMUP) / excitatory / seconds
    b2.device.reinit()
    b2.set_device("runtime")
    return total - simulated, simulated / seconds, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--networks", nargs="+", default=["brunel", "cuba", "coba"],
                        choices=["brunel", "cuba", "coba", "microcircuit"])
    parser.add_argument("--microcircuit", help="checkout of INM-6/microcircuit-PD14-model at f79f8ac")
    parser.add_argument("--simulators", nargs="+", default=["nest", "brian2-cython", "brian2-standalone"],
                        choices=["nest", "brian2-cython", "brian2-standalone"])
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--results", type=Path, help="append the measurements to this JSON lines file")
    args = parser.parse_args()
    import brian2
    import nest

    print(f"NEST {nest.__version__}, Brian2 {brian2.__version__}, {args.threads} threads")
    rows: list[results.Measurement] = []
    for name in args.networks:
        for simulator in args.simulators:
            if name == "microcircuit" and simulator != "nest":
                continue
            if name == "microcircuit":
                built, per_second, rate = bench_nest_microcircuit(args.microcircuit, args.threads,
                                                                  args.seconds)
            elif simulator == "nest":
                built, per_second, rate = bench_nest(name, args.threads, args.seconds)
            elif simulator == "brian2-cython":
                built, per_second, rate = bench_brian2_runtime(name, args.seconds)
            else:
                built, per_second, rate = bench_brian2_standalone(name, args.threads, args.seconds)
            print(f"{name:7s} {simulator:18s} build {built:6.2f} s  "
                  f"per simulated second {per_second:6.2f} s  excitatory rate {rate:5.1f} Hz", flush=True)
            rows.append({"network": name, "simulator": simulator, "build_s": built,
                         "seconds_per_simulated_second": per_second, "rate_hz": rate,
                         "simulated_s": args.seconds, "threads": args.threads, "dt_ms": DT})
    # conda's NEST has no package metadata, so its version is its module's.
    taken = results.conditions(("brian2", "numpy"), ["cpu"] * args.threads,
                               known={"nest-simulator": nest.__version__})
    results.write(args.results, "bench_reference_simulators", rows, taken)


if __name__ == "__main__":
    main()
