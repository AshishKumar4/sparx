"""Time sparx's simulator on the standard benchmark networks, beside NEST and Brian2.

    JAX_PLATFORMS=cpu python benchmarks/bench_networks.py
    JAX_PLATFORMS=cpu python benchmarks/bench_networks.py --networks brunel --seconds 2 --format events

Brunel's (2000) network at the paper's size (12,500 neurons, asynchronous
irregular), Brette et al.'s (2007) CUBA and COBA (4,000 neurons), and
Potjans and Diesmann's (2014) cortical microcircuit at a fifth of its
neurons and inputs (15,435 neurons), all at `dt = 0.1` ms. For each it
prints the time to build the network and draw its connections (`init`),
the time to compile a chunk of 100 ms, the step `simulate` compiles and
runs chunk after chunk, and the wall time per simulated second of running
that chunk from the state it leaves, as `simulate` does, after one chunk to
settle (five for the microcircuit, whose onset lasts 500 ms). It also
prints an excitatory population's mean rate over the timed chunks (layer
2/3's in the microcircuit), to show the run is the network
`tools/bench_reference_simulators.py` runs in NEST and Brian2. `--format`
sets every projection's format in place of the builders' `"auto"`.
"""

import argparse
import dataclasses
import time

import jax
import numpy as np

from sparx.graph import PopulationRate
from sparx.graph.models import brunel, coba, cuba, microcircuit

NETWORKS = {"brunel": lambda: brunel(2500, g=5.0, eta=2.0), "cuba": cuba, "coba": coba,
            "microcircuit": lambda: microcircuit(0.2, 0.2)}
WATCHED = {"microcircuit": "L23E"}
"""The population whose rate is printed, when it is not `"e"`."""
SETTLE = {"microcircuit": 5}
"""Chunks run before timing, when more than one."""
CHUNK = 100.0
"""ms; `simulate`'s default."""


def bench(name: str, seconds: float, format: str | None) -> None:
    start = time.perf_counter()
    network = NETWORKS[name]()
    if format is not None:
        network = dataclasses.replace(network, projections=tuple(dataclasses.replace(p, format=format)
                                                                 for p in network.projections))
    variables = jax.block_until_ready(network.init(jax.random.key(0)))
    built = time.perf_counter() - start
    fixed = {key: value for key, value in variables.items() if key != "state"}
    steps = round(CHUNK / network.dt)

    @jax.jit
    def chunk(state, key):
        records, updates = network.apply({**fixed, "state": state}, steps=steps,
                                         monitors={"rate": PopulationRate(WATCHED.get(name, "e"))},
                                         rngs={"noise": key},
                                         mutable=["state"])
        return records["rate"], updates["state"]

    start = time.perf_counter()
    compiled = chunk.lower(variables["state"], jax.random.key(1)).compile()
    compiling = time.perf_counter() - start
    state = variables["state"]
    for k in range(SETTLE.get(name, 1)):
        rates, state = compiled(state, jax.random.key(1000 + k))
    jax.block_until_ready(state)
    count = round(1000.0 * seconds / CHUNK)
    rates = []
    start = time.perf_counter()
    for k in range(count):
        rate, state = compiled(state, jax.random.key(2 + k))
        rates.append(rate)
    jax.block_until_ready(state)
    per_second = (time.perf_counter() - start) / seconds
    rate = float(np.mean(np.concatenate([np.asarray(r) for r in rates])))
    print(f"{name:7s} {format or 'auto':6s} build {built:6.2f} s  compile {compiling:6.2f} s  "
          f"per simulated second {per_second:6.2f} s  excitatory rate {rate:5.1f} Hz", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--networks", nargs="+", default=list(NETWORKS), choices=list(NETWORKS))
    parser.add_argument("--seconds", type=float, default=2.0, help="simulated seconds timed per network")
    parser.add_argument("--format", choices=["edges", "dense", "events"], default=None)
    args = parser.parse_args()
    print(f"jax {jax.__version__} on {jax.devices()[0].device_kind}, {jax.device_count()} device(s)")
    for name in args.networks:
        bench(name, args.seconds, args.format)


if __name__ == "__main__":
    main()
