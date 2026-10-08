"""Canonical networks, built as published, for science and as validation targets (design.md section 5.3).

Each builder has a short name in `sparx.registry.networks`, so a run's
record names the network it simulates and `from_record` rebuilds it in
another process.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from dew.registry import Record

from sparx.dynamics.neurons import LeakyIntegrateAndFire
from sparx.dynamics.synapses import Delta, Exponential, Receptor
from sparx.graph.connectivity import FixedInDegree, FixedProbability
from sparx.graph.network import Network, PerNeuron, PoissonInput, Population, Projection
from sparx.registry import networks

__all__ = ["brunel", "coba", "cuba", "from_record"]


def from_record(record: Record) -> Network:
    """The network a `{"class": builder, "fields": {...}}` record names, built from its fields.

    The builder is a short name of `sparx.registry.networks` or an import
    path. A field that is itself a record names its class or function by
    import path and is built first, so a model on a connectome is configured
    by its reader and paths: `{"class": "shiu2024", "fields": {"connectome":
    {"class": "sparx.graph.connectome:Connectome.from_shiu", "fields":
    {"completeness": ..., "connectivity": ...}}, "stimuli": ...}}`.
    """
    return networks.from_record(record)


def brunel(order: int = 2500, *, g: float = 5.0, eta: float = 2.0, j: float = 0.1, delay: float = 1.5,
           epsilon: float = 0.1, dt: float = 0.1) -> Network:
    """Brunel's (J. Comput. Neurosci. 2000) sparse network of excitatory and inhibitory LIF neurons, model A.

    `4 order` excitatory and `order` inhibitory neurons (`order=2500` is
    the paper's 12,500); each receives `epsilon` of each population's
    neurons, with delta synapses of `j` mV (excitatory) and `-g j`
    (inhibitory), all after `delay` ms, and Poisson input from `C_E`
    external neurons at `eta` times the rate that brings a free membrane to
    threshold. Membranes: 20 ms, threshold 20 mV, reset 10 mV, 2 ms
    refractory, rest 0. The regimes of his Figure 8: `g=3, eta=2`
    synchronous regular; `g=5, eta=2` asynchronous irregular; `g=6, eta=4`
    synchronous irregular, fast; `g=4.5, eta=0.9` synchronous irregular,
    slow. These are the parameters of NEST's `brunel_delta_nest.py`, and
    as there inputs are drawn with replacement, self-connections allowed
    (NEST's `fixed_indegree` defaults).
    """
    excitatory, inhibitory = 4 * order, order
    c_e, c_i = round(epsilon * excitatory), round(epsilon * inhibitory)
    tau_m, theta = 20.0, 20.0
    nu_threshold = theta / (j * c_e * tau_m)  # 1/ms
    external = eta * nu_threshold * 1000.0 * c_e  # Hz, all C_E sources together
    neuron = LeakyIntegrateAndFire(tau_m=tau_m, c_m=250.0, e_l=0.0, v_th=theta, v_reset=10.0, t_ref=2.0)
    receptors = {"ampa": Receptor(Delta()), "gaba_a": Receptor(Delta())}
    populations = (Population("e", excitatory, neuron, receptors),
                   Population("i", inhibitory, neuron, receptors))
    projections = tuple(
        Projection(pre, post, FixedInDegree(c_e if pre == "e" else c_i, autapses=True, multapses=True),
                   weight=j if pre == "e" else -g * j,
                   delay=delay, receptor="ampa" if pre == "e" else "gaba_a")
        for pre in ("e", "i") for post in ("e", "i"))
    inputs = tuple(PoissonInput(name, rate=external, weight=j, receptor="ampa") for name in ("e", "i"))
    return Network(populations, projections, inputs, dt=dt)


def _vogels_abbott(neuron: LeakyIntegrateAndFire, receptors: Mapping[str, Receptor],
                   weights: tuple[float, float], initial: Mapping[str, PerNeuron], dt: float) -> Network:
    populations = (Population("e", 3200, neuron, receptors, initial=initial),
                   Population("i", 800, neuron, receptors, initial=initial))
    projections = tuple(
        Projection(pre, post, FixedProbability(0.02, autapses=True), weight=weights[pre == "i"], delay=0.0,
                   receptor="ampa" if pre == "e" else "gaba_a")
        for pre in ("e", "i") for post in ("e", "i"))
    return Network(populations, projections, dt=dt)


def _random_voltage(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.uniform(-60.0, -50.0, size)


def cuba(dt: float = 0.1) -> Network:
    """Vogels and Abbott's (2005) network with current-based synapses: Brette et al.'s (2007) CUBA benchmark.

    As Brian2's `examples/CUBA.py`: 3,200 excitatory and 800 inhibitory
    neurons connected with probability 0.02, without delay; membranes of
    20 ms resting at -49 mV, above the -50 mV threshold, reset to -60 mV,
    5 ms refractory; exponential currents of 5 and 10 ms. Brian2 states
    the synapses in volts, 1.62 and -9 mV over the membrane time constant;
    on a 200 pF membrane they are 16.2 and -90 pA. Voltages start uniform
    between reset and threshold.
    """
    neuron = LeakyIntegrateAndFire(tau_m=20.0, c_m=200.0, e_l=-49.0, v_th=-50.0, v_reset=-60.0, t_ref=5.0)
    receptors = {"ampa": Receptor(Exponential(5.0)), "gaba_a": Receptor(Exponential(10.0))}
    return _vogels_abbott(neuron, receptors, (16.2, -90.0), {"v": _random_voltage}, dt)


def coba(dt: float = 0.1) -> Network:
    """Vogels and Abbott's (2005) network with conductance-based synapses (Brette et al.'s COBA benchmark).

    The CUBA network's structure with conductances of 6 and 67 nS, decaying
    in 5 and 10 ms, reversing at 0 and -80 mV, on membranes of 200 pF and
    10 nS resting at -60 mV (the reversal potentials are `LeakyIntegrateAndFire`'s defaults for
    `ampa` and `gaba_a`). Activity is sustained from random initial
    voltages and conductances (excitatory `N(40, 15)` nS, inhibitory
    `N(200, 120)` nS), as Brian's example sets them.
    """
    neuron = LeakyIntegrateAndFire(tau_m=20.0, c_m=200.0, e_l=-60.0, v_th=-50.0, v_reset=-60.0, t_ref=5.0)
    receptors = {"ampa": Receptor(Exponential(5.0), "conductance"),
                 "gaba_a": Receptor(Exponential(10.0), "conductance")}
    initial = {"v": _random_voltage,
               "ampa": lambda rng, size: (rng.normal(size=size) * 1.5 + 4) * 10,
               "gaba_a": lambda rng, size: (rng.normal(size=size) * 12 + 20) * 10}
    return _vogels_abbott(neuron, receptors, (6.0, 67.0), initial, dt)
