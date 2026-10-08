"""Canonical networks, built as published, for science and as validation targets (design.md section 5.3).

Each builder has a short name in `sparx.registry.networks`, so a run's
record names the network it simulates and `from_record` rebuilds it in
another process.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Literal

import numpy as np
from dew.registry import Record

from sparx.dynamics.neurons import LeakyIntegrateAndFire
from sparx.dynamics.synapses import Delta, Exponential, Receptor
from sparx.graph.connectivity import FixedInDegree, FixedProbability, FixedTotalNumber
from sparx.graph.network import Network, PerEdge, PerNeuron, PoissonInput, Population, Projection
from sparx.registry import networks

__all__ = ["brunel", "coba", "cuba", "from_record", "microcircuit"]


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


MICROCIRCUIT_POPULATIONS = ("L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I")
MICROCIRCUIT_SIZES = np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948])
MICROCIRCUIT_RATES = np.array([0.903, 2.965, 4.414, 5.876, 7.569, 8.633, 1.105, 7.829])
"""Each population's rate in the full model, Hz, which `microcircuit` uses to keep the input of a network
with fewer inputs per neuron (the reference's single NEST run at full scale)."""
MICROCIRCUIT_PROBABILITIES = np.array([  # [target, source], in MICROCIRCUIT_POPULATIONS' order
    [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0],
    [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0],
    [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.0],
    [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0],
    [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
    [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.0],
    [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
    [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443],
])
MICROCIRCUIT_EXTERNAL = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])
"""Cortico-cortical inputs per neuron, each at 8 Hz."""
MICROCIRCUIT_V0 = (np.array([-68.28, -63.16, -63.33, -63.45, -63.11, -61.66, -66.72, -61.43]),
                   np.array([5.36, 4.57, 4.74, 4.94, 4.94, 4.55, 5.46, 4.48]))
"""The mean and standard deviation of each population's initial voltage, mV (the reference's "optimized"
start, which shortens the burst at onset)."""


def _psc_over_psp(c_m: float, tau_m: float, tau_syn: float) -> float:
    """The current (pA) of an exponential synapse whose potential peaks at 1 mV (Hanuschkin et al. 2010)."""
    sub = 1.0 / (tau_syn - tau_m)
    pre = tau_m * tau_syn / c_m * sub
    frac = (tau_m / tau_syn) ** sub
    return 1.0 / (pre * (frac ** tau_m - frac ** tau_syn))


def _redrawn(rng: np.random.Generator, mean: float, std: float, low: float, high: float, count: int
             ) -> np.ndarray:
    """`count` normal draws, each drawn again until it falls inside `[low, high]` (NEST's `redraw`)."""
    values = rng.normal(mean, std, count)
    outside = (values < low) | (values > high)
    while outside.any():
        values[outside] = rng.normal(mean, std, int(outside.sum()))
        outside = (values < low) | (values > high)
    return values


def microcircuit(neurons: float = 1.0, indegrees: float = 1.0, *,
                 background: Literal["dc", "poisson"] = "dc", dt: float = 0.1) -> Network:
    """Potjans and Diesmann's (Cereb. Cortex 2014) cortical microcircuit: 1 mm^2 of cortex, layers 2/3 to 6.

    Eight populations, an excitatory and an inhibitory one per layer, of
    `iaf_psc_exp` neurons (10 ms, 250 pF, rest and reset -65 mV, threshold
    -50 mV, 2 ms refractory, 0.5 ms currents), 77,169 at full scale. Each
    pair of populations is joined by a fixed total number of synapses,
    drawn with replacement, self-connections allowed, set so that a pair of
    neurons is connected at least once with the measured probability.
    Weights are normal around a current that peaks at 0.15 mV (twice that
    from L4E to L2/3E, -4 times it from inhibitory neurons), 10% standard
    deviation, drawn again where they change sign; delays are normal around
    1.5 ms (excitatory) and 0.75 ms (inhibitory), 50% standard deviation,
    drawn again below half a step and rounded to steps, as NEST rounds them.
    The cortico-cortical input is a constant current per population
    (`background="dc"`, the reference's default) or Poisson spikes at 8 Hz
    from each of its inputs.

    `neurons` scales the population sizes and `indegrees` the inputs per
    neuron. With fewer inputs, each weight grows by one over the square
    root of the scale and a current makes up the mean input lost at the
    full model's rates, which keeps each population's rate. At a fifth of
    both (`microcircuit(0.2, 0.2)`: 15,435 neurons, 12 million synapses)
    the populations fire as in the full model; at a tenth, the currents
    leave all but L6I below threshold and the network falls silent, in NEST
    as here. Every number is from the PyNEST implementation of
    INM-6/microcircuit-PD14-model (commit f79f8ac), which
    `tests/test_microcircuit.py` compares against.
    """
    c_m, tau_m, tau_syn = 250.0, 10.0, 0.5
    psc = _psc_over_psp(c_m, tau_m, tau_syn)
    count = len(MICROCIRCUIT_POPULATIONS)
    excitatory = np.arange(count) % 2 == 0
    psp = np.where(excitatory, 0.15, -4 * 0.15)[None, :].repeat(count, 0)
    psp[0, 2] = 2 * 0.15  # L4E onto L2/3E
    pairs = np.outer(MICROCIRCUIT_SIZES, MICROCIRCUIT_SIZES).astype(np.float64)
    full_synapses = np.log(1.0 - MICROCIRCUIT_PROBABILITIES) / np.log((pairs - 1.0) / pairs)
    sizes = np.round(MICROCIRCUIT_SIZES * neurons).astype(int)
    synapses = np.round(full_synapses * neurons * indegrees).astype(int)
    weight, weight_external = psp * psc, 0.15 * psc
    dc = (8.0 * MICROCIRCUIT_EXTERNAL * weight_external * tau_syn * 1e-3 if background == "dc"
          else np.zeros(count))
    if indegrees != 1:
        # The mean input a neuron loses with fewer, stronger inputs, at the full model's rates.
        indegree = full_synapses / MICROCIRCUIT_SIZES[:, None]
        lost = np.sum(weight * indegree * MICROCIRCUIT_RATES, axis=1)
        if background == "poisson":
            lost = lost + weight_external * MICROCIRCUIT_EXTERNAL * 8.0
        dc = dc + 1e-3 * tau_syn * (1.0 - np.sqrt(indegrees)) * lost
        weight, weight_external = weight / np.sqrt(indegrees), weight_external / np.sqrt(indegrees)
    receptors = {"ampa": Receptor(Exponential(tau_syn)), "gaba_a": Receptor(Exponential(tau_syn))}
    neuron = LeakyIntegrateAndFire(tau_m=tau_m, c_m=c_m, e_l=-65.0, v_th=-50.0, v_reset=-65.0, t_ref=2.0)
    populations = tuple(
        Population(name, int(sizes[i]), dataclasses.replace(neuron, i_e=float(dc[i])), receptors,
                   initial={"v": _normal(MICROCIRCUIT_V0[0][i], MICROCIRCUIT_V0[1][i])})
        for i, name in enumerate(MICROCIRCUIT_POPULATIONS))
    projections = tuple(
        Projection(MICROCIRCUIT_POPULATIONS[source], MICROCIRCUIT_POPULATIONS[target],
                   FixedTotalNumber(int(synapses[target, source]), autapses=True, multapses=True),
                   weight=_signed_weights(float(weight[target, source])),
                   delay=_delays(1.5 if excitatory[source] else 0.75, dt),
                   receptor="ampa" if excitatory[source] else "gaba_a")
        for target in range(count) for source in range(count) if synapses[target, source] > 0)
    inputs = () if background == "dc" else tuple(
        PoissonInput(name, rate=8.0, weight=float(weight_external), receptor="ampa",
                     count=int(np.round(MICROCIRCUIT_EXTERNAL[i] * indegrees)))
        for i, name in enumerate(MICROCIRCUIT_POPULATIONS))
    return Network(populations, projections, inputs, dt=dt)


def _normal(mean: float, std: float) -> PerNeuron:
    return lambda rng, size: rng.normal(mean, std, size)


def _signed_weights(mean: float) -> PerEdge:
    """Normal weights of 10% standard deviation, drawn again where their sign differs from the mean's."""
    low, high = (0.0, np.inf) if mean >= 0 else (-np.inf, 0.0)
    return lambda rng, n: _redrawn(rng, mean, abs(mean) * 0.1, low, high, n)


def _delays(mean: float, dt: float) -> PerEdge:
    """Normal delays of 50% standard deviation, drawn again below half a step, rounded to steps."""
    return lambda rng, n: dt * np.floor(_redrawn(rng, mean, mean * 0.5, dt / 2, np.inf, n) / dt + 0.5)
