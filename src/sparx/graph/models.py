"""Canonical networks, built as published, for science and as validation targets (design.md section 5.3)."""

from __future__ import annotations

from sparx.dynamics.neurons import LIF
from sparx.dynamics.synapses import Delta, Receptor
from sparx.graph.connectivity import FixedInDegree
from sparx.graph.network import Network, PoissonInput, Population, Projection

__all__ = ["brunel"]


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
    neuron = LIF(tau_m=tau_m, c_m=250.0, e_l=0.0, v_th=theta, v_reset=10.0, t_ref=2.0)
    receptors = {"ex": Receptor(Delta()), "in": Receptor(Delta())}
    populations = (Population("e", excitatory, neuron, receptors),
                   Population("i", inhibitory, neuron, receptors))
    projections = tuple(
        Projection(pre, post, FixedInDegree(c_e if pre == "e" else c_i, autapses=True, multapses=True),
                   weight=j if pre == "e" else -g * j,
                   delay=delay, receptor="ex" if pre == "e" else "in")
        for pre in ("e", "i") for post in ("e", "i"))
    inputs = tuple(PoissonInput(name, rate=external, weight=j, receptor="ex") for name in ("e", "i"))
    return Network(populations, projections, inputs, dt=dt)
