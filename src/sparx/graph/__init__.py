"""Circuits and connectomes: populations of neurons joined by projections, simulated on one clock.

`sparx.dynamics` holds the models of single neurons, synapses and
plasticity rules; this package wires them into networks (design.md
sections 5 and 6), builds the canonical ones and the published models on
connectomes, and simulates them over dew's meshes and checkpoints.
"""

from sparx.graph.connectivity import (
    AllToAll,
    Connectivity,
    EdgeList,
    FixedInDegree,
    FixedOutDegree,
    FixedProbability,
    FromEdges,
    OneToOne,
)
from sparx.graph.connectome import FLYWIRE_630_MEDIAN_INPUTS, SIGNS, Connectome, matched_w_syn, shiu2024
from sparx.graph.models import brunel, coba, cuba, from_record
from sparx.graph.network import (
    ArrivalInput,
    CurrentInput,
    Drive,
    Monitor,
    Network,
    NetworkState,
    PlasticState,
    PoissonInput,
    Population,
    PopulationRate,
    PopulationState,
    Projection,
    ReleaseState,
    SpikeCounts,
    SpikeRaster,
    SpikeTimes,
    StateMonitor,
)
from sparx.graph.simulate import LAYOUT, RULES, Simulation, simulate

__all__ = [
    "FLYWIRE_630_MEDIAN_INPUTS",
    "LAYOUT",
    "RULES",
    "SIGNS",
    "AllToAll",
    "ArrivalInput",
    "Connectivity",
    "Connectome",
    "CurrentInput",
    "Drive",
    "EdgeList",
    "FixedInDegree",
    "FixedOutDegree",
    "FixedProbability",
    "FromEdges",
    "Monitor",
    "Network",
    "NetworkState",
    "OneToOne",
    "PlasticState",
    "PoissonInput",
    "Population",
    "PopulationRate",
    "PopulationState",
    "Projection",
    "ReleaseState",
    "Simulation",
    "SpikeCounts",
    "SpikeRaster",
    "SpikeTimes",
    "StateMonitor",
    "brunel",
    "coba",
    "cuba",
    "from_record",
    "matched_w_syn",
    "shiu2024",
    "simulate",
]
