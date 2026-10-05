"""Circuits and connectomes: populations of neurons joined by projections, simulated on one clock.

`sparx.dynamics` holds the models of single neurons, synapses and
plasticity rules; this package wires them into networks (design.md
sections 5 and 6).
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
from sparx.graph.network import (
    ArrivalInput,
    CurrentInput,
    Monitor,
    Network,
    PoissonInput,
    Population,
    PopulationRate,
    Projection,
    SpikeCounts,
    SpikeRaster,
    SpikeTimes,
    StateMonitor,
)
from sparx.graph.simulate import Simulation, simulate

__all__ = [
    "AllToAll",
    "ArrivalInput",
    "Connectivity",
    "CurrentInput",
    "EdgeList",
    "FixedInDegree",
    "FixedOutDegree",
    "FixedProbability",
    "FromEdges",
    "Monitor",
    "Network",
    "OneToOne",
    "PoissonInput",
    "Population",
    "PopulationRate",
    "Projection",
    "Simulation",
    "SpikeCounts",
    "SpikeRaster",
    "SpikeTimes",
    "StateMonitor",
    "simulate",
]
