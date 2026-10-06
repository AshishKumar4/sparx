"""Circuits and connectomes: populations of neurons joined by projections, simulated on one clock.

`sparx.dynamics` holds the models of single neurons, synapses and
plasticity rules; this package wires them into networks (design.md
sections 5 and 6), builds the canonical ones and the published models on
connectomes, and simulates them over dew's meshes and checkpoints.

The names here are the ones a model and a run are written with. The types
of the state, the layout rules and the connectome constants stay in their
modules (`sparx.graph.network`, `sparx.graph.simulate`,
`sparx.graph.connectome`).
"""

from sparx.graph.connectivity import (
    AllToAll,
    FixedInDegree,
    FixedOutDegree,
    FixedProbability,
    FromEdges,
    OneToOne,
)
from sparx.graph.connectome import Connectome, matched_w_syn, shiu2024
from sparx.graph.models import brunel, coba, cuba, from_record
from sparx.graph.network import (
    ArrivalInput,
    CurrentInput,
    GapJunction,
    Modulator,
    ModulatorTrace,
    Monitor,
    Network,
    OutputTrace,
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
    "Connectome",
    "CurrentInput",
    "FixedInDegree",
    "FixedOutDegree",
    "FixedProbability",
    "FromEdges",
    "GapJunction",
    "Modulator",
    "ModulatorTrace",
    "Monitor",
    "Network",
    "OneToOne",
    "OutputTrace",
    "PoissonInput",
    "Population",
    "PopulationRate",
    "Projection",
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
