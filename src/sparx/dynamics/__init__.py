"""Biophysical neuron, synapse and plasticity models in physical units.

`sparx.cells` holds the dimensionless, one-step-is-one-unit dynamics deep
spiking networks train with. This package holds models of neurons as
biology measures them: membrane equations in mV and ms with conductances,
refractoriness and reversal potentials, synapses with receptor kinetics,
and plasticity rules. They are what `sparx.graph` builds circuits and
connectomes from (design.md sections 4 and 5).
"""

from sparx.dynamics.core import (
    NeuronModel,
    Spikes,
    SynapticInput,
    Term,
    crossing,
    exact_linear,
    integrate,
    response,
    rk4,
    substeps,
)
from sparx.dynamics.neurons import (
    IZHIKEVICH_2003,
    LIF,
    RECEPTORS,
    AdEx,
    AdExState,
    HodgkinHuxley,
    HodgkinHuxleyState,
    Izhikevich,
    IzhikevichState,
    LIFState,
    MgBlock,
    izhikevich_2003,
)
from sparx.dynamics.plasticity import (
    PairSTDP,
    STDPTraces,
    TripletSTDP,
    TripletTraces,
    TsodyksMarkram,
    TsodyksMarkramState,
)
from sparx.dynamics.synapses import (
    Alpha,
    Arrivals,
    BiExponential,
    Delta,
    Exponential,
    PointNeuron,
    Receptor,
    SynapseModel,
)

__all__ = [
    "IZHIKEVICH_2003",
    "LIF",
    "RECEPTORS",
    "AdEx",
    "AdExState",
    "Alpha",
    "Arrivals",
    "BiExponential",
    "Delta",
    "Exponential",
    "HodgkinHuxley",
    "HodgkinHuxleyState",
    "Izhikevich",
    "IzhikevichState",
    "LIFState",
    "MgBlock",
    "NeuronModel",
    "PairSTDP",
    "PointNeuron",
    "Receptor",
    "STDPTraces",
    "Spikes",
    "SynapseModel",
    "SynapticInput",
    "Term",
    "TripletSTDP",
    "TripletTraces",
    "TsodyksMarkram",
    "TsodyksMarkramState",
    "crossing",
    "exact_linear",
    "integrate",
    "izhikevich_2003",
    "response",
    "rk4",
    "substeps",
]
