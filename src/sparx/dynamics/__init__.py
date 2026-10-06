"""Neuron, synapse and plasticity models, and the runner that scans them over time.

Every neuron model meets one contract (`sparx.dynamics.core`):
`init_state(shape, dtype)` and `step(state, SynapticInput, dt) -> (state,
Output)`, and `run(model, inputs)` scans one over time. A model's output is
its spikes, or a graded value each step when the model is `graded`. Two families meet
it. `sparx.dynamics.ml` holds the dimensionless, one-step-is-one-unit
models deep spiking networks train with: soft resets, detached resets,
learnable decays, input as a jump of the membrane. `sparx.dynamics.neurons`
holds models of neurons as biology measures them: membrane equations in mV
and ms with conductances, refractoriness and reversal potentials. Synapses
with receptor kinetics and plasticity rules complete them; `sparx.nn` builds
layers from these models and `sparx.graph` builds circuits and connectomes
(design.md sections 4 and 5).

This package exports the models and the contract. The arithmetic the
models share (`fire`, `exact_linear`, `rk4`, `substeps` and the rest), which
a new model is written with, stays in `sparx.dynamics.core`.
"""

from sparx.dynamics.core import Model, NeuronModel, Output, Reset, SynapticInput, Term, decay, run
from sparx.dynamics.ml import (
    ACTIVATIONS,
    ALIFCell,
    ALIFState,
    LICell,
    LIFCell,
    MembraneState,
    RateCell,
    RateState,
    RecurrentCell,
    RecurrentState,
    Serial,
)
from sparx.dynamics.neurons import (
    IZHIKEVICH_2003,
    IZHIKEVICH_2004,
    LIF,
    RECEPTORS,
    AdEx,
    AdExState,
    GradedPotential,
    GradedPotentialState,
    HodgkinHuxley,
    HodgkinHuxleyState,
    Izhikevich,
    IzhikevichState,
    LIFState,
    MgBlock,
    izhikevich_2003,
    izhikevich_2004,
)
from sparx.dynamics.plasticity import (
    PairSTDP,
    Plasticity,
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
    Graded,
    GradedState,
    Landing,
    PointNeuron,
    PointNeuronState,
    Receptor,
    SynapseModel,
)

__all__ = [
    "ACTIVATIONS",
    "IZHIKEVICH_2003",
    "IZHIKEVICH_2004",
    "LIF",
    "RECEPTORS",
    "ALIFCell",
    "ALIFState",
    "AdEx",
    "AdExState",
    "Alpha",
    "Arrivals",
    "BiExponential",
    "Delta",
    "Exponential",
    "Graded",
    "GradedPotential",
    "GradedPotentialState",
    "GradedState",
    "HodgkinHuxley",
    "HodgkinHuxleyState",
    "Izhikevich",
    "IzhikevichState",
    "LICell",
    "LIFCell",
    "LIFState",
    "Landing",
    "MembraneState",
    "MgBlock",
    "Model",
    "NeuronModel",
    "Output",
    "PairSTDP",
    "Plasticity",
    "PointNeuron",
    "PointNeuronState",
    "RateCell",
    "RateState",
    "Receptor",
    "RecurrentCell",
    "RecurrentState",
    "Reset",
    "STDPTraces",
    "Serial",
    "SynapseModel",
    "SynapticInput",
    "Term",
    "TripletSTDP",
    "TripletTraces",
    "TsodyksMarkram",
    "TsodyksMarkramState",
    "decay",
    "izhikevich_2003",
    "izhikevich_2004",
    "run",
]
