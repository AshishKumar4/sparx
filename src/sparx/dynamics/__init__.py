"""Biophysical neuron, synapse and plasticity models in physical units.

`sparx.cells` holds the dimensionless, one-step-is-one-unit dynamics deep
spiking networks train with. This package holds models of neurons as
biology measures them: membrane equations in mV and ms with conductances,
refractoriness and reversal potentials, synapses with receptor kinetics,
and plasticity rules. They are what `sparx.graph` builds circuits and
connectomes from (design.md sections 4 and 5).
"""

from sparx.dynamics.core import NeuronModel, Spikes, SynapticInput, crossing, exact_linear, integrate
from sparx.dynamics.neurons import LIF, RECEPTORS, LIFState

__all__ = ["LIF", "RECEPTORS", "LIFState", "NeuronModel", "Spikes", "SynapticInput", "crossing",
           "exact_linear", "integrate"]
