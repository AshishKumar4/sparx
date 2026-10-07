"""Flax linen layers for spiking networks, over time-major inputs `[T, ...]`."""

from .delays import DelayedDense, delay_kernel
from .neurons import (
    ALIF,
    IF,
    LI,
    LIF,
    RATES,
    STATE,
    Dynamics,
    Izhikevich,
    Modelled,
    Neuron,
    Rate,
    Recurrent,
    Synaptic,
    adopt,
    history_window,
    record_rates,
)
from .parallel import PSN, MaskedPSN, SlidingPSN, band_mask
from .plastic import DecayingTrace, HebbianTrace, ModulatedTrace, OjaTrace, Plastic, RetroactiveTrace
from .reshape import BatchMajor, Flatten, Flattens

__all__ = [
    "ALIF",
    "IF",
    "LI",
    "LIF",
    "PSN",
    "RATES",
    "STATE",
    "BatchMajor",
    "DecayingTrace",
    "DelayedDense",
    "Dynamics",
    "Flatten",
    "Flattens",
    "HebbianTrace",
    "Izhikevich",
    "MaskedPSN",
    "Modelled",
    "ModulatedTrace",
    "Neuron",
    "OjaTrace",
    "Plastic",
    "Rate",
    "Recurrent",
    "RetroactiveTrace",
    "SlidingPSN",
    "Synaptic",
    "adopt",
    "band_mask",
    "delay_kernel",
    "history_window",
    "record_rates",
]
