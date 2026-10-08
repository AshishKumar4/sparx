"""Flax linen layers for spiking networks, over time-major inputs `[T, ...]`."""

from .delays import DelayedDense, delay_kernel
from .hebbian import DecayingTrace, HebbianTrace, ModulatedTrace, OjaTrace, RetroactiveTrace
from .neurons import (
    ALIF,
    IF,
    LI,
    LIF,
    RATES,
    STATE,
    Dynamics,
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
    "MaskedPSN",
    "Modelled",
    "ModulatedTrace",
    "Neuron",
    "OjaTrace",
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
