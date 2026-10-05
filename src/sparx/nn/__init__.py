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
    Neuron,
    Recurrent,
    Synaptic,
    adopt,
    history_window,
    record_rates,
)
from .parallel import PSN, MaskedPSN, SlidingPSN, band_mask
from .reshape import Flatten

__all__ = [
    "ALIF",
    "IF",
    "LI",
    "LIF",
    "PSN",
    "RATES",
    "STATE",
    "DelayedDense",
    "Dynamics",
    "Flatten",
    "Izhikevich",
    "MaskedPSN",
    "Neuron",
    "Recurrent",
    "SlidingPSN",
    "Synaptic",
    "adopt",
    "band_mask",
    "delay_kernel",
    "history_window",
    "record_rates",
]
