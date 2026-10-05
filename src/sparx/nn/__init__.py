"""Flax linen layers for spiking networks, over time-major inputs `[T, ...]`."""

from .delays import DelayedDense, delay_kernel
from .neurons import (
    ALIF,
    IF,
    LI,
    LIF,
    RATES,
    STATE,
    Izhikevich,
    Neuron,
    Recurrent,
    Synaptic,
    decay,
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
    "Flatten",
    "Izhikevich",
    "MaskedPSN",
    "Neuron",
    "Recurrent",
    "SlidingPSN",
    "Synaptic",
    "band_mask",
    "decay",
    "delay_kernel",
    "record_rates",
]
