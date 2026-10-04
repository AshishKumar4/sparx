"""Sparx: spiking neural networks in JAX and Flax.

Networks are Flax linen modules over time-major spike trains `[T, ...]`;
neuron dynamics are pure JAX cells (`sparx.cells`) that `sparx.nn` wraps as
layers. `sparx.dew` adapts them to dew's `Trainer`.
"""

from sparx import cells, nn, surrogate
from sparx.cells import run
from sparx.surrogate import spike

__version__ = "0.1.0"

__all__ = ["__version__", "cells", "nn", "run", "spike", "surrogate"]
