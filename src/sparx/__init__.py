"""Sparx: spiking neural networks in JAX and Flax.

Networks are Flax linen modules over time-major spike trains `[T, ...]`.

- `sparx.surrogate`: the spike and its surrogate gradients.
- `sparx.cells`: neuron dynamics as pure JAX, and `run`, which scans them over time.
- `sparx.nn`: Flax layers over those cells, parallel spiking neurons and delayed synapses.
- `sparx.models`: architectures built from them (SEW ResNet).
- `sparx.encode`: data to spike trains.
- `sparx.losses` and `sparx.rates`: losses over time, firing-rate readouts and penalties.
- `sparx.dew`: spiking objectives for dew's `Trainer` (needs dew installed).
"""

from sparx import cells, encode, losses, models, nn, rates, surrogate
from sparx.cells import run
from sparx.rates import firing_rates, rate_penalty
from sparx.surrogate import spike

__version__ = "0.1.0"

__all__ = ["__version__", "cells", "encode", "firing_rates", "losses", "models", "nn", "rate_penalty",
           "rates", "run", "spike", "surrogate"]
