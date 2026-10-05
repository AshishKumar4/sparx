"""Sparx: spiking neural networks in JAX and Flax.

Networks are Flax linen modules over time-major spike trains `[T, ...]`.

- `sparx.surrogate`: the spike and its surrogate gradients.
- `sparx.dynamics`: neuron models as pure JAX, the dimensionless family deep networks train with and
  the physical one, and `run`, which scans any of them over time.
- `sparx.nn`: Flax layers over those models, parallel spiking neurons and delayed synapses.
- `sparx.models`: architectures built from them (SEW ResNet).
- `sparx.encode`: the registered encoders that turn data into spike trains.
- `sparx.losses` and `sparx.rates`: losses over time, firing-rate readouts and penalties.
- `sparx.spiketrains`: statistics of and distances between recorded spike trains.
- `sparx.dew`: spiking objectives for dew's `Trainer` (needs dew installed).
"""

from sparx import dynamics, encode, losses, models, nn, rates, spiketrains, surrogate
from sparx.dynamics import run
from sparx.rates import firing_rates, rate_penalty
from sparx.surrogate import spike

__version__ = "0.1.0"

__all__ = ["__version__", "dynamics", "encode", "firing_rates", "losses", "models", "nn", "rate_penalty",
           "rates", "run", "spike", "spiketrains", "surrogate"]
