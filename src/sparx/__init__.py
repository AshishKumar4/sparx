"""Sparx: spiking neural networks in JAX and Flax, trained, simulated and served through dew.

Networks are Flax linen modules over time-major spike trains `[T, ...]`.

Deep spiking networks:

- `sparx.nn`: Flax layers over the neuron models, parallel spiking neurons and delayed synapses.
- `sparx.models`: architectures built from them (`SEWResNet`, `SpikingMLP`).
- `sparx.surrogate`: the spike and its surrogate gradients.
- `sparx.encode`: the registered encoders that turn data into spike trains.
- `sparx.losses` and `sparx.rates`: losses over time, firing-rate readouts and penalties.
- `sparx.learn`: rules beyond backpropagation through time (e-prop, OTTT, EventProp, conversion).

Circuits in physical units:

- `sparx.dynamics`: neuron, synapse and plasticity models as pure JAX, the dimensionless family deep
  networks train with and the physical one, and `run`, which scans any of them over time.
- `sparx.graph`: populations and projections wired into a `Network`, `simulate`, and connectomes.
- `sparx.spiketrains`: statistics of and distances between recorded spike trains.

Around them:

- `sparx.objectives`: the objectives that train spiking networks under dew's `Trainer`, with
  `sparx.metrics` (their accuracy), `sparx.tasks` (the trained classifier `dew.pipeline` loads) and
  `sparx.optim` (the schedules and per-group Adam SNN-delays needs).
- `sparx.datasets`: spiking datasets as dew datasets (SHD).
- `sparx.serve`: `StreamServer`, many streaming sessions in one batch.
- `sparx.nir`: exchange through the Neuromorphic Intermediate Representation.

`sparx.graph`, `sparx.learn`, `sparx.objectives`, `sparx.metrics`,
`sparx.tasks`, `sparx.optim`, `sparx.datasets`, `sparx.serve` and
`sparx.nir` load on first access (`sparx.graph.Network` after `import
sparx`). The graph, the objectives and the datasets import dew's trainer
and data stack, about 0.9 s on a 4-core CPU, which a script that only
trains a network in its own loop does not need.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

from sparx import dynamics, encode, losses, models, nn, rates, spiketrains, surrogate
from sparx.dynamics import run
from sparx.rates import firing_rates, rate_penalty
from sparx.surrogate import spike

if TYPE_CHECKING:
    from sparx import datasets, graph, learn, metrics, nir, objectives, optim, serve, tasks

__version__ = "0.1.0"

_LAZY = ("datasets", "graph", "learn", "metrics", "nir", "objectives", "optim", "serve", "tasks")

__all__ = ["__version__", "datasets", "dynamics", "encode", "firing_rates", "graph", "learn", "losses",
           "metrics", "models", "nir", "nn", "objectives", "optim", "rate_penalty", "rates", "run", "serve",
           "spike", "spiketrains", "surrogate", "tasks"]


def __getattr__(name: str) -> ModuleType:
    # PEP 562: importing the submodule binds it on the package, so this runs once per name.
    if name in _LAZY:
        return importlib.import_module(f"sparx.{name}")
    raise AttributeError(f"module 'sparx' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
