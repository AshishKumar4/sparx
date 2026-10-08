"""Short names for the sparx classes and builders that sparx's own code reads by name.

Dew's records name a class or function by its import path,
`{"class": "sparx.nn.neurons:ALIF", "fields": {...}}`, so a run's record
rebuilds a spiking model, neuron, surrogate or encoder with nothing
registered: a field typed `Neuron` takes the record of any class derived
from it. A run of sparx loads in a fresh process once the reader trusts the
package, `dew.pipeline(run_dir, trust=("sparx",))` or `--trust sparx`.

Two kinds also have short names, each a dew `Aliases` table, for the places
where a person writes the record:

- `spike_encoders`: how a batch field becomes a spike train
  (`sparx.encode`), the recipe's `encoder:rate` subcommand.
- `networks`: the builders of `sparx.graph.Network`s, canonical circuits
  and models on connectomes (`sparx.graph.models`, `sparx.graph.connectome`),
  which `sparx.graph.from_record` reads: `{"class": "brunel", "fields":
  {"order": 2500, "g": 5.0}}`.

A record nested in another names its class or function by import path, as
every dew record does: a model on a connectome names the reader of its
tables, `{"class": "shiu2024", "fields": {"connectome": {"class":
"sparx.graph.connectome:Connectome.from_shiu", "fields": {...}}}}`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from dew.registry import Aliases

if TYPE_CHECKING:
    from sparx.encode import SpikeEncoder
    from sparx.graph.network import Network

__all__ = ["networks", "spike_encoders"]

spike_encoders: Aliases[type[SpikeEncoder], SpikeEncoder] = Aliases("spike_encoder", {
    "delta": "sparx.encode:DeltaEncoder",
    "direct": "sparx.encode:DirectEncoder",
    "events": "sparx.encode:EventsEncoder",
    "latency": "sparx.encode:LatencyEncoder",
    "rate": "sparx.encode:RateEncoder",
}, base="sparx.encode:SpikeEncoder")
networks: Aliases[Callable[..., Network], Network] = Aliases("network", {
    "brunel": "sparx.graph.models:brunel",
    "coba": "sparx.graph.models:coba",
    "cuba": "sparx.graph.models:cuba",
    "shiu2024": "sparx.graph.connectome:shiu2024",
}, base="sparx.graph.network:Network")
