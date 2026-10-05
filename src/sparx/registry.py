"""The kinds sparx adds to dew's registry.

Dew's registry names everything a run is made of, so a run's record can be
rebuilt in another process. Sparx registers its models, objectives and
datasets into dew's own tables (`dew.registry.models`, `objectives`,
`datasets`), and adds five kinds dew does not have:

- `surrogates`: the derivatives spikes train through (`sparx.surrogate`).
- `neurons`: neuron layers (`sparx.nn`), which a model holds as a field.
- `spike_encoders`: how a batch field becomes a spike train (`sparx.encode`).
- `networks`: the builders of `sparx.graph.Network`s, canonical circuits
  and models on connectomes (`sparx.graph.models`, `sparx.graph.connectome`).
- `connectomes`: the readers of connectome tables (`sparx.graph.connectome`),
  which a network built on a connectome names in its record.

A network's record names its builder and the builder's arguments,
`{"name": "brunel", "fields": {"order": 2500, "g": 5.0}}`; a model on a
connectome names the reader of its tables in the field `connectome`,
`{"name": "shiu2024", "fields": {"connectome": {"name": "flywire",
"fields": {...}}, ...}}` (design.md section 8). `sparx.graph.from_record`
rebuilds the network.

Each table records a member as dew records every registered member,
`{"name": name, "fields": {...}}`, and is shared with dew
(`Registry.share`), so a model field declared as `Surrogate` or `Neuron`
rebuilds from its record and writes back by name. Sparx names
itself a dew plugin: `[project.entry-points."dew.plugins"]` in
`pyproject.toml` names `sparx.plugin`, which dew imports when a lookup
misses its own index, so a run naming sparx's members loads in a process
that never imported sparx.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dew.registry import Registry

if TYPE_CHECKING:
    from sparx.graph.connectome import Connectome
    from sparx.graph.network import Network

__all__ = ["connectomes", "networks", "neurons", "spike_encoders", "surrogates"]

surrogates: Registry[Any, Any] = Registry("surrogate").share()
neurons: Registry[Any, Any] = Registry("neuron").share()
spike_encoders: Registry[Any, Any] = Registry("spike_encoder").share()
networks: Registry[Callable[..., Network], Network] = Registry("network").share()
connectomes: Registry[Callable[..., Connectome], Connectome] = Registry("connectome").share()
