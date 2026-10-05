"""The kinds sparx adds to dew's registry.

Dew's registry names everything a run is made of, so a run's record can be
rebuilt in another process. Sparx registers its models, objectives and
datasets into dew's own tables (`dew.registry.models`, `objectives`,
`datasets`), and adds three kinds dew does not have:

- `surrogates`: the derivatives spikes train through (`sparx.surrogate`).
- `neurons`: neuron layers (`sparx.nn`), which a model holds as a field.
- `spike_encoders`: how a batch field becomes a spike train (`sparx.dew`).

Each table records a member as `{"kind": name, **fields}` and is shared with
dew (`Registry.share`), so a model field declared as `Surrogate` or
`Neuron` rebuilds from its record and writes back by kind. Sparx names
itself a dew plugin: `[project.entry-points."dew.plugins"]` in
`pyproject.toml` names `sparx.plugin`, which dew imports when a lookup
misses its own index, so a run naming sparx's members loads in a process
that never imported sparx.
"""

from typing import Any

from dew.registry import Registry

__all__ = ["neurons", "spike_encoders", "surrogates"]

surrogates: Registry[Any, Any] = Registry("surrogate", record="kind").share()
neurons: Registry[Any, Any] = Registry("neuron", record="kind").share()
spike_encoders: Registry[Any, Any] = Registry("spike_encoder", record="kind").share()
