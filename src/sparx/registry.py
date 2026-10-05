"""The kinds sparx adds to dew's registry.

Dew's registry names everything a run is made of, so a run's record can be
rebuilt in another process. Sparx registers its models, objectives and
datasets into dew's own tables (`dew.registry.models`, `objectives`,
`datasets`), and adds three kinds dew does not have:

- `surrogates`: the derivatives spikes train through (`sparx.surrogate`).
- `neurons`: neuron layers (`sparx.nn`), which a model holds as a field.
- `spike_encoders`: how a batch field becomes a spike train (`sparx.dew`).

Each table records a member as `{"kind": name, **fields}` and is shared with
dew (`dew.registry.share`), so a model field declared as `Surrogate` or
`Neuron` rebuilds from its record and writes back by kind. Sparx names
itself a dew plugin (`[project.entry-points."dew.plugins"]` in
`pyproject.toml`), so dew finds these registrations without importing sparx
first.
"""

from typing import Any

from dew.registry import Registry, share

__all__ = ["neurons", "spike_encoders", "surrogates"]

surrogates: Registry[Any, Any] = share(Registry("surrogate", record="kind"))
neurons: Registry[Any, Any] = share(Registry("neuron", record="kind"))
spike_encoders: Registry[Any, Any] = share(Registry("spike_encoder", record="kind"))
