"""Trained spiking networks as inference tasks, which `dew.pipeline` loads from a run.

`SpikingClassification` is what a run of `SpikingClassifierObjective` loads
as (its `saved_task`). It follows dew's `SavedTask` protocol, so
`dew.pipeline(run_dir, trust=("sparx",))` builds it in a fresh process from
the run's record and checkpoint, and `objective.pipeline(state)` builds it
from a state still in memory.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import flax.linen as nn
import jax
import jax.numpy as jnp
from dew.objectives.base import Variables, thaw
from dew.registry import from_record
from dew.training.optim import ScheduleBase

from sparx.encode import SpikeEncoder
from sparx.losses import READOUTS, Readout, readout_logits

if TYPE_CHECKING:
    from dew.training.distributed import Layout, MeshSpec
    from jax.typing import DTypeLike

__all__ = ["SpikingClassification", "bound_call"]


def bound_call(model: nn.Module, *, train: bool,
               kwargs: Mapping[str, jax.Array | float]) -> functools.partial[jax.Array]:
    """The model's `__call__` with the keyword arguments `kwargs` bound, and `train` when it takes one, as
    `apply`'s method.

    A model with BatchNorm or dropout takes `train`, as dew's models do, so
    they know which pass they are in. A stack without them, a flax
    `nn.Sequential` of sparx layers say, need not.
    """
    call = type(model).__call__
    if "train" in inspect.signature(call).parameters:
        return functools.partial(call, train=train, **kwargs)
    return functools.partial(call, **kwargs)


@dataclass(frozen=True)
class SpikingClassification:
    """A trained spiking classifier: class scores and predictions for a batch field.

    `call` holds the model keyword arguments it runs with (a trained
    `DelayedDense`'s `sigma`, say); `0` for `sigma` reads the rounded
    delays, the network as deployed.
    """

    model: nn.Module
    variables: Variables
    encoder: SpikeEncoder
    readout: Readout = "mean"
    call: Mapping[str, float] = field(default_factory=dict)

    @functools.cached_property
    def _logits(self):
        method = bound_call(self.model, train=False, kwargs=self.call)

        def logits(variables: Variables, x: jax.Array, key: jax.Array) -> jax.Array:
            outputs = self.model.apply(variables, self.encoder(key, x), method=method)
            # `mutable` is unset, so apply returns the outputs alone.
            assert not isinstance(outputs, tuple)
            return readout_logits(self.readout, outputs)

        return jax.jit(logits)

    def logits(self, x: jax.Array, key: jax.Array | int = 0) -> jax.Array:
        """Class scores `[B, classes]` for a batch field `[B, ...]`; `key` drives a random encoder."""
        key = jax.random.key(key) if isinstance(key, int) else key
        return self._logits(self.variables, jnp.asarray(x), key)

    def __call__(self, x: jax.Array, key: jax.Array | int = 0) -> jax.Array:
        """Predicted classes `[B]` for a batch field `[B, ...]`."""
        return jnp.argmax(self.logits(x, key), axis=-1)

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: MeshSpec | None = None, layout: Layout | None = None,
                 dtype: DTypeLike | None = None,
                 param_dtype: DTypeLike | None = None) -> SpikingClassification:
        """Load the classifier a run of `SpikingClassifierObjective` in `directory` saved.

        The model is the one its record names, over the selected checkpoint's
        weights (the average when the run kept one, unless `ema` is False),
        placed on `mesh` under `layout`; the record and the weights are read
        from one pinned step (`dew.inference.tasks.run_record`). `dtype`
        replaces the recorded compute dtype and `param_dtype` the dtype the
        weights are read in.
        """
        from dew.checkpoints import Checkpoints
        from dew.inference.tasks import run_record, saved_model
        from dew.records import number, record as named_fields

        record, pinned = run_record(directory, step)
        config = saved_model(record, dtype)
        variables = Checkpoints(directory).variables(ema=ema, step=pinned, mesh=mesh, layout=layout,
                                                     param_dtype=param_dtype)
        encoder = from_record(SpikeEncoder, named_fields(record["encoder"], "encoder"))
        steps = record.get("schedule_steps")
        call: dict[str, float] = {}
        if isinstance(steps, int):
            for name, value in named_fields(record["schedules"], "schedules").items():
                schedule = from_record(ScheduleBase, named_fields(value, name))
                call[name] = float(jnp.asarray(schedule.schedule(steps)(steps)))
        deployed = record.get("deployed") or {}
        for name, value in named_fields(deployed, "deployed").items():
            call[name] = number(value, f"deployed {name}")
        for readout in READOUTS:
            if readout == record["readout"]:
                return cls(config.build(), thaw(variables), encoder, readout, call)
        raise ValueError(f"the run records an unknown readout {record['readout']!r}")
