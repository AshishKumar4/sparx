"""Train a spiking classifier with dew's recipe machinery.

    python recipes/snn/train.py data:shd --data.channels 140 --trainer.batch-size 64 \\
        --trainer.steps 3000 --trainer.eval-every 500 --trainer.checkpoint-dir runs --trainer.name shd \\
        --model.config '{"hidden": [128], "classes": 20, "delays": 15,
                         "neuron": {"name": "alif", "fields": {"tau": 5.0, "tau_adapt": 20.0, "beta": 0.2,
                                    "learn_tau": true, "detach_reset": true}}}' \\
        --schedules '{"sigma": {"name": "linear", "fields": {"peak": 7.5, "end": 0.5}}}'
    JAX_PLATFORMS=cpu python recipes/snn/train.py --smoke --trainer.checkpoint-dir /tmp/snn-smoke

The configuration is dew's `RunConfig` (model, data, optimizer, trainer) plus the
classifier's own fields, and `run.json` records all of it, so `dew.pipeline(run_dir)`
loads the trained classifier back. `data` is any registered dataset (`data:shd`), the
model any registered spiking model (`spiking_mlp`, `sew_resnet`) and the encoder any
registered spike encoder (`encoder:rate --encoder.steps 8` for static data). The
schedules are records of dew's schedules, as the model's neuron is a record of a
registered neuron.

`--smoke` trains a small network for a few steps on synthetic recordings in SHD's
layout (`sparx.datasets.write_synthetic_shd`), written under the checkpoint
directory, so it needs no download.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import tyro
from dew.config import JsonDict, ModelConfig, OptimConfig, RunConfig
from dew.data import Loading
from dew.inputs import Field
from dew.records import record

# Remove `from_record` when AshishKumar4/dew#37 merges: the groups it rebuilds become `optim.param_groups`.
from dew.registry import datasets, from_record, schedules
from dew.training import TrainState, prepare_process, run_timestamp
from dew.training.optim import ScheduleBase

import sparx.datasets
from sparx.encode import Events, SpikeEncoder
from sparx.losses import Readout
from sparx.metrics import Accuracy
from sparx.objectives import RateBand, SpikingClassifierObjective
from sparx.optim import GroupAdam
from sparx.registry import spike_encoders

if TYPE_CHECKING:
    # tyro reads the runtime annotation, a Union of the registered members, and
    # a type checker cannot read a variable in a type expression, so both get
    # what they need, as dew.config's DataSpec does.
    type DataSpec = sparx.datasets.SHD
    type EncoderSpec = SpikeEncoder
    type ScheduleSpec = ScheduleBase
else:
    DataSpec = datasets.union
    EncoderSpec = spike_encoders.union
    ScheduleSpec = schedules.union


def _schedules(args: list[str]) -> dict[str, ScheduleSpec]:
    return {name: schedules.from_record(record(value, name)) for name, value in json.loads(args[0]).items()}


def _schedule_records(value: dict[str, ScheduleSpec]) -> list[str]:
    # Remove when AshishKumar4/dew#34 merges: dew.config.to_json, the same function made public.
    from dew.config import _to_json
    return [json.dumps({name: _to_json(schedule, ScheduleSpec) for name, schedule in value.items()})]


ScheduleMap = Annotated[
    dict[str, ScheduleSpec],
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1, metavar="JSON", instance_from_str=_schedules,
        is_instance=lambda value: isinstance(value, dict), str_from_instance=_schedule_records),
]
"""Schedules by name, written on the command line as one JSON object of schedule records.

tyro has no flag form for a mapping of registered records, so the object is
read here, and `RunConfig.to_dict` and `from_dict` write and rebuild each
entry by its declared type."""


@dataclass(frozen=True)
class SNNRunConfig(RunConfig):
    """A run, plus the spiking classifier's own fields."""

    objective: str = "spiking_classifier"
    model: ModelConfig = field(default_factory=lambda: ModelConfig("spiking_mlp", {
        "hidden": [128], "classes": 20,
        "neuron": {"name": "alif", "fields": {"tau": 5.0, "tau_adapt": 20.0, "beta": 0.2, "learn_tau": True,
                                               "detach_reset": True}}}, dtype="float32"))
    data: DataSpec = field(default_factory=sparx.datasets.SHD)
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(learning_rate=2e-3, clip_grads=1.0))
    encoder: EncoderSpec = field(default_factory=Events)
    """How a batch field becomes spikes; `encoder:rate --encoder.steps 8` for static data."""
    sample: str = "spikes"
    """The batch field the encoder reads."""
    labels: str = "label"
    readout: Readout = "max"
    rates: RateBand | None = field(default_factory=lambda: RateBand(lower=0.01, upper=0.3))
    schedules: ScheduleMap = field(default_factory=dict)
    """Model keyword arguments on a schedule over the run, by name: records of dew schedules."""
    schedule_every: int = 1
    """Steps between advances of every schedule; an epoch's steps give torch's per-epoch schedulers."""
    deployed: dict[str, float] = field(default_factory=dict)
    """Model keyword arguments evaluation runs with in place of the schedules', such as `sigma 0`."""
    # Remove when AshishKumar4/dew#37 merges: `optim.param_groups`, each a `ParamGroup` with its own
    # schedule, b1, bounds and weight decay, so the run has one optimizer field.
    groups: JsonDict = field(default_factory=dict)
    """Parameter groups with optimizers of their own, by name, each the fields of a `sparx.optim.GroupAdam`.

    `optim` updates the parameters no group matches.
    """
    smoke: bool = False
    """Train a small network for a few steps on synthetic SHD-layout recordings; nothing is downloaded."""


def smoke_config(config: SNNRunConfig) -> SNNRunConfig:
    """`config` shrunk to a run of a few seconds on CPU, reading synthetic recordings.

    The recordings go under the checkpoint directory, so the run reads them
    there and leaves nothing elsewhere.
    """
    cache = sparx.datasets.write_synthetic_shd(Path(config.trainer.checkpoint_dir) / "synthetic-shd")
    data = sparx.datasets.SHD(steps=20, channels=70, cache=str(cache),
                              loading=Loading(workers=0, threads=1, read_buffer=1))
    model = dataclasses.replace(config.model, config={
        **config.model.config, "hidden": [16], "classes": 2})
    trainer = dataclasses.replace(config.trainer, batch_size=16, steps=8, log_every=4, eval_every=8,
                                  checkpoint_every=8, multi_host=False, compilation_cache_dir=None,
                                  name=config.trainer.name or "smoke")
    return dataclasses.replace(config, data=data, model=model, trainer=trainer)


def sample_field(config: SNNRunConfig) -> Field:
    """The batch field the encoder reads, at the per-record shape the dataset writes."""
    spec = config.data
    if isinstance(spec, sparx.datasets.SHD):
        return Field(config.sample, (spec.steps, spec.channels))
    raise ValueError(f"the recipe knows the record shape of shd, not {datasets.name_of(type(spec))}")


def main(config: SNNRunConfig) -> TrainState:
    if config.smoke:
        config = smoke_config(config)
    prepare_process(config.trainer.wandb, config.trainer.multi_host, config.trainer.xla_flags,
                    config.trainer.compilation_cache_dir, layout=config.trainer.layout)
    data = config.data.load(batch=config.trainer.batch_size)
    steps = config.trainer.total_steps(data)
    groups = {name: from_record(GroupAdam, value) for name, value in config.groups.items()}
    objective = SpikingClassifierObjective(
        config.model.build(), sample_field(config), config.encoder, labels=config.labels,
        readout=config.readout, rates=config.rates, schedules=config.schedules, schedule_steps=steps,
        schedule_every=config.schedule_every, deployed=config.deployed, groups=groups)
    name = config.trainer.name or (
        f"snn-{datasets.name_of(type(config.data))}/{config.model.architecture}/date-{run_timestamp()}")
    return config.train(objective, data, name=name, metrics=[Accuracy()],
                        summary={"model": dict(config.model.fields()), "readout": config.readout,
                                 "dataset": dataclasses.asdict(config.data)})


if __name__ == "__main__":
    main(SNNRunConfig.cli())
