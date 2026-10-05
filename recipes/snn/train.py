"""Train a spiking classifier with dew's recipe machinery.

    python recipes/snn/train.py data:shd --data.channels 140 --trainer.batch-size 64 \\
        --trainer.steps 3000 --trainer.eval-every 500 --trainer.checkpoint-dir runs --trainer.name shd \\
        --model.config '{"hidden": [128], "classes": 20, "delays": 15,
                         "neuron": {"name": "alif", "fields": {"tau": 5.0, "tau_adapt": 20.0, "beta": 0.2,
                                    "learn_tau": true, "detach_reset": true}}}' \\
        --schedules '{"sigma": {"name": "linear", "fields": {"peak": 7.5, "end": 0.5}}}'

The configuration is dew's `RunConfig` (model, data, optimizer, trainer) plus the
classifier's own fields, and `run.json` records all of it, so `dew.pipeline(run_dir)`
loads the trained classifier back. The model is any registered spiking model
(`spiking_mlp`, `sew_resnet`); the encoder and schedules are records of registered
members, as the model's neuron is.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import tyro
from dew.config import JsonDict, ModelConfig, OptimConfig, RunConfig
from dew.inputs import Field
from dew.records import record
from dew.registry import datasets, schedules
from dew.training import TrainState, prepare_process, run_timestamp

import sparx.datasets
from sparx.dew import RateBand, SpikingClassifier, accuracy
from sparx.registry import spike_encoders


@dataclass(frozen=True)
class SNNRunConfig(RunConfig):
    """A run, plus the spiking classifier's own fields."""

    objective: str = "spiking_classifier"
    model: ModelConfig = field(default_factory=lambda: ModelConfig("spiking_mlp", {
        "hidden": [128], "classes": 20,
        "neuron": {"name": "alif", "fields": {"tau": 5.0, "tau_adapt": 20.0, "beta": 0.2, "learn_tau": True,
                                               "detach_reset": True}}}, dtype="float32"))
    data: sparx.datasets.SHD = field(default_factory=sparx.datasets.SHD)
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(learning_rate=2e-3, clip_grads=1.0))
    encoder: JsonDict = field(default_factory=lambda: {"name": "events", "fields": {}})
    """The spike encoder's record, `{"name": "rate", "fields": {"steps": 8}}` for static data."""
    sample: str = "spikes"
    """The batch field the encoder reads."""
    labels: str = "label"
    readout: Literal["mean", "max", "sum", "per_step"] = "max"
    rates: RateBand | None = field(default_factory=lambda: RateBand(lower=0.01, upper=0.3))
    schedules: JsonDict = field(default_factory=dict)
    """Model keyword arguments on a schedule over the run, by name: records of dew schedules."""


def sample_field(config: SNNRunConfig) -> Field:
    """The batch field the encoder reads, at the per-record shape the dataset writes."""
    spec = config.data
    if isinstance(spec, sparx.datasets.SHD):
        return Field(config.sample, (spec.steps, spec.channels))
    raise ValueError(f"the recipe knows the record shape of shd, not {datasets.name_of(type(spec))}")


def main(config: SNNRunConfig) -> TrainState:
    prepare_process(config.trainer.wandb, config.trainer.multi_host, config.trainer.xla_flags,
                    config.trainer.compilation_cache_dir, layout=config.trainer.layout)
    data = config.data.load(batch=config.trainer.batch_size)
    steps = config.trainer.total_steps(data)
    encoder = spike_encoders.from_record(record(config.encoder, "encoder"))
    scheduled = {name: schedules.from_record(record(value, name)) for name, value in config.schedules.items()}
    objective = SpikingClassifier(config.model.build(), sample_field(config), encoder, labels=config.labels,
                                  readout=config.readout, rates=config.rates, schedules=scheduled,
                                  schedule_steps=steps)
    name = config.trainer.name or (
        f"snn-{datasets.name_of(type(config.data))}/{config.model.architecture}/date-{run_timestamp()}")
    return config.train(objective, data, name=name, metrics=[accuracy],
                        summary={"model": dict(config.model.fields()), "readout": config.readout,
                                 "dataset": dataclasses.asdict(config.data)})


if __name__ == "__main__":
    main(tyro.cli(tyro.conf.CascadeSubcommandArgs[SNNRunConfig]))
