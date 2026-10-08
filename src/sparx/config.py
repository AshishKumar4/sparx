"""The run a spiking classifier trains as, one typed record that `run.json` holds.

    python recipes/snn/train.py --data.channels 140 --trainer.batch-size 64 --trainer.steps 3000 \\
        --model.hidden 128 --model.classes 20 --model.delays 15 \\
        --objective.schedules '{"sigma": {"class": "linear", "fields": {"peak": 7.5, "end": 0.5}}}'
    dew train runs/<name>/run.json --trust sparx --set trainer.steps=6000   # the same run, trained on

`SNNRunConfig` is dew's `RunConfig` with the data, encoder and sample field
a spiking classifier reads. Its model is any spiking model over `[T, B,
channels]` by import path (`--model my_package.models:Net`), each of its
fields a flag and the neuron template a record (`--model.neuron '{"class":
"sparx.nn.neurons:LIF", "fields": {"tau": 3.0}}'`). The objective is
`SpikingClassifierObjective`, each of its keyword arguments a flag
(`--objective.readout max`), and `prepare` builds it around the model, the
sample field and the encoder. Parameter groups with optimizers of their own
are dew's (`--optim.param-groups`).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, Prepared, RunConfig
from dew.data import Loading
from dew.inputs import Field
from dew.registry import import_path

from sparx.datasets import SHD, write_synthetic_shd
from sparx.encode import EventsEncoder, SpikeEncoder
from sparx.metrics import Accuracy
from sparx.models import SpikingMLP
from sparx.nn.neurons import ALIF
from sparx.objectives import SpikingClassifierObjective
from sparx.registry import spike_encoders

if TYPE_CHECKING:
    # tyro reads the runtime annotation, a union of the encoders, and a type
    # checker cannot read a variable in a type expression, so both get what
    # they need, as dew.config's DataSpec does.
    type EncoderSpec = SpikeEncoder
else:
    EncoderSpec = spike_encoders.union

__all__ = ["SNNRunConfig"]


@dataclasses.dataclass(frozen=True)
class SNNRunConfig(RunConfig):
    """A run of `SpikingClassifierObjective` on SHD, with the encoder and the field it reads."""

    objective: ObjectiveConfig = dataclasses.field(default_factory=lambda: ObjectiveConfig(
        import_path(SpikingClassifierObjective), {"readout": "max", "rates": {"lower": 0.01, "upper": 0.3}}))
    """The objective and its keyword arguments (`--objective.readout mean`)."""
    model: ModelConfig = dataclasses.field(default_factory=lambda: ModelConfig(import_path(SpikingMLP), {
        "hidden": [128], "classes": 20, "dtype": "float32",
        "neuron": {"class": import_path(ALIF), "fields": {
            "tau": 5.0, "tau_adapt": 20.0, "beta": 0.2, "learn_tau": True, "detach_reset": True}}}))
    data: SHD = dataclasses.field(default_factory=SHD)
    optim: OptimConfig = dataclasses.field(default_factory=lambda: OptimConfig(learning_rate=2e-3,
                                                                                clip_grads=1.0))
    encoder: EncoderSpec = dataclasses.field(default_factory=EventsEncoder)
    """How a batch field becomes spikes; `encoder:rate --encoder.steps 8` for static data."""
    sample: str = "spikes"
    """The batch field the encoder reads."""
    smoke: bool = False
    """Train a small network for a few steps on synthetic SHD-layout recordings; nothing is downloaded."""

    def smoked(self) -> SNNRunConfig:
        """This run shrunk to a few seconds on CPU, reading synthetic recordings.

        The recordings go under the checkpoint directory, so the run reads
        them there and leaves nothing elsewhere. The run it returns is no
        longer a smoke run: it records the small run as it trains, so its
        `run.json` trains that run again.
        """
        cache = write_synthetic_shd(Path(self.trainer.checkpoint_dir) / "synthetic-shd")
        synthetic = SHD(steps=20, channels=70, cache=str(cache),
                        loading=Loading(workers=0, threads=1, read_buffer=1))
        model = dataclasses.replace(self.model, fields={**self.model.fields, "hidden": [16], "classes": 2})
        trainer = dataclasses.replace(self.trainer, batch_size=16, steps=8, log_every=4, eval_every=8,
                                      checkpoint_every=8)
        return dataclasses.replace(self, data=synthetic, model=model, trainer=trainer, smoke=False)

    def sample_field(self) -> Field:
        """The batch field the encoder reads, at the per-record shape the dataset writes."""
        return Field(self.sample, (self.data.steps, self.data.channels))

    def prepare(self) -> Prepared:
        """The objective the run names, around its model, the sample field and the encoder, on SHD.

        Its schedules run over the run's steps unless the run states
        `schedule_steps`.
        """
        run = self.smoked() if self.smoke else self
        dataset = run.data.load(batch=run.trainer.batch_size)
        derived = ({} if "schedule_steps" in run.objective.fields
                   else {"schedule_steps": run.trainer.total_steps(dataset)})
        objective = run.objective.build(model=run.model.build(), sample=run.sample_field(),
                                        encoder=run.encoder, **derived)
        return Prepared(run, lambda name: run.train(objective, dataset, name=name, metrics=[Accuracy()]))
