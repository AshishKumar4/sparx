"""Spiking networks as dew objectives, trained by dew's `Trainer`.

Dew splits a run into a model (a Flax module), an objective (parameters,
loss, evaluation) and a trainer (mesh, compiled step, EMA, checkpoints,
logging). A sparx network is a Flax module, so the one piece this module adds
is the objective: `SpikingClassifier` encodes a batch into spikes, runs the
network over time, and reads its outputs as class scores.

    import optax
    from dew import Field, Trainer
    from dew.data import Dataset
    from sparx.dew import SpikingClassifier, accuracy
    from sparx.encode import Rate

    objective = SpikingClassifier(net, Field("image", (28, 28, 1)), Rate(steps=16))
    trainer = Trainer(objective, optax.adam(1e-3), key=0)
    state = trainer.fit(Dataset.from_records({"image": x, "label": y}, batch=128),
                        steps=2000, metrics=[accuracy])

Importing this module needs dew, which sparx installs.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from dew.artifacts import TokenScores
from dew.eval import Mean
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, Batch, EMASpec, Objective, Ratio, Shown, Step, Variables, thaw
from dew.records import JSON
from dew.registry import objectives, schedules as dew_schedules
from dew.training.optim import ScheduleBase

from sparx.encode import SpikeEncoder
from sparx.losses import per_step_cross_entropy, van_rossum
from sparx.nn import RATES
from sparx.rates import firing_rates, rate_penalty
from sparx.registry import spike_encoders

__all__ = [
    "ActivityFit",
    "RateBand",
    "Readout",
    "SpikingClassification",
    "SpikingClassifier",
    "accuracy",
]


type Readout = Literal["mean", "max", "sum", "per_step"]
"""How the outputs `[T, B, C]` are scored against the labels.

- `mean`: cross entropy of the time-averaged outputs, a firing rate for
  spikes or the mean membrane of a leaky integrator readout.
- `max`: cross entropy of each class's largest output over time, the readout
  Cramer et al. (IEEE TNNLS 2020) use on a leaky integrator for SHD.
- `sum`: cross entropy of the summed outputs, the spike count.
- `per_step`: cross entropy at every step, averaged (`sparx.losses.per_step_cross_entropy`).
  Predictions use the mean.
"""


@dataclass(frozen=True)
class RateBand:
    """Keep each spiking neuron's rate within `[lower, upper]`, adding `weight * sparx.rate_penalty`."""

    lower: float = 0.0
    upper: float = 1.0
    weight: float = 1.0


def _takes_train(model: nn.Module) -> bool:
    return "train" in inspect.signature(type(model).__call__).parameters


@objectives("spiking_classifier")
class SpikingClassifier(Objective[Ratio]):
    """Classify a batch field with a spiking network.

    `model` maps the encoder's time-major input `[T, B, ...]` to outputs
    `[T, B, classes]`: spikes, or the membrane of a `sparx.nn.LI` readout.
    `sample` names the field and its per-example shape; `labels` names the
    integer class field. `encoder` is one of `sparx.encode`'s, which read a
    uint8 field of intensities as `x / 255`.

    The loss is the mean cross entropy of the `readout` over the batch, plus
    the `rates` penalty when given. Metrics report the batch accuracy, each
    spiking layer's mean firing rate (`rate/<layer>`) and the penalty.
    Evaluation returns `TokenScores` with one row per example (loss, weight
    1, whether the argmax is the label), which `sparx.dew.accuracy` reads.

    A model whose `__call__` takes `train` receives `train=True` in the loss
    and `False` in evaluation, and `rngs={"dropout": ...}` in the loss. A
    model that keeps `batch_stats` (BatchNorm) has them updated by the loss.
    `ema_decay` keeps an exponential moving average of the parameters, which
    `evaluate` scores when the trainer passes it.

    `schedules` names model keyword arguments that follow a schedule over
    `schedule_steps` steps, as dew's learning-rate schedules do:
    `{"sigma": Linear(peak=7.5, end=0.5)}` anneals a
    `sparx.nn.DelayedDense`, `{"masking": ...}` a `MaskedPSN`. The value is
    read at `Step.step`, the count of accepted microbatches, in the loss and
    in evaluation alike.

    The objective is registered as `spiking_classifier`, records the model,
    encoder, readout and schedules with every checkpoint
    (`inference_record`), and loads back as a `SpikingClassification`
    (`SpikingClassification.from_run`, or `pipeline(state)` after training).
    """

    artifact = TokenScores
    shown: Mapping[str, Shown] = {"accuracy": Shown(better="higher", percent=True)}

    def __init__(self, model: nn.Module, sample: Field, encoder: SpikeEncoder, *, labels: str = "label",
                 readout: Readout = "mean", rates: RateBand | None = None, ema_decay: float | None = None,
                 schedules: Mapping[str, ScheduleBase] | None = None, schedule_steps: int | None = None):
        if readout not in ("mean", "max", "sum", "per_step"):
            raise ValueError(f"readout must be mean, max, sum or per_step, not {readout!r}")
        if schedules and schedule_steps is None:
            raise ValueError("schedules run over schedule_steps steps; give it")
        self.model = model
        self.sample = sample
        self.encoder = encoder
        self.labels = labels
        self.readout: Readout = readout
        self.rates = rates
        self.schedules = dict(schedules or {})
        self.schedule_steps = schedule_steps
        self.inputs = InputSpec(sample=sample)
        self.ema = None if ema_decay is None else EMASpec(decay=optax.constant_schedule(ema_decay))
        self._train = _takes_train(model)

    def _scheduled(self, step: jax.Array) -> dict[str, jax.Array]:
        if self.schedule_steps is None:
            return {}
        return {name: jnp.asarray(schedule.schedule(self.schedule_steps)(step))
                for name, schedule in self.schedules.items()}

    def _method(self, step: jax.Array, *, train: bool) -> functools.partial[jax.Array]:
        """The model's `__call__` with `train` (when it takes one) and the scheduled arguments bound."""
        return _bound(self.model, self._train, train=train, kwargs=self._scheduled(step))

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        encode_key, init_key = jax.random.split(key)
        x = self.encoder(encode_key, jnp.zeros((1, *self.sample.shape), jnp.float32))
        return dict(self.model.init(init_key, x, method=self._method(jnp.zeros((), jnp.int32), train=False)))

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        encode_key, dropout_key = jax.random.split(step.key)
        x = self.encoder(encode_key, jnp.asarray(batch[self.sample.key]))
        labels = jnp.asarray(batch[self.labels])
        mutable = [RATES, *(["batch_stats"] if "batch_stats" in variables else [])]
        rngs = {"dropout": dropout_key}
        result = self.model.apply(variables, x, rngs=rngs, mutable=mutable,
                                  method=self._method(step.step, train=True))
        # With mutable collections, apply returns the outputs and the collections.
        assert isinstance(result, tuple)
        outputs, updated = result
        losses = _losses(self.readout, outputs, labels)
        total = jnp.sum(losses)
        metrics = {"accuracy": jnp.mean(jnp.argmax(_logits(self.readout, outputs), -1) == labels)}
        metrics |= {f"rate/{name}": rate for name, rate in firing_rates(updated).items()}
        if self.rates is not None:
            penalty = rate_penalty(updated, self.rates.lower, self.rates.upper)
            metrics["rate_penalty"] = penalty
            total = total + self.rates.weight * penalty * labels.shape[0]
        kept = {name: value for name, value in updated.items() if name != RATES}
        stats = Ratio(total, jnp.asarray(labels.shape[0], jnp.float32))
        return stats, Aux(metrics=metrics, variables=kept or None)

    @functools.cached_property
    def _scores(self):
        def scores(variables, field, labels, key, step):
            x = self.encoder(key, field)
            outputs = self.model.apply(variables, x, method=self._method(step, train=False))
            # `mutable` is unset, so apply returns the outputs alone, not a pair.
            assert not isinstance(outputs, tuple)
            losses = _losses(self.readout, outputs, labels)
            correct = jnp.argmax(_logits(self.readout, outputs), -1) == labels
            return losses[:, None], jnp.ones_like(losses)[:, None], correct[:, None]

        return jax.jit(scores)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        variables = params if step.ema is None else step.ema
        field, labels = jnp.asarray(batch[self.sample.key]), jnp.asarray(batch[self.labels])
        losses, weights, correct = self._scores(variables, field, labels, step.key, step.step)
        return TokenScores(losses=losses, weights=weights, correct=correct)

    def inference_record(self) -> JSON:
        """The model, encoder, readout and schedules a saved run rebuilds its classifier from."""
        from dew.config import ModelConfig, _to_json
        if not any(member is type(self) for member in objectives.values()):
            return None
        return {
            "objective": objectives.name_of(type(self)),
            "model": _to_json(ModelConfig.from_model(self.model), ModelConfig),
            "sample": {"key": self.sample.key, "shape": list(self.sample.shape)},
            "encoder": _to_json(self.encoder, SpikeEncoder),
            "readout": self.readout,
            "labels": self.labels,
            "schedules": {name: _to_json(schedule, ScheduleBase)
                          for name, schedule in self.schedules.items()},
            "schedule_steps": self.schedule_steps,
        }

    def pipeline(
            self, state, *, ema: bool | None = None) -> SpikingClassification:
        """The trained classifier over `state`'s weights, with the schedules at their final values."""
        variables = self._pipeline_weights(state, ema)
        final = jnp.asarray(self.schedule_steps or 0)
        return SpikingClassification(self.model, thaw(variables), self.encoder, self.readout,
                                     {name: float(value) for name, value in self._scheduled(final).items()})


def _bound(model: nn.Module, takes_train: bool, *, train: bool,
           kwargs: Mapping[str, jax.Array | float]) -> functools.partial[jax.Array]:
    bound: dict[str, jax.Array | float | bool] = dict(kwargs)
    if takes_train:
        bound["train"] = train
    return functools.partial(type(model).__call__, **bound)


def _logits(readout: Readout, outputs: jax.Array) -> jax.Array:
    outputs = outputs.astype(jnp.float32)
    if readout == "max":
        return jnp.max(outputs, axis=0)
    if readout == "sum":
        return jnp.sum(outputs, axis=0)
    return jnp.mean(outputs, axis=0)


def _losses(readout: Readout, outputs: jax.Array, labels: jax.Array) -> jax.Array:
    if readout == "per_step":
        return per_step_cross_entropy(outputs, labels)
    return optax.softmax_cross_entropy_with_integer_labels(_logits(readout, outputs), labels)


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
        method = _bound(self.model, _takes_train(self.model), train=False, kwargs=self.call)

        def logits(variables, x, key):
            outputs = self.model.apply(variables, self.encoder(key, x), method=method)
            assert not isinstance(outputs, tuple)
            return _logits(self.readout, outputs)

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
                 mesh=None, layout=None, dtype=None, param_dtype=None) -> SpikingClassification:
        """Load the classifier a `spiking_classifier` run in `directory` saved: the model its
        record names over the selected checkpoint's weights (the average when the run kept one,
        unless `ema` is False), placed on `mesh` under `layout`."""
        from dew.checkpoints import Checkpoints
        from dew.config import ModelConfig
        from dew.inference.tasks import run_record
        from dew.records import record as named_fields

        record = run_record(directory, step)
        config = ModelConfig.from_dict(named_fields(record["model"], "model"))
        if dtype is not None:
            config = dataclasses.replace(config, dtype=dtype)
        variables = Checkpoints(directory).variables(ema=ema, step=step, mesh=mesh, layout=layout,
                                                     param_dtype=param_dtype)
        encoder = spike_encoders.from_record(named_fields(record["encoder"], "encoder"))
        steps = record.get("schedule_steps")
        call: dict[str, float] = {}
        if isinstance(steps, int):
            for name, value in named_fields(record["schedules"], "schedules").items():
                schedule = dew_schedules.from_record(named_fields(value, name))
                call[name] = float(jnp.asarray(schedule.schedule(steps)(steps)))
        readout = record["readout"]
        if readout not in ("mean", "max", "sum", "per_step"):
            raise ValueError(f"the run records an unknown readout {readout!r}")
        return cls(config.build(), thaw(variables), encoder, readout, call)


SpikingClassifier.saved_task = SpikingClassification


accuracy = Mean(lambda scores, batch: scores.correct[:, 0], name="accuracy", better="higher",
                reads=TokenScores)
"""Validation accuracy from a `SpikingClassifier`'s evaluation, as `val/accuracy`."""


@objectives("activity_fit")
class ActivityFit(Objective[Ratio]):
    """Fit a network's spiking to recorded spike trains: model fitting to recordings.

    Each example holds a stimulus, `[T, in]` under `stimulus.key`, and the
    spikes recorded in response, `[T, N]` under `recording`; `model` maps
    the time-major stimulus `[T, B, in]` to spikes `[T, B, N]` of the
    recorded neurons (through surrogate gradients, so it trains).

    `loss="van_rossum"` sums van Rossum's (2001) distance over neurons and
    examples (`sparx.losses.van_rossum`, time constant `tau` ms, steps of
    `dt` ms): it compares spike timing at the scale `tau`, and its
    gradient moves spikes toward their recorded times. `loss="psth"` takes
    the squared difference of the trial-averaged rates over the batch,
    smoothed over `window` steps: for recordings repeated over trials,
    where only the rate is reproducible. The loss is reported per example;
    metrics give the model's and the recording's mean rates (spikes per step).
    Evaluation returns one `TokenScores` row per example, its loss.
    """

    artifact = TokenScores
    shown: Mapping[str, Shown] = {"distance": Shown(better="lower")}

    def __init__(self, model: nn.Module, stimulus: Field, *, recording: str = "spikes",
                 loss: Literal["van_rossum", "psth"] = "van_rossum", tau: float = 10.0, dt: float = 1.0,
                 window: int = 10):
        if loss not in ("van_rossum", "psth"):
            raise ValueError(f"loss must be van_rossum or psth, not {loss!r}")
        self.model = model
        self.stimulus = stimulus
        self.recording = recording
        self.kind = loss
        self.tau, self.dt, self.window = tau, dt, window
        self.inputs = InputSpec(sample=stimulus)

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        x = jnp.zeros((self.stimulus.shape[0], 1, *self.stimulus.shape[1:]), jnp.float32)
        return dict(self.model.init(key, x))

    def _distances(self, variables: Variables, batch: Batch) -> tuple[jax.Array, jax.Array, jax.Array]:
        x = jnp.swapaxes(jnp.asarray(batch[self.stimulus.key], jnp.float32), 0, 1)
        target = jnp.swapaxes(jnp.asarray(batch[self.recording], jnp.float32), 0, 1)
        spikes = self.model.apply(variables, x)
        assert not isinstance(spikes, tuple)
        if self.kind == "van_rossum":
            distance = jax.vmap(lambda s, r: van_rossum(s, r, self.tau, self.dt), in_axes=1)
            per_example = distance(spikes, target)
        else:
            kernel = jnp.ones(self.window) / self.window

            def smooth(train):
                return jnp.convolve(train, kernel, mode="same")

            def psth(trains):  # [T, B, N] -> [T, N]
                return jax.vmap(smooth, in_axes=1, out_axes=1)(trains.mean(1))

            error = jnp.sum((psth(spikes) - psth(target)) ** 2)
            per_example = jnp.full(spikes.shape[1], error / spikes.shape[1])
        return per_example, spikes, target

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        per_example, spikes, target = self._distances(variables, batch)
        metrics = {"distance": jnp.mean(per_example), "rate": jnp.mean(spikes),
                   "recorded_rate": jnp.mean(target)}
        stats = Ratio(jnp.sum(per_example), jnp.asarray(per_example.shape[0], jnp.float32))
        return stats, Aux(metrics=metrics)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        per_example, _, _ = self._distances(params if step.ema is None else step.ema, batch)
        ones = jnp.ones_like(per_example)[:, None]
        return TokenScores(losses=per_example[:, None], weights=ones, correct=jnp.zeros_like(ones, bool))

    def inference_record(self) -> JSON:
        return None
