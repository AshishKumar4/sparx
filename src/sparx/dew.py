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
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dew.artifacts import TokenScores
from dew.data import Dataset
from dew.data.dataset import Reader
from dew.eval import Mean
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, Batch, EMASpec, Objective, Ratio, Shown, Step, Variables, thaw
from dew.records import JSON
from dew.registry import objectives, schedules as dew_schedules
from dew.training.optim import ParamGroup, ScheduleBase, param_labels
from jax.typing import ArrayLike

from sparx.encode import SpikeEncoder
from sparx.losses import per_step_cross_entropy, softmax_sum, softmax_sum_cross_entropy, van_rossum
from sparx.nn import RATES
from sparx.rates import firing_rates, rate_penalty
from sparx.registry import spike_encoders

__all__ = [
    "WEIGHT",
    "ActivityFit",
    "ExponentialDecay",
    "GroupAdam",
    "OneCycle",
    "RateBand",
    "Readout",
    "SpikingClassification",
    "SpikingClassifier",
    "accuracy",
    "evaluation_pass",
    "holdout",
    "keep_within",
    "stepped",
    "whole_batches",
]


type Readout = Literal["mean", "max", "sum", "softmax_sum", "per_step"]
"""How the outputs `[T, B, C]` are scored against the labels.

- `mean`: cross entropy of the time-averaged outputs, a firing rate for
  spikes or the mean membrane of a leaky integrator readout.
- `max`: cross entropy of each class's largest output over time, the readout
  Cramer et al. (IEEE TNNLS 2020) use on a leaky integrator for SHD.
- `sum`: cross entropy of the summed outputs, the spike count.
- `softmax_sum`: the softmax of every step summed over time, scored as
  logits (`sparx.losses.softmax_sum_cross_entropy`), SNN-delays' `loss='sum'`.
- `per_step`: cross entropy at every step, averaged (`sparx.losses.per_step_cross_entropy`).
  Predictions use the mean.
"""

READOUTS: tuple[Readout, ...] = ("mean", "max", "sum", "softmax_sum", "per_step")

WEIGHT = "weight"
"""The batch field evaluation weighs each example by, when a batch holds it.

`whole_batches` writes it: 1 for a record, 0 for the copies that fill the
last batch, so a split of any size is scored over exactly its records."""


# Stopgaps for two schedule records dew lacks (dew.training.optim has cosine, power and linear).
# They belong in dew's table as `one_cycle` and `exponential`; until then they are registered under
# sparx-prefixed names, so a dew release that adds its own cannot collide with them, and a run
# records them under those names.


@dew_schedules("sparx_one_cycle")
@dataclass(frozen=True)
class OneCycle(ScheduleBase):
    """torch's `OneCycleLR` with its default cosine annealing and two phases.

    The value follows a half cosine from `start` to `peak` until step
    `warmup * steps - 1`, then another to `end` at step `steps - 1`, and
    stays there. The step boundaries are torch's, so stepped once an epoch
    over a run of epochs (`SpikingClassifier`'s `schedule_every`) it gives
    torch's values. For a learning rate torch starts at `peak / 25` and ends
    at `start / 1e4`; its Adam momentum cycle (`b1`) is
    `OneCycle(peak=0.85, start=0.95, end=0.95)`.
    """

    peak: float
    start: float
    end: float
    warmup: float = 0.3

    def schedule(self, steps: int) -> optax.Schedule:
        rise, last = self.warmup * steps - 1, steps - 1
        if not 0 < rise < last:
            raise ValueError(f"a one-cycle schedule over {steps} steps with warmup {self.warmup} has no "
                             "rise or no fall; give it more steps")

        def anneal(start: float, end: float, done: jax.Array) -> jax.Array:
            return end + (start - end) / 2 * (jnp.cos(jnp.pi * done) + 1)

        def value(count: ArrayLike) -> jax.Array:
            step = jnp.minimum(jnp.asarray(count, jnp.float32), last)
            return jnp.where(step <= rise, anneal(self.start, self.peak, step / rise),
                             anneal(self.peak, self.end, (step - rise) / (last - rise)))

        return value


@dew_schedules("sparx_exponential_decay")
@dataclass(frozen=True)
class ExponentialDecay(ScheduleBase):
    """A geometric decay from `offset + start` to `offset + end` over `decay_steps`, then constant.

    The value is `offset + start * (end / start) ** (min(step, decay_steps) / decay_steps)`,
    `decay_steps` being the run when None. SNN-delays shrinks DCLS's raw
    width this way, from `max_delay // 2` to 0.23 over the first quarter of
    its epochs; DCLS's width is the raw one plus 0.27, so the
    `sparx.nn.DelayedDense` width is `ExponentialDecay(12, 0.23, epochs // 4, offset=0.27)`
    for their 25-step kernels.
    """

    start: float
    end: float
    decay_steps: int | None = None
    offset: float = 0.0

    def schedule(self, steps: int) -> optax.Schedule:
        span = steps if self.decay_steps is None else self.decay_steps
        if span <= 0 or self.start <= 0 or self.end <= 0:
            raise ValueError("an exponential decay needs positive steps, start and end")
        ratio = self.end / self.start

        def value(count: ArrayLike) -> jax.Array:
            done = jnp.minimum(jnp.asarray(count, jnp.float32), span) / span
            return self.offset + self.start * ratio ** done

        return value


def stepped(schedule: ScheduleBase, steps: int, every: int = 1) -> optax.Schedule:
    """`schedule` advancing once every `every` steps, over `steps // every` steps of its own.

    A torch scheduler stepped once an epoch holds its value through the
    epoch. With `every` set to the steps of an epoch, a step reads the
    schedule's value for the epoch it falls in.
    """
    if every < 1:
        raise ValueError(f"a schedule advances every 1 or more steps, not {every}")
    values = schedule.schedule(max(steps // every, 1))
    if every == 1:
        return values
    return lambda count: values(jnp.asarray(count) // every)


def keep_within(lower: float, upper: float) -> optax.GradientTransformation:
    """Shorten each update so the parameter lands within `[lower, upper]`.

    DCLS clamps its positions to the kernel after every step
    (`clamp_parameters`), and clamping a `DelayedDense` delay to
    `[0, max_delay]` is the same. A delay left outside that range would
    have no gradient, since the kernel clips its center, and would stay
    there.
    """
    def init(params: optax.Params) -> optax.EmptyState:
        return optax.EmptyState()

    def update(updates: optax.Updates, state: optax.OptState,
               params: optax.Params | None = None) -> tuple[optax.Updates, optax.OptState]:
        if params is None:
            raise ValueError("keep_within reads the parameters; pass them to update")
        kept = jax.tree.map(lambda u, p: jnp.clip(p + u, lower, upper) - p, updates, params)
        return kept, state

    return optax.GradientTransformation(init, update)


@dataclass(frozen=True)
class GroupAdam:
    """Adam for the parameters whose paths match `patterns`, on schedules of its own.

    A path is the parameter's dict keys joined by `/` (`delayed_0/delay`),
    matched by `fnmatch` with `*` matching `/` too, as dew's `ParamGroup`
    matches. `learning_rate` and `b1` (0.9 when None) are dew schedule
    records. `weight_decay` adds `weight_decay * param` to the gradient
    before Adam's moments, as torch's `Adam(weight_decay=...)` does: an L2
    penalty, which differs from AdamW's decoupled decay. `bounds` keeps
    every parameter within `[lower, upper]` after each update
    (`keep_within`).

    SNN-delays trains its weights with Adam on a one-cycle schedule and its
    delay positions with Adam at 100 times the rate on a cosine, without
    weight decay, clamped to the kernel.
    """

    patterns: tuple[str, ...]
    learning_rate: ScheduleBase
    b1: ScheduleBase | None = None
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    bounds: tuple[float, float] | None = None

    def build(self, steps: int, every: int = 1) -> optax.GradientTransformation:
        """The optimizer over a run of `steps` updates, its schedules advancing every `every` (`stepped`)."""
        if self.b1 is None:
            adam = optax.scale_by_adam(b1=0.9, b2=self.b2, eps=self.eps)
        else:
            adam = optax.inject_hyperparams(optax.scale_by_adam)(
                b1=stepped(self.b1, steps, every), b2=self.b2, eps=self.eps)
        chain = [optax.add_decayed_weights(self.weight_decay)] if self.weight_decay else []
        chain += [adam, optax.scale_by_learning_rate(stepped(self.learning_rate, steps, every))]
        if self.bounds is not None:
            chain.append(keep_within(*self.bounds))
        return optax.chain(*chain)


_TRAINERS = "trainer"
"""The group of the parameters no `GroupAdam` matches, which the trainer's optimizer updates."""


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
    in evaluation alike. `schedule_every` advances every schedule, these and
    the groups' learning rates, once every that many steps (`stepped`): the
    steps of an epoch reproduce a torch scheduler stepped once an epoch.
    `deployed` holds model keyword arguments that evaluation and the trained
    classifier run with in place of the schedules' values, such as
    `{"sigma": 0}` to score every delay rounded to a whole step, the network
    as deployed and as SNN-delays evaluates it.

    `groups` gives parameter groups their own optimizers, by name: each
    `GroupAdam` updates the parameters its patterns match, the first
    matching group winning, over `schedule_steps` updates. The trainer's
    optimizer updates the rest, under `optax.multi_transform`
    (`optimizer`). Delay positions learn at their own rate this way.

    Evaluation weighs each example by the batch's `WEIGHT` field when it has
    one (`whole_batches`), so the loss and `accuracy` cover a split of any
    size exactly.

    The objective is registered as `spiking_classifier`, records the model,
    encoder, readout, schedules and deployed arguments with every checkpoint
    (`inference_record`), and loads back as a `SpikingClassification`
    (`SpikingClassification.from_run`, or `pipeline(state)` after training).
    """

    artifact = TokenScores
    shown: Mapping[str, Shown] = {"accuracy": Shown(better="higher", percent=True)}

    def __init__(self, model: nn.Module, sample: Field, encoder: SpikeEncoder, *, labels: str = "label",
                 readout: Readout = "mean", rates: RateBand | None = None, ema_decay: float | None = None,
                 schedules: Mapping[str, ScheduleBase] | None = None, schedule_steps: int | None = None,
                 schedule_every: int = 1, deployed: Mapping[str, float] | None = None,
                 groups: Mapping[str, GroupAdam] | None = None):
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {', '.join(READOUTS)}, not {readout!r}")
        if (schedules or groups) and schedule_steps is None:
            raise ValueError("schedules and groups run over schedule_steps steps; give it")
        if schedule_every < 1:
            raise ValueError(f"schedules advance every 1 or more steps, not {schedule_every}")
        if groups and _TRAINERS in groups:
            raise ValueError(f"{_TRAINERS!r} names the parameters no group matches; name the group otherwise")
        self.model = model
        self.sample = sample
        self.encoder = encoder
        self.labels = labels
        self.readout: Readout = readout
        self.rates = rates
        self.schedules = dict(schedules or {})
        self.schedule_steps = schedule_steps
        self.schedule_every = schedule_every
        self.deployed = {name: float(value) for name, value in (deployed or {}).items()}
        self.groups = dict(groups or {})
        self.inputs = InputSpec(sample=sample)
        self.ema = None if ema_decay is None else EMASpec(decay=optax.constant_schedule(ema_decay))
        self._train = _takes_train(model)

    def _scheduled(self, step: jax.Array) -> dict[str, jax.Array]:
        if self.schedule_steps is None:
            return {}
        return {name: jnp.asarray(stepped(schedule, self.schedule_steps, self.schedule_every)(step))
                for name, schedule in self.schedules.items()}

    def _method(self, step: jax.Array, *, train: bool) -> functools.partial[jax.Array]:
        """The model's `__call__` with `train` (when it takes one) and the scheduled arguments bound.

        Out of training, the deployed arguments replace the scheduled ones.
        """
        kwargs: dict[str, jax.Array | float] = dict(self._scheduled(step))
        if not train:
            kwargs |= self.deployed
        return _bound(self.model, self._train, train=train, kwargs=kwargs)

    def optimizer(self, tx: optax.GradientTransformation, *,
                  accumulation: int) -> optax.GradientTransformation:
        """`tx` for the parameters no group matches and each group's `GroupAdam` for its own."""
        if not self.groups:
            return tx
        assert self.schedule_steps is not None  # __init__ refuses groups without it
        solvers: dict[Hashable, optax.GradientTransformation] = {
            name: group.build(self.schedule_steps, self.schedule_every)
            for name, group in self.groups.items()}
        solvers[_TRAINERS] = tx
        labels = param_labels([*(ParamGroup(name, group.patterns) for name, group in self.groups.items()),
                               ParamGroup(_TRAINERS, ("*",))])
        return optax.multi_transform(solvers, labels)

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
        def scores(variables, field, labels, weights, key, step):
            x = self.encoder(key, field)
            outputs = self.model.apply(variables, x, method=self._method(step, train=False))
            # `mutable` is unset, so apply returns the outputs alone, not a pair.
            assert not isinstance(outputs, tuple)
            losses = _losses(self.readout, outputs, labels)
            correct = jnp.argmax(_logits(self.readout, outputs), -1) == labels
            return losses[:, None], weights[:, None], correct[:, None]

        return jax.jit(scores)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        variables = params if step.ema is None else step.ema
        field, labels = jnp.asarray(batch[self.sample.key]), jnp.asarray(batch[self.labels])
        weights = (jnp.asarray(batch[WEIGHT], jnp.float32) if WEIGHT in batch
                   else jnp.ones(labels.shape, jnp.float32))
        losses, weights, correct = self._scores(variables, field, labels, weights, step.key, step.step)
        return TokenScores(losses=losses, weights=weights, correct=correct)

    def inference_record(self) -> JSON:
        """The model, encoder, readout, schedules and deployed arguments of the classifier."""
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
            "schedule_every": self.schedule_every,
            "deployed": dict(self.deployed),
        }

    def pipeline(
            self, state, *, ema: bool | None = None) -> SpikingClassification:
        """The trained classifier over `state`'s weights, with the schedules at their final values
        and the deployed arguments over them."""
        variables = self._pipeline_weights(state, ema)
        final = jnp.asarray(self.schedule_steps or 0)
        call = {name: float(value) for name, value in self._scheduled(final).items()} | self.deployed
        return SpikingClassification(self.model, thaw(variables), self.encoder, self.readout, call)


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
    if readout == "softmax_sum":
        return softmax_sum(outputs)
    return jnp.mean(outputs, axis=0)


def _losses(readout: Readout, outputs: jax.Array, labels: jax.Array) -> jax.Array:
    if readout == "per_step":
        return per_step_cross_entropy(outputs, labels)
    if readout == "softmax_sum":
        return softmax_sum_cross_entropy(outputs, labels)
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
        every = record.get("schedule_every", 1)
        call: dict[str, float] = {}
        if isinstance(steps, int) and isinstance(every, int):
            for name, value in named_fields(record["schedules"], "schedules").items():
                schedule = dew_schedules.from_record(named_fields(value, name))
                call[name] = float(jnp.asarray(stepped(schedule, steps, every)(steps)))
        deployed = record.get("deployed") or {}
        for name, value in named_fields(deployed, "deployed").items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"the run records a deployed {name}={value!r}, not a number")
            call[name] = float(value)
        readout = record["readout"]
        if readout not in READOUTS:
            raise ValueError(f"the run records an unknown readout {readout!r}")
        return cls(config.build(), thaw(variables), encoder, readout, call)


SpikingClassifier.saved_task = SpikingClassification


def _weighted_correct(scores: TokenScores, batch: Batch) -> tuple[float, float]:
    weights = np.asarray(scores.weights[:, 0], np.float64)
    return float(np.sum(np.asarray(scores.correct[:, 0]) * weights)), float(np.sum(weights))


accuracy = Mean(_weighted_correct, name="accuracy", better="higher", reads=TokenScores)
"""Accuracy from a `SpikingClassifier`'s evaluation, as `val/accuracy` (or `<split>/accuracy`).

Each example counts by its weight, so the copies `whole_batches` adds count
for nothing."""


def holdout(records: Mapping[str, np.ndarray], fraction: float,
            seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split `records` (columns of equal length) into a random `1 - fraction` and `fraction`.

    SHD has no validation split, and SNN-delays selects its epochs on the
    test set; holding out part of the training set gives a validation set
    to select on, so the test accuracy stays an estimate.
    """
    sizes = {len(column) for column in records.values()}
    if len(sizes) != 1:
        raise ValueError(f"columns of one length split together, not lengths {sorted(sizes)}")
    total = sizes.pop()
    held = round(fraction * total)
    if not 0 < held < total:
        raise ValueError(f"holding out {fraction} of {total} records leaves one side empty")
    order = np.random.default_rng(seed).permutation(total)
    kept, out = np.sort(order[held:]), np.sort(order[:held])
    return ({name: column[kept] for name, column in records.items()},
            {name: column[out] for name, column in records.items()})


def whole_batches(records: Mapping[str, np.ndarray], batch: int) -> dict[str, np.ndarray]:
    """`records` filled to whole batches with copies of its first record, weighted under `WEIGHT`.

    dew scores a split in whole batches only, so the records past the last
    whole one would go unscored. Each record has weight 1 and each copy 0,
    which `SpikingClassifier`'s evaluation and `accuracy` honor.
    """
    total = len(next(iter(records.values())))
    fill = -total % batch
    padded = {name: np.concatenate([column, np.repeat(column[:1], fill, axis=0)])
              for name, column in records.items()}
    padded[WEIGHT] = np.concatenate([np.ones(total, np.float32), np.zeros(fill, np.float32)])
    return padded


def evaluation_pass(records: Mapping[str, np.ndarray], batch: int) -> Reader:
    """One pass over every record, in order, for `Trainer.fit(validation={...})`.

    The records are filled to whole batches first (`whole_batches`), so a
    split such as SHD's 2264 test recordings is scored over all of them.
    """
    padded = whole_batches(records, batch)
    reader = Dataset.from_records(padded, batch=batch, validation=padded).val
    assert reader is not None  # from_records reads a validation split it is given
    return reader


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
