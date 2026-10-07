"""Spiking networks as dew objectives, trained by dew's `Trainer`.

Dew splits a run into a model (a Flax module), an objective (parameters,
loss, evaluation) and a trainer (mesh, compiled step, EMA, checkpoints,
logging). A sparx network is a Flax module, so what sparx adds is the
objectives:

- `SpikingClassifierObjective` encodes a batch into spikes, runs the network
  over time, and reads its outputs as class scores.
- `ActivityFitObjective` fits a network's spikes to recorded ones.
- `EPropObjective` trains a recurrent spiking layer with e-prop's online
  gradients (`sparx.learn.eprop`) in place of backpropagation.
- `PredictiveCodingObjective` trains a stack of layers with predictive
  coding's or PC-ALM's local weight updates (`sparx.learn.PredictiveCoding`).

    import optax
    from dew import Field, Trainer
    from dew.data import Dataset
    from sparx.encode import Rate
    from sparx.metrics import Accuracy
    from sparx.objectives import SpikingClassifierObjective

    objective = SpikingClassifierObjective(net, Field("image", (28, 28, 1)), Rate(steps=16))
    trainer = Trainer(objective, optax.adam(1e-3), key=0)
    state = trainer.fit(Dataset.from_records({"image": x, "label": y}, batch=128),
                        steps=2000, metrics=[Accuracy()])

A run's record names each by import path, `sparx.objectives:EPropObjective`,
as dew records any objective. Each counts a batch's rows through
`Objective.row_mean`, so the repeats that fill a validation split's last
batch count for nothing in the loss.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from dew.artifacts import TokenScores
from dew.inputs import Field, InputSpec
from dew.objectives.base import (
    OMITTED,
    VALID_ROWS,
    Aux,
    Batch,
    EMASpec,
    Objective,
    Omitted,
    Ratio,
    Shown,
    Step,
    Variables,
    thaw,
)
from dew.registry import to_record
from dew.training.optim import ScheduleBase

from sparx.dynamics import NeuronModel
from sparx.encode import SpikeEncoder
from sparx.learn import (
    EPropParams,
    PredictiveCoding,
    bptt_loss,
    eprop,
    eprop_forward,
    sequential_blocks,
    squared_error,
)
from sparx.losses import READOUTS, Readout, readout_logits, readout_losses, van_rossum
from sparx.nn import RATES
from sparx.rates import firing_rates, rate_penalty
from sparx.tasks import SpikingClassification, bound_call

if TYPE_CHECKING:
    from dew.inference.tasks import Processor
    from dew.records import JSON

__all__ = ["ActivityFitObjective", "EPropObjective", "PredictiveCodingObjective", "RateBand",
           "SpikingClassifierObjective"]


def _row_weights(batch: Batch, rows: int) -> jax.Array:
    """1 for each real row of `batch` and 0 for each repeat (`VALID_ROWS`); all 1 in training."""
    valid = batch.get(VALID_ROWS)
    return jnp.ones(rows, jnp.float32) if valid is None else jnp.asarray(valid, jnp.float32)


@dataclass(frozen=True)
class RateBand:
    """Keep each spiking neuron's rate within `[lower, upper]`, adding `weight * sparx.rate_penalty`."""

    lower: float = 0.0
    upper: float = 1.0
    weight: float = 1.0


class SpikingClassifierObjective(Objective[Ratio]):
    """Classify a batch field with a spiking network.

    `model` maps the encoder's time-major input `[T, B, ...]` to outputs
    `[T, B, classes]`: spikes, or the membrane of a `sparx.nn.LI` readout.
    `sample` names the field and its per-example shape; `labels` names the
    integer class field. `encoder` is one of `sparx.encode`'s, which read a
    uint8 field of intensities as `x / 255`.

    The model takes `train`, as dew's models do: `True` in the loss, with
    `rngs={"dropout": ...}`, and `False` in evaluation. A model that keeps
    `batch_stats` (BatchNorm) has them updated by the loss. `ema_decay`
    keeps an exponential moving average of the parameters, which `evaluate`
    scores when the trainer passes it.

    The loss is the mean cross entropy of the `readout` over the batch's
    rows, plus the `rates` penalty when given. Metrics report the batch
    accuracy, each spiking layer's mean firing rate (`rate/<layer>`) and the
    penalty. Evaluation returns `TokenScores` with one row per example (loss,
    weight, whether the argmax is the label), which `sparx.metrics.Accuracy`
    reads.

    `schedules` names model keyword arguments that follow one of dew's
    schedules over `schedule_steps` steps, as a learning rate does:
    `{"sigma": Linear(peak=7.5, end=0.5)}` anneals a
    `sparx.nn.DelayedDense`, `{"masking": ...}` a `MaskedPSN`. The value is
    read at `Step.step`, the count of accepted microbatches, in the loss and
    in evaluation alike, and a schedule's `every` holds each value for that
    many steps, as a torch scheduler stepped once an epoch does. `deployed`
    holds model keyword arguments that evaluation and the trained classifier
    run with in place of the schedules' values, such as `{"sigma": 0}` to
    score every delay rounded to a whole step, the network as deployed and
    as SNN-delays evaluates it.

    Parameter groups with optimizers of their own, as SNN-delays trains its
    delay positions, are the trainer's: `OptimConfig(param_groups=...)`,
    each a dew `ParamGroup` with its own schedule, momentum, weight decay
    and bounds.

    Every checkpoint records the model, encoder, readout, schedules and
    deployed arguments (`task_record`), and the run loads back as a
    `sparx.tasks.SpikingClassification` (`dew.pipeline(run_dir,
    trust=("sparx",))`, or `pipeline(state)` after training).
    """

    artifact = TokenScores
    saved_task = SpikingClassification
    shown: Mapping[str, Shown] = {"accuracy": Shown(better="higher", percent=True)}

    def __init__(self, model: nn.Module, sample: Field, encoder: SpikeEncoder, *, labels: str = "label",
                 readout: Readout = "mean", rates: RateBand | None = None, ema_decay: float | None = None,
                 schedules: Mapping[str, ScheduleBase] | None = None, schedule_steps: int | None = None,
                 deployed: Mapping[str, float] | None = None):
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {', '.join(READOUTS)}, not {readout!r}")
        if schedules and schedule_steps is None:
            raise ValueError("schedules run over schedule_steps steps; give it")
        self.model = self.bind_model(model)
        self.sample = sample
        self.encoder = encoder
        self.labels = labels
        self.readout: Readout = readout
        self.rates = rates
        self.schedules = dict(schedules or {})
        self.schedule_steps = schedule_steps
        self.deployed = {name: float(value) for name, value in (deployed or {}).items()}
        self.inputs = InputSpec(sample=sample)
        self.ema = EMASpec.constant(ema_decay)

    def _scheduled(self, step: jax.Array) -> dict[str, jax.Array]:
        if self.schedule_steps is None:
            return {}
        return {name: jnp.asarray(schedule.schedule(self.schedule_steps)(step))
                for name, schedule in self.schedules.items()}

    def _method(self, step: jax.Array, *, train: bool) -> functools.partial[jax.Array]:
        """The model's `__call__` with `train` and the scheduled arguments bound.

        Out of training, the deployed arguments replace the scheduled ones.
        """
        kwargs: dict[str, jax.Array | float] = dict(self._scheduled(step))
        if not train:
            kwargs |= self.deployed
        return bound_call(self.model, train=train, kwargs=kwargs)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        encode_key, init_key = jax.random.split(key)
        x = self.encoder(encode_key, jnp.zeros((1, *self.sample.shape), jnp.float32))
        return dict(self.model.init(init_key, x, method=self._method(jnp.zeros((), jnp.int32), train=False)))

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        encode_key, dropout_key = jax.random.split(step.key)
        x = self.encoder(encode_key, jnp.asarray(batch[self.sample.key]))
        labels = jnp.asarray(batch[self.labels])
        mutable = [RATES, *(["batch_stats"] if "batch_stats" in variables else [])]
        rngs = {"dropout": dropout_key}
        applied = self.model.apply(variables, x, rngs=rngs, mutable=mutable,
                                   method=self._method(step.step, train=True))
        # With mutable collections, apply returns the outputs and the collections.
        assert isinstance(applied, tuple)
        outputs, updated = applied
        stats = self.row_mean(readout_losses(self.readout, outputs, labels), batch)
        correct = jnp.argmax(readout_logits(self.readout, outputs), -1) == labels
        metrics = {"accuracy": self.accuracy(correct.astype(jnp.float32), batch).mean()[0]}
        metrics |= {f"rate/{name}": rate for name, rate in firing_rates(updated).items()}
        if self.rates is not None:
            penalty = rate_penalty(updated, self.rates.lower, self.rates.upper,
                                   rows=_row_weights(batch, labels.shape[0]))
            metrics["rate_penalty"] = penalty
            stats = Ratio(stats.total + self.rates.weight * penalty * stats.mass, stats.mass)
        kept = {name: value for name, value in updated.items() if name != RATES}
        return stats, Aux(metrics=metrics, variables=kept or None)

    @functools.cached_property
    def _scores(self):
        def scores(variables: Variables, field: jax.Array, labels: jax.Array, key: jax.Array,
                   step: jax.Array) -> tuple[jax.Array, jax.Array]:
            x = self.encoder(key, field)
            outputs = self.model.apply(variables, x, method=self._method(step, train=False))
            # `mutable` is unset, so apply returns the outputs alone, not a pair.
            assert not isinstance(outputs, tuple)
            losses = readout_losses(self.readout, outputs, labels)
            correct = jnp.argmax(readout_logits(self.readout, outputs), -1) == labels
            return losses[:, None], correct[:, None]

        return jax.jit(scores)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        field, labels = jnp.asarray(batch[self.sample.key]), jnp.asarray(batch[self.labels])
        losses, correct = self._scores(self.evaluation_variables(params, step), field, labels, step.key,
                                       step.step)
        return TokenScores(losses=losses, weights=jnp.ones_like(losses), correct=correct)

    def task_record(self) -> Mapping[str, JSON]:
        """The encoder, readout, schedules and deployed arguments of the classifier."""
        return {
            "sample": {"key": self.sample.key, "shape": list(self.sample.shape)},
            "encoder": to_record(self.encoder, SpikeEncoder),
            "readout": self.readout,
            "labels": self.labels,
            "schedules": {name: to_record(schedule, ScheduleBase)
                          for name, schedule in self.schedules.items()},
            "schedule_steps": self.schedule_steps,
            "deployed": dict(self.deployed),
        }

    def build_task(self, variables: Variables, *,
                   processor: Processor | None | Omitted = OMITTED) -> SpikingClassification:
        """The trained classifier over `variables`.

        The schedules stand at their final values, with the deployed
        arguments over them. A spiking classifier reads no text, so it takes
        no `processor`.
        """
        if processor is not OMITTED:
            raise TypeError("a spiking classifier reads no text, so it takes no processor")
        final = jnp.asarray(self.schedule_steps or 0)
        call = {name: float(value) for name, value in self._scheduled(final).items()} | self.deployed
        return SpikingClassification(self.model, thaw(variables), self.encoder, self.readout, call)


class ActivityFitObjective(Objective[Ratio]):
    """Fit a network's spiking to recorded spike trains: model fitting to recordings.

    Each example holds a stimulus, `[T, in]` under `stimulus.key`, and the
    spikes recorded in response, `[T, N]` under `recording`; `model` maps
    the time-major stimulus `[T, B, in]` to spikes `[T, B, N]` of the
    recorded neurons (through surrogate gradients, so it trains).

    `loss="van_rossum"` sums van Rossum's (2001) distance over neurons and
    examples (`sparx.losses.van_rossum`, time constant `tau` ms, steps of
    `dt` ms). It compares spike timing at the scale `tau`, and its gradient
    moves spikes toward their recorded times. `loss="psth"` takes the
    squared difference of the trial-averaged rates over the batch, smoothed
    over `window` steps, for recordings repeated over trials where only the
    rate is reproducible, over the batch's real rows. The loss is reported
    per example; metrics give the model's and the recording's mean rates
    (spikes per step). Evaluation returns one `TokenScores` row per example,
    its loss.
    """

    artifact = TokenScores
    shown: Mapping[str, Shown] = {"distance": Shown(better="lower")}

    def __init__(self, model: nn.Module, stimulus: Field, *, recording: str = "spikes",
                 loss: Literal["van_rossum", "psth"] = "van_rossum", tau: float = 10.0, dt: float = 1.0,
                 window: int = 10):
        if loss not in ("van_rossum", "psth"):
            raise ValueError(f"loss must be van_rossum or psth, not {loss!r}")
        self.model = self.bind_model(model)
        self.stimulus = stimulus
        self.recording = recording
        self.kind = loss
        self.tau, self.dt, self.window = tau, dt, window
        self.inputs = InputSpec(sample=stimulus)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
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

            def smooth(train: jax.Array) -> jax.Array:
                return jnp.convolve(train, kernel, mode="same")

            weights = _row_weights(batch, spikes.shape[1])

            def psth(trains: jax.Array) -> jax.Array:  # [T, B, N] -> [T, N]
                mean = jnp.einsum("tbn,b->tn", trains, weights) / jnp.sum(weights)
                return jax.vmap(smooth, in_axes=1, out_axes=1)(mean)

            error = jnp.sum((psth(spikes) - psth(target)) ** 2)
            per_example = jnp.full(spikes.shape[1], error / jnp.sum(weights))
        return per_example, spikes, target

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        per_example, spikes, target = self._distances(variables, batch)
        stats = self.row_mean(per_example, batch)
        metrics = {"distance": stats.mean()[0], "rate": jnp.mean(spikes), "recorded_rate": jnp.mean(target)}
        return stats, Aux(metrics=metrics)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        per_example, _, _ = self._distances(self.evaluation_variables(params, step), batch)
        ones = jnp.ones_like(per_example)[:, None]
        return TokenScores(losses=per_example[:, None], weights=ones, correct=jnp.zeros_like(ones, bool))


type EPropRule = Literal["eprop", "random", "bptt"]
"""Where `EPropObjective`'s gradient comes from.

- `eprop`: e-prop with the readout weights as feedback (symmetric e-prop).
- `random`: e-prop with fixed random feedback weights (random e-prop).
- `bptt`: backpropagation through time on the same network, for comparison.
"""


class EPropObjective(Objective[Ratio]):
    """Classify recordings with a recurrent spiking layer whose gradients are e-prop's.

    The network is `sparx.learn.eprop_forward`'s: a recurrent layer of
    `cell` with `hidden` neurons over the `[T, channels]` field `sample`, and
    a leaky readout of time constant `tau` with `classes` outputs, all
    stepped at `dt`. Each step's cross entropy, divided by the steps, is
    summed over time, so the loss is the per-step mean, averaged over the
    batch's rows. The class is the argmax of the readout averaged over time.

    The loss runs `sparx.learn.eprop`, which computes the loss and its
    gradients online in memory that does not grow with the recording, and
    hands the gradients to dew's trainer as the loss's own
    (`Objective.with_gradients`), so the trainer's one gradient path applies
    them with its accumulation, sharding and logging. `rule="bptt"`
    differentiates `sparx.learn.bptt_loss` instead. With `rule="random"`
    the feedback weights are drawn once by `init` and kept in the
    `feedback` collection, which no update touches.

    The recurrent layer has no self-connections, as in Bellec et al.: the
    diagonal of `w_rec` starts at 0, is masked in the forward pass and gets
    no gradient. Evaluation returns `TokenScores` for `sparx.metrics.Accuracy`.
    """

    artifact = TokenScores
    shown: Mapping[str, Shown] = {"accuracy": Shown(better="higher", percent=True)}

    def __init__(self, cell: NeuronModel, sample: Field, *, hidden: int, classes: int, tau: float,
                 dt: float = 1.0, labels: str = "label", rule: EPropRule = "eprop"):
        if rule not in ("eprop", "random", "bptt"):
            raise ValueError(f"rule must be eprop, random or bptt, not {rule!r}")
        if len(sample.shape) != 2:
            raise ValueError(f"the sample is one recording [T, channels], not shape {sample.shape}")
        self.cell = cell
        self.sample = sample
        self.hidden, self.classes = hidden, classes
        self.tau, self.dt = tau, dt
        self.labels = labels
        self.rule: EPropRule = rule
        self.inputs = InputSpec(sample=sample)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        channels = self.sample.shape[1]
        w_in, w_rec, w_out, feedback = jax.random.split(key, 4)
        h, c = self.hidden, self.classes
        params = {
            "w_in": jax.random.normal(w_in, (channels, h)) / jnp.sqrt(channels),
            "w_rec": jax.random.normal(w_rec, (h, h)) / jnp.sqrt(h) * (1 - jnp.eye(h)),
            "w_out": jax.random.normal(w_out, (h, c)) / jnp.sqrt(h),
            "b_out": jnp.zeros(c),
        }
        tree: dict[str, dict[str, jax.Array]] = {"params": params}
        if self.rule == "random":
            tree["feedback"] = {"weight": jax.random.normal(feedback, (h, c)) / jnp.sqrt(h)}
        return tree

    def _network(self, variables: Variables, batch: Batch) -> tuple[EPropParams, jax.Array, jax.Array]:
        """The masked parameters, the time-major inputs `[T, B, channels]` and the labels `[B]`."""
        held = variables["params"]
        no_self = 1 - jnp.eye(self.hidden)
        params = EPropParams(held["w_in"], held["w_rec"] * no_self, held["w_out"], held["b_out"])
        inputs = jnp.swapaxes(jnp.asarray(batch[self.sample.key], jnp.float32), 0, 1)
        return params, inputs, jnp.asarray(batch[self.labels])

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        params, inputs, labels = self._network(variables, batch)
        steps = inputs.shape[0]
        targets = jnp.broadcast_to(labels, (steps, *labels.shape))
        weights = _row_weights(batch, labels.shape[0])

        def step_loss(y: jax.Array, label: jax.Array) -> jax.Array:
            return jnp.sum(weights * optax.softmax_cross_entropy_with_integer_labels(y, label)) / steps

        if self.rule == "bptt":
            total = bptt_loss(self.cell, params, inputs, targets, step_loss, tau=self.tau, dt=self.dt)
            return Ratio(total, jnp.sum(weights)), Aux(metrics={})
        feedback = variables["feedback"]["weight"] if self.rule == "random" else None
        # The trainer's gradient reaches the parameters through `with_gradients` alone, so it never
        # linearizes e-prop's scan, whose memory would then grow with the recording.
        total, grads = eprop(self.cell, jax.lax.stop_gradient(params), inputs, targets, step_loss,
                             tau=self.tau, dt=self.dt, feedback=feedback)
        no_self = 1 - jnp.eye(self.hidden)
        rule = {"w_in": grads.w_in, "w_rec": grads.w_rec * no_self, "w_out": grads.w_out,
                "b_out": grads.b_out}
        stats = Ratio(total, jnp.sum(weights))
        # The rule mirrors the statistics' tree: the total's gradient, and none for the row count.
        gradients = jax.tree.unflatten(jax.tree.structure(stats), [rule, None])
        return self.with_gradients(stats, gradients, variables["params"]), Aux(metrics={})

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        network, inputs, labels = self._network(self.evaluation_variables(params, step), batch)
        outputs, _ = eprop_forward(self.cell, network, inputs, tau=self.tau, dt=self.dt)
        logits = jnp.mean(outputs, axis=0)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        ones = jnp.ones_like(losses)[:, None]
        return TokenScores(losses=losses[:, None], weights=ones,
                           correct=(jnp.argmax(logits, -1) == labels)[:, None])


class PredictiveCodingObjective(Objective[Ratio]):
    """Classify with a stack of layers whose weight updates are predictive coding's or PC-ALM's.

    `model` is a Flax `nn.Sequential` over the field `sample`, flattened per
    example, each element one layer of the stack (`sparx.learn.residual_mlp`
    builds Seely and Gould's residual MLP). Its output scores `classes`
    classes against the one-hot labels under `labels` by half the squared
    error, the loss of their experiments, averaged over the batch's rows.
    `rule` (`sparx.learn.PredictiveCoding`) relaxes the hidden activity from
    the forward pass, and its weight update, the energy's gradient at the
    relaxed activity, goes to dew's trainer as the loss's own
    (`Objective.with_gradients`), so the trainer's optimizer, accumulation
    and logging apply it. `rule=None` trains the same stack by
    backpropagation, their baseline. The reported loss is the forward
    pass's whatever the rule, beside the batch's accuracy. Evaluation
    returns `TokenScores` for `sparx.metrics.Accuracy`.
    """

    artifact = TokenScores
    shown: Mapping[str, Shown] = {"accuracy": Shown(better="higher", percent=True)}

    def __init__(self, model: nn.Sequential, sample: Field, *, classes: int,
                 rule: PredictiveCoding | None = None, labels: str = "label"):
        if not isinstance(model, nn.Sequential):
            raise TypeError(f"predictive coding relaxes each layer of a stack: an nn.Sequential, not "
                            f"{type(model).__name__}")
        self.model = self.bind_model(model)
        self.stack = model
        self.sample = sample
        self.classes = classes
        self.rule = rule
        self.labels = labels
        self.inputs = InputSpec(sample=sample)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return dict(self.model.init(key, jnp.zeros((1, math.prod(self.sample.shape)), jnp.float32)))

    def _scored(self, variables: Variables,
                batch: Batch) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """The flattened inputs `[B, in]`, the labels `[B]`, their one-hot targets and the outputs."""
        x = jnp.asarray(batch[self.sample.key], jnp.float32)
        x = x.reshape(x.shape[0], -1)
        labels = jnp.asarray(batch[self.labels])
        output = self.model.apply(variables, x)
        assert not isinstance(output, tuple)  # no mutable collection, so apply returns the output alone
        return x, labels, jax.nn.one_hot(labels, self.classes, dtype=x.dtype), output

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        x, labels, target, output = self._scored(variables, batch)
        weights = _row_weights(batch, x.shape[0])
        stats = Ratio(jnp.sum(weights * squared_error(output, target)), jnp.sum(weights))
        correct = (jnp.argmax(output, axis=-1) == labels).astype(jnp.float32)
        metrics = {"accuracy": jnp.sum(weights * correct) / jnp.sum(weights)}
        if self.rule is None:
            return stats, Aux(metrics=metrics)
        held = variables["params"]
        blocks, params = sequential_blocks(self.stack, jax.lax.stop_gradient(held))
        grads = self.rule.gradient(blocks, params, x, target, rows=weights)
        # The update mirrors the parameters: a layer without any, an activation, has none to update.
        update = {f"layers_{k}": grad for k, grad in enumerate(grads) if f"layers_{k}" in held}
        gradients = jax.tree.unflatten(jax.tree.structure(stats), [update, None])
        return self.with_gradients(stats, gradients, held), Aux(metrics=metrics)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        _, labels, target, output = self._scored(self.evaluation_variables(params, step), batch)
        losses = squared_error(output, target)
        ones = jnp.ones_like(losses)[:, None]
        return TokenScores(losses=losses[:, None], weights=ones,
                           correct=(jnp.argmax(output, axis=-1) == labels)[:, None])

