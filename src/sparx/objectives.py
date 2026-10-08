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
- `RNeuralNetObjective` rewards an `RNeuralNet`'s choice and learns from the
  reward by reward diffusion or by REINFORCE (`sparx.learn.RNeuralNet`).

    import optax
    from dew import Field, Trainer
    from dew.data import Dataset
    from sparx.encode import RateEncoder
    from sparx.metrics import Accuracy
    from sparx.objectives import SpikingClassifierObjective

    objective = SpikingClassifierObjective(net, Field("image", (28, 28, 1)), RateEncoder(steps=16))
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from dew.artifacts import TokenScores
from dew.inputs import Field, InputSpec
from dew.nn.precision import at_least_fp32
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
from sparx.encode import EventsEncoder, SpikeEncoder
from sparx.learn import (
    EPropParams,
    PredictiveCoding,
    RNeuralNet,
    bptt_loss,
    eprop,
    reward_diffusion,
    sequential_blocks,
    squared_error,
)
from sparx.losses import READOUTS, Readout, readout_logits, readout_losses, van_rossum
from sparx.models import SpikingMLP
from sparx.nn import RATES
from sparx.rates import firing_rates, rate_penalty
from sparx.tasks import SpikingClassification, bound_call

if TYPE_CHECKING:
    from dew.inference.tasks import Processor
    from dew.records import JSON

__all__ = ["ActivityFitObjective", "EPropObjective", "PredictiveCodingObjective", "RNeuralNetObjective",
           "RateBand", "SpikingClassifierObjective"]


def _per_example(losses: jax.Array, correct: jax.Array | None = None) -> TokenScores:
    """`TokenScores` with one row per example: its loss `[B]`, counted once, and whether its prediction
    is its label (`correct`, `[B]`; False for an objective that predicts no class)."""
    rows = losses[:, None]
    hits = jnp.zeros_like(rows, bool) if correct is None else correct[:, None]
    return TokenScores(losses=rows, weights=jnp.ones_like(rows), correct=hits)


def _with_rule(objective: Objective[Ratio], stats: Ratio, update: Variables, params: Variables) -> Ratio:
    """`stats` whose total's gradient in `params` is `update`, a learning rule's own, and whose row count
    has none, for dew's trainer to apply as it applies a loss's gradient (`Objective.with_gradients`)."""
    gradients = jax.tree.unflatten(jax.tree.structure(stats), [update, None])
    return objective.with_gradients(stats, gradients, params)


def _row_weights(batch: Batch, rows: int) -> jax.Array:
    """1 for each real row of `batch` and 0 for each repeat (`VALID_ROWS`); all 1 in training."""
    valid = batch.get(VALID_ROWS)
    return jnp.ones(rows, jnp.float32) if valid is None else jnp.asarray(valid, jnp.float32)


def _fields(batch: Batch, *keys: str) -> dict[str, jax.Array]:
    """The fields `keys` of `batch`, and its `VALID_ROWS` when it has them, as arrays: what a compiled
    evaluation takes. Dew calls `evaluate` outside `jit`, so each objective compiles its own scoring,
    which would otherwise dispatch one operation at a time."""
    keys = (*keys, VALID_ROWS) if VALID_ROWS in batch else keys
    return {key: jnp.asarray(batch[key]) for key in keys}


def _classification_record(sample: Field, encoder: SpikeEncoder, readout: Readout, labels: str, *,
                           schedules: Mapping[str, ScheduleBase] | None = None,
                           schedule_steps: int | None = None,
                           deployed: Mapping[str, float] | None = None) -> dict[str, JSON]:
    """The settings `SpikingClassification.from_run` rebuilds a classifier with, beside its model."""
    return {
        "sample": {"key": sample.key, "shape": list(sample.shape)},
        "encoder": to_record(encoder, SpikeEncoder),
        "readout": readout,
        "labels": labels,
        "schedules": {name: to_record(schedule, ScheduleBase)
                      for name, schedule in (schedules or {}).items()},
        "schedule_steps": schedule_steps,
        "deployed": dict(deployed or {}),
    }


@dataclass(frozen=True)
class RateBand:
    """Keep each spiking neuron's rate, in spikes per step, within `[lower, upper]`, adding
    `weight * sparx.rate_penalty`."""

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

    A model that takes `train`, as dew's models do, gets `True` in the loss,
    with `rngs={"dropout": ...}`, and `False` in evaluation; one without
    BatchNorm or dropout need not take it, so an `nn.Sequential` stack of
    sparx layers trains as it is, and exports to NIR (`sparx.nir.to_nir`). A model that keeps
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
            return losses, jnp.argmax(readout_logits(self.readout, outputs), -1) == labels

        return jax.jit(scores)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        field, labels = jnp.asarray(batch[self.sample.key]), jnp.asarray(batch[self.labels])
        losses, correct = self._scores(self.evaluation_variables(params, step), field, labels, step.key,
                                       step.step)
        return _per_example(losses, correct)

    def task_record(self) -> Mapping[str, JSON]:
        """The encoder, readout, schedules and deployed arguments of the classifier."""
        return _classification_record(self.sample, self.encoder, self.readout, self.labels,
                                      schedules=self.schedules, schedule_steps=self.schedule_steps,
                                      deployed=self.deployed)

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

    @functools.cached_property
    def _compiled_distances(self) -> Callable[[Variables, Batch], tuple[jax.Array, jax.Array, jax.Array]]:
        return jax.jit(self._distances)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        fields = _fields(batch, self.stimulus.key, self.recording)
        per_example, _, _ = self._compiled_distances(self.evaluation_variables(params, step), fields)
        return _per_example(per_example)


type EPropRule = Literal["eprop", "random", "bptt"]
"""Where `EPropObjective`'s gradient comes from.

- `eprop`: e-prop with the readout weights as feedback (symmetric e-prop).
- `random`: e-prop with fixed random feedback weights (random e-prop).
- `bptt`: backpropagation through time on the same network, for comparison.
"""


class EPropObjective(Objective[Ratio]):
    """Classify recordings with a recurrent spiking network whose gradients are e-prop's.

    `model` is Bellec et al.'s (2020) network as a `sparx.models.SpikingMLP`
    with one recurrent hidden layer (`recurrent=True`) and dense synapses:
    an input synapse, a recurrent layer of `model.neuron`, and a synapse
    into a leaky readout, over the `[T, channels]` field `sample`. Each
    step's cross entropy, divided by the steps, is summed over time, so the
    loss is the per-step mean, averaged over the batch's rows. The class is
    the argmax of the readout averaged over time.

    The loss runs `sparx.learn.eprop` on the model's weights, which computes
    the loss and its gradients online in memory that does not grow with the
    recording, and hands the gradients to dew's trainer as the loss's own
    (`Objective.with_gradients`), so the trainer's one gradient path applies
    them with its accumulation, sharding and logging. A bias is a synapse
    from an input that is always 1, and e-prop trains it as one. e-prop
    computes no gradient for a time constant, so a model that learns one is
    refused, as are delays, batch norm, dropout and plasticity rules. It
    computes as the model does: in the model's `dtype` when it has one,
    float32 or wider, else in the widest of float32 and the parameters'
    dtypes, at the model's `precision`.
    `rule="bptt"` differentiates `sparx.learn.bptt_loss` instead. With
    `rule="random"` the feedback weights are drawn once by `init` and kept
    in the `feedback` collection, which no update touches.

    The recurrent layer has no self-connections, as in Bellec et al.: `init`
    zeroes the recurrent matrix's diagonal, which is masked in the forward
    pass and gets no gradient. The trained network is the `SpikingMLP`
    itself, so `pipeline(state)` and `dew.pipeline(run_dir,
    trust=("sparx",))` load it as a `sparx.tasks.SpikingClassification`,
    which streams, serves and exports as any other. Evaluation runs the
    model and returns `TokenScores` for `sparx.metrics.Accuracy`.
    """

    artifact = TokenScores
    saved_task = SpikingClassification
    shown: Mapping[str, Shown] = {"accuracy": Shown(better="higher", percent=True)}

    def __init__(self, model: SpikingMLP, sample: Field, *, labels: str = "label", rule: EPropRule = "eprop"):
        if rule not in ("eprop", "random", "bptt"):
            raise ValueError(f"rule must be eprop, random or bptt, not {rule!r}")
        if len(sample.shape) != 2:
            raise ValueError(f"the sample is one recording [T, channels], not shape {sample.shape}")
        _check_eprop_network(model, sample.shape[1])
        self.model = self.bind_model(model)
        self.sample = sample
        self.labels = labels
        self.rule: EPropRule = rule
        self.inputs = InputSpec(sample=sample)
        (self.hidden,) = model.hidden
        # Every layer of a `SpikingMLP` steps at its neuron's dt, the readout included.
        self.dt, self.tau = model.neuron.dt, model.readout_tau
        self.cell: NeuronModel = model.neuron.bind({}).model(jnp.zeros((1, self.hidden)))

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        model_key, feedback_key = jax.random.split(key)
        params = thaw(self.model.init(model_key, jnp.zeros((1, 1, self.sample.shape[1]))))["params"]
        params["recurrent_0"]["recurrent"] *= 1 - jnp.eye(self.hidden)
        tree: dict[str, Variables] = {"params": params}
        if self.rule == "random":
            h, c = self.hidden, self.model.classes
            tree["feedback"] = {"weight": jax.random.normal(feedback_key, (h, c)) / jnp.sqrt(h)}
        return tree

    def _network(self, params: Variables, batch: Batch) -> tuple[EPropParams, jax.Array, jax.Array]:
        """e-prop's parameters, the time-major inputs `[T, B, channels]` and the labels `[B]`, in the
        model's dtype when it has one, as its flax layers cast them. With biases, the input bias is a last
        row of `w_in` and the inputs gain a last channel of ones."""
        dtype = self.model.dtype
        inputs = jnp.asarray(batch[self.sample.key], jnp.float32 if dtype is None else dtype)
        inputs = jnp.swapaxes(inputs, 0, 1)
        w_in, readout = params["dense_0"]["kernel"], params["readout"]
        b_out = readout["bias"] if self.model.use_bias else jnp.zeros(self.model.classes)
        if self.model.use_bias:
            w_in = jnp.concatenate([w_in, params["dense_0"]["bias"][None]])
            inputs = jnp.concatenate([inputs, jnp.ones((*inputs.shape[:2], 1), inputs.dtype)], axis=-1)
        w_rec = params["recurrent_0"]["recurrent"] * (1 - jnp.eye(self.hidden))
        network = EPropParams(w_in, w_rec, readout["kernel"], b_out)
        if dtype is not None:
            network = EPropParams(*(p.astype(dtype) for p in network))
        return network, inputs, jnp.asarray(batch[self.labels])

    def _update(self, grads: EPropParams, params: Variables) -> Variables:
        """e-prop's gradients as a tree of the model's parameters `params`, each in its parameter's dtype,
        the recurrent diagonal's 0."""
        channels = self.sample.shape[1]
        update = {"dense_0": {"kernel": grads.w_in[:channels]},
                  "recurrent_0": {"recurrent": grads.w_rec * (1 - jnp.eye(self.hidden))},
                  "readout": {"kernel": grads.w_out}}
        if self.model.use_bias:
            update["dense_0"]["bias"] = grads.w_in[channels]
            update["readout"]["bias"] = grads.b_out
        return jax.tree.map(lambda g, p: g.astype(p.dtype), update, params)

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        params, inputs, labels = self._network(variables["params"], batch)
        steps = inputs.shape[0]
        targets = jnp.broadcast_to(labels, (steps, *labels.shape))
        weights = _row_weights(batch, labels.shape[0])

        def step_loss(y: jax.Array, label: jax.Array) -> jax.Array:
            return jnp.sum(weights * optax.softmax_cross_entropy_with_integer_labels(y, label)) / steps

        precision = self.model.precision
        if self.rule == "bptt":
            total = bptt_loss(self.cell, params, inputs, targets, step_loss, tau=self.tau, dt=self.dt,
                              precision=precision)
            return Ratio(total, jnp.sum(weights)), Aux(metrics={})
        feedback = variables["feedback"]["weight"] if self.rule == "random" else None
        # The trainer's gradient reaches the parameters through `with_gradients` alone, so it never
        # linearizes e-prop's scan, whose memory would then grow with the recording.
        total, grads = eprop(self.cell, jax.lax.stop_gradient(params), inputs, targets, step_loss,
                             tau=self.tau, dt=self.dt, feedback=feedback, precision=precision)
        stats = Ratio(total, jnp.sum(weights))
        update = self._update(grads, variables["params"])
        return _with_rule(self, stats, update, variables["params"]), Aux(metrics={})

    @functools.cached_property
    def _scores(self) -> Callable[[Variables, Batch], tuple[jax.Array, jax.Array]]:
        def scores(params: Variables, batch: Batch) -> tuple[jax.Array, jax.Array]:
            inputs = jnp.swapaxes(jnp.asarray(batch[self.sample.key], jnp.float32), 0, 1)
            outputs = self.model.apply({"params": params}, inputs)
            # `mutable` is unset, so apply returns the outputs alone, not a pair.
            assert not isinstance(outputs, tuple)
            labels = batch[self.labels]
            losses = readout_losses("mean", outputs, labels)
            return losses, jnp.argmax(readout_logits("mean", outputs), -1) == labels

        return jax.jit(scores)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        params = self.evaluation_variables(params, step)["params"]
        return _per_example(*self._scores(params, _fields(batch, self.sample.key, self.labels)))

    def task_record(self) -> Mapping[str, JSON]:
        """The recordings as they are, scored by the readout averaged over time."""
        return _classification_record(self.sample, EventsEncoder(), "mean", self.labels)

    def build_task(self, variables: Variables, *,
                   processor: Processor | None | Omitted = OMITTED) -> SpikingClassification:
        """The trained network as a classifier over `variables`; it reads no text, so takes no
        `processor`."""
        if processor is not OMITTED:
            raise TypeError("a spiking classifier reads no text, so it takes no processor")
        return SpikingClassification(self.model, {"params": thaw(variables)["params"]}, EventsEncoder())


def _check_eprop_network(model: SpikingMLP, channels: int) -> None:
    """Raise unless e-prop trains every parameter of `model`: one recurrent hidden layer, dense
    synapses, fixed time constants, no batch norm and no dropout, computing in float32 or wider."""
    if model.dtype is not None and at_least_fp32(jnp.dtype(model.dtype)) != jnp.dtype(model.dtype):
        raise ValueError(f"e-prop's eligibility traces accumulate in float32 or wider, and this SpikingMLP "
                         f"computes in {jnp.dtype(model.dtype).name}; give it dtype=None or a wider one")
    sample = jax.ShapeDtypeStruct((1, 1, channels), jnp.float32)
    shapes = jax.eval_shape(model.init, jax.random.key(0), sample)
    bias = {"bias"} if model.use_bias else set()
    wanted = {"dense_0": {"kernel"} | bias, "recurrent_0": {"recurrent"}, "readout": {"kernel"} | bias}
    found = {layer: set(leaves) for layer, leaves in shapes.get("params", {}).items()}
    if set(shapes) != {"params"} or found != wanted or model.dropout:
        raise ValueError(
            "e-prop trains a SpikingMLP with one recurrent hidden layer (recurrent=True), dense synapses, "
            "fixed time constants and no batch norm or dropout; this one has the parameters "
            f"{ {name: sorted(leaves) for name, leaves in found.items()} } and dropout {model.dropout}")


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
        stats = self.row_mean(squared_error(output, target), batch)
        correct = (jnp.argmax(output, axis=-1) == labels).astype(jnp.float32)
        metrics = {"accuracy": self.accuracy(correct, batch).mean()[0]}
        if self.rule is None:
            return stats, Aux(metrics=metrics)
        held = variables["params"]
        blocks, params = sequential_blocks(self.stack, jax.lax.stop_gradient(held))
        grads = self.rule.gradient(blocks, params, x, target, rows=_row_weights(batch, x.shape[0]))
        # The update mirrors the parameters: a layer without any, an activation, has none to update.
        update = {f"layers_{k}": grad for k, grad in enumerate(grads) if f"layers_{k}" in held}
        return _with_rule(self, stats, update, held), Aux(metrics=metrics)

    @functools.cached_property
    def _compiled_scored(self) -> Callable[[Variables, Batch], tuple[jax.Array, ...]]:
        return jax.jit(self._scored)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        fields = _fields(batch, self.sample.key, self.labels)
        _, labels, target, output = self._compiled_scored(self.evaluation_variables(params, step), fields)
        return _per_example(squared_error(output, target), jnp.argmax(output, axis=-1) == labels)



type RewardRule = Literal["first", "all", "gated", "agrel", "reinforce"]
"""How `RNeuralNetObjective` learns from a reward; its docstring describes each."""


class RNeuralNetObjective(Objective[Ratio]):
    """Reward an `RNeuralNet`'s choice among its output neurons after a sequence, and learn from the reward.

    The network (`sparx.learn.RNeuralNet`) runs over the field `sample`,
    one sequence of input values `[T, inputs]` per example, from nothing on
    its way. At the last tick it chooses one of its output neurons: in
    training it draws the choice from a softmax of `beta` times their
    outputs, in evaluation it takes the largest. A choice that matches the
    label under `labels` earns a reward of 1 and any other -1, the one
    scalar per example that every rule receives. The weights of all its
    connections are the parameters.

    `rule="first"` or `"all"` is reward diffusion (`sparx.learn.reward_diffusion`,
    with `discount`): each example's reward spreads from the feeder over its
    last tick's outputs, and the mean of the examples' weight changes goes
    to dew's trainer as the loss's gradient with its sign flipped
    (`Objective.with_gradients`), so `optax.sgd(0.01 * batch)` applies the
    original's `W_CONST` of 0.01 for every reward.

    Two rules add the changes attention-gated reinforcement learning (AGREL,
    Roelfsema and van Ooyen 2005) makes to a spread of reward. Both spread
    the reward prediction error `R - b`, with `b` the mean reward of the
    batch's other examples, where AGREL takes an expansive function of its
    error. `rule="gated"` spreads it from the chosen output neuron instead
    of the feeder, along every path (`paths="all"`, with `discount`), still
    shared by the softmax of absolute activity. `rule="agrel"` also sends it
    back through the connections' weights and the neurons' slopes along
    every delayed path, so each weight changes by `R - b` times the
    derivative of the chosen output's last activity. AGREL's feedback
    computes that update layer by layer in a layered network (Pozzi, Bohte
    and Roelfsema 2020); here it is differentiated through the network over
    time. `rule="reinforce"` is REINFORCE (Williams 1992) on the choice,
    `-(R - b) log p(choice)` per example, differentiated the same way. Whatever the rule, the reported loss
    is the negative mean reward of the choices drawn, and evaluation returns
    `TokenScores` for `sparx.metrics.Accuracy`.
    """

    artifact = TokenScores
    shown: Mapping[str, Shown] = {"accuracy": Shown(better="higher", percent=True)}

    def __init__(self, net: RNeuralNet, sample: Field, *, rule: RewardRule = "first", discount: float = 1.0,
                 beta: float = 4.0, labels: str = "label"):
        if rule not in ("first", "all", "gated", "agrel", "reinforce"):
            raise ValueError(f"rule must be first, all, gated, agrel or reinforce, not {rule!r}")
        if sample.shape[1:] != (net.inputs,):
            raise ValueError(f"the sample is one sequence [T, {net.inputs}], not shape {sample.shape}")
        self.net = net
        self.sample = sample
        self.rule: RewardRule = rule
        self.discount, self.beta = discount, beta
        self.labels = labels
        self.inputs = InputSpec(sample=sample)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return {"params": {"weight": self.net.wiring.weight}}

    def _last(self, weight: jax.Array, batch: Batch) -> jax.Array:
        """Every unit's output at the last tick, `[B, size]`, with the connections weighted by `weight`."""
        net = replace(self.net, cell=replace(self.net.cell, wiring=replace(self.net.wiring, weight=weight)))
        x = jnp.swapaxes(jnp.asarray(batch[self.sample.key], jnp.float32), 0, 1)
        return net.run(x)[0][-1]

    def _chosen(self, weight: jax.Array, batch: Batch,
                key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """The last tick's outputs, the log-probabilities of the choices, the choices and their rewards."""
        last = self._last(weight, batch)
        logits = self.beta * last[:, self.net.outputs]
        choice = jax.random.categorical(key, logits)
        reward = jnp.where(choice == jnp.asarray(batch[self.labels]), 1.0, -1.0)
        log_p = jnp.take_along_axis(jax.nn.log_softmax(logits), choice[:, None], axis=-1)[:, 0]
        return last, log_p, choice, reward

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        weight = jax.lax.stop_gradient(variables["params"]["weight"])
        rows = _row_weights(batch, len(batch[self.labels]))

        def error(r: jax.Array) -> jax.Array:
            """Each reward less the mean reward of the batch's other real examples."""
            return r - (jnp.sum(rows * r) - rows * r) / jnp.maximum(jnp.sum(rows) - 1, 1)

        if self.rule in ("reinforce", "agrel"):
            def surrogate(w: jax.Array) -> tuple[jax.Array, jax.Array]:
                last, log_p, choice, r = self._chosen(w, batch, step.key)
                chosen = jnp.take_along_axis(last[:, self.net.outputs], choice[:, None], axis=-1)[:, 0]
                score = log_p if self.rule == "reinforce" else chosen
                return -jnp.sum(rows * jax.lax.stop_gradient(error(r)) * score), r

            update, reward = jax.grad(surrogate, has_aux=True)(weight)
        else:
            last, _, choice, reward = self._chosen(weight, batch, step.key)
            wiring = replace(self.net.wiring, weight=weight)
            paths: Literal["first", "all"] = "first" if self.rule == "first" else "all"
            gated = self.rule == "gated"
            roots = jnp.asarray(self.net.outputs)[choice] if gated else jnp.full_like(choice, self.net.feeder)
            signal = error(reward) if gated else reward

            def change(activity: jax.Array, r: jax.Array, root: jax.Array) -> jax.Array:
                return reward_diffusion(wiring, activity, r, root=root, paths=paths, discount=self.discount,
                                        eta=1.0).change

            update = -jnp.sum(rows[:, None] * jax.vmap(change)(last, signal, roots), axis=0)
        stats = self.row_mean(-reward, batch)
        metrics = {"reward": -stats.mean()[0]}
        return _with_rule(self, stats, {"weight": update}, variables["params"]), Aux(metrics=metrics)

    @functools.cached_property
    def _compiled_last(self) -> Callable[[jax.Array, Batch], jax.Array]:
        return jax.jit(self._last)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        weight = self.evaluation_variables(params, step)["params"]["weight"]
        outputs = self._compiled_last(weight, _fields(batch, self.sample.key))[:, self.net.outputs]
        correct = jnp.argmax(outputs, axis=-1) == jnp.asarray(batch[self.labels])
        return _per_example(jnp.where(correct, -1.0, 1.0), correct)
