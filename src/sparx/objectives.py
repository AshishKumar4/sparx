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

Each is registered in dew's `objectives` table under the method's name, as
dew's objectives are.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from dew.artifacts import TokenScores
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, Batch, EMASpec, Objective, Ratio, Shown, Step, Variables, thaw
from dew.records import JSON
from dew.registry import objectives
from dew.training.optim import ParamGroup, ScheduleBase

from sparx.datasets import WEIGHT
from sparx.dynamics import NeuronModel
from sparx.encode import SpikeEncoder
from sparx.learn import EPropParams, bptt_loss, eprop, eprop_forward
from sparx.losses import READOUTS, Readout, readout_logits, readout_losses, van_rossum
from sparx.nn import RATES
from sparx.optim import GroupAdam, stepped
from sparx.rates import firing_rates, rate_penalty
from sparx.tasks import SpikingClassification, bound_call

if TYPE_CHECKING:
    from dew.training.state import TrainState

__all__ = ["ActivityFitObjective", "EPropObjective", "RateBand", "SpikingClassifierObjective"]


_TRAINERS = "trainer"
"""The group of the parameters no `GroupAdam` matches, which the trainer's optimizer updates."""


@dataclass(frozen=True)
class RateBand:
    """Keep each spiking neuron's rate within `[lower, upper]`, adding `weight * sparx.rate_penalty`."""

    lower: float = 0.0
    upper: float = 1.0
    weight: float = 1.0


@objectives("spiking_classifier")
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

    The loss is the mean cross entropy of the `readout` over the batch, plus
    the `rates` penalty when given. Metrics report the batch accuracy, each
    spiking layer's mean firing rate (`rate/<layer>`) and the penalty.
    Evaluation returns `TokenScores` with one row per example (loss, weight,
    whether the argmax is the label), which `sparx.metrics.Accuracy` reads.

    `schedules` names model keyword arguments that follow a schedule over
    `schedule_steps` steps, as dew's learning-rate schedules do:
    `{"sigma": Linear(peak=7.5, end=0.5)}` anneals a
    `sparx.nn.DelayedDense`, `{"masking": ...}` a `MaskedPSN`. The value is
    read at `Step.step`, the count of accepted microbatches, in the loss and
    in evaluation alike. `schedule_every` advances every schedule, these and
    the groups' learning rates, once every that many steps
    (`sparx.optim.stepped`); the steps of an epoch reproduce a torch
    scheduler stepped once an epoch. `deployed` holds model keyword
    arguments that evaluation and the trained classifier run with in place
    of the schedules' values, such as `{"sigma": 0}` to score every delay
    rounded to a whole step, the network as deployed and as SNN-delays
    evaluates it.

    `groups` gives parameter groups their own optimizers, by name: each
    `sparx.optim.GroupAdam` updates the parameters its patterns match, the
    first matching group winning, over `schedule_steps` updates. The
    trainer's optimizer updates the rest, under `optax.multi_transform`
    (`optimizer`). Delay positions learn at their own rate this way.

    Evaluation weighs each example by the batch's `sparx.datasets.WEIGHT`
    field when it has one (`sparx.datasets.whole_batches`), so the loss and
    `Accuracy` cover a split of any size exactly.

    The objective records the model, encoder, readout, schedules and
    deployed arguments with every checkpoint (`inference_record`), and loads
    back as a `sparx.tasks.SpikingClassification`
    (`SpikingClassification.from_run`, or `pipeline(state)` after training).
    """

    artifact = TokenScores
    saved_task = SpikingClassification
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

    def _scheduled(self, step: jax.Array) -> dict[str, jax.Array]:
        if self.schedule_steps is None:
            return {}
        return {name: jnp.asarray(stepped(schedule, self.schedule_steps, self.schedule_every)(step))
                for name, schedule in self.schedules.items()}

    def _method(self, step: jax.Array, *, train: bool) -> functools.partial[jax.Array]:
        """The model's `__call__` with `train` and the scheduled arguments bound.

        Out of training, the deployed arguments replace the scheduled ones.
        """
        kwargs: dict[str, jax.Array | float] = dict(self._scheduled(step))
        if not train:
            kwargs |= self.deployed
        return bound_call(self.model, train=train, kwargs=kwargs)

    def optimizer(self, tx: optax.GradientTransformation, *,
                  accumulation: int) -> optax.GradientTransformation:
        """`tx` for the parameters no group matches and each group's `GroupAdam` for its own."""
        # Remove when AshishKumar4/dew#37 merges: `OptimConfig.param_groups` with per-group schedules,
        # b1 and bounds, so the trainer's optimizer covers every group and this override goes.
        if not self.groups:
            return tx
        assert self.schedule_steps is not None  # __init__ refuses groups without it
        solvers: dict[Hashable, optax.GradientTransformation] = {
            name: group.build(self.schedule_steps, self.schedule_every)
            for name, group in self.groups.items()}
        solvers[_TRAINERS] = tx
        # Remove when AshishKumar4/dew#39 merges: `param_labels` in dew.training.optim's `__all__`.
        from dew.training.optim import param_labels
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
        applied = self.model.apply(variables, x, rngs=rngs, mutable=mutable,
                                   method=self._method(step.step, train=True))
        # With mutable collections, apply returns the outputs and the collections.
        assert isinstance(applied, tuple)
        outputs, updated = applied
        losses = readout_losses(self.readout, outputs, labels)
        total = jnp.sum(losses)
        metrics = {"accuracy": jnp.mean(jnp.argmax(readout_logits(self.readout, outputs), -1) == labels)}
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
        def scores(variables: Variables, field: jax.Array, labels: jax.Array, weights: jax.Array,
                   key: jax.Array, step: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            x = self.encoder(key, field)
            outputs = self.model.apply(variables, x, method=self._method(step, train=False))
            # `mutable` is unset, so apply returns the outputs alone, not a pair.
            assert not isinstance(outputs, tuple)
            losses = readout_losses(self.readout, outputs, labels)
            correct = jnp.argmax(readout_logits(self.readout, outputs), -1) == labels
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
        # Remove when AshishKumar4/dew#34 merges: dew.config.to_json, the same function made public.
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

    def pipeline(self, state: TrainState, *, ema: bool | None = None) -> SpikingClassification:
        """The trained classifier over `state`'s weights.

        The schedules stand at their final values, with the deployed
        arguments over them.
        """
        # Remove when AshishKumar4/dew#39 merges: `Objective.pipeline_variables(state, ema=ema)`.
        variables = self._pipeline_weights(state, ema)
        final = jnp.asarray(self.schedule_steps or 0)
        call = {name: float(value) for name, value in self._scheduled(final).items()} | self.deployed
        return SpikingClassification(self.model, thaw(variables), self.encoder, self.readout, call)


@objectives("activity_fit")
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
    rate is reproducible. The loss is reported per example; metrics give the
    model's and the recording's mean rates (spikes per step). Evaluation
    returns one `TokenScores` row per example, its loss.
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

            def smooth(train: jax.Array) -> jax.Array:
                return jnp.convolve(train, kernel, mode="same")

            def psth(trains: jax.Array) -> jax.Array:  # [T, B, N] -> [T, N]
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


type EPropRule = Literal["eprop", "random", "bptt"]
"""Where `EPropObjective`'s gradient comes from.

- `eprop`: e-prop with the readout weights as feedback (symmetric e-prop).
- `random`: e-prop with fixed random feedback weights (random e-prop).
- `bptt`: backpropagation through time on the same network, for comparison.
"""


@objectives("eprop")
class EPropObjective(Objective[Ratio]):
    """Classify recordings with a recurrent spiking layer whose gradients are e-prop's.

    The network is `sparx.learn.eprop_forward`'s: a recurrent layer of
    `cell` with `hidden` neurons over the `[T, channels]` field `sample`, and
    a leaky readout of time constant `tau` with `classes` outputs, all
    stepped at `dt`. Each step's cross entropy, divided by the steps, is
    summed over time, so the loss is the per-step mean, averaged over the
    batch. The class is the argmax of the readout averaged over time.

    Dew's trainer differentiates the loss, and an objective has no hook to
    hand it a gradient of its own. The loss therefore carries e-prop's
    gradient as its custom VJP (`jax.custom_vjp`): the forward pass of a
    differentiated loss runs `sparx.learn.eprop`, which computes the
    gradients online in memory that does not grow with the recording, and
    the backward pass scales them by the cotangent. `rule="bptt"`
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

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
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

        def step_loss(y: jax.Array, label: jax.Array) -> jax.Array:
            return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(y, label)) / steps

        if self.rule == "bptt":
            total = bptt_loss(self.cell, params, inputs, targets, step_loss, tau=self.tau, dt=self.dt)
        else:
            feedback = variables["feedback"]["weight"] if self.rule == "random" else None
            total = self._online(params, inputs, targets, step_loss, feedback)
        stats = Ratio(total, jnp.asarray(labels.shape[0], jnp.float32))
        return stats, Aux(metrics={})

    def _online(self, params: EPropParams, inputs: jax.Array, targets: jax.Array,
                step_loss: Callable[[jax.Array, jax.Array], jax.Array],
                feedback: jax.Array | None) -> jax.Array:
        """The summed loss, whose gradient by `params` is e-prop's."""
        cell, tau, dt = self.cell, self.tau, self.dt
        no_self = 1 - jnp.eye(self.hidden)

        @jax.custom_vjp
        def summed(params: EPropParams, inputs: jax.Array, targets: jax.Array,
                   feedback: jax.Array | None) -> jax.Array:
            return bptt_loss(cell, params, inputs, targets, step_loss, tau=tau, dt=dt)

        def forward(params: EPropParams, inputs: jax.Array, targets: jax.Array,
                    feedback: jax.Array | None) -> tuple[jax.Array, EPropParams]:
            return eprop(cell, params, inputs, targets, step_loss, tau=tau, dt=dt, feedback=feedback)

        def backward(grads: EPropParams, cotangent: jax.Array) -> tuple[EPropParams, None, None, None]:
            # The data and the fixed feedback weights take no gradient.
            grads = grads._replace(w_rec=grads.w_rec * no_self)
            return jax.tree.map(lambda grad: cotangent * grad, grads), None, None, None

        summed.defvjp(forward, backward)
        return summed(params, inputs, targets, feedback)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        network, inputs, labels = self._network(params if step.ema is None else step.ema, batch)
        outputs, _ = eprop_forward(self.cell, network, inputs, tau=self.tau, dt=self.dt)
        logits = jnp.mean(outputs, axis=0)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        ones = jnp.ones_like(losses)[:, None]
        return TokenScores(losses=losses[:, None], weights=ones,
                           correct=(jnp.argmax(logits, -1) == labels)[:, None])

    def inference_record(self) -> JSON:
        return None
