"""Spiking networks as dew objectives, trained by dew's `Trainer`.

Dew splits a run into a model (a Flax module), an objective (parameters,
loss, evaluation) and a trainer (mesh, compiled step, EMA, checkpoints,
logging). A sparx network is a Flax module, so the one piece this module adds
is the objective: `SpikingClassifier` encodes a batch into spikes, runs the
network over time, and reads its outputs as class scores.

    import optax
    from dew import Field, Trainer
    from dew.data import Dataset
    from sparx.dew import Rate, SpikingClassifier, accuracy

    objective = SpikingClassifier(net, Field("image", (28, 28, 1)), Rate(steps=16))
    trainer = Trainer(objective, optax.adam(1e-3), key=0)
    state = trainer.fit(Dataset.from_records({"image": x, "label": y}, batch=128),
                        steps=2000, metrics=[accuracy])

Importing this module needs dew installed (`pip install "sparxml[dew]"`).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from dew.artifacts import TokenScores
from dew.eval import Mean
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, Batch, EMASpec, Objective, Ratio, Shown, Step, Variables

from sparx import encode
from sparx.losses import per_step_cross_entropy
from sparx.nn import RATES
from sparx.rates import firing_rates, rate_penalty

__all__ = [
    "Direct",
    "Encoder",
    "Events",
    "Latency",
    "Rate",
    "RateBand",
    "Readout",
    "SpikingClassifier",
    "accuracy",
]


class Encoder(Protocol):
    """Turns one batch field `[B, ...]` into the network's time-major input `[T, B, ...]`."""

    def __call__(self, key: jax.Array, x: jax.Array) -> jax.Array: ...


def _intensities(x: jax.Array) -> jax.Array:
    """uint8 pixels as [0, 1]; other values unchanged, as float32."""
    if x.dtype == jnp.uint8:
        return x.astype(jnp.float32) / 255
    return x.astype(jnp.float32)


@dataclass(frozen=True)
class Direct:
    """The values themselves as the input current at each of `steps` steps (`sparx.encode.repeat`)."""

    steps: int

    def __call__(self, key: jax.Array, x: jax.Array) -> jax.Array:
        return encode.repeat(_intensities(x), self.steps)


@dataclass(frozen=True)
class Rate:
    """Bernoulli spikes at the value's probability for `steps` steps (`sparx.encode.rate`).

    A fresh draw every step of training, from the step's key.
    """

    steps: int

    def __call__(self, key: jax.Array, x: jax.Array) -> jax.Array:
        return encode.rate(key, _intensities(x), self.steps)


@dataclass(frozen=True)
class Latency:
    """One spike per value, earlier for larger values, over `steps` steps (`sparx.encode.latency`)."""

    steps: int
    threshold: float = 0.01

    def __call__(self, key: jax.Array, x: jax.Array) -> jax.Array:
        return encode.latency(_intensities(x), self.steps, self.threshold)


@dataclass(frozen=True)
class Events:
    """Data that already holds spikes or currents over time, on axis `time_axis` of each record.

    A record `[T, F]` arrives batched as `[B, T, F]`; the default moves its
    time axis to the front. `dtype` is what the network receives.
    """

    time_axis: int = 0
    dtype: jnp.dtype = jnp.float32

    def __call__(self, key: jax.Array, x: jax.Array) -> jax.Array:
        return jnp.moveaxis(x, self.time_axis + 1, 0).astype(self.dtype)


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


class SpikingClassifier(Objective[Ratio]):
    """Classify a batch field with a spiking network.

    `model` maps the encoder's time-major input `[T, B, ...]` to outputs
    `[T, B, classes]`: spikes, or the membrane of a `sparx.nn.LI` readout.
    `sample` names the field and its per-example shape; `labels` names the
    integer class field. uint8 fields are read as `x / 255`.

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
    """

    artifact = TokenScores
    shown: Mapping[str, Shown] = {"accuracy": Shown(better="higher", percent=True)}

    def __init__(self, model: nn.Module, sample: Field, encoder: Encoder, *, labels: str = "label",
                 readout: Readout = "mean", rates: RateBand | None = None, ema_decay: float | None = None):
        if readout not in ("mean", "max", "sum", "per_step"):
            raise ValueError(f"readout must be mean, max, sum or per_step, not {readout!r}")
        self.model = model
        self.sample = sample
        self.encoder = encoder
        self.labels = labels
        self.readout = readout
        self.rates = rates
        self.inputs = InputSpec(sample=sample)
        self.ema = None if ema_decay is None else EMASpec(decay=optax.constant_schedule(ema_decay))
        self._train = _takes_train(model)

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        encode_key, init_key = jax.random.split(key)
        x = self.encoder(encode_key, jnp.zeros((1, *self.sample.shape), jnp.float32))
        kwargs = {"train": False} if self._train else {}
        return dict(self.model.init(init_key, x, **kwargs))

    def _logits(self, outputs: jax.Array) -> jax.Array:
        outputs = outputs.astype(jnp.float32)
        if self.readout == "max":
            return jnp.max(outputs, axis=0)
        if self.readout == "sum":
            return jnp.sum(outputs, axis=0)
        return jnp.mean(outputs, axis=0)

    def _losses(self, outputs: jax.Array, labels: jax.Array) -> jax.Array:
        if self.readout == "per_step":
            return per_step_cross_entropy(outputs, labels)
        return optax.softmax_cross_entropy_with_integer_labels(self._logits(outputs), labels)

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        encode_key, dropout_key = jax.random.split(step.key)
        x = self.encoder(encode_key, jnp.asarray(batch[self.sample.key]))
        labels = jnp.asarray(batch[self.labels])
        mutable = [RATES, *(["batch_stats"] if "batch_stats" in variables else [])]
        kwargs = {"train": True} if self._train else {}
        outputs, updated = self.model.apply(variables, x, rngs={"dropout": dropout_key}, mutable=mutable,
                                            **kwargs)
        losses = self._losses(outputs, labels)
        total = jnp.sum(losses)
        metrics = {"accuracy": jnp.mean(jnp.argmax(self._logits(outputs), -1) == labels)}
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
        kwargs = {"train": False} if self._train else {}

        def scores(variables, field, labels, key):
            outputs = self.model.apply(variables, self.encoder(key, field), **kwargs)
            losses = self._losses(outputs, labels)
            correct = jnp.argmax(self._logits(outputs), -1) == labels
            return losses[:, None], jnp.ones_like(losses)[:, None], correct[:, None]

        return jax.jit(scores)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        variables = params if step.ema is None else step.ema
        losses, weights, correct = self._scores(
            variables, jnp.asarray(batch[self.sample.key]), jnp.asarray(batch[self.labels]), step.key)
        return TokenScores(losses=losses, weights=weights, correct=correct)


accuracy = Mean(lambda scores, batch: scores.correct[:, 0], name="accuracy", better="higher", reads=TokenScores)
"""Validation accuracy from a `SpikingClassifier`'s evaluation, as `val/accuracy`."""
