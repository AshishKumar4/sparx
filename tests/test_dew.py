"""SpikingClassifier trained by dew's own Trainer, on CPU."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

dew = pytest.importorskip("dew")  # the optional dependency `sparxml[dew]`

from dew import Checkpoints, Field, Trainer  # noqa: E402
from dew.data import Dataset, Loading  # noqa: E402
from dew.objectives.base import Step  # noqa: E402

from sparx.dew import Direct, Events, Rate, RateBand, SpikingClassifier, accuracy  # noqa: E402
from sparx.nn import LI, LIF  # noqa: E402

LOADING = Loading(workers=0, threads=1, read_buffer=1)


def halves(count, seed):
    """8x8 uint8 images, bright in the top half (class 0) or the bottom half (class 1), with noise."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, count)
    images = rng.integers(0, 90, (count, 8, 8, 1))
    rows = np.arange(8)[None, :, None, None]
    bright = np.where(labels[:, None, None, None] == 0, rows < 4, rows >= 4)
    images = np.where(bright, images + 150, images).astype(np.uint8)
    return {"image": images, "label": labels.astype(np.int32)}


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape(*x.shape[:2], -1)
        x = LIF(tau=2.0)(nn.Dense(32)(x))
        return LI(tau=2.0)(nn.Dense(2)(x))


class NormNet(nn.Module):
    """With BatchNorm and dropout, which need `train`."""

    @nn.compact
    def __call__(self, x, train: bool):
        x = x.reshape(*x.shape[:2], -1)
        x = nn.BatchNorm(use_running_average=not train)(nn.Dense(32)(x))
        x = nn.Dropout(0.1, deterministic=not train)(LIF()(x))
        return LI()(nn.Dense(2)(x))


def fit(objective, tmp_path, steps=40, learning_rate=1e-2):
    data = Dataset.from_records(halves(256, 0), batch=32, validation=halves(64, 1), loading=LOADING)
    trainer = Trainer(objective, optax.adam(learning_rate), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path / "run")))
    state = trainer.fit(data, steps=steps, log_every=10, eval_every=steps, metrics=[accuracy])
    trainer.checkpoints.wait()
    return trainer, state


def test_a_rate_coded_spiking_classifier_learns_through_dews_trainer(tmp_path):
    trainer, state = fit(SpikingClassifier(Net(), Field("image", (8, 8, 1)), Rate(steps=8)), tmp_path)
    assert trainer._display.evaluations["val"][-1].scores["val/accuracy"] >= 0.95
    # The validation score is the model's own accuracy on the validation set.
    val = halves(64, 1)
    x = Rate(8)(jax.random.key(5), jnp.asarray(val["image"]))
    predicted = jnp.argmax(jnp.mean(Net().apply(state.variables, x), axis=0), -1)
    assert float(jnp.mean(predicted == val["label"])) >= 0.95


def test_batch_norm_dropout_ema_and_per_step_readout_train_together(tmp_path):
    objective = SpikingClassifier(NormNet(), Field("image", (8, 8, 1)), Direct(steps=4),
                                  readout="per_step", ema_decay=0.9)
    trainer, state = fit(objective, tmp_path)
    assert set(state.variables) == {"params", "batch_stats"}
    # The running statistics moved off their initial zero mean.
    assert float(jnp.abs(state.variables["batch_stats"]["BatchNorm_0"]["mean"]).max()) > 0.1
    assert state.ema is not None
    assert trainer._display.evaluations["val"][-1].scores["val/accuracy"] >= 0.95


def test_the_loss_is_the_mean_cross_entropy_of_the_readout_plus_the_rate_penalty():
    objective = SpikingClassifier(Net(), Field("image", (8, 8, 1)), Direct(steps=4), readout="max",
                                  rates=RateBand(lower=0.3, upper=0.4, weight=2.0))
    batch = {key: jnp.asarray(value) for key, value in halves(16, 2).items()}
    variables = objective.init(jax.random.key(0))
    stats, aux = objective.loss(variables, batch, Step(jnp.asarray(0), jax.random.key(1), None))

    x = Direct(4)(jax.random.key(1), batch["image"])
    outputs, sown = Net().apply(variables, x, mutable=["spike_rates"])
    ce = optax.softmax_cross_entropy_with_integer_labels(jnp.max(outputs, 0), batch["label"])
    (rate,) = sown["spike_rates"]["LIF_0"]["rate"]
    per_neuron = rate.mean(0)
    penalty = jnp.mean(jax.nn.relu(per_neuron - 0.4) ** 2 + jax.nn.relu(0.3 - per_neuron) ** 2)
    value, _ = stats.mean()
    np.testing.assert_allclose(value, ce.mean() + 2.0 * penalty, rtol=1e-5)
    np.testing.assert_allclose(aux.metrics["rate_penalty"], penalty, rtol=1e-5)
    np.testing.assert_allclose(aux.metrics["rate/LIF_0"], rate.mean(), rtol=1e-6)
    assert penalty > 0  # the band is narrow enough to bind


def test_event_data_moves_its_time_axis_to_the_front():
    events = jnp.arange(2 * 5 * 3).reshape(2, 5, 3)
    out = Events(time_axis=0)(jax.random.key(0), events)
    assert out.shape == (5, 2, 3) and out.dtype == jnp.float32
    np.testing.assert_array_equal(out[:, 1], events[1])


def test_an_unknown_readout_is_refused():
    with pytest.raises(ValueError, match="readout"):
        SpikingClassifier(Net(), Field("image", (8, 8, 1)), Direct(4), readout="last")  # type: ignore[arg-type]


class Scaled(nn.Module):
    """Net, with its readout scaled by a keyword the objective schedules."""

    @nn.compact
    def __call__(self, x, scale):
        return Net()(x) * scale


def test_scheduled_call_arguments_reach_the_model_in_loss_and_evaluation():
    objective = SpikingClassifier(Scaled(), Field("image", (8, 8, 1)), Direct(steps=4),
                                  call=lambda step: {"scale": 1.0 + step.astype(jnp.float32)})
    batch = {key: jnp.asarray(value) for key, value in halves(16, 3).items()}
    variables = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(2), jax.random.key(1), None)
    stats, _ = objective.loss(variables, batch, step)
    outputs = Scaled().apply(variables, Direct(4)(jax.random.key(1), batch["image"]), 3.0)
    ce = optax.softmax_cross_entropy_with_integer_labels(jnp.mean(outputs, 0), batch["label"])
    np.testing.assert_allclose(stats.mean()[0], ce.mean(), rtol=1e-5)
    scores = objective.evaluate(variables, batch, step)
    np.testing.assert_allclose(scores.losses[:, 0], ce, rtol=1e-5)
