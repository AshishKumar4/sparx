"""The spiking objectives trained by dew's own Trainer, on CPU."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from dew import Checkpoints, Field, Trainer
from dew.artifacts import TokenScores
from dew.config import OptimConfig
from dew.data import Dataset, Loading
from dew.objectives.base import VALID_ROWS, Step
from dew.training.optim import Exponential, Linear, OneCycle, ParamGroup

import sparx
from sparx.datasets import holdout
from sparx.encode import DirectEncoder, EventsEncoder, RateEncoder
from sparx.learn import (
    PredictiveCoding,
    RNeuralNet,
    residual_mlp,
    reward_diffusion,
    sequential_blocks,
    squared_error,
)
from sparx.metrics import Accuracy
from sparx.nn import LI, LIF
from sparx.objectives import (
    ActivityFitObjective,
    EPropObjective,
    PredictiveCodingObjective,
    RateBand,
    RNeuralNetObjective,
    SpikingClassifierObjective,
)
from sparx.tasks import SpikingClassification

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
    def __call__(self, x, train: bool = False):
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


def fit(objective, tmp_path, steps=40, optimizer=None):
    data = Dataset.from_records(halves(256, 0), batch=32, validation=halves(64, 1), loading=LOADING)
    trainer = Trainer(objective, optax.adam(1e-2) if optimizer is None else optimizer,
                      key=jax.random.key(0), checkpoints=Checkpoints(str(tmp_path / "run")))
    state = trainer.fit(data, steps=steps, log_every=10, eval_every=steps, metrics=[Accuracy()])
    trainer.checkpoints.wait()
    return trainer, state


def test_a_rate_coded_spiking_classifier_learns_through_dews_trainer(tmp_path):
    objective = SpikingClassifierObjective(Net(), Field("image", (8, 8, 1)), RateEncoder(steps=8))
    trainer, state = fit(objective, tmp_path)
    assert trainer._display.evaluations["val"][-1].scores["val/accuracy"] >= 0.95
    # The validation score is the model's own accuracy on the validation set.
    val = halves(64, 1)
    x = RateEncoder(8)(jax.random.key(5), jnp.asarray(val["image"]))
    predicted = jnp.argmax(jnp.mean(Net().apply(state.variables, x), axis=0), -1)
    assert float(jnp.mean(predicted == val["label"])) >= 0.95


def test_batch_norm_dropout_ema_and_per_step_readout_train_together(tmp_path):
    objective = SpikingClassifierObjective(NormNet(), Field("image", (8, 8, 1)), DirectEncoder(steps=4),
                                           readout="per_step", ema_decay=0.9)
    trainer, state = fit(objective, tmp_path)
    assert set(state.variables) == {"params", "batch_stats"}
    # The running statistics moved off their initial zero mean.
    assert float(jnp.abs(state.variables["batch_stats"]["BatchNorm_0"]["mean"]).max()) > 0.1
    assert state.ema is not None
    assert trainer._display.evaluations["val"][-1].scores["val/accuracy"] >= 0.95


def test_the_loss_is_the_mean_cross_entropy_of_the_readout_plus_the_rate_penalty():
    objective = SpikingClassifierObjective(Net(), Field("image", (8, 8, 1)), DirectEncoder(steps=4),
                                           readout="max", rates=RateBand(lower=0.3, upper=0.4, weight=2.0))
    batch = {key: jnp.asarray(value) for key, value in halves(16, 2).items()}
    variables = objective.init(jax.random.key(0))
    stats, aux = objective.loss(variables, batch, Step(jnp.asarray(0), jax.random.key(1), None))

    x = DirectEncoder(4)(jax.random.key(1), batch["image"])
    outputs, sown = Net().apply(variables, x, mutable=["spike_rates"])
    ce = optax.softmax_cross_entropy_with_integer_labels(jnp.max(outputs, 0), batch["label"])
    (rate,) = sown["spike_rates"]["LIF_0"]["rate"]
    per_neuron = rate.mean(0)
    penalty = jnp.mean(jax.nn.relu(per_neuron - 0.4) ** 2 + jax.nn.relu(0.3 - per_neuron) ** 2)
    value, _ = stats.mean()
    np.testing.assert_allclose(value, ce.mean() + 2.0 * penalty, rtol=1e-5)  # observed 0 relative
    np.testing.assert_allclose(aux.metrics["rate_penalty"], penalty, rtol=1e-5)  # observed 0 relative
    np.testing.assert_allclose(aux.metrics["rate/LIF_0"], rate.mean(), rtol=1e-6)  # observed 0 relative
    assert penalty > 0  # the band is narrow enough to bind


def test_event_data_moves_its_time_axis_to_the_front():
    events = jnp.arange(2 * 5 * 3).reshape(2, 5, 3)
    out = EventsEncoder(time_axis=0)(jax.random.key(0), events)
    assert out.shape == (5, 2, 3) and out.dtype == jnp.float32
    np.testing.assert_array_equal(out[:, 1], events[1])


def test_an_unknown_readout_is_refused():
    with pytest.raises(ValueError, match="readout"):
        SpikingClassifierObjective(Net(), Field("image", (8, 8, 1)), DirectEncoder(4), readout="last")  # type: ignore[arg-type]


class Scaled(nn.Module):
    """Net, with its readout scaled by a keyword the objective schedules."""

    @nn.compact
    def __call__(self, x, scale, train: bool = False):
        return Net()(x) * scale


def test_scheduled_call_arguments_reach_the_model_in_loss_and_evaluation():
    # A linear ramp from 3 to 1 over 4 steps is 2 at step 2.
    objective = SpikingClassifierObjective(Scaled(), Field("image", (8, 8, 1)), DirectEncoder(steps=4),
                                           schedules={"scale": Linear(peak=3.0, end=1.0)}, schedule_steps=4)
    batch = {key: jnp.asarray(value) for key, value in halves(16, 3).items()}
    variables = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(2), jax.random.key(1), None)
    stats, _ = objective.loss(variables, batch, step)
    outputs = Scaled().apply(variables, DirectEncoder(4)(jax.random.key(1), batch["image"]), 2.0)
    ce = optax.softmax_cross_entropy_with_integer_labels(jnp.mean(outputs, 0), batch["label"])
    np.testing.assert_allclose(stats.mean()[0], ce.mean(), rtol=1e-5)  # observed 0 relative
    scores = objective.evaluate(variables, batch, step)
    np.testing.assert_allclose(scores.losses[:, 0], ce, rtol=1e-5)  # observed 0 relative


def test_schedules_need_their_horizon():
    with pytest.raises(ValueError, match="schedule_steps"):
        SpikingClassifierObjective(Scaled(), Field("image", (8, 8, 1)), DirectEncoder(4),
                                   schedules={"scale": Linear(peak=1.0)})


def test_deployed_arguments_replace_the_schedules_in_evaluation_and_the_trained_classifier():
    objective = SpikingClassifierObjective(Scaled(), Field("image", (8, 8, 1)), DirectEncoder(steps=4),
                                           schedules={"scale": Linear(peak=3.0, end=1.0)}, schedule_steps=4,
                                           deployed={"scale": 5.0})
    batch = {key: jnp.asarray(value) for key, value in halves(16, 3).items()}
    variables = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(2), jax.random.key(1), None)
    x = DirectEncoder(4)(jax.random.key(1), batch["image"])

    def ce(scale):
        outputs = Scaled().apply(variables, x, scale)
        return optax.softmax_cross_entropy_with_integer_labels(jnp.mean(outputs, 0), batch["label"])

    stats, _ = objective.loss(variables, batch, step)
    # Observed 0 relative.
    np.testing.assert_allclose(stats.mean()[0], ce(2.0).mean(), rtol=1e-5)  # training follows the schedule
    # Observed 3.4e-7 relative.
    np.testing.assert_allclose(objective.evaluate(variables, batch, step).losses[:, 0], ce(5.0), rtol=1e-5)
    assert objective.task_record()["deployed"] == {"scale": 5.0}


def test_a_schedule_advances_once_every_its_every_steps():
    # Over 8 steps advancing every 4, the ramp from 3 to 1 runs over 2 of its own steps: 3, then 2.
    objective = SpikingClassifierObjective(Scaled(), Field("image", (8, 8, 1)), DirectEncoder(steps=4),
                                           schedules={"scale": Linear(peak=3.0, end=1.0, every=4)},
                                           schedule_steps=8)
    values = [float(objective._scheduled(jnp.asarray(step))["scale"]) for step in range(8)]
    assert values == [3.0] * 4 + [2.0] * 4


def test_the_softmax_sum_readout_is_scored_and_predicted_by_the_summed_probabilities():
    objective = SpikingClassifierObjective(Net(), Field("image", (8, 8, 1)), DirectEncoder(steps=4),
                                           readout="softmax_sum")
    batch = {key: jnp.asarray(value) for key, value in halves(16, 4).items()}
    variables = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(1), None)
    outputs = Net().apply(variables, DirectEncoder(4)(jax.random.key(1), batch["image"]))
    probabilities = jnp.sum(jax.nn.softmax(outputs, -1), 0)
    expected = optax.softmax_cross_entropy_with_integer_labels(probabilities, batch["label"])
    stats, _ = objective.loss(variables, batch, step)
    np.testing.assert_allclose(stats.mean()[0], expected.mean(), rtol=1e-5)  # observed 0 relative
    scores = objective.evaluate(variables, batch, step)
    np.testing.assert_array_equal(scores.correct[:, 0], jnp.argmax(probabilities, -1) == batch["label"])


def delayed_net():
    from sparx.models import SpikingMLP
    return SpikingMLP(hidden=(16,), classes=2, neuron=LIF(tau=2.0), delays=(3, 3), batch_norm=True)


def delay_groups(delays):
    """SNN-delays' split: the delays on their own schedule, clamped to the kernel; the rest at 1e-2."""
    return OptimConfig(optimizer="adam", learning_rate=1e-2, param_groups=(
        ParamGroup("delays", ("*/delay",), schedule=delays, bounds=(0.0, 3.0)), ParamGroup("rest", ("*",))))


def test_dews_parameter_groups_find_the_delays_and_keep_them_within_bounds(tmp_path):
    # A rate of 1 per step pushes delays past their range at once; the bounds clamp them.
    objective = SpikingClassifierObjective(delayed_net(), Field("image", (8, 8, 1)), DirectEncoder(steps=4),
                                           readout="max", schedules={"sigma": Linear(peak=1.0, end=1.0)},
                                           schedule_steps=10)
    _, state = fit(objective, tmp_path, steps=10, optimizer=delay_groups(Linear(peak=1.0, end=1.0)))
    delays = np.concatenate([np.ravel(state.variables["params"][layer]["delay"])
                             for layer in ("delayed_0", "readout")])
    assert delays.min() >= 0 and delays.max() <= 3
    assert np.mean((delays == 0) | (delays == 3)) > 0.3


def test_every_record_of_a_split_is_scored_once_whatever_the_batch(tmp_path):
    # 70 records in batches of 32: dew pads the last batch with repeats, which count for nothing.
    objective = SpikingClassifierObjective(Net(), Field("image", (8, 8, 1)), DirectEncoder(steps=4))
    val, test = halves(70, 1), halves(45, 5)

    def split(records):
        return Dataset.from_records(records, batch=32, validation=records, loading=LOADING).val

    data = Dataset.from_records(halves(256, 0), batch=32, loading=LOADING)
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0))
    state = trainer.fit(data, steps=3, log_every=3, eval_every=3, metrics=[Accuracy()],
                        validation={"val": split(val), "test": split(test)})
    classifier = objective.pipeline(state)
    for name, records in (("val", val), ("test", test)):
        expected = np.mean(np.asarray(classifier(records["image"])) == records["label"])
        evaluation = trainer._display.evaluations[name][-1]
        np.testing.assert_allclose(evaluation.scores[f"{name}/accuracy"], expected)  # observed 0 relative
        assert evaluation.records == len(records["label"])


def padded_batch(batch, repeats):
    """`batch` with its first `repeats` rows repeated after it, marked as dew marks a split's last batch."""
    padded = {key: jnp.concatenate([value, value[:repeats]]) for key, value in batch.items()}
    rows = len(next(iter(batch.values())))
    return padded | {VALID_ROWS: jnp.arange(rows + repeats) < rows}


def test_a_repeated_row_counts_for_nothing_in_the_loss():
    # A narrow band, so the rate penalty is part of the loss.
    objective = SpikingClassifierObjective(Net(), Field("image", (8, 8, 1)), DirectEncoder(steps=4),
                                           readout="max", rates=RateBand(lower=0.3, upper=0.4, weight=2.0))
    real = {key: jnp.asarray(value) for key, value in halves(12, 2).items()}
    variables = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(1), None)
    # Repeating the first 4 rows shifts the batch's rates, so the repeats would move the penalty if they
    # counted.
    stats, aux = objective.loss(variables, padded_batch(real, 4), step)
    unpadded, unpadded_aux = objective.loss(variables, real, step)
    assert float(stats.mass) == 12 and aux.metrics["rate_penalty"] > 0
    np.testing.assert_allclose(stats.mean()[0], unpadded.mean()[0], rtol=1e-6)  # observed 0 relative
    np.testing.assert_allclose(aux.metrics["rate_penalty"], unpadded_aux.metrics["rate_penalty"],
                               rtol=1e-6)  # observed 0 relative
    np.testing.assert_allclose(aux.metrics["accuracy"], unpadded_aux.metrics["accuracy"],
                               rtol=1e-6)  # observed 0 relative


def test_a_holdout_splits_the_records_without_overlap():
    records = {"x": np.arange(100), "y": np.arange(100) * 2}
    train, held = holdout(records, 0.1, seed=3)
    assert len(held["x"]) == 10 and len(train["x"]) == 90
    assert sorted(np.concatenate([train["x"], held["x"]]).tolist()) == list(range(100))
    np.testing.assert_array_equal(held["y"], held["x"] * 2)
    with pytest.raises(ValueError, match="empty"):
        holdout(records, 0.001)


@pytest.mark.parametrize("delayed", [False, True])
def test_a_saved_run_loads_back_through_dew_pipeline_in_a_fresh_process(tmp_path, delayed):
    import json
    import os
    import subprocess
    import sys

    from sparx.models import SpikingMLP

    neuron = LIF(tau=2.0, surrogate=sparx.surrogate.FastSigmoid(50.0))
    optimizer = optax.adam(1e-2)
    if delayed:
        # SNN-delays' shape: every synapse delayed and extended, batch norm, the softmax-sum readout,
        # a width shrinking once every 2 steps, the delays on their own optimizer, and the rounded
        # delays deployed.
        net = SpikingMLP(hidden=(16,), classes=2, neuron=neuron, delays=(3, 2), extend=True, batch_norm=True,
                         use_bias=False, weight_init="kaiming_uniform", dropout=0.2, dropout_mask="sequence")
        objective = SpikingClassifierObjective(
            net, Field("image", (8, 8, 1)), RateEncoder(steps=6), readout="softmax_sum",
            schedules={"sigma": Exponential(init=1.5, end=0.23, offset=0.27, every=2)}, schedule_steps=8,
            deployed={"sigma": 0})
        optimizer = delay_groups(OneCycle(peak=0.1, init=0.01, end=0.0))
    else:
        net = SpikingMLP(hidden=(16,), classes=2, neuron=neuron)
        objective = SpikingClassifierObjective(net, Field("image", (8, 8, 1)), RateEncoder(steps=6),
                                               readout="max")
    run = tmp_path / "run"
    data = Dataset.from_records(halves(64, 0), batch=32, loading=LOADING)
    trainer = Trainer(objective, optimizer, key=jax.random.key(0), checkpoints=Checkpoints(str(run)))
    state = trainer.fit(data, steps=4, log_every=4, checkpoint_every=4)
    trainer.checkpoints.wait()
    images = halves(8, 7)["image"]
    expected = objective.pipeline(state).logits(images, key=3)
    program = ("import json, sys\n"
               "import dew\n"
               f"task = dew.pipeline({str(run)!r}, trust=('sparx',))\n"
               "assert 'sparx' in sys.modules\n"
               "import numpy as np\n"
               f"images = np.asarray(json.loads({json.dumps(images.tolist())!r}), np.uint8)\n"
               "print(type(task).__name__, json.dumps(task.call), "
               "json.dumps(np.asarray(task.logits(images, key=3)).tolist()))\n")
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=600,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr[-3000:]
    name, rest = done.stdout.strip().split(" ", 1)
    call, end = json.JSONDecoder().raw_decode(rest)
    logits = rest[end:]
    assert name == SpikingClassification.__name__
    # The deployed width replaces the schedule's last value in the loaded classifier.
    assert call == ({"sigma": 0.0} if delayed else {})
    np.testing.assert_array_equal(np.asarray(json.loads(logits), np.float32), np.asarray(expected))


def test_activity_fit_recovers_a_teachers_spiking():
    # A teacher network's spikes are the recording; a student with other
    # weights fits them by the van Rossum distance and gets closer to the
    # teacher's weights' behavior than where it started.
    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            return LIF(tau=4.0)(nn.Dense(6)(x))

    rng = np.random.default_rng(0)
    stimulus = (rng.random((32, 40, 8)) < 0.2).astype(np.float32)  # [B, T, in]
    teacher = Net()
    teacher_vars = teacher.init(jax.random.key(1), jnp.zeros((40, 1, 8)))
    teacher_vars = jax.tree.map(lambda w: 2.5 * w, teacher_vars)
    recording = np.swapaxes(np.asarray(teacher.apply(teacher_vars, jnp.swapaxes(stimulus, 0, 1))), 0, 1)
    assert recording.mean() > 0.02
    objective = ActivityFitObjective(Net(), Field("stimulus", (40, 8)), recording="spikes", tau=5.0)
    variables = objective.init(jax.random.key(2))
    batch = {"stimulus": stimulus, "spikes": recording}
    step = Step(jnp.asarray(0), jax.random.key(0), None)
    optimizer = optax.adam(3e-2)
    opt_state = optimizer.init(variables)

    def total(variables):
        stats, _ = objective.loss(variables, batch, step)
        return stats.total / stats.mass

    start = float(total(variables))
    gradient = jax.jit(jax.grad(total))
    for _ in range(150):
        grads = gradient(variables)
        updates, opt_state = optimizer.update(grads, opt_state)
        variables = optax.apply_updates(variables, updates)
    assert float(total(variables)) < 0.4 * start


def test_the_accuracy_metric_counts_each_example_by_its_weight():
    scores = TokenScores(losses=jnp.zeros((3, 1)), weights=jnp.asarray([[1.0], [1.0], [0.0]]),
                         correct=jnp.asarray([[True], [False], [True]]))
    assert Accuracy()(scores, {}) == (1.0, 2.0)


def eprop_model(dt=1.0, **kwargs):
    from sparx.models import SpikingMLP
    from sparx.nn import ALIF
    from sparx.surrogate import Triangle

    neuron = ALIF(tau=4.0, tau_adapt=20.0, beta=0.2, detach_reset=True, surrogate=Triangle(scale=0.3), dt=dt)
    return SpikingMLP(hidden=(6,), classes=3, neuron=neuron, recurrent=True, readout_tau=4.0, **kwargs)


def eprop_objective(rule="eprop", **kwargs):
    return EPropObjective(eprop_model(**kwargs), Field("spikes", (12, 5)), rule=rule)


def eprop_batch(seed):
    rng = np.random.default_rng(seed)
    return {"spikes": (rng.random((8, 12, 5)) < 0.3).astype(np.uint8),
            "label": rng.integers(0, 3, 8).astype(np.int32)}


def objective_gradients(objective, variables, batch):
    step = Step(jnp.asarray(0), jax.random.key(1), None)
    return jax.value_and_grad(lambda v: objective.scalar_loss(v, batch, step)[0])(variables)


@pytest.mark.parametrize("rule", ["eprop", "random"])
def test_the_eprop_objectives_gradient_is_eprops(rule):
    from sparx.learn import EPropParams, eprop

    objective = eprop_objective(rule)
    variables = objective.init(jax.random.key(0))
    batch = {key: jnp.asarray(value) for key, value in eprop_batch(1).items()}
    _, grads = objective_gradients(objective, variables, batch)

    # e-prop trains a bias as the weight of an input that is always 1.
    held = variables["params"]
    w_in = jnp.concatenate([held["dense_0"]["kernel"], held["dense_0"]["bias"][None]])
    params = EPropParams(w_in, held["recurrent_0"]["recurrent"], held["readout"]["kernel"],
                         held["readout"]["bias"])
    spikes = jnp.swapaxes(batch["spikes"].astype(jnp.float32), 0, 1)
    inputs = jnp.concatenate([spikes, jnp.ones((12, 8, 1))], axis=-1)
    targets = jnp.broadcast_to(batch["label"], (12, 8))

    def step_loss(y, label):
        return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(y, label)) / 12

    feedback = variables["feedback"]["weight"] if rule == "random" else None
    _, expected = eprop(objective.cell, params, inputs, targets, step_loss, tau=4.0, feedback=feedback)
    no_self = 1 - jnp.eye(6)
    want = {"dense_0": {"kernel": expected.w_in[:5], "bias": expected.w_in[5]},
            "recurrent_0": {"recurrent": expected.w_rec * no_self},
            "readout": {"kernel": expected.w_out, "bias": expected.b_out}}
    # The trainer differentiates the mean over the batch of 8.
    jax.tree.map(lambda got, w: np.testing.assert_allclose(got, w / 8, rtol=1e-6, atol=1e-8),  # observed 0
                 grads["params"], want)
    assert np.all(np.diag(grads["params"]["recurrent_0"]["recurrent"]) == 0)
    assert all(np.any(np.asarray(g) != 0) for g in jax.tree.leaves(grads["params"]))


@pytest.mark.parametrize("dt", [1.0, 2.0])
def test_the_eprop_objective_trains_the_spiking_mlp_it_was_given(dt):
    # By BPTT the objective's loss and gradient are those of the model's own forward pass, so the
    # network e-prop trains is the one `pipeline` returns, biases and time step included.
    objective = eprop_objective("bptt", dt=dt)
    variables = objective.init(jax.random.key(0))
    assert np.all(np.diag(variables["params"]["recurrent_0"]["recurrent"]) == 0)
    batch = {key: jnp.asarray(value) for key, value in eprop_batch(1).items()}
    loss, grads = objective_gradients(objective, variables, batch)

    def models_loss(params):
        spikes = jnp.swapaxes(batch["spikes"].astype(jnp.float32), 0, 1)
        outputs = objective.model.apply({"params": params}, spikes)
        labels = jnp.broadcast_to(batch["label"], outputs.shape[:2])
        per_step = optax.softmax_cross_entropy_with_integer_labels(outputs, labels)
        return jnp.mean(jnp.mean(per_step, axis=0))

    want, want_grads = jax.value_and_grad(models_loss)(variables["params"])
    np.testing.assert_allclose(loss, want, rtol=1e-6)  # observed 1.1e-7
    # The model's gradient reaches the recurrent diagonal, which the objective keeps at 0.
    off = 1 - jnp.eye(6)
    want_grads["recurrent_0"]["recurrent"] = want_grads["recurrent_0"]["recurrent"] * off
    jax.tree.map(lambda got, w: np.testing.assert_allclose(got, w, rtol=1e-5, atol=1e-7),  # observed 1.5e-7
                 grads["params"], want_grads)
    assert all(np.any(np.asarray(g) != 0) for g in jax.tree.leaves(grads["params"]))


@pytest.mark.parametrize("rule", ["eprop", "bptt"])
def test_a_repeated_row_counts_for_nothing_in_the_eprop_objectives_loss_and_gradient(rule):
    objective = eprop_objective(rule)
    variables = objective.init(jax.random.key(0))
    real = {key: jnp.asarray(value) for key, value in eprop_batch(1).items()}
    want, want_grads = objective_gradients(objective, variables, real)
    got, got_grads = objective_gradients(objective, variables, padded_batch(real, 3))
    np.testing.assert_allclose(got, want, rtol=1e-6)  # observed 0
    jax.tree.map(lambda g, w: np.testing.assert_allclose(g, w, rtol=1e-6, atol=1e-8),  # observed 0
                 got_grads["params"], want_grads["params"])


def test_an_eprop_run_loads_back_as_its_spiking_mlp(tmp_path):
    import dew

    objective = eprop_objective()
    run = tmp_path / "run"
    data = Dataset.from_records(eprop_batch(2), batch=8, validation=eprop_batch(3), loading=LOADING)
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0), checkpoints=Checkpoints(str(run)))
    state = trainer.fit(data, steps=3, log_every=3, eval_every=3, checkpoint_every=3, metrics=[Accuracy()])
    trainer.checkpoints.wait()
    assert "val/accuracy" in trainer._display.evaluations["val"][-1].scores
    params = state.variables["params"]
    assert np.all(np.diag(params["recurrent_0"]["recurrent"]) == 0)

    spikes = eprop_batch(4)["spikes"]
    classifier = objective.pipeline(state)
    assert isinstance(classifier, SpikingClassification) and classifier.model is objective.model
    outputs = objective.model.apply({"params": params}, jnp.swapaxes(spikes.astype(np.float32), 0, 1))
    np.testing.assert_allclose(classifier.logits(spikes), jnp.mean(outputs, axis=0), rtol=1e-6)  # observed 0
    loaded = dew.pipeline(str(run), trust=("sparx",))
    np.testing.assert_array_equal(np.asarray(loaded.logits(spikes)), np.asarray(classifier.logits(spikes)))


@pytest.mark.parametrize("change", [{"recurrent": False}, {"hidden": (6, 6)}, {"delays": 2},
                                    {"batch_norm": True}, {"dropout": 0.1}, {"learn_readout_tau": True}])
def test_the_eprop_objective_refuses_parameters_eprop_does_not_train(change):
    with pytest.raises(ValueError, match="e-prop trains a SpikingMLP"):
        EPropObjective(eprop_model().clone(**change), Field("spikes", (12, 5)))


def test_the_eprop_objective_refuses_a_learned_time_constant():
    from sparx.nn import ALIF

    model = eprop_model().clone(neuron=ALIF(learn_tau=True))
    with pytest.raises(ValueError, match="e-prop trains a SpikingMLP"):
        EPropObjective(model, Field("spikes", (12, 5)))


def test_the_eprop_objective_trains_a_network_without_biases():
    objective = EPropObjective(eprop_model(use_bias=False), Field("spikes", (12, 5)))
    variables = objective.init(jax.random.key(0))
    params = variables["params"]
    assert set(params["dense_0"]) == {"kernel"} and set(params["readout"]) == {"kernel"}
    batch = {key: jnp.asarray(value) for key, value in eprop_batch(1).items()}
    _, grads = objective_gradients(objective, variables, batch)
    assert jax.tree.structure(grads["params"]) == jax.tree.structure(variables["params"])


PC_ALM = PredictiveCoding(8, 0.2, alpha=1.0)


def pc_objective(rule=PC_ALM):
    return PredictiveCodingObjective(residual_mlp(6, 4, 12, 3, "tanh"), Field("image", (3, 4)), classes=3,
                                     rule=rule)


def pc_batch(seed):
    rng = np.random.default_rng(seed)
    return {"image": rng.normal(size=(8, 3, 4)).astype(np.float32),
            "label": rng.integers(0, 3, 8).astype(np.int32)}


@pytest.mark.parametrize("rule", [PC_ALM, None])
def test_the_predictive_coding_objectives_gradient_is_its_rules(rule):
    objective = pc_objective(rule)
    variables = objective.init(jax.random.key(0))
    batch = {key: jnp.asarray(value) for key, value in pc_batch(1).items()}
    step = Step(jnp.asarray(0), jax.random.key(1), None)
    grads = jax.grad(lambda v: objective.scalar_loss(v, batch, step)[0])(variables)["params"]
    x, target = batch["image"].reshape(8, 12), jax.nn.one_hot(batch["label"], 3)
    if rule is None:  # backpropagation through the stack
        expected = jax.grad(lambda v: jnp.mean(squared_error(objective.model.apply(v, x), target)))(variables)
        expected = expected["params"]
    else:
        blocks, params = sequential_blocks(objective.stack, variables["params"])
        # The trainer differentiates the mean over the batch of 8.
        expected = {f"layers_{k}": g for k, g in enumerate(rule.gradient(blocks, params, x, target))}
        expected = jax.tree.map(lambda g: g / 8, expected)
    for got, want in zip(jax.tree.leaves(grads), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)  # observed 0
    assert all(np.any(np.asarray(g) != 0) for g in jax.tree.leaves(grads))


def test_a_repeated_row_counts_for_nothing_in_the_predictive_coding_objective():
    objective = pc_objective()
    variables = objective.init(jax.random.key(0))
    real = {key: jnp.asarray(value) for key, value in pc_batch(1).items()}
    step = Step(jnp.asarray(0), jax.random.key(1), None)

    def value_and_grad(batch):
        return jax.value_and_grad(lambda v: objective.scalar_loss(v, batch, step)[0])(variables)

    want, want_grads = value_and_grad(real)
    got, got_grads = value_and_grad(padded_batch(real, 3))
    np.testing.assert_allclose(got, want, rtol=1e-6)  # observed 0
    for a, b in zip(jax.tree.leaves(got_grads), jax.tree.leaves(want_grads), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)  # observed 0


def test_the_predictive_coding_objective_trains_through_dews_trainer():
    objective = pc_objective()
    data = Dataset.from_records(pc_batch(2), batch=8, validation=pc_batch(3), loading=LOADING)
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0))
    trainer.fit(data, steps=3, log_every=3, eval_every=3, metrics=[Accuracy()])
    assert "val/accuracy" in trainer._display.evaluations["val"][-1].scores


def test_the_predictive_coding_objective_refuses_a_model_that_is_not_a_stack():
    with pytest.raises(TypeError, match=r"nn\.Sequential"):
        PredictiveCodingObjective(LIF(), Field("image", (4,)), classes=2)



RNEURALNET = RNeuralNet.random(0, 24, 2, 2, fan=4, input_fan=2)


def rneuralnet_batch(seed, rows=8):
    rng = np.random.default_rng(seed)
    return {"cues": (rng.random((rows, 24, 2)) < 0.4).astype(np.float32) * 3,
            "label": rng.integers(0, 2, rows).astype(np.int32)}


@pytest.mark.parametrize("rule", ["first", "all", "gated", "agrel", "reinforce"])
def test_the_rneuralnet_objectives_gradient_is_its_rules(rule):
    objective = RNeuralNetObjective(RNEURALNET, Field("cues", (24, 2)), rule=rule, discount=0.9)
    variables = objective.init(jax.random.key(0))
    batch = {key: jnp.asarray(value) for key, value in rneuralnet_batch(1).items()}
    step = Step(jnp.asarray(0), jax.random.key(1), None)
    grads = jax.grad(lambda v: objective.scalar_loss(v, batch, step)[0])(variables)["params"]["weight"]
    outputs = RNEURALNET.outputs

    def network(weight):
        wiring = RNEURALNET.wiring.replace(weight=weight)
        return RNEURALNET.replace(cell=RNEURALNET.cell.replace(wiring=wiring))

    def chosen(weight):
        last = network(weight).run(batch["cues"].swapaxes(0, 1))[0][-1]
        logits = 4.0 * last[:, outputs]
        choice = jax.random.categorical(step.key, logits)
        reward = jnp.where(choice == batch["label"], 1.0, -1.0)
        return last, jax.nn.log_softmax(logits)[jnp.arange(8), choice], choice, reward

    weight = variables["params"]["weight"]
    last, _, choice, reward = chosen(weight)
    error = reward - (jnp.sum(reward) - reward) / 7  # each reward less the mean of the other seven
    # The trainer descends the mean over the batch of 8 of each rule's update.
    if rule == "reinforce":
        def surrogate(w):
            _, log_p, _, r = chosen(w)
            return -jnp.mean((r - (jnp.sum(r) - r) / 7) * log_p)

        expected = jax.grad(surrogate)(weight)
    elif rule == "agrel":
        # Each example's error times the gradient of its chosen output's last activity.
        def chosen_output(w, cues, unit):
            return network(w).run(cues[:, None])[0][-1, 0, unit]

        slope = jax.vmap(jax.grad(chosen_output), in_axes=(None, 0, 0))
        slopes = slope(weight, batch["cues"], outputs[choice])
        expected = -jnp.mean(error[:, None] * slopes, axis=0)
    else:
        paths = "first" if rule == "first" else "all"
        roots = outputs[choice] if rule == "gated" else jnp.full(8, RNEURALNET.feeder)
        signal = error if rule == "gated" else reward

        def change(a, r, root):
            return reward_diffusion(RNEURALNET.wiring, a, r, root=root, paths=paths, discount=0.9,
                                    eta=1.0).change

        expected = -jnp.mean(jax.vmap(change)(last, signal, roots), axis=0)
    np.testing.assert_allclose(grads, expected, rtol=1e-5, atol=1e-8)  # observed 0; 5.6e-9 for agrel
    assert np.any(np.asarray(grads) != 0) and 0 < np.sum(np.asarray(reward) > 0) < 8
    _, aux = objective.loss(variables, batch, step)
    np.testing.assert_allclose(aux.metrics["reward"], jnp.mean(reward), rtol=1e-6)


def test_the_rneuralnet_objective_trains_through_dews_trainer():
    objective = RNeuralNetObjective(RNEURALNET, Field("cues", (24, 2)), rule="first")
    data = Dataset.from_records(rneuralnet_batch(2), batch=8, validation=rneuralnet_batch(3, 11),
                                loading=LOADING)
    trainer = Trainer(objective, optax.sgd(0.08), key=jax.random.key(0))
    state = trainer.fit(data, steps=3, log_every=3, eval_every=3, metrics=[Accuracy()])
    assert "val/accuracy" in trainer._display.evaluations["val"][-1].scores
    assert np.any(np.asarray(state.variables["params"]["weight"]) != np.asarray(RNEURALNET.wiring.weight))


def test_the_rneuralnet_objective_refuses_a_sample_that_is_not_one_sequence_of_its_inputs():
    with pytest.raises(ValueError, match="sequence"):
        RNeuralNetObjective(RNEURALNET, Field("cues", (24, 3)))
