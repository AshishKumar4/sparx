"""Train a spiking network on the Spiking Heidelberg Digits with dew's Trainer.

    pip install -e ".[datasets]"
    python examples/train_shd.py --steps 3000
    python examples/train_shd.py --recipe snn-delays --epochs 150

Downloads SHD (169 MB) into ~/.cache/sparx on first use.

The default recipe, `alif`, bins SHD into 100 steps of 14 ms over 700
channels and trains a layer of adaptive LIF neurons with a leaky integrator
readout scored by its maximum over time. The test split serves as the
validation set (SHD has no separate one) and the script prints the accuracy
over all 2264 test recordings at the end. `--recurrent` feeds the hidden
layer's spikes back to itself. Backpropagation through that loop explodes
with the heavy-tailed ATan surrogate once the recurrent weights grow
(gradient norms past 1e8 within 300 steps), so pair it with
`--surrogate superspike`, whose derivative falls off steeply. `--delays K`
replaces the input layer with `sparx.nn.DelayedDense`, which learns a delay
of 0 to K steps for every synapse. Its Gaussian width falls from K / 2 to 0.5
over training, and the final test accuracy is measured with every delay
rounded to a whole step (`sigma=0`), the network as deployed. `--channels`
pools adjacent input channels (700 must be a multiple).

`--recipe snn-delays` is Hammouamri et al.'s SHD recipe ("Learning Delays in
Spiking Neural Networks using Dilated Convolutions with Learnable Spacings",
ICLR 2024; their `best_config_SHD.py`), which reaches about 95%:

- SHD binned as their SpikingJelly bins it, 10 ms steps opened at events
  (`sparx.datasets.Binning`), with the 700 channels summed in fives to 140;
- two hidden layers of 256 LIF neurons, every synapse (the readout's too) a
  `DelayedDense` with delays of 0 to 24 steps (their 25-step kernel), each
  input extended by 12 steps of zeros as they pad it on the right, batch
  norm before each hidden neuron, no bias, Kaiming-uniform weights;
- SpikingJelly's LIF with `tau` 1.005 steps and `decay_input=False` (a decay
  of `1 - 1 / tau` a step), hard reset to 0, threshold 1, ATan(5) and
  `detach_reset`; dropout 0.4 with one mask a recording;
- a non-spiking LIF readout of the same `tau`, scored by the softmax of every
  step summed over time (`readout="softmax_sum"`, their `loss='sum'`);
- Adam with an L2 weight decay of 1e-5 on the weights, torch's one-cycle
  schedule to 5e-3 and its momentum cycle; Adam on the delays at 0.1 on a
  cosine, clamped to the kernel; batch norm on the weights' schedule without
  decay; all stepped once an epoch, as their schedulers are;
- the Gaussian width shrinks exponentially from 12.27 to 0.5 steps over the
  first quarter of the epochs (`decrease_sig`), and every evaluation rounds
  the delays (`sigma=0`), as their `eval_model` does;
- batches of 256 for 150 epochs, seed 0.

Their script validates on the test set and reports the best test accuracy
over epochs. This one holds out `--validation` of the training set (10% by
default, 0 to train on all of it as they do), scores the validation and test
sets after every epoch, and reports the test accuracy at the epoch of best
validation accuracy beside their number, each labelled.

What still differs from their code:

- Their batches are padded to the batch's longest recording (about 105 steps
  on average, at most 124); here every recording is padded to 124 steps, so
  the summed softmax and the batch statistics also see the extra silent
  steps.
- dew drops the last partial batch of an epoch (31 steps of 256 out of
  8156, where they take 32 with a last batch of 220), and with a 10%
  holdout trains on 7340 recordings in 28 steps an epoch.
- flax's BatchNorm keeps the biased variance in its running average where
  torch keeps the unbiased one, a factor `n / (n - 1)` for `n` = steps x
  batch, about 1 + 3e-5 here.
- The LIF decay is `exp(-1 / tau')` with `tau'` chosen to equal their
  `1 - 1 / 1.005`, so it matches to float32 rounding.
- Their test accuracy is the mean of per-batch accuracies (the last batch of
  216 weighs as much as the others); here it is the accuracy over all 2264
  recordings.
- Random streams differ: initialization, shuffling and dropout masks.
"""

import argparse
import math
import time
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset
from dew.training.optim import Cosine, Linear

import sparx
from sparx.datasets import shd
from sparx.dew import (
    ExponentialDecay,
    GroupAdam,
    OneCycle,
    RateBand,
    SpikingClassifier,
    accuracy,
    evaluation_pass,
    holdout,
)
from sparx.encode import Events
from sparx.models import SpikingMLP

STEPS_MS = 10
STEPS = 124  # the longest recording of either split at 10 ms steps opened at events
MAX_DELAY = 24  # 250 ms // 10 ms = 25 taps, odd already, so delays of 0 to 24 steps
TAU_SPIKINGJELLY = (10.05 + 1e-9) / STEPS_MS  # their init_tau, in steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", choices=["alif", "snn-delays"], default="alif")
    parser.add_argument("--steps", type=int, default=3000, help="alif: training steps")
    parser.add_argument("--epochs", type=int, default=150, help="snn-delays: epochs the schedules run over")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="snn-delays: stop after this many epochs of the schedules, a short check")
    parser.add_argument("--validation", type=float, default=0.1,
                        help="snn-delays: fraction of the training set held out to select epochs on")
    parser.add_argument("--batch", type=int, default=None, help="64 for alif, 256 for snn-delays")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--tau", type=float, default=5.0, help="alif: membrane time constant, in 14 ms steps")
    parser.add_argument("--tau-adapt", type=float, default=20.0, help="alif: adaptation time constant")
    parser.add_argument("--dropout", type=float, default=0.1, help="alif: dropout on hidden spikes")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="alif: peak learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="alif: global gradient norm limit")
    parser.add_argument("--recurrent", action="store_true")
    parser.add_argument("--surrogate", choices=["atan", "superspike"], default="atan")
    parser.add_argument("--delays", type=int, default=0, help="alif: largest learnable delay, in steps")
    parser.add_argument("--channels", type=int, default=700, help="alif: input channels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", default=None, help="the run's directory; it resumes from a checkpoint there")
    args = parser.parse_args()
    if args.recipe == "snn-delays":
        if args.epochs < 4:
            parser.error("the one-cycle schedule and the width decay need at least 4 epochs")
        snn_delays(args)
    else:
        alif(args)


def alif(args: argparse.Namespace) -> None:
    batch = args.batch or 64
    train, test = shd("train", channels=args.channels), shd("test", channels=args.channels)
    data = Dataset.from_records(train, batch=batch, validation=test)
    surrogate = sparx.surrogate.ATan() if args.surrogate == "atan" else sparx.surrogate.FastSigmoid(100.0)
    neuron = sparx.nn.ALIF(tau=args.tau, tau_adapt=args.tau_adapt, beta=0.2, learn_tau=True,
                           detach_reset=True, surrogate=surrogate)
    net = SpikingMLP(hidden=(args.hidden,), classes=20, neuron=neuron, recurrent=args.recurrent,
                     delays=args.delays, dropout=args.dropout, readout_tau=args.tau, learn_readout_tau=True)
    width = {"sigma": Linear(peak=args.delays / 2, end=0.5)} if args.delays else None
    objective = SpikingClassifier(net, Field("spikes", train["spikes"].shape[1:]), Events(), readout="max",
                                  rates=RateBand(lower=0.01, upper=0.3, weight=1.0),
                                  schedules=width, schedule_steps=args.steps)
    schedule = optax.cosine_decay_schedule(args.learning_rate, args.steps)
    optimizer = optax.chain(optax.clip_by_global_norm(args.clip), optax.adamw(schedule, weight_decay=1e-4))
    trainer = Trainer(objective, optimizer, key=jax.random.key(args.seed),
                      checkpoints=Checkpoints(args.run or "runs/shd"))
    start = time.perf_counter()
    state = trainer.fit(data, steps=args.steps, log_every=100, eval_every=500, metrics=[accuracy])
    trained = time.perf_counter() - start

    @jax.jit
    def predict(variables, spikes):
        outputs = net.apply(variables, jnp.moveaxis(spikes, 1, 0).astype(jnp.float32))
        return jnp.argmax(jnp.max(outputs, axis=0), -1)

    predictions = np.concatenate([predict(state.variables, test["spikes"][i:i + 256])
                                  for i in range(0, len(test["label"]), 256)])
    print(f"test accuracy {np.mean(predictions == test['label']):.4f} over {len(predictions)} recordings "
          f"after {args.steps} steps ({trained:.0f} s on {jax.devices()[0].device_kind})")


class History:
    """A dew tracker that keeps every logged scalar by step, and when it arrived."""

    def __init__(self):
        self.scalars: dict[int, dict[str, float]] = {}
        self.arrived: dict[int, float] = {}

    def log(self, scalars: Mapping[str, float], step: int) -> None:
        self.scalars.setdefault(step, {}).update({name: float(value) for name, value in scalars.items()})
        self.arrived[step] = time.perf_counter()

    def artifact(self, value: object, step: int) -> None:
        pass

    def close(self) -> None:
        pass


def snn_delays(args: argparse.Namespace) -> None:
    batch = args.batch or 256
    binned = {"steps": STEPS, "max_time": STEPS * STEPS_MS / 1000, "channels": 140, "binning": "events"}
    train, test = shd("train", **binned), shd("test", **binned)
    splits = {}
    if args.validation:
        train, val = holdout(train, args.validation, seed=args.seed)
        splits["val"] = evaluation_pass(val, batch)
    splits["test"] = evaluation_pass(test, batch)
    data = Dataset.from_records(train, batch=batch, seed=args.seed)
    per_epoch = data.steps_per_epoch
    assert per_epoch is not None  # records held in memory have a count
    steps = args.epochs * per_epoch

    # SpikingJelly's decay_input=False LIF keeps 1 - 1 / tau of its membrane a step, sparx exp(-1 / tau).
    tau = -1 / math.log(1 - 1 / TAU_SPIKINGJELLY)
    neuron = sparx.nn.LIF(tau=tau, threshold=1.0, reset="zero", surrogate=sparx.surrogate.ATan(5.0),
                          detach_reset=True)
    net = SpikingMLP(hidden=(256, 256), classes=20, neuron=neuron, delays=(MAX_DELAY,) * 3, extend=True,
                     batch_norm=True, use_bias=False, weight_init="kaiming_uniform", dropout=0.4,
                     dropout_mask="sequence", readout_tau=tau)
    # torch's OneCycleLR(max_lr=5e-3) starts at max_lr / 25 and ends 1e4 times lower, cycling Adam's
    # beta1 between 0.95 and 0.85.
    rate = OneCycle(peak=5e-3, start=5e-3 / 25, end=5e-3 / 25 / 1e4)
    momentum = OneCycle(peak=0.85, start=0.95, end=0.95)
    groups = {
        "delays": GroupAdam(("*/delay",), Cosine(peak=0.1, warmup_steps=0), bounds=(0.0, float(MAX_DELAY))),
        "weights": GroupAdam(("*/kernel",), rate, b1=momentum, weight_decay=1e-5),
        "norms": GroupAdam(("*",), rate, b1=momentum),
    }
    # DCLS's raw width falls from 25 // 2 to 0.23 over the first quarter; its effective width adds 0.27.
    width = ExponentialDecay(start=float((MAX_DELAY + 1) // 2), end=0.23, decay_steps=args.epochs // 4,
                        offset=0.27)
    objective = SpikingClassifier(net, Field("spikes", (STEPS, 140)), Events(), readout="softmax_sum",
                                  schedules={"sigma": width}, schedule_steps=steps, schedule_every=per_epoch,
                                  deployed={"sigma": 0}, groups=groups)
    history = History()
    # Every parameter belongs to a group, so the trainer's own optimizer updates nothing.
    trainer = Trainer(objective, optax.set_to_zero(), key=jax.random.key(args.seed),
                      checkpoints=Checkpoints(args.run or "runs/shd-snn-delays"), tracker=history)
    print(f"snn-delays: {len(train['label'])} training recordings in {per_epoch} steps of {batch} an epoch, "
          f"{args.epochs} epochs; scoring {', '.join(splits)} after each")
    started = time.perf_counter()
    trained = (args.stop_after or args.epochs) * per_epoch
    trainer.fit(data, steps=trained, log_every=per_epoch, eval_every=per_epoch,
                checkpoint_every=5 * per_epoch, metrics=[accuracy], validation=splits)
    report(history, per_epoch, started)


def report(history: History, per_epoch: int, started: float) -> None:
    """Each epoch's scores, the test accuracy at the best validation epoch, and SNN-delays' number."""
    evaluated = sorted(step for step, scalars in history.scalars.items() if "test/accuracy" in scalars)
    rows, previous = [], started
    print(f"{'epoch':>5} {'val acc':>8} {'test acc':>8} {'seconds':>8}")
    for step in evaluated:
        scalars = history.scalars[step]
        seconds = history.arrived[step] - previous
        previous = history.arrived[step]
        rows.append((step // per_epoch, scalars.get("val/accuracy", math.nan), scalars["test/accuracy"],
                     seconds))
        print(f"{rows[-1][0]:>5} {rows[-1][1]:>8.4f} {rows[-1][2]:>8.4f} {seconds:>8.1f}")
    if not rows:
        return
    epochs, val, test, seconds = (np.asarray(column) for column in zip(*rows, strict=True))
    if not np.isnan(val).all():
        chosen = int(np.nanargmax(val))
        print(f"selected on validation: epoch {epochs[chosen]}, validation accuracy {val[chosen]:.4f}, "
              f"test accuracy {test[chosen]:.4f}")
    best = int(np.argmax(test))
    print(f"best test accuracy over epochs (SNN-delays' reported number, chosen on the test set): "
          f"{test[best]:.4f} at epoch {epochs[best]}")
    device = jax.devices()[0].device_kind
    print(f"{np.median(seconds):.1f} s an epoch (median, training and evaluation) on {device}")


if __name__ == "__main__":
    main()
