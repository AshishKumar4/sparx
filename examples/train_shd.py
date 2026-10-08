"""Train a spiking network on the Spiking Heidelberg Digits with dew's Trainer.

    pip install -e ".[datasets]"
    python examples/train_shd.py --steps 3000
    python examples/train_shd.py --recipe snn-delays --epochs 150
    JAX_PLATFORMS=cpu python examples/train_shd.py --smoke --out /tmp/shd-smoke
    JAX_PLATFORMS=cpu python examples/train_shd.py --recipe snn-delays --smoke --out /tmp/shd-smoke

Downloads SHD (169 MB) into ~/.cache/sparx on first use. `--smoke` trains a
small network for a few steps on synthetic recordings in SHD's layout
(`sparx.datasets.write_synthetic_shd`) instead, and downloads nothing.

The default recipe, `alif`, bins SHD into 100 steps of 14 ms over 700
channels and trains a layer of adaptive LIF neurons with a leaky integrator
readout scored by its maximum over time. The test split serves as the
validation set (SHD has no separate one), and at the end the script loads
the trained classifier (`objective.pipeline(state)`) and prints its accuracy
over all 2264 test recordings. `--recurrent` feeds the hidden layer's spikes
back to itself. Backpropagation through that loop explodes with the
heavy-tailed ATan surrogate once the recurrent weights grow (gradient norms
past 1e8 within 300 steps), so pair it with `--surrogate superspike`, whose
derivative falls off steeply. `--delays K` replaces the input layer with
`sparx.nn.DelayedDense`, which learns a delay of 0 to K steps for every
synapse. Its Gaussian width falls from K / 2 to 0.5 over training, and
evaluation and the trained classifier round every delay to a whole step
(`sigma=0`), the network as deployed. `--channels` pools adjacent input
channels (700 must be a multiple).

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
  decay; all stepped once an epoch, as their schedulers are (dew's
  `OptimConfig` with a `ParamGroup` each);
- the Gaussian width shrinks exponentially from 12.27 to 0.5 steps over the
  first quarter of the epochs (`decrease_sig`), and every evaluation rounds
  the delays (`sigma=0`), as their `eval_model` does;
- batches of 256 for 150 epochs, seed 0.

Their script validates on the test set and reports the best test accuracy
over epochs. This one holds out `--validation` of the training set (10% by
default, 0 to train on all of it as they do) and scores the validation and
test sets after every epoch. dew's `Best` keeps the checkpoint of best
validation accuracy, whose recorded test accuracy the script reports beside
their number, each labelled. The per-epoch table is read back from the run's
tracking journal (dew's `LocalTracker`).

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

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import jax
import optax
import tyro
from dew import Best, Checkpoints, Field, LocalTracker, Trainer
from dew.config import OptimConfig
from dew.data import Dataset, Loading
from dew.training import Evaluation
from dew.training.optim import Cosine, Exponential, Linear, OneCycle, ParamGroup

import sparx
from sparx.datasets import holdout, shd, write_synthetic_shd
from sparx.encode import EventsEncoder
from sparx.metrics import Accuracy
from sparx.models import SpikingMLP
from sparx.objectives import RateBand, SpikingClassifierObjective

STEPS_MS = 10
STEPS = 124  # the longest recording of either split at 10 ms steps opened at events
MAX_DELAY = 24  # 250 ms // 10 ms = 25 taps, odd already, so delays of 0 to 24 steps
TAU_SPIKINGJELLY = (10.05 + 1e-9) / STEPS_MS  # their init_tau, in steps


@dataclass
class Config:
    recipe: Literal["alif", "snn-delays"] = "alif"
    steps: int = 3000
    """alif: training steps."""
    epochs: int = 150
    """snn-delays: epochs the schedules run over."""
    stop_after: int | None = None
    """snn-delays: stop after this many epochs of the schedules, a short check."""
    validation: float = 0.1
    """snn-delays: fraction of the training set held out to select epochs on."""
    batch: int | None = None
    """64 for alif, 256 for snn-delays."""
    hidden: int = 256
    """Neurons in each hidden layer."""
    tau: float = 5.0
    """alif: membrane time constant, in 14 ms steps."""
    tau_adapt: float = 20.0
    """alif: adaptation time constant."""
    dropout: float = 0.1
    """alif: dropout on hidden spikes."""
    learning_rate: float = 2e-3
    """alif: peak learning rate."""
    clip: float = 1.0
    """alif: global gradient norm limit."""
    recurrent: bool = False
    surrogate: Literal["atan", "superspike"] = "atan"
    delays: int = 0
    """alif: largest learnable delay, in steps."""
    channels: int = 700
    """alif: input channels."""
    seed: int = 0
    out: Path | None = None
    """The run's directory, runs/shd or runs/shd-snn-delays by default; a run there resumes."""
    cache: Path | None = None
    """Where SHD is read from and downloaded to, ~/.cache/sparx by default."""
    smoke: bool = False
    """Train a small network for a few steps on synthetic recordings; nothing is downloaded."""


def main(config: Config) -> None:
    if config.recipe == "snn-delays" and config.epochs < 4:
        raise ValueError("the one-cycle schedule and the width decay need at least 4 epochs")
    if config.smoke:
        out = config.out or Path("runs/shd-smoke")
        config = replace(config, out=out, cache=write_synthetic_shd(out / "synthetic-shd"), steps=8,
                         epochs=4, batch=16, hidden=16, channels=70)
    if config.recipe == "snn-delays":
        snn_delays(config)
    else:
        alif(config)


def loading(config: Config) -> Loading:
    """Few workers for a smoke run, which reads a few records; dew's default otherwise."""
    return Loading(workers=0, threads=1, read_buffer=1) if config.smoke else Loading()


def alif(config: Config) -> None:
    batch = config.batch or 64
    train = shd("train", channels=config.channels, cache=config.cache)
    test = shd("test", channels=config.channels, cache=config.cache)
    data = Dataset.from_records(train, batch=batch, seed=config.seed, validation=test,
                                loading=loading(config))
    surrogate = sparx.surrogate.ATan() if config.surrogate == "atan" else sparx.surrogate.FastSigmoid(100.0)
    neuron = sparx.nn.ALIF(tau=config.tau, tau_adapt=config.tau_adapt, beta=0.2, learn_tau=True,
                           detach_reset=True, surrogate=surrogate)
    net = SpikingMLP(hidden=(config.hidden,), classes=20, neuron=neuron, recurrent=config.recurrent,
                     delays=config.delays, dropout=config.dropout, readout_tau=config.tau,
                     learn_readout_tau=True)
    width = {"sigma": Linear(peak=config.delays / 2, end=0.5)} if config.delays else None
    objective = SpikingClassifierObjective(
        net, Field("spikes", train["spikes"].shape[1:]), EventsEncoder(), readout="max",
        rates=RateBand(lower=0.01, upper=0.3, weight=1.0), schedules=width, schedule_steps=config.steps,
        deployed={"sigma": 0} if config.delays else None)
    schedule = optax.cosine_decay_schedule(config.learning_rate, config.steps)
    optimizer = optax.chain(optax.clip_by_global_norm(config.clip), optax.adamw(schedule, weight_decay=1e-4))
    trainer = Trainer(objective, optimizer, key=jax.random.key(config.seed),
                      checkpoints=Checkpoints(str(config.out or "runs/shd")))
    evaluations = config.steps // 2 if config.smoke else 500
    state = trainer.fit(data, steps=config.steps, log_every=min(100, config.steps), eval_every=evaluations,
                        metrics=[Accuracy()], validation={"test": data.val})
    final = Evaluation.run(objective, state.variables, data.val, key=config.seed, metrics=[Accuracy()],
                           step=config.steps, split="test")
    print(f"test accuracy {final.scores['test/accuracy']:.4f} over {final.records} recordings "
          f"after {config.steps} steps on {jax.devices()[0].device_kind}")


def snn_delays(config: Config) -> None:
    batch = config.batch or 256
    binned = {"steps": STEPS, "max_time": STEPS * STEPS_MS / 1000, "channels": 140, "binning": "events"}
    train, test = shd("train", cache=config.cache, **binned), shd("test", cache=config.cache, **binned)
    if config.validation:
        train, val = holdout(train, config.validation, seed=config.seed)
    data = Dataset.from_records(train, batch=batch, seed=config.seed, validation=test,
                                loading=loading(config))
    splits = {"test": data.val}
    if config.validation:
        # dew reads a validation split beside the records it trains on; this pair's reader scores the
        # holdout.
        held = Dataset.from_records(train, batch=batch, seed=config.seed, validation=val,
                                    loading=loading(config))
        splits = {"val": held.val, **splits}
    per_epoch = data.steps_per_epoch
    assert per_epoch is not None  # records held in memory have a count
    steps = config.epochs * per_epoch

    # SpikingJelly's decay_input=False LIF keeps 1 - 1 / tau of its membrane a step, sparx exp(-1 / tau).
    tau = -1 / math.log(1 - 1 / TAU_SPIKINGJELLY)
    neuron = sparx.nn.LIF(tau=tau, threshold=1.0, reset="zero", surrogate=sparx.surrogate.ATan(5.0),
                          detach_reset=True)
    net = SpikingMLP(hidden=(config.hidden, config.hidden), classes=20, neuron=neuron,
                     delays=(MAX_DELAY,) * 3, extend=True, batch_norm=True, use_bias=False,
                     weight_init="kaiming_uniform", dropout=0.4, dropout_mask="sequence", readout_tau=tau)
    # torch's OneCycleLR(max_lr=5e-3) starts at max_lr / 25 and ends 1e4 times lower, cycling Adam's
    # beta1 between 0.95 and 0.85. Every schedule steps once an epoch.
    rate = OneCycle(peak=5e-3, every=per_epoch)
    momentum = OneCycle(peak=0.85, init=0.95, end=0.95, every=per_epoch)
    groups = (
        ParamGroup("delays", ("*/delay",), schedule=Cosine(peak=0.1, warmup_steps=0, every=per_epoch),
                   bounds=(0.0, float(MAX_DELAY))),
        ParamGroup("weights", ("*/kernel",), schedule=rate, b1=momentum, weight_decay=1e-5),
        ParamGroup("norms", ("*",), schedule=rate, b1=momentum),
    )
    # Built over the whole schedule, so --stop-after ends the run partway through it.
    optimizer = OptimConfig(optimizer="adam", param_groups=groups).build(steps)
    # DCLS's raw width falls from 25 // 2 to 0.23 over the first quarter; its effective width adds 0.27.
    width = Exponential(init=float((MAX_DELAY + 1) // 2), end=0.23, decay_steps=config.epochs // 4,
                        offset=0.27, every=per_epoch)
    objective = SpikingClassifierObjective(
        net, Field("spikes", (STEPS, 140)), EventsEncoder(), readout="softmax_sum",
        schedules={"sigma": width}, schedule_steps=steps, deployed={"sigma": 0})
    run = config.out or Path("runs/shd-snn-delays")
    journal = LocalTracker(run / "tracking")
    checkpoints = Checkpoints(str(run))
    trainer = Trainer(objective, optimizer, key=jax.random.key(config.seed), checkpoints=checkpoints,
                      tracker=journal)
    print(f"snn-delays: {len(train['label'])} training recordings in {per_epoch} steps of {batch} an epoch, "
          f"{config.epochs} epochs; scoring {', '.join(splits)} after each")
    trained = (config.stop_after or config.epochs) * per_epoch
    accuracy = Accuracy()
    trainer.fit(data, steps=trained, log_every=per_epoch, eval_every=per_epoch,
                checkpoint_every=5 * per_epoch, metrics=[accuracy], validation=splits,
                best=Best(accuracy, split="val") if "val" in splits else None)
    checkpoints.wait()
    report(journal.directory / "scalars.jsonl", checkpoints, per_epoch)


def report(journal: Path, checkpoints: Checkpoints, per_epoch: int) -> None:
    """Each epoch's scores, the test accuracy of the best validation checkpoint, and SNN-delays' number."""
    # Each split's scores arrive as a row of their own; an epoch is every row of its step.
    epochs: dict[int, dict[str, float]] = {}
    arrived: dict[int, float] = {}
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    for row in rows:
        epochs.setdefault(row["step"], {}).update(row["scalars"])
        arrived[row["step"]] = row["time"]
    evaluated = sorted(step for step, scalars in epochs.items() if "test/accuracy" in scalars)
    if not evaluated:
        return
    print(f"{'epoch':>5} {'val acc':>8} {'test acc':>8} {'seconds':>8}")
    previous = rows[0]["time"]
    for step in evaluated:
        scalars = epochs[step]
        print(f"{step // per_epoch:>5} {scalars.get('val/accuracy', math.nan):>8.4f} "
              f"{scalars['test/accuracy']:>8.4f} {arrived[step] - previous:>8.1f}")
        previous = arrived[step]
    best = checkpoints.best
    if best is not None:
        kept = {checkpoint.step: checkpoint.metrics for checkpoint in checkpoints.kept()}
        print(f"selected on validation: epoch {best // per_epoch}, validation accuracy "
              f"{kept[best]['val/accuracy']:.4f}, test accuracy {kept[best]['test/accuracy']:.4f}")
    top = max(evaluated, key=lambda step: epochs[step]["test/accuracy"])
    print(f"best test accuracy over epochs (SNN-delays' reported number, chosen on the test set): "
          f"{epochs[top]['test/accuracy']:.4f} at epoch {top // per_epoch}")


if __name__ == "__main__":
    main(tyro.cli(Config))
