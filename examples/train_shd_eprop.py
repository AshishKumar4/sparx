"""Train a recurrent spiking network on the Spiking Heidelberg Digits with e-prop, through dew's Trainer.

    python examples/train_shd_eprop.py --rule eprop --epochs 5
    python examples/train_shd_eprop.py --rule random --epochs 5      # random feedback weights
    python examples/train_shd_eprop.py --rule bptt --epochs 5        # the same network by BPTT
    JAX_PLATFORMS=cpu python examples/train_shd_eprop.py --smoke --out /tmp/eprop-smoke

A layer of recurrent adaptive LIF neurons (Bellec et al. 2020) reads SHD,
binned into 100 steps of 14 ms with adjacent channels pooled to 140, and a
leaky readout scores the 20 classes. The network is a
`sparx.models.SpikingMLP`, the model `examples/train_shd.py` trains by
backpropagation. e-prop (`sparx.learn.eprop`) computes
the gradients as the recording runs: each synapse keeps an eligibility
trace, and each step's cross entropy weights it through the readout
(`--rule eprop`) or through fixed random weights (`--rule random`). Its
memory does not grow with the recording's length. `--rule bptt` trains the
same network by backpropagation through time, for comparison. The class
is the argmax of the readout averaged over time.

`sparx.objectives.EPropObjective` is the dew objective. Its loss hands
e-prop's gradient to dew's trainer as the loss's own
(`Objective.with_gradients`); the trainer, its optimizer, checkpoints and
evaluation are dew's as for any other objective. The test set is scored
after each epoch, all 2264 recordings. The run loads back as the trained
`SpikingMLP` (`dew.pipeline(out, trust=("sparx",))`), which streams and
serves as any other. `--smoke` trains a small layer for a few steps on synthetic
recordings in SHD's layout and downloads nothing.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import jax
import optax
import tyro
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset, Loading

from sparx.datasets import shd, write_synthetic_shd
from sparx.metrics import Accuracy
from sparx.models import SpikingMLP
from sparx.nn import ALIF
from sparx.objectives import EPropObjective
from sparx.surrogate import Triangle

DT = 14.0
"""One step of the binned recordings, in ms."""
TAU_READOUT = 20.0
"""The readout's time constant, in ms."""


@dataclass
class Config:
    rule: Literal["eprop", "random", "bptt"] = "eprop"
    epochs: int = 5
    hidden: int = 128
    channels: int = 140
    batch: int = 64
    learning_rate: float = 2e-3
    seed: int = 0
    out: Path = Path("runs/shd-eprop")
    """The run's directory; a run there resumes."""
    cache: Path | None = None
    """Where SHD is read from and downloaded to, ~/.cache/sparx by default."""
    smoke: bool = False
    """Train a small layer for a few steps on synthetic recordings; nothing is downloaded."""


def main(config: Config) -> None:
    loading = Loading()
    if config.smoke:
        config = replace(config, epochs=1, hidden=16, batch=16,
                         cache=write_synthetic_shd(config.out / "synthetic-shd"))
        loading = Loading(workers=0, threads=1, read_buffer=1)
    train = shd("train", channels=config.channels, cache=config.cache)
    test = shd("test", channels=config.channels, cache=config.cache)
    data = Dataset.from_records(train, batch=config.batch, seed=config.seed, validation=test, loading=loading)
    per_epoch = data.steps_per_epoch
    assert per_epoch is not None  # records held in memory have a count
    # Time in ms, in steps of 14 ms: membrane 20 ms, adaptation 200 ms and
    # readout 20 ms, as Bellec et al. take them for speech (TIMIT), and two
    # steps of refractoriness.
    neuron = ALIF(tau=20.0, tau_adapt=200.0, beta=0.2, detach_reset=True, surrogate=Triangle(scale=0.3),
                  refractory=2 * DT, dt=DT)
    model = SpikingMLP(hidden=(config.hidden,), classes=20, neuron=neuron, recurrent=True,
                       readout_tau=TAU_READOUT)
    objective = EPropObjective(model, Field("spikes", train["spikes"].shape[1:]), rule=config.rule)
    trainer = Trainer(objective, optax.adam(config.learning_rate), key=jax.random.key(config.seed),
                      checkpoints=Checkpoints(str(config.out)))
    trainer.fit(data, steps=config.epochs * per_epoch, log_every=per_epoch, eval_every=per_epoch,
                metrics=[Accuracy()], validation={"test": data.val})


if __name__ == "__main__":
    main(tyro.cli(Config))
