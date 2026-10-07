"""Train a deep residual MLP with PC-ALM's local weight updates, through dew's Trainer.

    python examples/train_pcalm.py                  # their headline Fashion-MNIST cell, by PC-ALM
    python examples/train_pcalm.py --method pc      # predictive coding at the same budget
    python examples/train_pcalm.py --method bp      # backpropagation, their baseline
    JAX_PLATFORMS=cpu python examples/train_pcalm.py --smoke --out /tmp/pcalm-smoke

Seely and Gould's (2026) headline cell (arXiv 2605.31022, the README of
github.com/SakanaAI/pc-alm): a ReLU residual MLP of width 32 and depth 32
at their mean-field scales (`sparx.learn.residual_mlp`) on Fashion-MNIST,
images scaled to [0, 1] and standardized by the dataset's mean and
deviation, one epoch in batches of 64, Adam at `1e-3 * sqrt(width / depth)`
on half the squared error to one-hot targets. PC and PC-ALM relax the
hidden activity for `budget = 2 * depth` steps of `state_lr`, their
`eta_best_by_cell.csv` value for this cell, with `rho = 1`; PC-ALM's
multipliers step at `alpha = 1`. Their CPU reference run reports a test
accuracy of 78.66 % for BP, 68.13 % for PC and 77.75 % for PC-ALM.

`sparx.objectives.PredictiveCodingObjective` holds the network: the loss it
reports is the forward pass's, and PC's or PC-ALM's weight update goes to
dew's trainer as that loss's gradient, so the trainer's Adam, checkpoints
and evaluation are dew's as for any objective. The weights start from dew's
draw of the same distribution as theirs, so a run matches theirs in
distribution, not draw for draw. `--smoke` trains a small network on random
images for a few steps and downloads nothing.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import jax
import numpy as np
import optax
import tyro
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset, Loading

from sparx.datasets import mnist
from sparx.learn import PredictiveCoding, residual_mlp
from sparx.metrics import Accuracy
from sparx.objectives import PredictiveCodingObjective

STATISTICS = {False: (0.1307, 0.3081), True: (0.2860, 0.3530)}
"""Each dataset's pixel mean and deviation on [0, 1], MNIST's and Fashion-MNIST's, as theirs standardizes."""


@dataclass
class Config:
    method: Literal["pcalm", "pc", "bp"] = "pcalm"
    fashion: bool = True
    """Fashion-MNIST, their headline; MNIST otherwise."""
    width: int = 32
    depth: int = 32
    activation: Literal["linear", "tanh", "relu"] = "relu"
    budget: int | None = None
    """Activity steps per update; `2 * depth` by default, their `T = 2L`."""
    state_lr: float = 0.23588549900873967
    alpha: float = 1.0
    rho: float = 1.0
    epochs: int = 1
    batch: int = 64
    learning_rate: float | None = None
    """Adam's rate; `1e-3 * sqrt(width / depth)` by default, theirs."""
    seed: int = 0
    out: Path = Path("runs/pcalm")
    """The run's directory; a run there resumes."""
    smoke: bool = False
    """Train a small network on random images for a few steps; nothing is downloaded."""


def standardized(records: dict[str, np.ndarray], fashion: bool) -> dict[str, np.ndarray]:
    mean, deviation = STATISTICS[fashion]
    images = (records["image"].astype(np.float32) / 255.0 - mean) / deviation
    return {"image": images.reshape(len(images), -1), "label": records["label"]}


def main(config: Config) -> None:
    if config.smoke:
        config = replace(config, width=8, depth=4, batch=32)
        rng = np.random.default_rng(config.seed)
        train, test = ({"image": rng.integers(0, 256, (n, 28, 28), dtype=np.uint8),
                        "label": rng.integers(0, 10, n).astype(np.int32)} for n in (256, 64))
        loading = Loading(workers=0, threads=1, read_buffer=1)
    else:
        train, test = mnist("train", fashion=config.fashion), mnist("test", fashion=config.fashion)
        loading = Loading()
    data = Dataset.from_records(standardized(train, config.fashion), batch=config.batch, seed=config.seed,
                                validation=standardized(test, config.fashion), loading=loading)
    per_epoch = data.steps_per_epoch
    assert per_epoch is not None  # records held in memory have a count
    budget = config.budget if config.budget is not None else 2 * config.depth
    alpha = config.alpha if config.method == "pcalm" else 0.0
    rule = None if config.method == "bp" else PredictiveCoding(budget, config.state_lr, config.rho, alpha)
    model = residual_mlp(config.width, config.depth, 28 * 28, 10, config.activation)
    objective = PredictiveCodingObjective(model, Field("image", (28 * 28,)), classes=10, rule=rule)
    rate = config.learning_rate
    if rate is None:
        rate = 1e-3 * (config.width / config.depth) ** 0.5
    trainer = Trainer(objective, optax.adam(rate), key=jax.random.key(config.seed),
                      checkpoints=Checkpoints(str(config.out)))
    trainer.fit(data, steps=config.epochs * per_epoch, log_every=max(1, per_epoch // 10),
                eval_every=per_epoch, metrics=[Accuracy()], validation={"test": data.val})


if __name__ == "__main__":
    main(tyro.cli(Config))
