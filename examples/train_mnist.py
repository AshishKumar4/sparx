"""Train a spiking MNIST classifier with dew's Trainer.

    python examples/train_mnist.py --epochs 2
    JAX_PLATFORMS=cpu python examples/train_mnist.py --smoke --out /tmp/mnist-smoke

Downloads MNIST (11 MB) into ~/.cache/sparx on first use. The images are
encoded as Bernoulli spike trains, a fresh draw every step, and two dense
layers of LIF neurons with a leaky integrator readout (`SpikingMLP`) are
trained on the cross entropy of its time-averaged membrane
(`SpikingClassifierObjective`, readout `mean`). The test set is scored
after every epoch, every one of its 10,000 images, and at the end the
trained classifier (`objective.pipeline(state)`) predicts a few test
images. `dew.pipeline("runs/mnist", trust=("sparx",))` loads the same
classifier in another process. `--smoke` trains on 256 random images
for a few steps instead, and downloads nothing.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import jax
import numpy as np
import optax
import tyro
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset, Loading

import sparx
from sparx.datasets import mnist
from sparx.metrics import Accuracy
from sparx.models import SpikingMLP
from sparx.objectives import SpikingClassifierObjective


@dataclass
class Config:
    epochs: int = 2
    steps: int = 8
    """Time steps per image."""
    batch: int = 128
    learning_rate: float = 1e-3
    hidden: int = 512
    out: Path = Path("runs/mnist")
    """The run's directory; a run there resumes."""
    smoke: bool = False
    """Train on 256 random images for a few steps; nothing is downloaded."""


def random_images(count: int, seed: int) -> dict[str, np.ndarray]:
    """`count` random uint8 images and labels in MNIST's layout, for a smoke run."""
    rng = np.random.default_rng(seed)
    return {"image": rng.integers(0, 256, (count, 28, 28), dtype=np.uint8),
            "label": rng.integers(0, 10, count).astype(np.int32)}


def main(config: Config) -> None:
    if config.smoke:
        config = replace(config, epochs=1, batch=32, hidden=32)
        train, test = random_images(256, 0), random_images(64, 1)
        loading = Loading(workers=0, threads=1, read_buffer=1)
    else:
        train, test = mnist("train"), mnist("test")
        loading = Loading()
    data = Dataset.from_records(train, batch=config.batch, validation=test, loading=loading)
    per_epoch = data.steps_per_epoch
    assert per_epoch is not None  # records held in memory have a count
    net = SpikingMLP(hidden=(config.hidden, config.hidden), classes=10,
                     neuron=sparx.nn.LIF(tau=2.0, detach_reset=True), readout_tau=2.0)
    encoder = sparx.encode.RateEncoder(config.steps)
    objective = SpikingClassifierObjective(net, Field("image", (28, 28)), encoder)
    trainer = Trainer(objective, optax.adam(config.learning_rate), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(config.out)))
    state = trainer.fit(data, steps=config.epochs * per_epoch, log_every=min(100, per_epoch),
                        eval_every=per_epoch, metrics=[Accuracy()],
                        validation={"test": data.val})
    classifier = objective.pipeline(state)
    print(f"predicted {np.asarray(classifier(test['image'][:10])).tolist()} "
          f"for labels {test['label'][:10].tolist()}")


if __name__ == "__main__":
    main(tyro.cli(Config))
