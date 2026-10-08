"""RNeuralNet's reward diffusion against no learning and REINFORCE, on a delayed cue-order task.

    python examples/reward_diffusion.py --rule first        # the original's spread
    python examples/reward_diffusion.py --rule all          # every path, discounted
    python examples/reward_diffusion.py --rule reinforce    # REINFORCE through the network
    python examples/reward_diffusion.py --rule none         # no learning, and the readout
    JAX_PLATFORMS=cpu python examples/reward_diffusion.py --smoke --out /tmp/diffusion-smoke

The experiment the owner's research notes propose for RNeuralNet-Research
(chapter 2, section 6): observe two cues, endure a delay with distractors,
and report which cue came first. Cue A or B arrives at tick 0 and the
other `gap` ticks later, each a pulse of 3 on its input neuron; for
`delay` ticks after that a third input neuron pulses 3 with probability
`distractors` each tick; at the last tick the network chooses between its
two output neurons, and the choice earns 1 if it names the first cue and -1
otherwise. The network is `sparx.learn.RNeuralNet.random`: `neurons`
neurons drawn as the original draws them, 16 connections each with delays of
1 to 21 ticks, the three input neurons and two output neurons; `--myelin
0` gives every connection the shortest delay instead.

First, as the notes ask, whether the network keeps the order at all: a
linear readout of every neuron's output at the last tick, fit by least
squares on the training trials, on the test trials. Then each rule learns
from the same rewards on the same trials, through dew's Trainer
(`sparx.objectives.RNeuralNetObjective`): the original's reward diffusion
at its own rate (SGD at `0.01` per reward), every path's at the same rate,
or REINFORCE by Adam. `none` evaluates the network as drawn. Each run
reports the test accuracy of the network's own choice, the largest output.

On seeds 0 to 4 (the guide's results) the readout is right on every test
trial, REINFORCE teaches the output neurons on four seeds, and reward
diffusion leaves the choice as drawn on all five.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset, Loading

from sparx.learn import RNeuralNet
from sparx.metrics import Accuracy
from sparx.objectives import RNeuralNetObjective

INPUTS, PULSE = 3, 3.0


@dataclass
class Config:
    rule: Literal["first", "all", "reinforce", "none"] = "first"
    neurons: int = 256
    gap: int = 2
    delay: int = 10
    distractors: float = 0.2
    myelin: int | None = None
    """Every connection's `Myelin`, for one delay throughout; the original's draws by default."""
    trials: int = 19_200
    """Training trials, each one reward."""
    test: int = 512
    batch: int = 32
    discount: float = 0.9
    """Each connection's share of what it passes on, for `--rule all`."""
    learning_rate: float = 1e-3
    """Adam's rate for REINFORCE."""
    seed: int = 0
    out: Path = Path("runs/reward-diffusion")
    """The run's directory; a run there resumes."""
    smoke: bool = False
    """A small network for a few steps."""


def cue_order(count: int, config: Config, seed: int) -> dict[str, np.ndarray]:
    """`count` trials: the input neurons' values `[count, T, 3]` and which cue came first."""
    rng = np.random.default_rng(seed)
    ticks = config.gap + config.delay + 1
    cues = np.zeros((count, ticks, INPUTS), np.float32)
    first = rng.integers(0, 2, count)
    rows = np.arange(count)
    cues[rows, 0, first] = PULSE
    cues[rows, config.gap, 1 - first] = PULSE
    cues[:, config.gap + 1:, 2] = PULSE * (rng.random((count, config.delay)) < config.distractors)
    return {"cues": cues, "label": first.astype(np.int32)}


def readout(net: RNeuralNet, train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> float:
    """The test accuracy of a least-squares linear readout of every neuron at the last tick."""
    def features(trials: dict[str, np.ndarray]) -> np.ndarray:
        last = jax.jit(net.run)(jnp.swapaxes(trials["cues"], 0, 1))[0][-1, :, :net.neurons]
        return np.c_[np.asarray(last), np.ones(len(last))]

    weights = np.linalg.lstsq(features(train), 2.0 * train["label"] - 1, rcond=None)[0]
    return float(np.mean((features(test) @ weights > 0) == test["label"]))


def chosen(net: RNeuralNet, weight: jax.Array, test: dict[str, np.ndarray]) -> float:
    """The test accuracy of the network's own choice, its largest output, with the connections at `weight`."""
    net = replace(net, cell=replace(net.cell, wiring=replace(net.wiring, weight=jnp.asarray(weight))))
    last = jax.jit(net.run)(jnp.swapaxes(test["cues"], 0, 1))[0][-1]
    return float(np.mean(np.asarray(jnp.argmax(last[:, net.outputs], axis=-1)) == test["label"]))


def main(config: Config) -> None:
    if config.smoke:
        config = replace(config, neurons=48, trials=128, test=64, batch=16)
    loading = Loading(workers=0, threads=1, read_buffer=1)
    train = cue_order(config.trials, config, 2 * config.seed)
    test = cue_order(config.test, config, 2 * config.seed + 1)
    net = RNeuralNet.random(config.seed, config.neurons, INPUTS, 2, myelin=config.myelin)
    ticks = config.gap + config.delay + 1
    if config.rule == "none":
        print(f"none: test accuracy {chosen(net, net.wiring.weight, test):.4f}")
        print(f"linear readout of every neuron: test accuracy {readout(net, train, test):.4f}")
        return
    objective = RNeuralNetObjective(net, Field("cues", (ticks, INPUTS)), rule=config.rule,
                                    discount=config.discount)
    if config.rule == "reinforce":
        optimizer = optax.adam(config.learning_rate)
    else:
        optimizer = optax.sgd(0.01 * config.batch)  # the original's W_CONST for each reward of the batch
    data = Dataset.from_records(train, batch=config.batch, seed=config.seed, validation=test, loading=loading)
    steps = config.trials // config.batch
    trainer = Trainer(objective, optimizer, key=jax.random.key(config.seed),
                      checkpoints=Checkpoints(str(config.out)))
    state = trainer.fit(data, steps=steps, log_every=max(1, steps // 10), eval_every=max(1, steps // 4),
                        metrics=[Accuracy()], validation={"test": data.val})
    print(f"{config.rule}: test accuracy {chosen(net, state.variables['params']['weight'], test):.4f}")


if __name__ == "__main__":
    main(tyro.cli(Config))
