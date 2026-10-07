"""Train a network to complete a pattern it saw once in an episode, with fast weights, through dew's Trainer.

    python examples/pattern_completion.py                          # Miconi et al.'s task and settings
    python examples/pattern_completion.py --trace retroactive      # Backpropamine's eligibility trace
    python examples/pattern_completion.py --trace none             # the same network without fast weights
    JAX_PLATFORMS=cpu python examples/pattern_completion.py --smoke --out /tmp/completion-smoke

Miconi et al.'s (2018) pattern memorization (their section 4.2 and
`simple/simple.py`, github.com/uber-research/differentiable-plasticity).
An episode shows `patterns` patterns of `bits` values, half of them -1 and
half +1, each `cycles` times for `shown` steps with `blank` steps after,
then one of them with half its bits zeroed for `tested` steps; on the last
step the network must fill the zeroed bits in. A new episode has new
patterns, so the weights learned across episodes cannot hold them: the
network holds them in a Hebbian trace, which starts at zero every episode.
Every input is multiplied by 20, and one more unit receives a constant
input, their bias neuron. The defaults are theirs: 1000 bits, 5 patterns,
2 cycles of 6 steps with 4 blank, 6 test steps, Adam at 3e-4 over 2000
episodes of one, weights and plasticity drawn at `0.01 * randn`, `eta` at
0.01.

The network is one tanh unit per input without leak (`sparx.nn.Rate` with
`tau=0`, which also learns a bias per unit) behind `sparx.nn.Plastic`,
whose trace `--trace` picks: Miconi et al.'s decaying Hebbian trace or
Oja's rule, or Backpropamine's neuromodulated or retroactive trace (Miconi
et al. 2019); `none` is `sparx.nn.Recurrent` on the same units. dew's
`Supervised` objective trains it on their loss, the last step's squared
error on the pattern's bits, and reports the share of zeroed bits whose
sign comes out wrong. Episodes are drawn as dew reads them, each from its
index (`Episodes`), so nothing is held in memory and a resumed run sees the
same ones. `--smoke` trains a small network for a few steps.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from dew import Checkpoints, Field, Supervised, Trainer
from dew.data import Dataset, Loading
from dew.inputs import InputSpec

import sparx.nn as snn

TRACES: dict[str, snn.HebbianTrace] = {
    "decaying": snn.DecayingTrace(), "oja": snn.OjaTrace(), "modulated": snn.ModulatedTrace(),
    "retroactive": snn.RetroactiveTrace()}


@dataclass(frozen=True)
class Task:
    """One episode's layout, in their names: PATTERNSIZE, NBPATTERNS, NBPRESCYCLES, PRESTIME,
    INTERPRESDELAY, PRESTIMETEST, PROBADEGRADE and their input gain."""

    bits: int = 1000
    patterns: int = 5
    cycles: int = 2
    shown: int = 6
    blank: int = 4
    tested: int = 6
    zeroed: float = 0.5
    gain: float = 20.0

    @property
    def steps(self) -> int:
        return self.cycles * self.patterns * (self.shown + self.blank) + self.tested

    def episode(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """One episode as their `generateInputsAndTarget` draws it: its inputs `[steps, bits + 1]`,
        the pattern to complete `[bits]`, and which of its bits the test shows `[bits]`."""
        balanced = np.where(np.arange(self.bits) < self.bits // 2, -1.0, 1.0)
        patterns = [rng.permutation(balanced) for _ in range(self.patterns)]
        target = patterns[rng.integers(self.patterns)]
        shown = rng.permutation(np.arange(self.bits) >= int(self.zeroed * self.bits)).astype(np.float64)
        steps = []
        for _ in range(self.cycles):
            for i in rng.permutation(self.patterns):
                steps += [patterns[i]] * self.shown + [np.zeros(self.bits)] * self.blank
        steps += [target * shown] * self.tested
        inputs = np.concatenate([np.stack(steps), np.ones((len(steps), 1))], axis=1) * self.gain
        return {"episode": inputs.astype(np.float32), "target": target.astype(np.float32),
                "shown": shown.astype(np.float32)}


@dataclass(frozen=True)
class Episodes:
    """`count` episodes of `task`, each drawn from its index and `seed`: a source dew reads by index."""

    task: Task
    count: int
    seed: int

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return self.task.episode(np.random.default_rng((self.seed, index)))


def completion_loss(outputs: jax.Array, batch: dict[str, jax.Array]) -> jax.Array:
    """Their loss: the last step's squared error on the pattern's bits, per episode."""
    bits = batch["target"].shape[-1]
    return jnp.sum((outputs[:, -1, :bits] - batch["target"]) ** 2, axis=-1)


def zeroed_error(outputs: jax.Array, batch: dict[str, jax.Array]) -> jax.Array:
    """The share of the zeroed bits whose sign comes out wrong on the last step, per episode."""
    bits = batch["target"].shape[-1]
    wrong = (jnp.sign(outputs[:, -1, :bits]) != batch["target"]) & (batch["shown"] == 0)
    return jnp.sum(wrong, axis=-1) / jnp.sum(batch["shown"] == 0, axis=-1)


@dataclass
class Config:
    trace: Literal["decaying", "oja", "modulated", "retroactive", "none"] = "decaying"
    task: Task = Task()
    episodes: int = 2000
    """Training episodes, one pass."""
    batch: int = 1
    learning_rate: float = 3e-4
    holdout: int = 100
    """Episodes scored after training and every `eval_every` steps."""
    eval_every: int = 500
    seed: int = 0
    out: Path = Path("runs/pattern-completion")
    """The run's directory; a run there resumes."""
    smoke: bool = False
    """Train a small network for a few steps."""


def network(trace: str) -> nn.Module:
    """Their network over a batch of episodes `[B, T, bits + 1]`."""
    small = nn.initializers.normal(0.01)
    unit = snn.Rate(tau=0)
    if trace == "none":
        return snn.BatchMajor(snn.Recurrent(unit, kernel_init=small))
    return snn.BatchMajor(snn.Plastic(unit, rule=TRACES[trace], kernel_init=small, alpha_init=small))


def main(config: Config) -> None:
    loading = Loading()
    if config.smoke:
        config = replace(config, task=Task(bits=16, patterns=2, shown=3, blank=2, tested=3), episodes=64,
                         batch=8, learning_rate=1e-2, holdout=16, eval_every=8)
        loading = Loading(workers=0, threads=1, read_buffer=1)
    task = config.task
    data = Dataset.from_records(Episodes(task, config.episodes, config.seed), batch=config.batch,
                                seed=config.seed, validation=Episodes(task, config.holdout, config.seed + 1),
                                loading=loading)
    model = network(config.trace)
    objective = Supervised(model, completion_loss, [zeroed_error],
                           inputs=InputSpec(Field("episode", (task.steps, task.bits + 1))))
    trainer = Trainer(objective, optax.adam(config.learning_rate), key=jax.random.key(config.seed),
                      checkpoints=Checkpoints(str(config.out)))
    steps = config.episodes // config.batch
    trained = trainer.fit(data, steps=steps, log_every=max(1, steps // 20), eval_every=config.eval_every,
                          validation={"holdout": data.val})
    # dew's validation pass reports Supervised's loss alone, so the zeroed bits are scored here.
    holdout = Episodes(task, config.holdout, config.seed + 1)
    batch = {key: jnp.stack([holdout[i][key] for i in range(len(holdout))])
             for key in ("episode", "target", "shown")}
    error = jnp.mean(zeroed_error(model.apply(trained.variables, batch["episode"]), batch))
    print(f"holdout: {float(error):.1%} of the zeroed bits come out with the wrong sign")


if __name__ == "__main__":
    main(tyro.cli(Config))
