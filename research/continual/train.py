"""Train an agent to learn switch-to-door mappings within a session, and measure how fast it adapts.

    python research/continual/train.py --agent plastic          # the core with selective fast plasticity
    python research/continual/train.py --agent core             # the same core, no fast weights
    python research/continual/train.py --agent rnn              # a dense recurrent network of tanh units
    JAX_PLATFORMS=cpu python research/continual/train.py --smoke --out /tmp/continual-smoke

The notes' comparison (chapter 6, "The first task should expose whether the
mechanisms help"): the proposed agent against a conventional recurrent
network given the same observations and feedback, and against itself
without fast plasticity, which separates what rapid weight change adds from
what the recurrent state holds. Every agent is an input projection, a
recurrent core and a readout of switch scores; `core.ModularCore` is the
notes' core, with `--plastic` fast-plastic inputs per unit under `--agent
plastic` and none under `--agent core`; `--agent rnn` is
`sparx.nn.Recurrent(Rate)` with every unit connected to every unit.

Training is REINFORCE through whole sessions (`SessionObjective`, a dew
objective): the agent samples a switch at the end of each trial from the
softmax of its scores, and each choice is credited with its own trial's
reward and `--discount` times each later one's, compounding, less the mean
of the batch's other sessions at that trial. The gradient flows through the
recurrent state and the fast weights over the whole session, as the notes
ask, and not through the task. Evaluation presses the highest-scoring
switch and scores the last trial. After training the script prints each
trial's test accuracy, chance on the first trial and then the curve the
agent's learning within a session makes, beside the curve of a learner that
remembers every pair it has seen (`SwitchDoor.ideal`).
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from core import ModularCore
from dew import Checkpoints, Field, Trainer
from dew.artifacts import TokenScores
from dew.data import Dataset, Loading
from dew.inputs import InputSpec
from dew.objectives.base import Aux, Batch, Objective, Ratio, Shown, Step, Variables
from task import SwitchDoor

import sparx.nn as snn
from sparx.metrics import Accuracy


class Agent(nn.Module):
    """Observations `[T, B, observed]` to switch scores `[T, B, doors]`: an input projection onto `core`'s
    `size` units, the core, and a readout."""

    core: snn.Neuron
    size: int
    doors: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(self.doors, name="readout")(self.core(nn.Dense(self.size, name="input")(x)))


class SessionObjective(Objective[Ratio]):
    """REINFORCE through sessions of `task`, with an `entropy` bonus on each choice; see the module.

    A choice is credited with its own trial's reward and `discount` times
    each later one's, compounding, so 1 credits every later reward in full
    and 0 none.
    """

    artifact = TokenScores
    shown: Mapping[str, Shown] = {"accuracy": Shown(better="higher", percent=True)}

    def __init__(self, model: Agent, task: SwitchDoor, *, entropy: float = 0.1, discount: float = 0.0):
        self.model = self.bind_model(model)
        self.task = task
        self.entropy, self.discount = entropy, discount
        self.inputs = InputSpec(sample=Field("targets", (task.trials,)))

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        x = jnp.zeros((self.task.steps, 1, self.task.observed), jnp.float32)
        return {"params": self.model.init(key, x)["params"]}

    def _session(self, params: Variables, batch: Batch, key: jax.Array | None
                 ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Each trial's reward, the log-probability of the switch pressed and the entropy of the choice,
        `[trials, B]`; with `key` None the agent presses its highest-scoring switch."""
        mapping, targets = jnp.asarray(batch["mapping"]), jnp.asarray(batch["targets"])
        rows = jnp.arange(mapping.shape[0])
        keys = jnp.arange(self.task.trials) if key is None else jax.random.split(key, self.task.trials)

        def trial(carry, inputs):
            state, switch, opened, reward = carry
            target, k = inputs
            observed = self.task.observe(target, switch, opened, reward)
            scores, updates = self.model.apply({"params": params, **state}, observed, mutable=["state"])
            logits = scores[-1]
            pressed = jnp.argmax(logits, -1) if key is None else jax.random.categorical(k, logits)
            log_p = jax.nn.log_softmax(logits)
            door = mapping[rows, pressed]
            won = (door == target).astype(jnp.float32)
            entropy = -jnp.sum(jnp.exp(log_p) * log_p, -1)
            return (dict(updates), pressed, door, won), (won, log_p[rows, pressed], entropy)

        start = (self._initial_state(params, mapping.shape[0]), jnp.full(mapping.shape[0], -1),
                 jnp.zeros(mapping.shape[0], jnp.int32), jnp.zeros(mapping.shape[0]))
        _, (won, log_p, entropy) = jax.lax.scan(trial, start, (targets.T, keys))
        return won, log_p, entropy

    def _initial_state(self, params: Variables, batch: int) -> Variables:
        """The core's state at rest: zeros (activity, what is on its way, traces), in the structure a call
        with the `state` collection mutable leaves."""
        x = jnp.zeros((1, batch, self.task.observed), jnp.float32)
        _, updates = self.model.apply({"params": params}, x, mutable=["state"])
        return jax.tree.map(jnp.zeros_like, dict(updates))

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        won, log_p, entropy = self._session(variables["params"], batch, step.key)
        def back(later: jax.Array, reward: jax.Array) -> tuple[jax.Array, jax.Array]:
            value = reward + self.discount * later
            return value, value

        returns = jax.lax.scan(back, jnp.zeros_like(won[0]), won, reverse=True)[1]
        count = won.shape[1]
        baseline = (jnp.sum(returns, axis=1, keepdims=True) - returns) / max(count - 1, 1)
        advantage = jax.lax.stop_gradient(returns - baseline)
        total = -jnp.sum(advantage * log_p) - self.entropy * jnp.sum(entropy)
        metrics = {"reward": jnp.mean(won), "last_trial": jnp.mean(won[-1])}
        return Ratio(total, jnp.asarray(count, jnp.float32)), Aux(metrics=metrics)

    @functools.cached_property
    def _greedy(self):
        return jax.jit(lambda params, batch: self._session(params, batch, None)[0])

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        fields = {key: jnp.asarray(batch[key]) for key in ("mapping", "targets")}
        # The last trial: what the session has taught the agent.
        won = self._greedy(self.evaluation_variables(params, step)["params"], fields)[-1][:, None]
        return TokenScores(losses=-won, weights=jnp.ones_like(won), correct=won > 0)


@dataclass
class Config:
    agent: Literal["plastic", "core", "rnn"] = "plastic"
    modules: int = 4
    units: int = 64
    interface: int = 16
    plastic: int = 16
    rule: Literal["modulated", "retroactive"] = "modulated"
    rnn_units: int = 256
    """Units of `--agent rnn`."""
    doors: int = 4
    trials: int = 8
    steps: int = 3
    sessions: int = 20_000
    """Training sessions, each one gradient example."""
    test: int = 1024
    batch: int = 32
    learning_rate: float = 1e-3
    entropy: float = 0.1
    """The entropy bonus. At 0.01 and 0.05 every agent on two doors settled on a rule right in 6 of the 8
    situations a trial can show, and stopped exploring; at 0.1 the dense network found the whole rule."""
    discount: float = 0.0
    """How much of each later trial's reward a choice is credited with; with every later reward credited
    in full (1), no agent left chance on four doors in 20,000 sessions."""
    seed: int = 0
    out: Path = Path("runs/continual")
    """The run's directory; a run there resumes."""
    smoke: bool = False
    """A small agent for a few steps."""


def agent(config: Config) -> Agent:
    if config.agent == "rnn":
        core = snn.Recurrent(snn.Rate(tau=2.0))
        return Agent(core, config.rnn_units, config.doors)
    rule = snn.ModulatedTrace() if config.rule == "modulated" else snn.RetroactiveTrace()
    core = ModularCore(modules=config.modules, units=config.units, interface=config.interface,
                       plastic=config.plastic if config.agent == "plastic" else 0,
                       rule=rule if config.agent == "plastic" else None, seed=config.seed)
    return Agent(core, core.size, config.doors)


def per_trial(objective: SessionObjective, params: Variables, sessions: dict[str, np.ndarray]) -> np.ndarray:
    """Each trial's accuracy over `sessions`, the agent pressing its highest-scoring switch."""
    fields = {key: jnp.asarray(value) for key, value in sessions.items()}
    return np.asarray(objective._greedy(params, fields)).mean(axis=1)


def main(config: Config) -> None:
    if config.smoke:
        config = replace(config, modules=2, units=8, interface=4, plastic=4, rnn_units=16, sessions=64,
                         test=32, batch=16)
    task = SwitchDoor(config.doors, config.trials, config.steps)
    loading = Loading(workers=0, threads=1, read_buffer=1)
    data = Dataset.from_records(task.sessions(config.sessions, 2 * config.seed), batch=config.batch,
                                seed=config.seed, validation=task.sessions(config.test, 2 * config.seed + 1),
                                loading=loading)
    objective = SessionObjective(agent(config), task, entropy=config.entropy, discount=config.discount)
    steps = config.sessions // config.batch
    trainer = Trainer(objective, optax.adam(config.learning_rate), key=jax.random.key(config.seed),
                      checkpoints=Checkpoints(str(config.out)))
    state = trainer.fit(data, steps=steps, log_every=max(1, steps // 10), eval_every=max(1, steps // 4),
                        metrics=[Accuracy()], validation={"test": data.val})
    test = task.sessions(config.test, 2 * config.seed + 1)
    accuracy = per_trial(objective, state.variables["params"], test)
    print(f"{config.agent}: test accuracy by trial " + " ".join(f"{a:.3f}" for a in accuracy))
    print("a learner remembering every pair:   " + " ".join(f"{a:.3f}" for a in task.ideal()))


if __name__ == "__main__":
    main(tyro.cli(Config))
