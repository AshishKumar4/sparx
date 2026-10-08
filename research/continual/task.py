"""Switches and doors: a session of trials under one hidden mapping, which an agent can only learn by trying.

The notes' first test of learning within a new situation (chapter 7,
stage C): sample a new mapping from switches to doors, let the agent act
under it for several attempts while its adaptive memory persists, score the
whole session, then sample another mapping. Each trial here shows a target
door; the agent presses one of `doors` switches; the switch opens the door
the session's mapping assigns it, and the agent sees which door opened and
whether it was the target. The mapping is a permutation, new each session,
so the first trial is a guess, and what the agent sees narrows the rest.

A trial is `steps` steps of observation, `[target door, last switch, door it
opened, reward]` one-hot and scalar, with last trial's outcome on its first
step and zeros after, and the agent acts on the last.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SwitchDoor:
    """Sessions of `trials` trials of `steps` steps under a hidden mapping from `doors` switches to doors."""

    doors: int = 4
    trials: int = 8
    steps: int = 3

    @property
    def observed(self) -> int:
        """The width of one step's observation."""
        return 3 * self.doors + 1

    def sessions(self, count: int, seed: int) -> dict[str, np.ndarray]:
        """`count` sessions: each one's mapping, `mapping[s]` the door switch `s` opens, `[count, doors]`, and
        its trials' target doors `[count, trials]`."""
        rng = np.random.default_rng(seed)
        mapping = rng.random((count, self.doors)).argsort(axis=1).astype(np.int32)
        targets = rng.integers(0, self.doors, (count, self.trials)).astype(np.int32)
        return {"mapping": mapping, "targets": targets}

    def ideal(self) -> np.ndarray:
        """Each trial's expected accuracy, `[trials]`, for a learner that remembers every pair it has seen,
        presses the switch of a target door it knows and otherwise a switch it has not tried.

        A trial's chance of a hit depends only on how many pairs `k` the
        learner knows: `k / doors` that it knows the target's switch, and
        otherwise one in the `doors - k` untried switches, `(k + 1) / doors`
        in all. Each miss on what it knows teaches one more pair, and
        `doors - 1` pairs give the last one.
        """
        known = np.arange(self.doors)
        learns = np.where(known < self.doors - 1, 1 - known / self.doors, 0.0)
        p = np.eye(self.doors)[0]
        accuracy = []
        for _ in range(self.trials):
            accuracy.append(p @ (known + 1) / self.doors)
            p = p * (1 - learns) + np.roll(p * learns, 1)
        return np.asarray(accuracy)

    def observe(self, target: jax.Array, switch: jax.Array, opened: jax.Array,
                reward: jax.Array) -> jax.Array:
        """A trial's observations `[steps, B, observed]`: the target throughout, the last trial's switch, door
        and reward on the first step (zeros before the first trial: `switch < 0`)."""
        shown = switch >= 0
        outcome = jnp.concatenate([jax.nn.one_hot(switch, self.doors) * shown[:, None],
                                   jax.nn.one_hot(opened, self.doors) * shown[:, None],
                                   (reward * shown)[:, None]], axis=-1)
        cue = jax.nn.one_hot(target, self.doors)
        first = jnp.concatenate([cue, outcome], axis=-1)
        later = jnp.concatenate([cue, jnp.zeros_like(outcome)], axis=-1)
        return jnp.stack([first, *[later] * (self.steps - 1)])
