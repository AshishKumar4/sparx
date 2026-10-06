"""Optimizer pieces SNN-delays needs and the pinned dew lacks.

Hammouamri et al.'s SNN-delays trains its weights with Adam on torch's
one-cycle schedule, its delay positions with Adam at another rate on a
cosine, clamped to the kernel, and steps every scheduler once an epoch. The
dew sparx pins has neither a one-cycle nor an exponential schedule record,
no per-epoch stepping, and no per-group schedule, momentum or bounds, so
this module holds them until dew does. Each piece names the dew change that
replaces it, and `SpikingClassifierObjective.groups` is the one place that
reads `GroupAdam`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from dew.registry import schedules
from dew.training.optim import ScheduleBase
from jax.typing import ArrayLike

__all__ = ["ExponentialDecay", "GroupAdam", "OneCycle", "keep_within", "stepped"]


# Remove when AshishKumar4/dew#37 merges: dew's `one_cycle` schedule record, torch's
# OneCycleLR with its phase boundaries. Registered under a sparx-prefixed name so it
# cannot collide with dew's record, and a run records it under that name.
@schedules("sparx_one_cycle")
@dataclass(frozen=True)
class OneCycle(ScheduleBase):
    """torch's `OneCycleLR` with its default cosine annealing and two phases.

    The value follows a half cosine from `start` to `peak` until step
    `warmup * steps - 1`, then another to `end` at step `steps - 1`, and
    stays there. The step boundaries are torch's, so stepped once an epoch
    over a run of epochs (`SpikingClassifierObjective`'s `schedule_every`) it
    gives torch's values. For a learning rate torch starts at `peak / 25` and
    ends at `start / 1e4`; its Adam momentum cycle (`b1`) is
    `OneCycle(peak=0.85, start=0.95, end=0.95)`.
    """

    peak: float
    start: float
    end: float
    warmup: float = 0.3

    def schedule(self, steps: int) -> optax.Schedule:
        rise, last = self.warmup * steps - 1, steps - 1
        if not 0 < rise < last:
            raise ValueError(f"a one-cycle schedule over {steps} steps with warmup {self.warmup} has no "
                             "rise or no fall; give it more steps")

        def anneal(start: float, end: float, done: jax.Array) -> jax.Array:
            return end + (start - end) / 2 * (jnp.cos(jnp.pi * done) + 1)

        def value(count: ArrayLike) -> jax.Array:
            step = jnp.minimum(jnp.asarray(count, jnp.float32), last)
            return jnp.where(step <= rise, anneal(self.start, self.peak, step / rise),
                             anneal(self.peak, self.end, (step - rise) / (last - rise)))

        return value


# Remove when AshishKumar4/dew#37 merges: dew's `exponential` schedule record, a geometric
# decay to `end` over `decay_steps` with an `offset`.
@schedules("sparx_exponential_decay")
@dataclass(frozen=True)
class ExponentialDecay(ScheduleBase):
    """A geometric decay from `offset + start` to `offset + end` over `decay_steps`, then constant.

    The value is `offset + start * (end / start) ** (min(step, decay_steps) / decay_steps)`,
    `decay_steps` being the run when None. SNN-delays shrinks DCLS's raw
    width this way, from `max_delay // 2` to 0.23 over the first quarter of
    its epochs; DCLS's width is the raw one plus 0.27, so the
    `sparx.nn.DelayedDense` width is `ExponentialDecay(12, 0.23, epochs // 4, offset=0.27)`
    for their 25-step kernels.
    """

    start: float
    end: float
    decay_steps: int | None = None
    offset: float = 0.0

    def schedule(self, steps: int) -> optax.Schedule:
        span = steps if self.decay_steps is None else self.decay_steps
        if span <= 0 or self.start <= 0 or self.end <= 0:
            raise ValueError("an exponential decay needs positive steps, start and end")
        ratio = self.end / self.start

        def value(count: ArrayLike) -> jax.Array:
            done = jnp.minimum(jnp.asarray(count, jnp.float32), span) / span
            return self.offset + self.start * ratio ** done

        return value


def stepped(schedule: ScheduleBase, steps: int, every: int = 1) -> optax.Schedule:
    """`schedule` advancing once every `every` steps, over `steps // every` steps of its own.

    A torch scheduler stepped once an epoch holds its value through the
    epoch. With `every` set to the steps of an epoch, a step reads the
    schedule's value for the epoch it falls in.
    """
    # Remove when AshishKumar4/dew#37 merges: the `every` field every dew schedule record carries.
    if every < 1:
        raise ValueError(f"a schedule advances every 1 or more steps, not {every}")
    values = schedule.schedule(max(steps // every, 1))
    if every == 1:
        return values
    return lambda count: values(jnp.asarray(count) // every)


def keep_within(lower: float, upper: float) -> optax.GradientTransformation:
    """Shorten each update so the parameter lands within `[lower, upper]`.

    DCLS clamps its positions to the kernel after every step
    (`clamp_parameters`), and clamping a `DelayedDense` delay to
    `[0, max_delay]` is the same. A delay left outside that range would
    have no gradient, since the kernel clips its center, and would stay
    there.
    """
    # Remove when AshishKumar4/dew#37 merges: `ParamGroup.bounds`, dew's `keep_within`.
    def init(params: optax.Params) -> optax.EmptyState:
        return optax.EmptyState()

    def update(updates: optax.Updates, state: optax.OptState,
               params: optax.Params | None = None) -> tuple[optax.Updates, optax.OptState]:
        if params is None:
            raise ValueError("keep_within reads the parameters; pass them to update")
        kept = jax.tree.map(lambda u, p: jnp.clip(p + u, lower, upper) - p, updates, params)
        return kept, state

    return optax.GradientTransformation(init, update)


# Remove when AshishKumar4/dew#37 merges: a `ParamGroup` with its own `schedule`, `b1`,
# `bounds` and `weight_decay` (coupled L2 under optimizer='adam') in one `OptimConfig`.
@dataclass(frozen=True)
class GroupAdam:
    """Adam for the parameters whose paths match `patterns`, on schedules of its own.

    A path is the parameter's dict keys joined by `/` (`delayed_0/delay`),
    matched by `fnmatch` with `*` matching `/` too, as dew's `ParamGroup`
    matches. `learning_rate` and `b1` (0.9 when None) are dew schedule
    records. `weight_decay` adds `weight_decay * param` to the gradient
    before Adam's moments, as torch's `Adam(weight_decay=...)` does. That is
    an L2 penalty, which differs from AdamW's decoupled decay. `bounds` keeps
    every parameter within `[lower, upper]` after each update
    (`keep_within`).

    SNN-delays trains its weights with Adam on a one-cycle schedule and its
    delay positions with Adam at 100 times the rate on a cosine, without
    weight decay, clamped to the kernel.
    """

    patterns: tuple[str, ...]
    learning_rate: ScheduleBase
    b1: ScheduleBase | None = None
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    bounds: tuple[float, float] | None = None

    def build(self, steps: int, every: int = 1) -> optax.GradientTransformation:
        """The optimizer over a run of `steps` updates, its schedules advancing every `every` (`stepped`)."""
        if self.b1 is None:
            adam = optax.scale_by_adam(b1=0.9, b2=self.b2, eps=self.eps)
        else:
            adam = optax.inject_hyperparams(optax.scale_by_adam)(
                b1=stepped(self.b1, steps, every), b2=self.b2, eps=self.eps)
        chain = [optax.add_decayed_weights(self.weight_decay)] if self.weight_decay else []
        chain += [adam, optax.scale_by_learning_rate(stepped(self.learning_rate, steps, every))]
        if self.bounds is not None:
            chain.append(keep_within(*self.bounds))
        return optax.chain(*chain)
