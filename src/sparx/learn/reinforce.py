"""REINFORCE for a recurrent layer of stochastic spiking neurons, its gradient carried as eligibilities.

A layer of `sparx.dynamics.BernoulliCell`s draws its spikes `s` with
probabilities `p` its weights set. Williams' (1992) REINFORCE estimates the
gradient of the expected reward of a trajectory from samples of it,

    grad E[R] = E[R grad log P(s)],

and `log P(s)` is a sum over steps, `s_t log p_t + (1 - s_t) log(1 - p_t)`,
so its gradient is an eligibility that each synapse accumulates as the layer
runs, in memory that does not grow with the sequence: three factors meet at
a synapse, its presynaptic signal, its neuron's surprise `s_t - p_t`, and the
reward the caller multiplies in at the end (`policy_gradient`).

With every spike held at its sampled value, the membrane before the threshold
test, `v_t = decay ** dt * v_{t-1} + x_t`, depends on a weight through the
step's input `x_t` and the membrane the last reset left. For a weight that
carries the presynaptic signal `z_t` (the input `u_t` for `w_in`, the
layer's last spikes for `w_rec`), its derivative `eps_t = dv_t / dw` and the
weight's eligibility `e` follow

    eps_t = decay ** dt * r_{t-1} * eps_{t-1} + z_t
    e    += beta * (s_t - p_t) * eps_t

where `r_{t-1}` is the reset's derivative, `1 - s_{t-1}` for a reset to
zero and 1 for a subtraction or none, since `d log P(s_t) / dv_t` is
`beta (s_t - p_t)` for the cell's sigmoid. `tests/test_learn.py` checks
the identity on every trajectory of a small layer, enumerated.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from sparx.dynamics import BernoulliCell, SynapticInput
from sparx.dynamics.core import membrane_dtype

__all__ = ["ReinforceParams", "policy_gradient", "reinforce"]


class ReinforceParams(NamedTuple):
    """A recurrent layer, `x_t = u_t @ w_in + s_{t-1} @ w_rec`, `[in, N]` and `[N, N]`; or each
    example's eligibility for them, with a leading batch axis."""

    w_in: jax.Array
    w_rec: jax.Array


def reinforce(cell: BernoulliCell, params: ReinforceParams, inputs: jax.Array, noise: jax.Array, *,
              dt: float = 1.0) -> tuple[jax.Array, ReinforceParams]:
    """The spikes `[T, B, N]` the layer draws over `inputs` `[T, B, in]`, and each example's
    eligibility, `d log P(spikes) / d params`, `[B, in, N]` and `[B, N, N]`.

    `noise` `[T, B, N]` is uniform on [0, 1), `jax.random.uniform` of a key,
    and decides the spikes (`BernoulliCell`); noise `1 - s` replays the
    spikes `s`. The layer starts at rest with no spikes before the first
    step.
    """
    steps, batch, _ = inputs.shape
    size = params.w_rec.shape[0]
    dtype = membrane_dtype(inputs.dtype)
    if noise.shape != (steps, batch, size):
        raise ValueError(f"the noise is one draw per neuron per step, {(steps, batch, size)}, not "
                         f"{noise.shape}")
    leak = cell.decay ** dt

    def step(carry: tuple, given: tuple[jax.Array, jax.Array]) -> tuple[tuple, jax.Array]:
        state, last, kept_in, kept_rec, eligibility = carry
        u, drawn = given
        x = u @ params.w_in + last @ params.w_rec
        state, out = cell.step(state, SynapticInput(jump=x, noise=drawn), dt)
        s = out.value.astype(dtype)
        # dv_t / dw over the weights' presynaptic signals, and this step's d log P(s_t) / dv_t.
        eps_in = leak * kept_in + u[:, :, None]
        eps_rec = leak * kept_rec + last[:, :, None]
        surprise = (cell.beta * (s - state.p))[:, None, :]
        eligibility = ReinforceParams(eligibility.w_in + surprise * eps_in,
                                      eligibility.w_rec + surprise * eps_rec)
        # The derivative the reset passes to the next step's membrane.
        passed = (1 - s if cell.reset == "zero" else jnp.ones_like(s))[:, None, :]
        return (state, s, passed * eps_in, passed * eps_rec, eligibility), s

    start = (cell.init_state((batch, size), dtype), jnp.zeros((batch, size), dtype),
             jnp.zeros((batch, *params.w_in.shape), dtype), jnp.zeros((batch, size, size), dtype),
             ReinforceParams(jnp.zeros((batch, *params.w_in.shape), dtype),
                             jnp.zeros((batch, size, size), dtype)))
    (_, _, _, _, eligibility), spikes = jax.lax.scan(step, start, (jnp.asarray(inputs, dtype), noise))
    return spikes, eligibility


def policy_gradient(eligibility: ReinforceParams, rewards: jax.Array,
                    baseline: jax.Array | float = 0.0) -> ReinforceParams:
    """REINFORCE's estimate of the gradient of the expected reward: the mean over the batch of
    `(rewards - baseline) * eligibility`, for each example's reward `rewards` `[B]`.

    Every eligibility has mean zero over the spikes it was drawn with, so a
    `baseline` that does not depend on those spikes (a running mean of the
    reward, say) leaves the estimate unbiased and can shrink its variance.
    Ascend it to raise the reward.
    """
    advantage = jnp.asarray(rewards) - baseline
    return ReinforceParams(*(jnp.einsum("b,b...->...", advantage, e) / advantage.shape[0]
                             for e in eligibility))
