"""The training loop pilot.py and racer.py share.

Dew builds the optimizer: Adam on a warmup-cosine schedule from 0 to `lr` and
down to `lr / 20`, warming up over a tenth of the run or 100 steps, whichever
is fewer, with each gradient clipped to norm 1. The loop minimizes
`loss(params, key, progress) -> (loss, aux)` on a fresh key each step, where
`progress` runs from 0 to 1 over the run (for a curriculum), and logs every
`log_every` steps. Given a `score`, higher being better, it scores the
parameters every `score_every` steps and returns the best it scored rather
than the last: a choice made on data apart from the test's.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from dew.config import OptimConfig
from dew.training.optim import Cosine


def optimizer(lr: float, steps: int) -> optax.GradientTransformation:
    schedule = Cosine(peak=lr, warmup_steps=min(100, steps // 10), end=lr / 20)
    return OptimConfig(optimizer="adam", schedule=schedule, clip_grads=1.0).build(steps)


def train(loss: Callable, params, *, lr: float, steps: int, seed: int, log_every: int,
          score: Callable | None = None, score_every: int = 0):
    """Returns the trained parameters and the logged rows."""
    tx = optimizer(lr, steps)
    state = tx.init(params)

    @jax.jit
    def update(params, state, key, progress):
        (value, aux), grads = jax.value_and_grad(loss, has_aux=True)(params, key, progress)
        updates, state = tx.update(grads, state, params)
        return optax.apply_updates(params, updates), state, value, aux, optax.global_norm(grads)

    history, start = [], time.time()
    best, best_score = params, -float("inf")
    key = jax.random.key(seed)
    for step in range(1, steps + 1):
        key, sub = jax.random.split(key)
        params, state, value, aux, norm = update(params, state, sub, jnp.float32((step - 1) / steps))
        scored = score is not None and (step % score_every == 0 or step == steps)
        if step % log_every == 0 or step == 1 or scored:
            row = {"step": step, "loss": float(value), "grad_norm": float(norm),
                   "seconds": round(time.time() - start, 1)}
            row |= {name: float(v) for name, v in aux.items()}
            if scored:
                row["score"] = float(score(params))
                if row["score"] > best_score:
                    best, best_score = params, row["score"]
            history.append(row)
            print(json.dumps(row), flush=True)
    return (best if score is not None else params), history
