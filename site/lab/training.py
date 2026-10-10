"""The training loop pilot.py and racer.py share.

Dew builds the optimizer: Adam on a warmup-cosine schedule from 0 to `lr` and
down to `lr / 20`, warming up over a tenth of the run or 100 steps, whichever
is fewer, with each gradient clipped to norm 1. The loop minimizes
`loss(params, key) -> (loss, aux)` on a fresh key each step and logs every
`log_every` steps.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import jax
import optax
from dew.config import OptimConfig
from dew.training.optim import Cosine


def optimizer(lr: float, steps: int) -> optax.GradientTransformation:
    schedule = Cosine(peak=lr, warmup_steps=min(100, steps // 10), end=lr / 20)
    return OptimConfig(optimizer="adam", schedule=schedule, clip_grads=1.0).build(steps)


def train(loss: Callable, params, *, lr: float, steps: int, seed: int, log_every: int):
    """Returns the trained parameters and the logged rows."""
    tx = optimizer(lr, steps)
    state = tx.init(params)

    @jax.jit
    def update(params, state, key):
        (value, aux), grads = jax.value_and_grad(loss, has_aux=True)(params, key)
        updates, state = tx.update(grads, state, params)
        return optax.apply_updates(params, updates), state, value, aux, optax.global_norm(grads)

    history, start = [], time.time()
    key = jax.random.key(seed)
    for step in range(1, steps + 1):
        key, sub = jax.random.split(key)
        params, state, value, aux, norm = update(params, state, sub)
        if step % log_every == 0 or step == 1:
            row = {"step": step, "loss": float(value), "grad_norm": float(norm),
                   "seconds": round(time.time() - start, 1)}
            row |= {name: float(v) for name, v in aux.items()}
            history.append(row)
            print(json.dumps(row), flush=True)
    return params, history
