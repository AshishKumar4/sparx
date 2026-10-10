"""One training step of each racer arm on this GPU: compile time, step time and peak memory.

Batch 32 (or each of `--batch`), 300 steps of driving with gradients cut every 50, as the conv arms
would train. On the workstation it runs only through the shared GPU queue:

    ~/.cache/dew/dew-gpu-run <python with jax[cuda]> research/racer/timing.py [--batch 32 160] [--arms A B]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "site/lab")]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
import racer  # noqa: E402
import training  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, nargs="+", default=[32])
parser.add_argument("--arms", nargs="+", default=None, help="the arms' keys, all when absent")
options = parser.parse_args()
print(jax.devices(), flush=True)
arms = {
    "A": ("spiking conv on events", "conv", "lif", 1, "events"),
    "B": ("graded conv on events", "conv", "relu", 1, "events"),
    "C": ("graded conv on frames", "conv", "relu", 1, "frames"),
    "A2": ("2-bit spiking conv on events", "conv", "lif", 2, "events"),
    "A4": ("4-bit spiking conv on events", "conv", "lif", 4, "events"),
    "S": ("sigma-delta conv on events", "conv", "sigma-delta", 1, "events"),
    "D": ("spiking conv with dendrites on events", "conv", "dendritic", 1, "events"),
    "dense": ("dense stack on 24x12 events", "dense", "lif", 1, "events"),
}
for (key, (name, kind, neuron, bits, sees)), batch in itertools.product(arms.items(), options.batch):
    if options.arms and key not in options.arms:
        continue
    columns, rows, n = (64, 32, 2) if kind == "conv" else (24, 12, 1)
    world = racer.World(columns=columns, rows=rows, supersample=n, sees=sees)
    ground = tuple(jnp.asarray(g, jnp.float32) for g in world.ground())
    net = racer.vision(world, neuron, bits=bits) if kind == "conv" else racer.network()
    params = net.init(jax.random.key(0), jnp.zeros((1, 1, world.features)))["params"]
    tracks = racer.Tracks(bend=1.3, floor=0.35)

    def loss(params, key, net=net, world=world, ground=ground, tracks=tracks, batch=batch):
        k1, k2 = jax.random.split(key)
        points = tracks.draw(world, k1, batch)
        out = racer.race(net, params, world, ground, points, racer.starts(world, k2, points), 300, 50)
        return racer.cost(world, out, (0.02, 0.3), 2.0)

    tx = training.optimizer(1e-3, 3000)
    state = tx.init(params)

    @jax.jit
    def update(params, state, key, loss=loss, tx=tx):
        (value, _), grads = jax.value_and_grad(loss, has_aux=True)(params, key)
        updates, state = tx.update(grads, state, params)
        return optax.apply_updates(params, updates), state, value, optax.global_norm(grads)

    start = time.time()
    params, state, value, norm = jax.block_until_ready(update(params, state, jax.random.key(1)))
    compiled = time.time() - start
    times = []
    for k in range(5):
        start = time.time()
        params, state, value, norm = jax.block_until_ready(update(params, state, jax.random.key(2 + k)))
        times.append(time.time() - start)
    stats = jax.devices()[0].memory_stats() or {}
    print(json.dumps({"arm": f"{key}: {name}", "batch": batch, "compile_s": round(compiled, 1),
                      "step_s": round(sorted(times)[2], 3),
                      "peak_gb": round(stats.get("peak_bytes_in_use", 0) / 2**30, 2),
                      "loss": float(value), "grad_norm": float(norm)}), flush=True)
