"""One training step of each racer arm on this GPU: compile time, step time and peak memory.

Batch 32, 300 steps of driving with gradients cut every 50, as the conv arms would train. On the
workstation it runs only through the shared GPU queue:

    ~/.cache/dew/dew-gpu-run <python with jax[cuda]> research/racer/timing.py
"""
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

print(jax.devices(), flush=True)
arms = {
    "A spiking conv on events": ("conv", "lif", "events", 64, 32, 2),
    "B graded conv on events": ("conv", "relu", "events", 64, 32, 2),
    "C graded conv on frames": ("conv", "relu", "frames", 64, 32, 2),
    "dense stack on 24x12 events": ("dense", "lif", "events", 24, 12, 1),
}
for name, (kind, neuron, sees, columns, rows, n) in arms.items():
    world = racer.World(columns=columns, rows=rows, supersample=n, sees=sees)
    ground = tuple(jnp.asarray(g, jnp.float32) for g in world.ground())
    net = racer.vision(world, neuron) if kind == "conv" else racer.network()
    params = net.init(jax.random.key(0), jnp.zeros((1, 1, world.features)))["params"]
    tracks = racer.Tracks(bend=1.3, floor=0.35)

    def loss(params, key, net=net, world=world, ground=ground, tracks=tracks):
        k1, k2 = jax.random.split(key)
        points = tracks.draw(world, k1, 32)
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
    print(json.dumps({"arm": name, "compile_s": round(compiled, 1), "step_s": sorted(times)[2],
                      "peak_gb": round(stats.get("peak_bytes_in_use", 0) / 2**30, 2),
                      "loss": float(value), "grad_norm": float(norm)}), flush=True)
