"""Batch-1 latency of each racer arm's network on this machine's accelerator: the time from a step's input
to its readout, as a controller that acts every 20 ms would see it.

Each arm is the Phase 2 run that drove farthest on its held-out tracks (`research/racer/results/
phase2c-arms.json` names the seeds), loaded from its saved weights. Two timings each:

- one step: the step compiled alone and called once per input, with its state carried between calls,
  timed from the call to the readout on the host, over 2,000 calls after 200 of warm-up;
- a stream: 1,500 steps in one compiled scan, the per-step time when inputs arrive in a block.

The inputs are a recorded drive's own (`race` on one unseen track), so each arm sees the activity it
produced; the arm on frames is left out, since a drive records events, not frames. XLA's CUDA command
buffers are on by default; run again with `XLA_FLAGS=--xla_gpu_enable_command_buffer=` for them off. The
report records the flags. On the workstation it runs only through the shared GPU queue:

    ~/.cache/dew/dew-gpu-run --short <python with jax[cuda]> research/snn-ann/latency.py <runs dir> <out.json>
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "site/lab")]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import racer  # noqa: E402
from stack import act, at_rest  # noqa: E402


def best_runs(runs: Path) -> dict[str, Path]:
    """Each arm's run whose kept parameters scored highest on held-out tracks."""
    chosen: dict[str, tuple[float, Path]] = {}
    for model in sorted(runs.glob("*/racer.json")):
        meta = json.loads(model.read_text())["meta"]
        trained = meta["trained"]
        key = (trained["neuron"], trained["bits"], trained["sees"])
        name = {("lif", 1, "events"): "A, spiking", ("relu", 1, "events"): "B, graded, events",
                ("relu", 1, "frames"): "C, graded, frames", ("sigma-delta", 1, "events"): "sigma-delta",
                ("dendritic", 1, "events"): "dendritic"}.get(key)
        if name is None:
            continue
        score = max(row["score"] for row in meta["history"] if "score" in row)
        if name not in chosen or score > chosen[name][0]:
            chosen[name] = (score, model)
    return {name: path for name, (_, path) in chosen.items()}


def measure(model: dict) -> dict:
    net, params, world, ground = racer.from_json(model)
    ground = tuple(jnp.asarray(g) for g in ground)
    points = racer.Tracks().draw(world, jax.random.key(1000), 1)
    car0 = racer.at_start(points)
    steps = 1500
    # The drive's own inputs, rebuilt from its events and speed.
    drive = jax.jit(lambda p: racer.race(net, p, world, ground, points, car0, steps))
    cars, _, on, off, spikes, *_ = drive(params)
    image = jnp.concatenate([on, off], -1)
    # `race` returns each step's car after it moved; the network read the speed before the move.
    before = jnp.concatenate([car0[None], cars[:-1]])
    speed = before[:, :, 3:4] / world.max_speed
    inputs = jnp.concatenate([image, speed], -1)  # [T, 1, F]
    activity = float(jnp.mean(spikes != 0))

    step = jax.jit(lambda p, carried, x: act(net, p, carried, x)[:2])
    carried = at_rest(net, params, 1, world.features)
    for t in range(200):
        u, carried = step(params, carried, inputs[t % steps])
    jax.block_until_ready(u)
    times = []
    for t in range(2000):
        x = inputs[t % steps]
        start = time.perf_counter()
        u, carried = step(params, carried, x)
        np.asarray(u)  # the readout on the host, as the car's controller needs it
        times.append(time.perf_counter() - start)

    def stream(p, x):
        def body(carried, xt):
            u, carried = act(net, p, carried, xt)[:2]
            return carried, u
        return jax.lax.scan(body, at_rest(net, p, 1, world.features), x)[1]

    scan = jax.jit(stream)
    jax.block_until_ready(scan(params, inputs))
    runs = []
    for _ in range(10):
        start = time.perf_counter()
        jax.block_until_ready(scan(params, inputs))
        runs.append((time.perf_counter() - start) / steps)
    ms = [1000 * t for t in times]
    return {
        "units_active": activity,
        "step_ms": {"median": statistics.median(ms), "p99": float(np.percentile(ms, 99)), "min": min(ms)},
        "stream_ms_per_step": 1000 * statistics.median(runs),
    }


def main() -> None:
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    report = {"device": str(jax.devices()[0]), "jax": jax.__version__,
              "xla_flags": os.environ.get("XLA_FLAGS", ""), "arms": {}}
    for name, path in best_runs(runs).items():
        model = json.loads(path.read_text())
        if model["world"].get("sees") == "frames":
            continue
        report["arms"][name] = {"run": str(path.parent.name), **measure(model)}
        print(name, json.dumps(report["arms"][name]), flush=True)
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
