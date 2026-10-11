"""Measure the Pallas racer step against the trained Vision's dense XLA float32 step.

Run only through the racer GPU queue, for example (from the worktree):

    DEW_GPU_LANE=racer ~/.cache/dew/dew-gpu-run --minutes 20 env PYTHONPATH=src \
        /home/mrwhite0racle/Desktop/dew/.venv/bin/python research/snn-ann/measure.py

The float64 NumPy reference is evaluated in the same queued correctness job, never as an unwrapped
CPU job. The track seed and the pre-step speed, rather than the post-step speed, reproduce the
network's actual input in race(). No separate device-only time is inferred from wall time.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "site/lab")]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import kernel  # noqa: E402
import numpy as np  # noqa: E402
import racer  # noqa: E402
from stack import act, at_rest  # noqa: E402

RUNS = Path("/mnt/local/dew/tmp/sparx-site/racer-phase2c/artifacts")
STEPS = 1500
TRACK_SEED = 1000
TEST_BOUNDS = {
    "A": {name: {"rms_rel": 5e-8, "max_rel": 1e-7} for name in kernel.NAMES},
    "B": {name: {"rms_rel": 5e-8, "max_rel": 1e-7 if name == "hidden" else 1.5e-7}
          for name in kernel.NAMES},
}


def load_arm(arm: str, runs: Path = RUNS):
    path = runs / ("7" if arm == "A" else "29") / "racer.json"
    if not path.exists() and arm == "A":
        path = ROOT / "site/public/racer/racer.json"
    model = json.loads(path.read_text())
    params, config = kernel.from_model(model)
    return model, params, config, path


def recorded_inputs(model: dict, steps: int = STEPS):
    net, params, world, ground = racer.from_json(model)
    params = jax.tree.map(lambda p: jnp.asarray(p, jnp.float32), params)
    ground = tuple(jnp.asarray(g, jnp.float32) for g in ground)
    with jax.enable_x64(new_val=False):
        points = racer.Tracks().draw(world, jax.random.key(TRACK_SEED), 1)
        car0 = racer.at_start(points)

    def drive(p):
        cars, _, on, off, spikes, *_ = racer.race(net, p, world, ground, points, car0, steps)
        # race() emits the car after drive(), but inputs() reads its speed before drive().
        before = jnp.concatenate([car0[None], cars[:-1]], axis=0)
        image = jnp.concatenate([on, off], axis=-1)
        return jnp.concatenate([image, before[:, :, 3:4] / world.max_speed], axis=-1)[:, 0], spikes

    # Set the input-producing drive to float32 too even when the reference enables x64 globally.
    with jax.enable_x64(new_val=False):
        inputs, activity = jax.jit(drive)(params)
    jax.block_until_ready(inputs)
    return inputs, net, params, float(jnp.mean(activity != 0))


def xla_rest(net, params):
    return jax.tree.map(lambda x: jnp.zeros(x.shape, jnp.float32), at_rest(net, params, 1, 4097))


def xla_state(carried) -> kernel.State:
    units = [jax.tree.leaves(carried[f"units_{k}"])[0].reshape(kernel.SHAPES[k]) for k in range(3)]
    readout = jax.tree.leaves(carried["LI_0"])[0].reshape(2)
    return kernel.State(*units, readout)


def xla_function(net):
    def step(p, carried, x):
        u, carried, _ = act(net, p, carried, x[None])
        return carried, u[0]

    return step


def trajectory(params, inputs, config, skip, interpret=False):
    def body(state, x):
        after, _, activity = kernel.step_with_activity(
            params, state, x, config=config, skip=skip, interpret=interpret,
        )
        return after, (after, tuple(a != 0 for a in activity))

    return jax.jit(lambda xs: jax.lax.scan(body, kernel.at_rest(), xs)[1])(inputs)


def xla_trajectory(net, params, inputs):
    def body(carried, x):
        _, after, spikes = act(net, params, carried, x[None])
        return after, (xla_state(after), spikes[0] != 0)

    states, spikes = jax.jit(lambda xs: jax.lax.scan(body, xla_rest(net, params), xs)[1])(inputs)
    shapes = kernel.SHAPES[:3]
    offsets = (0, 8192, 12288)
    activity = tuple(spikes[:, start:start + int(np.prod(shape))].reshape((len(inputs), *shape))
                     for start, shape in zip(offsets, shapes, strict=True))
    return states, activity


def reference_trajectory(params, inputs, config):
    state = kernel.State(*(np.zeros(shape, np.float64) for shape in kernel.SHAPES))
    states = [[] for _ in kernel.NAMES]
    activity = [[] for _ in range(3)]
    for x in inputs:
        state, _, outputs = kernel.reference_step(params, state, x, config)
        for dest, value in zip(states, state, strict=True):
            dest.append(value)
        for dest, value in zip(activity, outputs, strict=True):
            dest.append(value != 0)
    return kernel.State(*(np.stack(xs) for xs in states)), tuple(np.stack(xs) for xs in activity)


def distances(actual, reference) -> dict:
    """Relative L2 (RMS-relative) and L-infinity errors, normalized over the entire drive.

    A zero reference and nonzero error gives infinity, never an epsilon-adjusted denominator.
    """
    measured = {}
    for name, a, r in zip(kernel.NAMES, actual, reference, strict=True):
        a, r = np.asarray(a, np.float64), np.asarray(r, np.float64)
        diff = a - r
        norm, max_abs = float(np.linalg.norm(r)), float(np.max(np.abs(r)))
        error, max_error = float(np.linalg.norm(diff)), float(np.max(np.abs(diff)))
        measured[name] = {
            "rms_rel": error / norm if norm else (0.0 if not error else float("inf")),
            "max_rel": max_error / max_abs if max_abs else (0.0 if not max_error else float("inf")),
        }
    return measured


def spike_flips(actual, reference) -> dict:
    return {name: int(np.count_nonzero(np.asarray(a) != np.asarray(r)))
            for name, a, r in zip(kernel.NAMES[:3], actual, reference, strict=True)}


def sparsity(inputs, activity) -> dict:
    # The dense layer consumes conv1's output in contiguous 16-input blocks. The speed is its own tail.
    x = np.asarray(activity[1]).reshape((len(inputs), 4096))
    return {
        "input_zero_fraction": float(np.mean(np.asarray(inputs)[:, :4096] == 0)),
        "population_active_fraction": {
            name: float(np.mean(np.asarray(a) != 0))
            for name, a in zip(kernel.NAMES[:3], activity, strict=True)
        },
        "dense_zero_block_fraction_excluding_speed": float(np.mean(~np.any(x.reshape(-1, 16) != 0, -1))),
    }


def correctness(host_params, params, config, net, xla_params, inputs):
    reference, ref_activity = reference_trajectory(host_params, np.asarray(inputs), config)
    xla, xla_activity = xla_trajectory(net, xla_params, inputs)
    result = {
        "steps": len(inputs), "metric": "L2(error)/L2(reference), Linf(error)/Linf(reference)",
        "xla_f32_per_output": distances(xla, reference),
        "test_xla_f32_per_output": distances(tuple(a[:32] for a in xla), tuple(a[:32] for a in reference)),
        "xla_f32_vs_reference_flips": spike_flips(xla_activity, ref_activity),
        "test_inputs": {
            "steps": min(32, len(inputs)), "event_bitorder": "big",
            "events_packbits_base64": base64.b64encode(
                np.packbits(np.asarray(inputs[:32, :4096], np.uint8), axis=-1).tobytes(),
            ).decode(),
            "speed_float32": np.asarray(inputs[:32, 4096]).tolist(),
        },
        "variants": {},
    }
    for skip in (True, False):
        states, activity = trajectory(params, inputs, config, skip)
        result["variants"]["skip" if skip else "dense"] = {
            "per_output": distances(states, reference),
            "test_per_output": distances(tuple(a[:32] for a in states), tuple(a[:32] for a in reference)),
            "spike_flips_vs_reference": spike_flips(activity, ref_activity),
            "spike_flips_vs_xla_f32": spike_flips(activity, xla_activity),
        }
        print("correctness", "skip" if skip else "dense", json.dumps(result["variants"][
            "skip" if skip else "dense"]), flush=True)
    result["sparsity"] = sparsity(inputs, ref_activity)
    return result


def latency(step, params, initial, inputs, calls=2000, warmup=200, repeats=10):
    alone = jax.jit(step)
    # Slicing and transferring the next input are outside the timer for every implementation.
    samples = [inputs[t] for t in range(len(inputs))]
    jax.block_until_ready(samples)
    state = initial
    for t in range(warmup):
        state, readout = alone(params, state, samples[t % len(samples)])
    jax.block_until_ready(readout)
    times = []
    for t in range(calls):
        x = samples[t % len(samples)]
        start = time.perf_counter_ns()
        state, readout = alone(params, state, x)
        np.asarray(readout)
        times.append((time.perf_counter_ns() - start) / 1e6)

    def stream(p, xs, state):
        return jax.lax.scan(lambda carry, x: step(p, carry, x), state, xs)[1]

    scan = jax.jit(stream)
    jax.block_until_ready(scan(params, inputs, initial))
    runs = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        jax.block_until_ready(scan(params, inputs, initial))
        runs.append((time.perf_counter_ns() - start) / 1e6 / len(inputs))
    return {
        "calls": calls, "warmup": warmup, "step_ms": {
            "median": statistics.median(times), "p99": float(np.percentile(times, 99)), "min": min(times),
        },
        "scan_steps": len(inputs), "scan_runs": repeats,
        "stream_ms_per_step": statistics.median(runs), "stream_run_ms_per_step": runs,
        "device_ms_per_step": None,
    }


def profile_device(step, params, initial, inputs, directory, calls=100):
    """CUPTI kernel active time, excluding gaps and copies, not an end-to-end device timer."""
    call = jax.jit(step)
    samples = [inputs[t] for t in range(calls)]
    jax.block_until_ready(samples)
    state, readout = call(params, initial, samples[0])
    jax.block_until_ready(readout)
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 0
    options.python_tracer_level = 0
    with jax.profiler.trace(directory, profiler_options=options):
        for x in samples:
            state, readout = call(params, state, x)
        jax.block_until_ready(readout)
    traces = sorted(directory.rglob("*.trace.json.gz"))
    if not traces:
        return {"calls": calls, "unavailable": "The profiler exported no Chrome trace"}
    with gzip.open(traces[-1], "rt") as f:
        events = json.load(f)["traceEvents"]
    gpu_pids = {e["pid"] for e in events if e.get("name") == "process_name"
                and "GPU" in e.get("args", {}).get("name", "")}
    gpu_events = [e for e in events if e.get("pid") in gpu_pids and e.get("ph") == "X" and "dur" in e]
    copy_events = [e for e in gpu_events if e.get("name", "").lower().startswith(("memcpy", "memset"))]
    kernels = [e for e in gpu_events if not e.get("name", "").lower().startswith(("memcpy", "memset"))]
    if not kernels:
        return {"calls": calls, "unavailable": "The profiler recorded no GPU kernel events"}
    total = sum(e["dur"] for e in kernels)
    by_kernel = {}
    for e in kernels:
        name = e["name"]
        entry = by_kernel.setdefault(name, {"events": 0, "total_us": 0.0})
        entry["events"] += 1
        entry["total_us"] += e["dur"]
    return {"calls": calls, "source": "jax.profiler Chrome trace / CUPTI",
            "definition": "sum of GPU kernel durations / calls; excludes idle gaps and copies",
            "kernel_active_ms_per_step": total / calls / 1000,
            "gpu_kernel_events": len(kernels), "kernels": by_kernel,
            "copy_events": len(copy_events),
            "copy_active_ms_per_step": sum(e["dur"] for e in copy_events) / calls / 1000,
            "observed_gpu_events_per_step": len(gpu_events) / calls}


def smoke(params, config, net, xla_params, inputs):
    for skip in (True, False):
        call = jax.jit(partial(kernel.step, config=config, skip=skip))
        compiled = call.lower(params, kernel.at_rest(), inputs[0]).compile()
        state, readout = compiled(params, kernel.at_rest(), inputs[0])
        jax.block_until_ready(readout)
        print("lowered", skip, [a.shape for a in state], np.asarray(readout).tolist(), flush=True)
    print("xla state", jax.tree.map(lambda a: (a.shape, str(a.dtype)), xla_rest(net, xla_params)), flush=True)


def measure_arm(arm, args):
    model, host_params, config, path = load_arm(arm, args.runs)
    params = jax.tree.map(jnp.asarray, host_params)
    inputs, net, xla_params, activity = recorded_inputs(model, args.steps)
    report = {
        "run": path.parent.name, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "inputs_sha256": hashlib.sha256(np.asarray(inputs).tobytes()).hexdigest(),
        "units_active": activity,
    }
    if args.mode == "smoke":
        smoke(params, config, net, xla_params, inputs)
        return report
    if args.mode in ("all", "correctness"):
        report["correctness"] = correctness(host_params, params, config, net, xla_params, inputs)
    if args.mode in ("all", "latency"):
        report["latency"] = {}
        for skip in (True, False):
            name = "skip" if skip else "dense"
            fn = partial(kernel.step, config=config, skip=skip)
            report["latency"][name] = latency(fn, params, kernel.at_rest(), inputs,
                                             args.calls, args.warmup, args.repeats)
            print(arm, name, json.dumps(report["latency"][name]), flush=True)
        report["latency"]["xla"] = latency(xla_function(net), xla_params, xla_rest(net, xla_params), inputs,
                                           args.calls, args.warmup, args.repeats)
        print(arm, "xla", json.dumps(report["latency"]["xla"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results/kernel-rtx4080.json")
    parser.add_argument("--mode", choices=("all", "smoke", "correctness", "latency", "profile", "verify"),
                        default="all")
    parser.add_argument("--arms", nargs="+", choices=("A", "B"), default=("A", "B"))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--calls", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", bool(1))
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError("Measurement requires the queued GPU, not a CPU backend")
    if args.mode == "profile":
        add_profiles(args)
        return
    if args.mode == "verify":
        verify_report(args)
        return
    report = {
        "device": str(device), "device_kind": device.device_kind, "jax": jax.__version__,
        **hardware_metadata(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""), "track_seed": TRACK_SEED,
        "input_speed": "pre-step car speed, as passed by racer.inputs inside racer.race",
        "kernel": {
            "backend": "Pallas/Triton",
            "fused_parts": ["conv0 + unit", "conv1 + unit", "split-K dense partial sums",
                            "dense final reduction + unit + readout"],
            "launches_per_step": 4, "state_dtype": "float32", "weight_dtype": "float32",
            "launch_count_scope": "Pallas compute launches. The standalone compiled return also has "
                                  "an XLA readout-copy kernel; CUPTI records it separately.",
            "accumulator_dtype": "float64", "dense_input_block": kernel.DENSE_BLOCK,
            "dense_splits": kernel.DENSE_SPLITS,
            "could_not_do": "A single launch needs a grid-wide barrier between convolutional populations, "
                            "which the Triton backend does not expose. A three-launch version put the "
                            "entire dense/readout in one CTA and serialized its weight reads; the fourth "
                            "launch parallelizes the dense reduction across 64 CTAs.",
        },
        "timing_note": "Host input indexing is outside the timer; readout transfer is inside. Scan time "
                       "is synchronized wall time, not a separately profiled device-only time.",
        "method": {"calls": args.calls, "warmup": args.warmup, "scan_steps": args.steps,
                   "scan_runs": args.repeats, "mode": args.mode},
        "test_bounds": TEST_BOUNDS, "test_bounds_steps": 32,
        "arms": {},
    }
    for arm in args.arms:
        report["arms"][arm] = measure_arm(arm, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("wrote", args.out, flush=True)


def hardware_metadata():
    hardware = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=compute_cap,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip().split(", ")
    return {"compute_capability": hardware[0], "driver_version": hardware[1]}


def verify_report(args):
    """Attach queued test results and fixed bounds without rerunning or replacing the latency sample."""
    report = json.loads(args.out.read_text())
    command = [sys.executable, "-m", "pytest", "-q", "research/snn-ann/test_kernel.py"]
    env = {**os.environ, "JAX_PLATFORMS": "cuda,cpu", "SPARX_RACER_RUNS": str(args.runs)}
    run = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, check=False)
    output = run.stdout + run.stderr
    print(output, flush=True)
    if run.returncode:
        raise RuntimeError("The fixed-bound CPU interpretation and GPU tests failed")
    matched = re.search(r"(\d+) passed.* in ([\d.]+)s", output)
    if matched is None:
        raise RuntimeError("The test output does not contain a passed-test summary")
    report.update(hardware_metadata())
    report["test_bounds"] = TEST_BOUNDS
    report["test_bounds_steps"] = 32
    report["verification"] = {"command": command, "jax_platforms": "cuda,cpu", "exit_code": run.returncode,
                              "passed": int(matched[1]), "pytest_seconds": float(matched[2])}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def add_profiles(args):
    report = json.loads(args.out.read_text())
    if report["jax"] != jax.__version__ or report["xla_flags"] != os.environ.get("XLA_FLAGS", ""):
        raise RuntimeError("Profile and latency must use the same JAX version and XLA_FLAGS")
    for arm in args.arms:
        model, host_params, config, _ = load_arm(arm, args.runs)
        params = jax.tree.map(jnp.asarray, host_params)
        inputs, net, xla_params, _ = recorded_inputs(model, args.steps)
        for name in ("skip", "dense", "xla"):
            fn = (xla_function(net) if name == "xla"
                  else partial(kernel.step, config=config, skip=name == "skip"))
            p = xla_params if name == "xla" else params
            initial = xla_rest(net, xla_params) if name == "xla" else kernel.at_rest()
            directory = args.out.parent / "profiles" / f"{arm}-{name}"
            measured = profile_device(fn, p, initial, inputs, directory)
            report["arms"][arm]["latency"][name]["device_profile"] = measured
            print(arm, name, "profile", json.dumps(measured), flush=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
