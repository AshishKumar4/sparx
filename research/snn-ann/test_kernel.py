"""Real-shape, real-input correctness, including CPU Pallas interpretation and GPU lowering.

On the workstation this runs only in the GPU wrapper, with JAX_PLATFORMS=cuda,cpu so the CPU
interpretation backend is available too. Events and pre-step speeds are the first 32 steps recorded
in kernel-rtx4080.json, not a random sparsity fixture. Arm B requires its trained run 29; arm A's run
7 is also the site's checked-in racer. SPARX_RACER_RUNS can name a copy of the training artifacts.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import kernel
import measure
import numpy as np
import pytest

REPORT = Path(__file__).parent / "results/kernel-rtx4080.json"

# Fixed before testing, and never increased to make a run pass. Each pair is (RMS-relative,
# max-relative). Every bound is below both XLA float32's 32-step and 1,500-step measured distance.
# XLA float32, 1,500 steps, RTX 4080, relative L2 / Linf:
# A conv0 1.2092180187874226e-7 / 2.367537547538301e-7;
#   conv1 1.1835679551541949e-7 / 2.6207548211662966e-7;
#   hidden 1.0077047215553178e-7 / 1.560636437548826e-7;
#   readout 6.88496379916845e-8 / 1.9039123377840745e-7.
# B conv0 1.1660798099118791e-7 / 2.623197910659325e-7;
#   conv1 1.9665406413968595e-7 / 3.1966992669648105e-7;
#   hidden 2.810289941087901e-7 / 3.9371092386243334e-7;
#   readout 2.4198716849769306e-7 / 3.548737194010247e-7.
# The Pallas 32-step distances are about 3.4e-8 to 3.9e-8 RMS and 3.8e-8 to 7.3e-8 max,
# with no spike/active-unit mismatches. Full-drive max errors remain below the same bounds.
BOUNDS = {
    "A": dict.fromkeys(kernel.NAMES, (5e-8, 1e-7)),
    "B": {"conv0": (5e-8, 1.5e-7), "conv1": (5e-8, 1.5e-7),
          "hidden": (5e-8, 1e-7), "readout": (5e-8, 1.5e-7)},
}


def gpu_device():
    """The GPU, or None on a machine without one, where the GPU tests skip."""
    gpus = [device for device in jax.devices() if device.platform == "gpu"]
    return gpus[0] if gpus else None


@pytest.fixture(scope="module", params=("A", "B"))
def arm(request):
    jax.config.update("jax_enable_x64", bool(1))
    name = request.param
    runs = Path(os.environ.get("SPARX_RACER_RUNS", str(measure.RUNS)))
    if name == "B" and not (runs / "29/racer.json").exists():
        pytest.skip("The trained graded arm's run 29 is required")
    model, params, config, path = measure.load_arm(name, runs)
    report = json.loads(REPORT.read_text())["arms"][name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["model_sha256"]
    recorded = report["correctness"]["test_inputs"]
    assert recorded["steps"] == 32
    events = np.frombuffer(base64.b64decode(recorded["events_packbits_base64"]), np.uint8)
    events = np.unpackbits(events.reshape(32, 512), axis=1, bitorder=recorded["event_bitorder"])
    speeds = np.asarray(recorded["speed_float32"], np.float32)
    inputs = np.concatenate([events.astype(np.float32), speeds[:, None]], axis=1)
    reference, activity = measure.reference_trajectory(params, inputs, config)
    return {"name": name, "model": model, "params": params, "config": config,
            "inputs": inputs, "reference": reference, "activity": activity,
            "reported": report["correctness"]}


def assert_distance(name, states, reference):
    measured = measure.distances(states, reference)
    for population, (rms, maximum) in BOUNDS[name].items():
        assert measured[population]["rms_rel"] <= rms, (name, population, measured[population])
        assert measured[population]["max_rel"] <= maximum, (name, population, measured[population])


@pytest.mark.parametrize("skip", [True, False], ids=["skip", "dense"])
@pytest.mark.parametrize("backend", ["cpu-interpret", "gpu"])
def test_real_drive(arm, skip, backend):
    device = jax.devices("cpu")[0] if backend == "cpu-interpret" else gpu_device()
    if device is None:
        pytest.skip("No GPU present")
    with jax.default_device(device):
        params = jax.device_put(arm["params"], device)
        inputs = jax.device_put(arm["inputs"], device)
        states, activity = measure.trajectory(params, inputs, arm["config"], skip,
                                              interpret=backend == "cpu-interpret")
        jax.block_until_ready(states)
    assert_distance(arm["name"], states, arm["reference"])
    assert measure.spike_flips(activity, arm["activity"]) == dict.fromkeys(kernel.NAMES[:3], 0)


def test_bounds_are_tighter_than_measured_xla(arm):
    for output, (rms, maximum) in BOUNDS[arm["name"]].items():
        assert measure.TEST_BOUNDS[arm["name"]][output] == {"rms_rel": rms, "max_rel": maximum}
        for field in ("xla_f32_per_output", "test_xla_f32_per_output"):
            assert rms <= arm["reported"][field][output]["rms_rel"]
            assert maximum <= arm["reported"][field][output]["max_rel"]


@pytest.mark.parametrize("skip", [True, False], ids=["skip", "dense"])
def test_gpu_lowering_and_spikes_against_xla(arm, skip):
    device = gpu_device()
    if device is None:
        pytest.skip("No GPU present")
    with jax.default_device(device):
        params = jax.device_put(arm["params"], device)
        inputs = jax.device_put(arm["inputs"], device)
        call = jax.jit(partial(kernel.step, config=arm["config"], skip=skip))
        compiled = call.lower(params, kernel.at_rest(), inputs[0]).compile()
        after, readout = compiled(params, kernel.at_rest(), inputs[0])
        jax.block_until_ready(readout)
        assert tuple(a.shape for a in after) == kernel.SHAPES
        assert all(a.dtype == jnp.float32 for a in after)
        net, xla_params, _, _ = measure.racer.from_json(arm["model"])
        xla_params = jax.tree.map(lambda a: jnp.asarray(a, jnp.float32), xla_params)
        _, xla_activity = measure.xla_trajectory(net, xla_params, inputs)
        _, activity = measure.trajectory(params, inputs, arm["config"], skip)
        assert measure.spike_flips(activity, xla_activity) == dict.fromkeys(kernel.NAMES[:3], 0)


def test_zero_blocks_do_not_skip_membrane_decay(arm):
    # A completely zero drive still leaks the carried state. Graded negative membranes are retained,
    # not reset to ReLU's output. Make all synaptic weights/biases zero to isolate these updates.
    device = jax.devices("cpu")[0]
    with jax.default_device(device):
        params = kernel.Parameters(*(jnp.zeros(p.shape, jnp.float32) for p in arm["params"]))
        state = kernel.State(*(jnp.full(shape, -0.25, jnp.float32) for shape in kernel.SHAPES))
        config = dataclasses.replace(arm["config"], graded=True)
        call = jax.jit(partial(kernel.step, config=config, skip=True, interpret=True))
        after, _ = call(params, state, jnp.zeros(4097, jnp.float32))
    for value in after[:3]:
        np.testing.assert_array_equal(value, np.full(value.shape, np.float32(-0.25 * config.decay)))
    np.testing.assert_array_equal(after.readout, np.full(2, np.float32(-0.25 * config.readout_decay)))
