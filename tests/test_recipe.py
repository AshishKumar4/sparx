"""recipes/snn/train.py trains, records run.json, and its run loads back through dew.pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fake_shd(cache: Path, seed: int, records: int) -> None:
    """Two SHD-layout files whose classes differ by which half of the channels fire."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(seed)
    for split in ("train", "test"):
        labels = rng.integers(0, 2, records).astype(np.uint16)
        with h5py.File(cache / f"shd_{split}.h5", "w") as file:
            times = file.create_dataset("spikes/times", (records,), dtype=h5py.vlen_dtype(np.float32))
            units = file.create_dataset("spikes/units", (records,), dtype=h5py.vlen_dtype(np.uint16))
            for i, label in enumerate(labels):
                times[i] = np.sort(rng.uniform(0, 1.4, 60)).astype(np.float32)
                units[i] = (rng.integers(0, 350, 60) + 350 * int(label)).astype(np.uint16)
            file.create_dataset("labels", data=labels)


def test_the_recipe_trains_and_its_run_loads_through_dew_pipeline(tmp_path):
    _fake_shd(tmp_path, 0, 64)
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    command = [sys.executable, str(ROOT / "recipes/snn/train.py"),
               "--data.cache", str(tmp_path), "--data.steps", "20", "--data.channels", "70",
               "--data.loading.workers", "0", "--data.loading.threads", "1",
               "--data.loading.read-buffer", "1",
               "--model.config", json.dumps({"hidden": [16], "classes": 2, "delays": 3,
                                             "neuron": {"name": "lif", "fields": {"tau": 3.0}}}),
               "--schedules", json.dumps({"sigma": {"name": "linear", "fields": {"peak": 1.5, "end": 0.5}}}),
               "--trainer.batch-size", "16", "--trainer.steps", "8", "--trainer.log-every", "4",
               "--trainer.eval-every", "8", "--trainer.checkpoint-every", "8",
               "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", "toy",
               "--trainer.multi-host", "False"]
    done = subprocess.run(command, capture_output=True, text=True, env=env, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    run = tmp_path / "runs" / "toy"
    recorded = json.loads((run / "run.json").read_text())
    assert recorded["objective"] == "spiking_classifier"
    assert recorded["model"]["config"]["neuron"]["name"] == "lif"
    program = ("import dew\n"
               f"task = dew.pipeline({str(run)!r})\n"
               "print(type(task).__name__, task.call)\n")
    loaded = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env,
                            timeout=600)
    assert loaded.returncode == 0, loaded.stderr[-3000:]
    # The width schedule ends at 0.5, which the loaded task runs with.
    assert loaded.stdout.strip() == "SpikingClassification {'sigma': 0.5}"
