"""recipes/snn/train.py trains, records run.json, and its run loads back through dew.pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


SNN_DELAYS_FLAGS = [
    "--data.binning", "events", "--readout", "softmax_sum", "--schedule-every", "2",
    "--model.config", json.dumps({"hidden": [16], "classes": 2, "delays": [3, 3], "extend": True,
                                  "batch_norm": True, "use_bias": False, "dropout_mask": "sequence",
                                  "neuron": {"name": "lif", "fields": {"tau": 3.0}}}),
    "--deployed", "sigma", "0",
    "--groups", json.dumps({"delays": {"patterns": ["*/delay"], "bounds": [0, 3],
                                       "learning_rate": {"name": "cosine", "fields": {"peak": 0.1,
                                                                                     "warmup_steps": 0}}}}),
]


@pytest.mark.parametrize("delayed", [False, True])
def test_the_recipe_trains_and_its_run_loads_through_dew_pipeline(tmp_path, delayed):
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    recipe = [sys.executable, str(ROOT / "recipes/snn/train.py")]
    width = {"sigma": {"name": "linear", "fields": {"peak": 1.5, "end": 0.5}}}
    schedules = ["--schedules", json.dumps(width)]
    runs = ["--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", "toy"]
    if delayed:
        # The documented form: the dataset named as a subcommand, every setting a flag.
        pytest.importorskip("h5py")
        from sparx.datasets import write_synthetic_shd
        write_synthetic_shd(tmp_path, records=64)
        command = [*recipe, "data:shd", "--data.cache", str(tmp_path), "--data.steps", "20",
                   "--data.channels", "70", "--data.loading.workers", "0", "--data.loading.threads", "1",
                   "--data.loading.read-buffer", "1", *SNN_DELAYS_FLAGS, *schedules,
                   "--trainer.batch-size", "16", "--trainer.steps", "8", "--trainer.log-every", "4",
                   "--trainer.eval-every", "8", "--trainer.checkpoint-every", "8",
                   "--trainer.multi-host", "False", *runs]
    else:
        model = {"delays": 3, "neuron": {"name": "lif", "fields": {"tau": 3.0}}}
        command = [*recipe, "--smoke", "--model.config", json.dumps(model), *schedules, *runs]
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
    # The width schedule ends at 0.5, which the loaded task runs with unless the rounded delays are deployed.
    assert loaded.stdout.strip() == f"SpikingClassification {{'sigma': {0.0 if delayed else 0.5}}}"
    if delayed:
        assert recorded["groups"]["delays"]["learning_rate"]["name"] == "cosine"
