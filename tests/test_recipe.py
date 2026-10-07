"""recipes/snn/train.py trains, records run.json, and its run loads back through dew in a fresh process."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

LIF = {"class": "sparx.nn.neurons:LIF", "fields": {"tau": 3.0}}

SNN_DELAYS_FLAGS = [
    "--data.binning", "events", "--objective.readout", "softmax_sum", "--objective.deployed", '{"sigma": 0}',
    "--model.hidden", "16", "--model.classes", "2", "--model.delays", "3", "3", "--model.extend",
    "--model.batch-norm", "--model.no-use-bias", "--model.dropout-mask", "sequence",
    "--model.neuron", json.dumps(LIF),
    "--optim.optimizer", "adam", "--optim.param-groups", json.dumps([
        {"name": "delays", "patterns": ["*/delay"], "bounds": [0, 3],
         "schedule": {"class": "cosine", "fields": {"peak": 0.1, "warmup_steps": 0}}},
        {"name": "rest", "patterns": ["*"]}]),
]


@pytest.mark.parametrize("delayed", [False, True])
def test_the_recipe_trains_and_its_run_loads_through_dew(tmp_path, delayed):
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    recipe = [sys.executable, str(ROOT / "recipes/snn/train.py")]
    # A width shrinking once every 2 steps.
    width = {"sigma": {"class": "linear", "fields": {"peak": 1.5, "end": 0.5, "every": 2}}}
    schedules = ["--objective.schedules", json.dumps(width)]
    runs = ["--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", "toy"]
    if delayed:
        # Every setting a flag, SNN-delays' among them.
        pytest.importorskip("h5py")
        from sparx.datasets import write_synthetic_shd
        write_synthetic_shd(tmp_path, records=64)
        command = [*recipe, "--data.cache", str(tmp_path), "--data.steps", "20", "--data.channels", "70",
                   "--data.loading.workers", "0", "--data.loading.threads", "1", "--data.loading.read-buffer",
                   "1", *SNN_DELAYS_FLAGS, *schedules, "--trainer.batch-size", "16", "--trainer.steps", "8",
                   "--trainer.log-every", "4", "--trainer.eval-every", "8", "--trainer.checkpoint-every", "8",
                   "--trainer.multi-host", "False", *runs]
    else:
        command = [*recipe, "--smoke", "--model.delays", "3", "--model.neuron", json.dumps(LIF), *schedules,
                   *runs]
    done = subprocess.run(command, capture_output=True, text=True, env=env, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    run = tmp_path / "runs" / "toy"
    recorded = json.loads((run / "run.json").read_text())
    assert recorded["class"] == "sparx.config:SNNRunConfig"
    fields = recorded["fields"]
    assert fields["objective"]["name"] == "sparx.objectives:SpikingClassifierObjective"
    assert fields["model"]["fields"]["neuron"]["class"] == "sparx.nn.neurons:LIF"
    assert not fields["smoke"]
    if delayed:
        assert fields["optim"]["param_groups"][0]["schedule"]["class"] == "dew.training.optim:Cosine"
    # A fresh process trusts sparx and reads the run back: the classifier it trained, and the run itself.
    program = ("import json\n"
               "import dew\n"
               "from dew.config import RunConfig\n"
               f"task = dew.pipeline({str(run)!r}, trust=('sparx',))\n"
               f"config = RunConfig.load({str(run)!r}, trust=('sparx',))\n"
               "print(type(task).__name__, json.dumps(task.call), type(config).__name__,\n"
               f"      config.record() == json.loads(open({str(run / 'run.json')!r}).read()))\n")
    loaded = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env,
                            timeout=600)
    assert loaded.returncode == 0, loaded.stderr[-3000:]
    # The width schedule ends at 0.5, which the loaded task runs with unless the rounded delays are deployed.
    width_at_end = 0.0 if delayed else 0.5
    assert loaded.stdout.split() == ["SpikingClassification", "{\"sigma\":", f"{width_at_end}}}",
                                     "SNNRunConfig", "True"]
