"""The research code runs in its `--smoke` mode, so a change to the sparx API it builds on fails here."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

RESEARCH = Path(__file__).parents[1] / "research"


def _continual_task():
    spec = importlib.util.spec_from_file_location("continual_task", RESEARCH / "continual" / "task.py")
    assert spec is not None and spec.loader is not None, "the file exists"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_continual_learning_agents_train_in_their_smoke_mode(tmp_path):
    for agent in ("plastic", "core", "rnn"):
        command = [sys.executable, str(RESEARCH / "continual" / "train.py"), "--agent", agent, "--smoke",
                   "--out", str(tmp_path / agent)]
        done = subprocess.run(command, capture_output=True, text=True, timeout=900, cwd=tmp_path,
                              env={**os.environ, "JAX_PLATFORMS": "cpu"})
        assert done.returncode == 0, done.stderr[-3000:]
        assert f"{agent}: test accuracy by trial" in done.stdout


def test_the_ideal_curve_is_what_a_learner_remembering_every_pair_scores_on_the_sessions():
    task = _continual_task().SwitchDoor(doors=4, trials=8)
    sessions = task.sessions(20_000, 0)
    rng = np.random.default_rng(1)
    hits = np.zeros(task.trials)
    for mapping, targets in zip(sessions["mapping"], sessions["targets"], strict=True):
        known = {}  # door: the switch that opened it
        for trial, target in enumerate(targets):
            untried = [s for s in range(task.doors) if s not in known.values()]
            switch = known[target] if target in known else untried[rng.integers(len(untried))]
            hits[trial] += mapping[switch] == target
            known[int(mapping[switch])] = switch
    np.testing.assert_allclose(hits / len(sessions["targets"]), task.ideal(), atol=0.015)  # observed 0.0027
