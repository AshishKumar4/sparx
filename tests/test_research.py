"""The research code runs in its `--smoke` mode, so a change to the sparx API it builds on fails here."""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.dynamics import SynapticInput

RESEARCH = Path(__file__).parents[1] / "research"


def _continual(name):
    spec = importlib.util.spec_from_file_location(f"continual_{name}", RESEARCH / "continual" / f"{name}.py")
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
    task = _continual("task").SwitchDoor(doors=4, trials=8)
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


# Always choosing ATan made the slope-25 and slope-100 cases fail, two of three.
@pytest.mark.parametrize("slope", [0.0, 25.0, 100.0])
def test_stream_surrogate_changes_the_gradient_and_keeps_the_spikes(slope):
    args = argparse.Namespace(tau=10.0, hidden=5, clip=0.0, lr=1e-3, seed=0, learner="bptt",
                              surrogate_slope=slope)
    cell = _continual("stream").Learner(args, steps=1).cell
    state = cell.init_state((5,), jnp.float32)
    values = jnp.asarray([0.4, 0.9, 1.0, 1.1, 1.6], jnp.float32)

    def step(x):
        return cell.step(state, SynapticInput(jump=x), 1.0)[1].value

    np.testing.assert_array_equal(step(values), np.asarray(values >= 1, np.float32))
    offsets = np.asarray(values, np.float64) - 1
    expected = (1 / (slope * np.abs(offsets) + 1) ** 2 if slope
                else 1 / (1 + (np.pi * offsets) ** 2))
    np.testing.assert_allclose(jax.grad(lambda x: jnp.sum(step(x)))(values), expected,
                               rtol=1e-6, atol=1e-7)
