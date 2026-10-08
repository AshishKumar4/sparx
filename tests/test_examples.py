"""Every example runs end to end in its `--smoke` mode, which trains a small network briefly and downloads
nothing, so a change to the API an example uses fails here."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
RUNS = [[path.name] for path in sorted(EXAMPLES.glob("*.py"))] + [["train_shd.py", "--recipe", "snn-delays"]]


@pytest.mark.parametrize("run", RUNS, ids=" ".join)
def test_every_example_runs_in_its_smoke_mode(run, tmp_path):
    script, *flags = run
    command = [sys.executable, str(EXAMPLES / script), *flags, "--smoke", "--out", str(tmp_path / "run")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=900, cwd=tmp_path,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr[-3000:]
