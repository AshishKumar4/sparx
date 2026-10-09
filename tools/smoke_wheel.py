#!/usr/bin/env python3
"""Install a built wheel into a new, empty virtual environment and use it.

    python tools/smoke_wheel.py dist/sparxml-*.whl

The wheel is installed from PyPI's index alone, so a requirement the index
cannot serve fails here. Then, from outside the checkout so nothing of it
is importable:

- every module the wheel installed compiles (`compileall`), so a syntax
  error anywhere in the package fails, imported or not;
- the README's first network trains two steps through dew's `Trainer`, and
  its parameters move and stay finite.

Exits nonzero on the first failure. The release workflow runs this on the
artifact it then uploads.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import venv
from pathlib import Path

TRAIN = """
import flax.linen as nn, jax, jax.numpy as jnp, numpy as np, optax
from dew import Field, Trainer
from dew.data import Dataset, Loading

import sparx
from sparx.encode import RateEncoder
from sparx.objectives import SpikingClassifierObjective

assert "site-packages" in sparx.__file__, sparx.__file__


class Net(nn.Module):
    @nn.compact
    def __call__(self, spikes):
        x = sparx.nn.LIF(tau=2.0)(nn.Dense(16)(spikes.reshape(*spikes.shape[:2], -1)))
        return sparx.nn.LI(tau=2.0)(nn.Dense(2)(x))


rng = np.random.default_rng(0)
records = {"image": rng.integers(0, 255, (32, 4, 4, 1)).astype(np.uint8),
           "label": rng.integers(0, 2, 32).astype(np.int32)}
data = Dataset.from_records(records, batch=16, loading=Loading(workers=0, threads=1, read_buffer=1))
objective = SpikingClassifierObjective(Net(), Field("image", (4, 4, 1)), RateEncoder(steps=4))
trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0))
before = objective.init(jax.random.key(0))["params"]
state = trainer.fit(data, steps=2, log_every=2)
after = state.variables["params"]
leaves = jax.tree.leaves(after)
assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
assert any(not np.array_equal(a, b) for a, b in zip(leaves, jax.tree.leaves(before)))
print(f"sparx {sparx.__version__} trained two steps from {sparx.__file__}")
"""


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        venv.create(root / "env", with_pip=True)
        python = str(root / "env" / "bin" / "python")
        run([python, "-m", "pip", "install", "--quiet", "--index-url", "https://pypi.org/simple", str(wheel)],
            root)
        where = "import sparx, pathlib; print(pathlib.Path(sparx.__file__).parent)"
        installed = subprocess.run([python, "-c", where], capture_output=True, text=True, check=True,
                                   cwd=root).stdout.strip()
        run([python, "-m", "compileall", "-q", installed], root)
        run([python, "-c", TRAIN], root)


if __name__ == "__main__":
    main()
