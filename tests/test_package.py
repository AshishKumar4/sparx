"""`import sparx` reaches every submodule, and loads the heavy ones only when they are used."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import sparx

LAZY = ("datasets", "graph", "learn", "metrics", "nir", "objectives", "optim", "serve", "tasks")


def test_import_sparx_leaves_the_lazy_submodules_unloaded_until_used():
    code = ("import sys, sparx\n"
            f"assert sparx.__file__ == {sparx.__file__!r}\n"
            f"assert not [m for m in {LAZY!r} if 'sparx.' + m in sys.modules]\n"
            "assert 'dew.training' not in sys.modules\n"
            "sparx.graph.Network\n"
            "assert 'sparx.graph' in sys.modules and 'sparx.nir' not in sys.modules\n")
    # The child reads this tree's sparx, not whichever one the environment installs.
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(Path(sparx.__file__).parents[1])}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]


@pytest.mark.parametrize("name", LAZY)
def test_each_lazy_submodule_is_an_attribute_of_the_package(name):
    module = getattr(sparx, name)
    assert module.__name__ == f"sparx.{name}"
    assert name in dir(sparx) and name in sparx.__all__


def test_an_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'graphs'"):
        sparx.graphs  # noqa: B018
