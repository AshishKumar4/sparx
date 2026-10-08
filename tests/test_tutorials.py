"""The code of the tutorials in `docs/tutorials` runs as written and does what their text says."""

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
NEST = np.load(ROOT / "tests" / "fixtures" / "microcircuit.npz")


def _blocks(page: str) -> list[str]:
    return re.findall(r"```python\n(.*?)\n```", (ROOT / "docs" / "tutorials" / page).read_text(), re.S)


def _block(page: str, start: str) -> str:
    (found,) = [block for block in _blocks(page) if block.startswith(start)]
    return found


def test_the_nest_and_brian2_tutorial_builds_its_networks_and_the_microcircuit_fires_as_nests():
    scope = {}
    exec(_block("nest-and-brian2.md", "import jax\nfrom sparx.dynamics"), scope)
    spikes = scope["spikes"]
    rate, cv = scope["rates_hz"](spikes, 0.1).mean(), scope["cv_isi"](spikes).mean()
    assert 10 < rate < 16 and 0.3 < cv < 0.5, (rate, cv)  # the text's 13 Hz and 0.4
    exec(_block("nest-and-brian2.md", "from sparx.graph import PopulationRate"), scope)
    sizes = {p.name: p.size for p in scope["network"].populations}
    for i, name in enumerate(scope["MICROCIRCUIT_POPULATIONS"]):
        nest = np.array([NEST[f"{seed}/{name}/mean"] for seed in NEST["meta/seeds"]], np.float64)
        rate = scope["rates"][name]
        assert 0.9 * nest.min() <= rate <= 1.1 * nest.max(), (name, rate, nest)
        assert sizes[name] == NEST["derived/neurons"][i]


def test_the_train_and_deploy_tutorial_trains_loads_serves_and_exports_one_classifier(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the run and the NIR file land here
    scope = {}
    for block in _blocks("train-and-deploy.md"):
        exec(block, scope)
    assert scope["accuracy"] >= 0.95
    whole = scope["classifier"].model.apply(scope["classifier"].variables, scope["recording"][:, None])[:, 0]
    np.testing.assert_allclose(scope["streamed"], whole, rtol=1e-6, atol=1e-6)  # observed 0
    assert scope["predicted"] == int(scope["classifier"](scope["test"]["spikes"][:1])[0])
    assert scope["same"]
    assert [type(scope["graph"].nodes[k]).__name__ for k in "0123"] == ["Affine", "LIF", "Affine", "LI"]
