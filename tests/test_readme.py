"""The README's code runs as written and does what its text says."""

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

BLOCKS = re.findall(r"```python\n(.*?)\n```", (Path(__file__).parents[1] / "README.md").read_text(), re.S)


def _block(start):
    (found,) = [block for block in BLOCKS if block.startswith(start)]
    return found


def _first_network():
    scope = {}
    exec(_block("import flax.linen as nn"), scope)
    return scope


def test_the_first_network_trains():
    scope = _first_network()
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(scope["grads"]))
    assert any(np.any(g != 0) for g in jax.tree.leaves(scope["grads"]))


def test_the_conv_network_runs_over_time_and_batch():
    scope = _first_network()
    exec(_block("class ConvNet"), scope)
    x = jnp.ones((4, 2, 8, 8, 1))
    net = scope["ConvNet"]()
    out, _ = net.apply(net.init(jax.random.key(0), x), x, mutable=["batch_stats"])
    assert out.shape == (4, 2, 10)


def test_the_firing_rate_snippet_reports_each_layer():
    scope = _first_network()
    exec(_block("outputs, sown = net.apply"), scope)
    assert set(scope["sparx"].firing_rates(scope["sown"])) == {"LIF_0"}
    assert scope["penalty"].shape == ()


def test_the_streaming_snippet_continues_across_chunks():
    scope = _first_network()
    spikes = scope["spikes"]
    scope["chunks"] = [spikes[:3], spikes[3:5], spikes[5:]]
    exec(_block("carried = {}"), scope)
    whole = scope["net"].apply(scope["params"], spikes)
    np.testing.assert_allclose(scope["out"], whole[5:], rtol=1e-6, atol=1e-6)


def test_the_pure_jax_cell_snippet_runs():
    scope = {"sparx": __import__("sparx"),
             "currents": jax.random.normal(jax.random.key(0), (20, 4, 6)),
             "next_currents": jax.random.normal(jax.random.key(1), (5, 4, 6)),
             "weight": jnp.eye(6) * 0.1}
    exec(_block("from sparx.cells import"), scope)
    assert scope["spikes"].shape == (20, 4, 6) and scope["more"].shape == (5, 4, 6)


def test_the_sew_resnet_snippet_returns_per_step_logits():
    import sparx

    frames = jnp.ones((2, 1, 32, 32, 3))
    net = sparx.models.sew_resnet18(10, width=32, stem="small")
    scope = {"sparx": sparx, "frames": frames,
             "variables": net.init(jax.random.key(0), frames, train=False)}
    exec(_block("net = sparx.models.sew_resnet18"), scope)
    assert scope["logits"].shape == (2, 1, 10)
