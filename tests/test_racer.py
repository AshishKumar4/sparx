"""The racer's few-bit payload, reset and firing-rate gradient follow the same level scaling."""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.dynamics import SynapticInput, run


@pytest.fixture(scope="module")
def stack():
    path = Path(__file__).parents[1] / "site/lab/stack.py"
    spec = importlib.util.spec_from_file_location("racer_stack", path)
    assert spec is not None and spec.loader is not None, "the file exists"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Omitting the level division failed both normalized cases above one level and the Vision test, 3 of 8.
@pytest.mark.parametrize("levels", [1, 3, 15])
@pytest.mark.parametrize("normalized", [False, True])
def test_fewbit_levels_scale_payload_rate_and_reset_gradients(stack, levels, normalized):
    cell = stack.FewBitCell(0.8, levels, normalize_levels=normalized)
    values = jnp.asarray([-0.3, 0.9, 1.0, 1.1, 2.9, 3.0, 3.1, 14.9, 15.0, 16.1], jnp.float32)
    state = cell.init_state(values.shape, jnp.float32)
    after, out = cell.step(state, SynapticInput(jump=values), 1.0)
    v = np.asarray(values, np.float64)
    np.testing.assert_array_equal(out.value, np.floor(np.clip(v, 0, levels)))
    np.testing.assert_array_equal(after.v, np.where(v >= 1, 0, v))
    np.testing.assert_array_equal(after.first, v >= 1)

    def outputs(x):
        new, fired = cell.step(state, SynapticInput(jump=x), 1.0)
        return new.v, new.first, fired.value

    gradients = jax.jvp(outputs, (values,), (jnp.ones_like(values),))[1]
    divisor = levels if normalized else 1
    derivatives = 1 / (1 + (np.pi * (v[:, None] - np.arange(1, levels + 1))) ** 2) / divisor
    expected = ((v < 1) - v * derivatives[:, 0], derivatives[:, 0], derivatives.sum(-1))
    for actual, reference in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=1e-7)
    reverse = jax.grad(lambda x: outputs(x)[2].sum())(values)
    np.testing.assert_allclose(reverse, expected[2], rtol=2e-6, atol=1e-7)


def test_scaled_fewbit_keeps_the_forward_stream_and_chunks(stack):
    inputs = jnp.asarray(np.random.default_rng(0).normal(0.7, 1.4, (17, 5)), jnp.float32)
    original = stack.FewBitCell(0.8, 15)
    scaled = stack.FewBitCell(0.8, 15, normalize_levels=True)
    whole, final = run(scaled, inputs)
    reference, ref_final = run(original, inputs)
    left, middle = run(scaled, inputs[:7])
    right, end = run(scaled, inputs[7:], state=middle)
    np.testing.assert_array_equal(whole.value, reference.value)
    np.testing.assert_array_equal(whole.value, jnp.concatenate([left.value, right.value]))
    for actual in (ref_final, end):
        for got, expected in zip(actual, final, strict=True):
            np.testing.assert_array_equal(got, expected)


def test_vision_records_and_restores_level_scaling(stack):
    net = stack.Vision((8, 16, 2), features=(2, 2), kernels=(1, 1), hidden=2, bits=2,
                       normalize_levels=True)
    inputs = jnp.ones((1, 1, 257), jnp.float32)
    params = net.init(jax.random.key(0), inputs)["params"]
    params["Dense_1"]["kernel"] = jnp.ones_like(params["Dense_1"]["kernel"])
    recorded = stack.vision_json(net, params)
    restored, weights = stack.vision_from_json(recorded)
    assert restored.normalize_levels
    del recorded["config"]["normalize_levels"]
    original, _ = stack.vision_from_json(recorded)
    assert not original.normalize_levels
    np.testing.assert_array_equal(restored.apply({"params": weights}, inputs),
                                  original.apply({"params": weights}, inputs))
    scaled = jax.grad(lambda x: restored.apply({"params": weights}, x).sum())(inputs)
    baseline = jax.grad(lambda x: original.apply({"params": weights}, x).sum())(inputs)
    assert float(jnp.max(jnp.abs(baseline[..., :256]))) > 1e-6
    assert float(jnp.max(jnp.abs(baseline[..., 256:]))) > 1e-6
    np.testing.assert_allclose(scaled[..., :256], baseline[..., :256] / 3**3, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(scaled[..., 256:], baseline[..., 256:] / 3, rtol=2e-5, atol=1e-8)
