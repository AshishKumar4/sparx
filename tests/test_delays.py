import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.nn import STATE, DelayedDense, delay_kernel

T, B, IN, OUT, K = 15, 3, 4, 5, 6
HIGHEST = jax.lax.Precision.HIGHEST


def inputs(seed=0, steps=T):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(steps, B, IN)), jnp.float32)


def delayed_reference(x, weight, delay, bias):
    """y[t, j] = bias[j] + sum_i weight[i, j] * x[t - delay[i, j], i], zero before the start."""
    x = np.asarray(x, np.float64)
    y = np.zeros((x.shape[0], x.shape[1], weight.shape[1])) + bias
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            d = int(delay[i, j])
            y[d:, :, j] += weight[i, j] * x[:x.shape[0] - d, :, i]
    return y


def test_zero_width_delays_shift_each_synapse_by_its_rounded_delay():
    x = inputs()
    layer = DelayedDense(OUT, K, precision=HIGHEST)
    params = layer.init(jax.random.key(0), x, 1.0)
    out = layer.apply(params, x, 0)
    p = params["params"]
    delays = np.clip(np.round(np.asarray(p["delay"])), 0, K)
    assert len(set(delays.ravel().tolist())) > 3  # many distinct delays are exercised
    expected = delayed_reference(x, np.asarray(p["kernel"], np.float64), delays, np.asarray(p["bias"]))
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_zero_delays_are_a_dense_layer():
    x = inputs(1)
    layer = DelayedDense(OUT, K, precision=HIGHEST)
    params = layer.init(jax.random.key(0), x, 1.0)
    params = {"params": {**params["params"], "delay": jnp.zeros((IN, OUT))}}
    p = params["params"]
    dense = jnp.matmul(x, p["kernel"], precision=HIGHEST) + p["bias"]
    np.testing.assert_allclose(layer.apply(params, x, 0), dense, rtol=1e-6, atol=1e-6)


def test_gaussian_kernels_sum_to_one_and_peak_at_the_delay():
    delay = jnp.asarray([[0.0, 2.0], [3.6, 9.0]])
    kernel = delay_kernel(delay, 5, 0.5)
    np.testing.assert_allclose(kernel.sum(0), np.ones((2, 2)), rtol=1e-6)
    np.testing.assert_array_equal(jnp.argmax(kernel, 0), [[0, 2], [4, 5]])  # 9 clips to the last lag


def test_delays_receive_gradients_while_the_gaussian_has_width():
    x = inputs(2)
    layer = DelayedDense(OUT, K)
    params = layer.init(jax.random.key(0), x, 1.0)
    grads = jax.grad(lambda p: jnp.sum(layer.apply(p, x, 1.5) ** 2))(params)["params"]["delay"]
    assert np.all(np.isfinite(grads)) and np.count_nonzero(np.asarray(grads)) == IN * OUT
    zero = jax.grad(lambda p: jnp.sum(layer.apply(p, x, 0) ** 2))(params)["params"]["delay"]
    np.testing.assert_array_equal(zero, 0)


def test_traced_widths_follow_a_schedule():
    x = inputs(3)
    layer = DelayedDense(OUT, K)
    params = layer.init(jax.random.key(0), x, 1.0)
    traced = jax.jit(lambda p, sigma: layer.apply(p, x, sigma))(params, jnp.asarray(1.5))
    np.testing.assert_allclose(traced, layer.apply(params, x, 1.5), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sigma", [0, 2.0])
def test_delayed_layers_stream_in_chunks(sigma):
    x = inputs(4, steps=17)
    layer = DelayedDense(OUT, K)
    params = layer.init(jax.random.key(0), x, 1.0)
    whole = layer.apply(params, x, sigma)
    outputs, carried = [], {}
    for chunk in (x[:1], x[1:3], x[3:12], x[12:]):
        out, carried = layer.apply({**params, **carried}, chunk, sigma, mutable=[STATE])
        outputs.append(out)
    np.testing.assert_allclose(jnp.concatenate(outputs), whole, rtol=1e-5, atol=1e-5)


def test_delayed_layers_are_causal():
    x = inputs(5)
    layer = DelayedDense(OUT, K)
    params = layer.init(jax.random.key(0), x, 1.0)
    changed = x.at[9:].set(4.0)
    np.testing.assert_array_equal(layer.apply(params, changed, 2.0)[:9], layer.apply(params, x, 2.0)[:9])
