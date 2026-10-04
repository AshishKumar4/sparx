from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sparx import encode

SNNTORCH = np.load(Path(__file__).parent / "fixtures" / "snntorch.npz")


def test_latency_matches_snntorch_linear_normalized_clipped():
    out = encode.latency(SNNTORCH["latency_input"], 9, threshold=0.01)
    np.testing.assert_array_equal(out, SNNTORCH["latency"])


def test_latency_fires_once_earlier_for_brighter_values():
    x = jnp.asarray([1.0, 0.5, 0.02, 0.0])
    out = encode.latency(x, 5)
    np.testing.assert_array_equal(out.sum(0), [1, 1, 1, 0])
    np.testing.assert_array_equal(jnp.argmax(out[:, :3], axis=0), [0, 2, 4])


def test_delta_matches_snntorch_without_padding():
    np.testing.assert_array_equal(encode.delta(SNNTORCH["delta_input"], 0.5, off_spikes=True), SNNTORCH["delta"])
    np.testing.assert_array_equal(encode.delta(SNNTORCH["delta_input"], 0.5), SNNTORCH["delta_on"])


def test_rate_spikes_with_the_given_probability():
    p = jnp.asarray([0.0, 0.1, 0.5, 0.9, 1.0, 1.7, -0.3])
    out = encode.rate(jax.random.key(0), p, 20_000)
    assert out.shape == (20_000, 7) and out.dtype == jnp.float32
    np.testing.assert_allclose(out.mean(0), [0, 0.1, 0.5, 0.9, 1, 1, 0], atol=0.01)


def test_rate_draws_differ_by_key_and_repeat_for_the_same_key():
    x = jnp.full((4, 4), 0.5)
    a, b = encode.rate(jax.random.key(0), x, 8), encode.rate(jax.random.key(1), x, 8)
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a, encode.rate(jax.random.key(0), x, 8))


def test_repeat_adds_a_leading_time_axis():
    x = jnp.arange(6.0).reshape(2, 3)
    out = encode.repeat(x, 4)
    assert out.shape == (4, 2, 3)
    np.testing.assert_array_equal(out[3], x)
