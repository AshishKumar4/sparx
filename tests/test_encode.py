from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dew.registry import from_record, to_record

from sparx.encode import DeltaEncoder, DirectEncoder, EventsEncoder, LatencyEncoder, RateEncoder, SpikeEncoder
from sparx.registry import spike_encoders

SNNTORCH = np.load(Path(__file__).parent / "fixtures" / "snntorch.npz")
KEY = jax.random.key(0)


def test_latency_matches_snntorch_linear_normalized_clipped():
    out = LatencyEncoder(9, threshold=0.01)(KEY, SNNTORCH["latency_input"])
    np.testing.assert_array_equal(out, SNNTORCH["latency"])


def test_latency_fires_once_earlier_for_brighter_values():
    x = jnp.asarray([1.0, 0.5, 0.02, 0.0])
    out = LatencyEncoder(5)(KEY, x)
    np.testing.assert_array_equal(out.sum(0), [1, 1, 1, 0])
    np.testing.assert_array_equal(jnp.argmax(out[:, :3], axis=0), [0, 2, 4])


def test_delta_matches_snntorch_without_padding():
    # snnTorch's input is one time-major signal [T, F]; as a batch of one record it is [1, T, F].
    signal = SNNTORCH["delta_input"][None]
    np.testing.assert_array_equal(DeltaEncoder(0.5, off_spikes=True)(KEY, signal)[:, 0], SNNTORCH["delta"])
    np.testing.assert_array_equal(DeltaEncoder(0.5)(KEY, signal)[:, 0], SNNTORCH["delta_on"])


def test_delta_reads_time_on_the_records_time_axis():
    signal = jnp.asarray(SNNTORCH["delta_input"])  # [T, F]
    feature_major = jnp.stack([signal.T, 2 * signal.T])  # [B, F, T]
    out = DeltaEncoder(0.5, off_spikes=True, time_axis=1)(KEY, feature_major)
    assert out.shape == (signal.shape[0], 2, signal.shape[1])
    np.testing.assert_array_equal(out[:, 0], SNNTORCH["delta"])


def test_rate_spikes_with_the_given_probability():
    p = jnp.asarray([0.0, 0.1, 0.5, 0.9, 1.0, 1.7, -0.3])
    out = RateEncoder(20_000)(KEY, p)
    assert out.shape == (20_000, 7) and out.dtype == jnp.float32
    np.testing.assert_allclose(out.mean(0), [0, 0.1, 0.5, 0.9, 1, 1, 0], atol=0.01)  # observed 8.5e-4


def test_rate_draws_differ_by_key_and_repeat_for_the_same_key():
    x = jnp.full((4, 4), 0.5)
    a, b = RateEncoder(8)(jax.random.key(0), x), RateEncoder(8)(jax.random.key(1), x)
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a, RateEncoder(8)(jax.random.key(0), x))


def test_direct_repeats_the_values_on_a_leading_time_axis():
    x = jnp.arange(6.0).reshape(2, 3)
    out = DirectEncoder(4)(KEY, x)
    assert out.shape == (4, 2, 3)
    np.testing.assert_array_equal(out[3], x)


@pytest.mark.parametrize("encoder", [DirectEncoder(3), RateEncoder(3), LatencyEncoder(3), DeltaEncoder(0.1)])
def test_intensity_encoders_read_uint8_as_a_fraction_of_255(encoder):
    pixels = np.random.default_rng(0).integers(0, 256, (2, 3, 4)).astype(np.uint8)
    np.testing.assert_array_equal(encoder(KEY, pixels), encoder(KEY, jnp.asarray(pixels, jnp.float32) / 255))


def test_events_pass_uint8_spike_counts_unscaled():
    counts = jnp.asarray([[[0, 2], [1, 0], [3, 1]]], jnp.uint8)  # [B=1, T=3, F=2]
    out = EventsEncoder()(KEY, counts)
    assert out.dtype == jnp.float32
    np.testing.assert_array_equal(out[:, 0], counts[0])


@pytest.mark.parametrize("encoder", [DirectEncoder(4), RateEncoder(8), LatencyEncoder(6, threshold=0.2),
                                     DeltaEncoder(0.3, off_spikes=True), EventsEncoder(time_axis=1)])
def test_every_encoder_rebuilds_from_its_record(encoder):
    record = to_record(encoder, SpikeEncoder)
    assert record["class"] == f"sparx.encode:{type(encoder).__name__}"
    assert from_record(SpikeEncoder, record) == encoder


def test_an_encoders_short_name_builds_it():
    assert spike_encoders.from_record({"class": "rate", "fields": {"steps": 8}}) == RateEncoder(8)
    assert set(spike_encoders) == {"delta", "direct", "events", "latency", "rate"}


ENCODERS = [DirectEncoder(3), RateEncoder(3), LatencyEncoder(3), DeltaEncoder(0.1), EventsEncoder()]


@pytest.mark.parametrize("encoder", ENCODERS)
def test_every_encoder_refuses_swapped_arguments_and_int_seeds(encoder):
    x = np.random.default_rng(0).random((2, 3, 4)).astype(np.float32)
    for first, second in ((x, KEY), (0, x)):
        with pytest.raises(TypeError, match=r"encoder\(key, x\) with a JAX PRNG key"):
            encoder(first, second)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_every_encoder_takes_typed_raw_and_traced_keys(encoder):
    x = np.random.default_rng(0).random((2, 3, 4)).astype(np.float32)
    expected = encoder(KEY, x)
    np.testing.assert_array_equal(jax.jit(encoder)(KEY, x), expected)
    np.testing.assert_array_equal(encoder(jax.random.key_data(KEY), x), expected)
