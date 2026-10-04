import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.nn import PSN, RATES, STATE, MaskedPSN, SlidingPSN, band_mask


def inputs(steps=12, seed=0):
    return jnp.asarray(np.random.default_rng(seed).normal(0.6, 1.0, (steps, 3, 5)), jnp.float32)


def test_band_mask_keeps_each_step_and_the_k_minus_one_before_it():
    np.testing.assert_array_equal(band_mask(4, 2), [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    with pytest.raises(ValueError, match="order"):
        band_mask(4, 0)


def test_fully_masked_psn_is_causal_within_its_order():
    x = inputs()
    layer = MaskedPSN(k=3)
    params = layer.init(jax.random.key(0), x)
    future = x.at[7:].set(5.0)
    distant = x.at[:4].set(-5.0)
    # Changing steps 7 onward leaves steps 0..6; changing steps 0..3 leaves
    # steps 6 onward, which are more than k - 1 = 2 steps later.
    np.testing.assert_array_equal(layer.apply(params, future)[:7], layer.apply(params, x)[:7])
    np.testing.assert_array_equal(layer.apply(params, distant)[6:], layer.apply(params, x)[6:])
    # Partial masking lets the future in.
    assert not np.array_equal(layer.apply(params, future, masking=0.5)[:7], layer.apply(params, x, masking=0.5)[:7])


def test_full_psn_reads_the_future():
    x = inputs()
    layer = PSN()
    params = layer.init(jax.random.key(0), x)
    assert not np.array_equal(layer.apply(params, x.at[-1].set(5.0))[0], layer.apply(params, x)[0])


@pytest.mark.parametrize("k", [1, 3, 5])
def test_sliding_psn_streams_in_chunks(k):
    x = inputs(13, seed=1)
    layer = SlidingPSN(k=k, exponential_init=False)
    params = layer.init(jax.random.key(2), x)
    whole = layer.apply(params, x)
    outputs, carried = [], {}
    for chunk in (x[:1], x[1:2], x[2:9], x[9:]):
        out, carried = layer.apply({**params, **carried}, chunk, mutable=[STATE])
        outputs.append(out)
    np.testing.assert_array_equal(jnp.concatenate(outputs), whole)


def test_sliding_psn_runs_on_any_length_with_the_same_parameters():
    params = SlidingPSN(k=3).init(jax.random.key(0), inputs(4))
    for steps in (1, 2, 40):
        assert SlidingPSN(k=3).apply(params, inputs(steps)).shape == (steps, 3, 5)


@pytest.mark.parametrize("layer", [PSN(), MaskedPSN(k=2)])
def test_fixed_length_psns_refuse_to_stream(layer):
    x = inputs()
    params = layer.init(jax.random.key(0), x)
    with pytest.raises(ValueError, match="cannot continue across calls"):
        layer.apply(params, x, mutable=[STATE])


def test_psn_layers_sow_firing_rates():
    x = inputs()
    for layer in (PSN(), MaskedPSN(k=2), SlidingPSN(k=2)):
        params = layer.init(jax.random.key(0), x)
        spikes, sown = layer.apply(params, x, mutable=[RATES])
        np.testing.assert_allclose(sown[RATES]["rate"][0], spikes.mean(0))


def test_psn_spikes_keep_bf16_and_train():
    x = inputs().astype(jnp.bfloat16)
    layer = SlidingPSN(k=4)
    params = layer.init(jax.random.key(0), x)
    assert layer.apply(params, x).dtype == jnp.bfloat16
    grads = jax.grad(lambda p: jnp.sum(layer.apply(p, x)).astype(jnp.float32))(params)
    assert np.any(np.asarray(grads["params"]["weight"]) != 0)
