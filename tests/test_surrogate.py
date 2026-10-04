import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.surrogate import (
    ATan,
    FastSigmoid,
    Gaussian,
    Rectangle,
    Sigmoid,
    StraightThrough,
    Surrogate,
    Triangle,
    spike,
)

POINTS = np.array([-3.0, -1.2, -0.6, -0.4, -0.1, 0.0, 0.05, 0.3, 0.45, 0.9, 2.5], np.float32)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Each derivative written out from its paper, independently of sparx's code.
REFERENCE = [
    (ATan(alpha=2.0), lambda x: 2.0 / 2 / (1 + (np.pi / 2 * 2.0 * x) ** 2)),
    (ATan(alpha=5.0), lambda x: 5.0 / 2 / (1 + (np.pi / 2 * 5.0 * x) ** 2)),
    (Sigmoid(alpha=4.0), lambda x: 4.0 * _sigmoid(4.0 * x) * (1 - _sigmoid(4.0 * x))),
    (FastSigmoid(slope=25.0), lambda x: 1 / (25.0 * np.abs(x) + 1) ** 2),
    (Triangle(width=1.0, scale=0.3), lambda x: 0.3 * np.maximum(0, 1 - np.abs(x))),
    (Triangle(width=0.5), lambda x: np.maximum(0, 1 - np.abs(x) / 0.5)),
    (Rectangle(width=1.0), lambda x: (np.abs(x) < 0.5) / 1.0),
    (Gaussian(sigma=0.5), lambda x: np.exp(-0.5 * (x / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))),
    (StraightThrough(), np.ones_like),
]


@pytest.mark.parametrize(("surrogate", "reference"), REFERENCE, ids=lambda s: repr(s)[:30])
def test_reverse_mode_gradient_is_the_surrogate_derivative(surrogate: Surrogate, reference):
    grad = jax.grad(lambda x: jnp.sum(spike(x, surrogate)))(jnp.asarray(POINTS))
    np.testing.assert_allclose(grad, reference(POINTS.astype(np.float64)), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(("surrogate", "reference"), REFERENCE, ids=lambda s: repr(s)[:30])
def test_forward_mode_tangent_is_the_surrogate_derivative(surrogate: Surrogate, reference):
    tangent = np.linspace(-1, 2, POINTS.size).astype(np.float32)
    _, jvp = jax.jvp(lambda x: spike(x, surrogate), (jnp.asarray(POINTS),), (jnp.asarray(tangent),))
    np.testing.assert_allclose(jvp, reference(POINTS.astype(np.float64)) * tangent, rtol=1e-5, atol=1e-6)


def test_forward_pass_is_the_heaviside_step_with_threshold_inclusive():
    np.testing.assert_array_equal(spike(jnp.asarray(POINTS), ATan()), (POINTS >= 0).astype(np.float32))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
def test_spikes_keep_the_input_dtype(dtype):
    x = jnp.asarray(POINTS, dtype)
    out = ATan()(x)
    assert out.dtype == dtype
    grad = jax.grad(lambda x: jnp.sum(ATan()(x)).astype(jnp.float32))(x)
    assert grad.dtype == dtype


def test_integer_membranes_are_refused():
    with pytest.raises(TypeError, match="floating"):
        spike(jnp.arange(3), ATan())


@pytest.mark.parametrize(("surrogate", "area"), [
    (ATan(alpha=2.0), 1.0),
    (Sigmoid(alpha=4.0), 1.0),
    (Rectangle(width=0.7), 1.0),
    (Gaussian(sigma=0.3), 1.0),
    (Triangle(width=0.8, scale=1 / 0.8), 1.0),
    (FastSigmoid(slope=25.0), 2 / 25.0),
], ids=lambda s: repr(s)[:30])
def test_surrogate_areas_match_their_documented_normalization(surrogate, area):
    # The area is what the surrogate step rises by across threshold, so a
    # wrong scale changes every gradient's size.
    # The spacing is taken from the grid's definition, not from two float32
    # points near 400, whose difference carries their rounding.
    count, half_width = 1_600_001, 400.0
    x = jnp.linspace(-half_width, half_width, count, dtype=jnp.float32)
    density = np.asarray(surrogate.derivative(x), np.float64)
    integral = float(density.sum() * (2 * half_width / (count - 1)))
    assert math.isclose(integral, area, rel_tol=2e-3)


def test_spike_batches_under_vmap():
    xs = jnp.asarray(POINTS).reshape(1, -1).repeat(3, 0) * jnp.asarray([[1.0], [2.0], [-1.0]])
    grads = jax.vmap(jax.grad(lambda x: jnp.sum(spike(x, Sigmoid()))))(xs)
    np.testing.assert_allclose(grads[1], jax.grad(lambda x: jnp.sum(spike(x, Sigmoid())))(xs[1]))
