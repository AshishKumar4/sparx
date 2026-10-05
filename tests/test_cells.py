import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reference

from sparx.cells import ALIFCell, IzhikevichCell, LICell, LIFCell, RecurrentCell, SynapticCell, run
from sparx.surrogate import ATan, Rectangle

T, B, F = 40, 3, 7


def currents(seed=0, scale=0.8, shape=(T, B, F)):
    return np.random.default_rng(seed).normal(0.3, scale, shape).astype(np.float32)


# Spikes are compared exactly, so the float32 scan and the float64 loop
# must not disagree on any threshold crossing. With these seeds and scales
# the closest approach of a membrane to its threshold is far above float32
# rounding of a 40-step sum.

@pytest.mark.parametrize("reset", ["subtract", "zero", "none"])
def test_lif_matches_the_reference_loop(reset):
    xs = currents()
    spikes, _ = run(LIFCell(0.8, 1.0, reset), jnp.asarray(xs))
    expected, _ = reference.lif(xs.astype(np.float64), 0.8, 1.0, reset)
    np.testing.assert_array_equal(spikes, expected)
    assert spikes.sum() > 50  # the comparison covers real spiking


def test_lif_final_membrane_matches_the_reference_loop():
    xs = currents(1)
    _, state = run(LIFCell(0.9, 1.2, "subtract"), jnp.asarray(xs))
    _, membranes = reference.lif(xs.astype(np.float64), 0.9, 1.2, "subtract")
    np.testing.assert_allclose(state.v, membranes[-1], rtol=1e-5, atol=1e-5)


def test_per_neuron_decays_and_thresholds_broadcast_over_the_last_axis():
    xs = currents(2)
    decay = np.linspace(0.5, 0.95, F).astype(np.float32)
    threshold = np.linspace(0.6, 1.5, F).astype(np.float32)
    spikes, _ = run(LIFCell(jnp.asarray(decay), jnp.asarray(threshold)), jnp.asarray(xs))
    expected, _ = reference.lif(xs.astype(np.float64), decay.astype(np.float64), threshold.astype(np.float64))
    np.testing.assert_array_equal(spikes, expected)


def test_li_returns_the_membrane_trace():
    xs = currents(3)
    out, _ = run(LICell(0.7), jnp.asarray(xs))
    np.testing.assert_allclose(out, reference.li(xs.astype(np.float64), 0.7), rtol=1e-5, atol=1e-5)


def test_synaptic_matches_the_reference_loop():
    xs = currents(4, scale=0.5)
    spikes, _ = run(SynapticCell(0.85, 0.6), jnp.asarray(xs))
    np.testing.assert_array_equal(spikes, reference.synaptic(xs.astype(np.float64), 0.85, 0.6))
    assert spikes.sum() > 50


def test_alif_matches_the_reference_loop():
    xs = currents(5, scale=1.0)
    spikes, _ = run(ALIFCell(0.9, 0.97, beta=0.5), jnp.asarray(xs))
    np.testing.assert_array_equal(spikes, reference.alif(xs.astype(np.float64), 0.9, 0.97, 0.5))
    # Adaptation must matter in this regime, or the test would pass a cell
    # that ignored it.
    lif_spikes, _ = run(LIFCell(0.9), jnp.asarray(xs))
    assert spikes.sum() < lif_spikes.sum()


def test_alif_refractoriness_is_bellecs_counter():
    xs = currents(5, scale=1.5)
    spikes, _ = run(ALIFCell(0.9, 0.97, beta=0.5, refractory=4), jnp.asarray(xs))
    expected = reference.alif(xs.astype(np.float64), 0.9, 0.97, 0.5, n_refractory=4)
    np.testing.assert_array_equal(spikes, expected)

    def closest(spikes):
        trains = np.asarray(spikes).reshape(len(spikes), -1).T
        return min(np.diff(np.flatnonzero(train)).min() for train in trains if train.sum() > 1)

    assert closest(spikes) >= 4  # a spike and three silent steps
    free, _ = run(ALIFCell(0.9, 0.97, beta=0.5), jnp.asarray(xs))
    assert closest(free) < 4


def test_izhikevich_matches_his_published_loop_spike_for_spike_in_float64():
    # The quadratic membrane amplifies rounding chaotically over thousands of
    # noisy steps, and compiled XLA rounds its fused arithmetic differently
    # from NumPy in the last bit (22% of elements of one step, measured). The
    # comparison therefore runs op by op in float64, where the same
    # arithmetic gives the same bits, and isolates the integration scheme.
    xs = np.full((800, 2, 2), 10.0) + currents(6, 2.0, (800, 2, 2)).astype(np.float64)
    with jax.enable_x64(new_val=True), jax.disable_jit():
        spikes, _ = run(IzhikevichCell(), jnp.asarray(xs))
        spikes = np.asarray(spikes)
    expected = reference.izhikevich(xs)
    np.testing.assert_array_equal(spikes, expected)
    assert expected.sum() > 50


def test_izhikevich_regular_spiking_fires_tonically_at_the_published_rate():
    # Izhikevich (2003), Fig. 2: a regular-spiking cell under a constant
    # input of 10 adapts, then fires tonically. In his scheme the first
    # interval is 27 ms and the rest stay within 47 to 62 ms (spike times
    # fall on 1 ms steps), about 18 Hz.
    with jax.enable_x64(new_val=True), jax.disable_jit():
        spikes, _ = run(IzhikevichCell(), jnp.full((600, 1), 10.0, jnp.float64))
        times = np.flatnonzero(np.asarray(spikes[:, 0]))
    expected = np.flatnonzero(reference.izhikevich(np.full((600, 1), 10.0))[:, 0])
    np.testing.assert_array_equal(times, expected)
    intervals = np.diff(times)
    assert len(times) >= 10
    assert intervals[0] < 30 and np.all((intervals[1:] >= 45) & (intervals[1:] <= 65))


def test_recurrent_lif_matches_the_reference_loop():
    xs = currents(7, scale=0.6)
    weight = np.random.default_rng(8).normal(0, 0.3, (F, F)).astype(np.float32)
    cell = RecurrentCell(LIFCell(0.8), jnp.asarray(weight), jax.lax.Precision.HIGHEST)
    spikes, _ = run(cell, jnp.asarray(xs))
    expected = reference.recurrent_lif(xs.astype(np.float64), weight.astype(np.float64), 0.8)
    np.testing.assert_array_equal(spikes, expected)
    without_feedback = reference.lif(xs.astype(np.float64), 0.8)[0]
    assert (expected != without_feedback).any()


CELLS = {
    "lif": LIFCell(0.8),
    "lif_zero": LIFCell(0.8, reset="zero"),
    "li": LICell(0.8),
    "synaptic": SynapticCell(0.8, 0.5),
    "alif": ALIFCell(0.9, 0.95, beta=0.3),
    "izhikevich": IzhikevichCell(),
    "recurrent_alif": RecurrentCell(
        ALIFCell(0.9, 0.95, beta=0.3),
        jnp.asarray(np.random.default_rng(9).normal(0, 0.3, (F, F)), jnp.float32)),
}


@pytest.mark.parametrize("name", list(CELLS))
@pytest.mark.parametrize("split", [1, 17, 39])
def test_running_in_two_chunks_equals_one_run(name, split):
    cell = CELLS[name]
    scale = 10.0 if name == "izhikevich" else 1.0
    xs = jnp.asarray(currents(10) * scale)
    whole, final = run(cell, xs)
    head, middle = run(cell, xs[:split])
    tail, end = run(cell, xs[split:], middle)
    np.testing.assert_array_equal(jnp.concatenate([head, tail]), whole)
    for a, b in zip(jax.tree.leaves(end), jax.tree.leaves(final), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("unroll", [2, 8, True])
def test_unrolling_changes_nothing(unroll):
    xs = jnp.asarray(currents(11))
    np.testing.assert_array_equal(run(LIFCell(0.8), xs, unroll=unroll)[0], run(LIFCell(0.8), xs)[0])


def test_bf16_inputs_give_bf16_spikes_over_a_float32_membrane():
    xs = jnp.asarray(currents(12), jnp.bfloat16)
    spikes, state = run(LIFCell(0.8), xs)
    assert spikes.dtype == jnp.bfloat16
    assert state.v.dtype == jnp.float32
    expected, _ = reference.lif(np.asarray(xs, np.float64), 0.8)
    np.testing.assert_array_equal(np.asarray(spikes, np.float64), expected)


def test_recurrent_bf16_carry_keeps_its_dtype():
    xs = jnp.asarray(currents(13), jnp.bfloat16)
    cell = RecurrentCell(LIFCell(0.8), jnp.eye(F, dtype=jnp.float32) * 0.2)
    spikes, state = run(cell, xs)
    assert spikes.dtype == jnp.bfloat16
    assert state.spikes.dtype == jnp.bfloat16


def _two_step_gradient(detach_reset):
    """d s[1] / d x[0] for an LIF that fires at step 0."""
    cell = LIFCell(0.8, 1.0, "subtract", Rectangle(width=2.0), detach_reset=detach_reset)

    def second_spike(x0):
        xs = jnp.stack([x0, jnp.asarray(0.9)])[:, None]
        return run(cell, xs)[0][1, 0]

    return jax.grad(second_spike)(jnp.asarray(1.3))


def test_reset_gradient_follows_the_chain_rule():
    # v0 = 1.3 fires; v1 = 0.8 * (1.3 - s0) + 0.9 = 1.14 fires. With a
    # rectangle of width 2 (height 1/2, so both crossings are inside it):
    # ds1/dx0 = g(v1 - 1) * 0.8 * (1 - g(v0 - 1)) = 0.5 * 0.8 * 0.5 = 0.2,
    # and with the reset detached the spike's path drops out:
    # ds1/dx0 = g(v1 - 1) * 0.8 = 0.4.
    np.testing.assert_allclose(_two_step_gradient(detach_reset=False), 0.2, rtol=1e-6)
    np.testing.assert_allclose(_two_step_gradient(detach_reset=True), 0.4, rtol=1e-6)


def test_gradients_reach_learnable_decays_and_thresholds():
    xs = jnp.asarray(currents(14))

    def loss(decay, threshold):
        spikes, _ = run(LIFCell(decay, threshold, surrogate=ATan()), xs)
        return jnp.sum(spikes)

    d_decay, d_threshold = jax.grad(loss, argnums=(0, 1))(jnp.full((F,), 0.8), jnp.full((F,), 1.0))
    assert np.all(np.asarray(d_decay) > 0)  # more memory, more spikes
    assert np.all(np.asarray(d_threshold) < 0)  # higher threshold, fewer spikes


def test_a_scalar_input_is_refused():
    with pytest.raises(ValueError, match="time-major"):
        run(LIFCell(0.8), jnp.asarray(1.0))


def test_an_unknown_reset_is_refused():
    with pytest.raises(ValueError, match="reset"):
        run(LIFCell(0.8, reset="soft"), jnp.ones((3, 2)))  # type: ignore[arg-type]


def test_alif_is_bellecs_model_with_its_reset_decayed():
    # Bellec et al.'s code subtracts the baseline threshold one step after a
    # spike, undecayed; sparx resets at the spike, so the reset has decayed by
    # the next step. The two are the same model when theirs subtracts
    # decay * threshold, spike for spike.
    import reference
    xs = currents(15, scale=1.0)
    spikes, _ = run(ALIFCell(0.9, 0.97, beta=0.5, threshold=1.0), jnp.asarray(xs))
    expected = reference.eprop_alif(xs.astype(np.float64), 0.9, 0.97, 0.5, 1.0, reset=0.9 * 1.0)
    np.testing.assert_array_equal(spikes, expected)
    assert expected.sum() > 50
    # With their undecayed reset the spike trains differ.
    undecayed = reference.eprop_alif(xs.astype(np.float64), 0.9, 0.97, 0.5, 1.0, reset=1.0)
    assert (undecayed != expected).any()
