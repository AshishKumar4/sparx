import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reference

from sparx.dynamics import (
    ALIFCell,
    BernoulliCell,
    Izhikevich,
    LICell,
    LIFCell,
    ModulatedHebb,
    PlasticRecurrentCell,
    RateCell,
    RecurrentCell,
    RetroactiveHebb,
    Serial,
    SparseRecurrentCell,
    SynapticInput,
    Term,
    run,
)
from sparx.surrogate import ATan, Rectangle

T, B, F = 40, 3, 7


def currents(seed=0, scale=0.8, shape=(T, B, F)):
    return np.random.default_rng(seed).normal(0.3, scale, shape).astype(np.float32)


def fired(model, xs, state=None, **kwargs):
    """The spikes `[T, ...]` of `model` over `xs`, and its final state."""
    spikes, state = run(model, xs, state, **kwargs)
    return spikes.value, state


# Spikes are compared exactly, so the float32 scan and the float64 loop
# must not disagree on any threshold crossing. With these seeds and scales
# the closest approach of a membrane to its threshold is far above float32
# rounding of a 40-step sum.

@pytest.mark.parametrize("reset", ["subtract", "zero", "none"])
def test_lif_matches_the_reference_loop(reset):
    xs = currents()
    spikes, _ = fired(LIFCell(0.8, 1.0, reset), jnp.asarray(xs))
    expected, _ = reference.lif(xs.astype(np.float64), 0.8, 1.0, reset)
    np.testing.assert_array_equal(spikes, expected)
    assert spikes.sum() > 50  # the comparison covers real spiking


def test_lif_final_membrane_matches_the_reference_loop():
    xs = currents(1)
    _, state = fired(LIFCell(0.9, 1.2, "subtract"), jnp.asarray(xs))
    _, membranes = reference.lif(xs.astype(np.float64), 0.9, 1.2, "subtract")
    np.testing.assert_allclose(state.v, membranes[-1], rtol=1e-5, atol=1e-5)  # observed 4.2e-7


def test_per_neuron_decays_and_thresholds_broadcast_over_the_last_axis():
    xs = currents(2)
    decay = np.linspace(0.5, 0.95, F).astype(np.float32)
    threshold = np.linspace(0.6, 1.5, F).astype(np.float32)
    spikes, _ = fired(LIFCell(jnp.asarray(decay), jnp.asarray(threshold)), jnp.asarray(xs))
    expected, _ = reference.lif(xs.astype(np.float64), decay.astype(np.float64), threshold.astype(np.float64))
    np.testing.assert_array_equal(spikes, expected)


def test_li_reports_its_membrane_trace():
    xs = currents(3)
    out, _ = fired(LICell(0.7), jnp.asarray(xs))
    # Observed 3.2e-7.
    np.testing.assert_allclose(out, reference.li(xs.astype(np.float64), 0.7), rtol=1e-5, atol=1e-5)


def test_li_then_lif_is_the_current_based_lif():
    xs = currents(4, scale=0.5)
    spikes, _ = fired(Serial(LICell(0.6), LIFCell(0.85)), jnp.asarray(xs))
    np.testing.assert_array_equal(spikes, reference.synaptic(xs.astype(np.float64), 0.85, 0.6))
    assert spikes.sum() > 50


def test_a_step_of_dt_decays_by_the_decay_to_the_power_dt():
    # Two steps of 0.5 leave what one step of 1 leaves: dt is a duration.
    xs = jnp.asarray(currents(16))
    halves = jnp.zeros((2 * T, B, F), xs.dtype).at[1::2].set(xs)
    _, whole = fired(LICell(0.7), xs)
    _, halved = fired(LICell(0.7), halves, dt=0.5)
    np.testing.assert_allclose(halved.v, whole.v, rtol=1e-5, atol=1e-6)  # observed 4.8e-7
    _, unchanged = fired(LICell(0.7), halves, dt=1.0)
    assert not np.allclose(unchanged.v, whole.v)


def test_a_dimensionless_model_refuses_currents():
    xs = jnp.asarray(currents(17))
    with pytest.raises(ValueError, match="jump"):
        run(LIFCell(0.8), SynapticInput(current=xs))
    with pytest.raises(ValueError, match="jump"):
        run(LIFCell(0.8), SynapticInput(jump=xs, currents=(Term(xs, xs, 5.0),)))


@pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid"])
def test_a_recurrent_rate_cell_is_flynns_recurrence(activation):
    xs = currents(18, scale=1.0)
    rng = np.random.default_rng(19)
    weight = rng.normal(0, 0.6, (F, F)).astype(np.float32)
    alpha = rng.uniform(0.3, 0.95, F).astype(np.float32)  # one leak per unit, as FLYNN's classes give them
    bias = rng.normal(0, 0.3, F).astype(np.float32)
    model = RecurrentCell(RateCell(jnp.asarray(alpha), jnp.asarray(bias), activation), jnp.asarray(weight),
                          jax.lax.Precision.HIGHEST)
    out, state = run(model, jnp.asarray(xs))
    expected = reference.flynn(*(a.astype(np.float64) for a in (xs, weight, alpha, bias)), activation)
    # Observed 1.2e-7 (tanh), 6.0e-8 (sigmoid), 1.5e-5 (relu, whose activity grows past 300).
    np.testing.assert_allclose(out.value, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state.output, expected[-1], rtol=1e-5, atol=1e-5)
    assert model.graded and np.all(out.offset == 1)
    # The feedback matters: without it the activity differs.
    alone = reference.flynn(xs.astype(np.float64), np.zeros((F, F)), alpha, bias, activation)
    assert np.abs(alone - expected).max() > 0.1


def test_a_rate_cells_leak_over_dt_is_exp_of_minus_dt_over_tau():
    # alpha = exp(-dt / tau): a step of dt leaves decay ** dt of the activity.
    tau, dt = 7.0, 0.25
    xs = currents(20)
    out, _ = run(RateCell(float(np.exp(-1 / tau))), jnp.asarray(xs), dt=dt)
    alpha = np.exp(-dt / tau)
    expected = reference.flynn(xs.astype(np.float64), np.zeros((F, F)), alpha, 0.0)
    np.testing.assert_allclose(out.value, expected, rtol=1e-5, atol=1e-6)  # observed 8.9e-8


def test_the_graded_models_say_so_and_the_spiking_ones_do_not():
    assert LICell(0.5).graded and RateCell(0.5).graded
    assert not LIFCell(0.5).graded and not ALIFCell(0.5, 0.9).graded
    assert Serial(LIFCell(0.5), LICell(0.5)).graded and not Serial(LICell(0.5), LIFCell(0.5)).graded
    assert RecurrentCell(RateCell(0.5), jnp.eye(2)).graded


@pytest.mark.parametrize("reset", ["subtract", "zero", "none"])
def test_bernoulli_matches_the_reference_loop(reset):
    xs = currents(12, scale=0.6)
    noise = np.asarray(jax.random.uniform(jax.random.key(3), xs.shape))
    model = BernoulliCell(0.8, threshold=1.0, beta=3.0, reset=reset)
    (spikes, probability), state = run(model, SynapticInput(jump=jnp.asarray(xs), noise=jnp.asarray(noise)),
                                       record=lambda state: state.p)
    expected, p, v = reference.bernoulli(xs.astype(np.float64), noise.astype(np.float64), 0.8, 3.0,
                                         reset=reset)
    # The closest draw lies 2.7e-5 from its probability, far above float32 rounding.
    np.testing.assert_array_equal(spikes.value, expected)
    np.testing.assert_allclose(probability, p, rtol=1e-5, atol=1e-6)  # observed 1.9e-7
    np.testing.assert_allclose(state.v, v, rtol=1e-5, atol=1e-5)  # observed 2.6e-7
    assert 0.1 < expected.mean() < 0.9  # the noise decides, neither all nor none fire


def test_a_noise_of_one_minus_the_spikes_replays_them():
    # p lies strictly between 0 and 1, so noise 0 always fires and noise 1 never does.
    spikes = (np.random.default_rng(5).random((T, B, F)) < 0.4).astype(np.float32)
    out, _ = run(BernoulliCell(0.9, beta=0.5), SynapticInput(jump=jnp.asarray(currents(13) * 4),
                                                             noise=jnp.asarray(1 - spikes)))
    np.testing.assert_array_equal(out.value, spikes)


def test_noise_goes_to_a_stochastic_model_and_only_there():
    xs = jnp.asarray(currents(14))
    with pytest.raises(ValueError, match="fires by its noise"):
        run(BernoulliCell(0.8), xs)
    with pytest.raises(ValueError, match="fires without noise"):
        run(LIFCell(0.8), SynapticInput(jump=xs, noise=jnp.zeros_like(xs)))


def test_alif_matches_the_reference_loop():
    xs = currents(5, scale=1.0)
    spikes, _ = fired(ALIFCell(0.9, 0.97, beta=0.5), jnp.asarray(xs))
    np.testing.assert_array_equal(spikes, reference.alif(xs.astype(np.float64), 0.9, 0.97, 0.5))
    # Adaptation must matter in this regime, or the test would pass a model
    # that ignored it.
    lif_spikes, _ = fired(LIFCell(0.9), jnp.asarray(xs))
    assert spikes.sum() < lif_spikes.sum()


def test_alif_refractoriness_is_bellecs_counter():
    xs = currents(5, scale=1.5)
    spikes, _ = fired(ALIFCell(0.9, 0.97, beta=0.5, refractory=4), jnp.asarray(xs))
    expected = reference.alif(xs.astype(np.float64), 0.9, 0.97, 0.5, n_refractory=4)
    np.testing.assert_array_equal(spikes, expected)

    def closest(spikes):
        trains = np.asarray(spikes).reshape(len(spikes), -1).T
        return min(np.diff(np.flatnonzero(train)).min() for train in trains if train.sum() > 1)

    assert closest(spikes) >= 4  # a spike and three silent steps
    free, _ = fired(ALIFCell(0.9, 0.97, beta=0.5), jnp.asarray(xs))
    assert closest(free) < 4


def test_izhikevich_matches_his_published_loop_spike_for_spike_in_float64():
    # The quadratic membrane amplifies rounding chaotically over thousands of
    # noisy steps, and compiled XLA rounds its fused arithmetic differently
    # from NumPy in the last bit (22% of elements of one step, measured). The
    # comparison therefore runs op by op in float64, where the same
    # arithmetic gives the same bits, and isolates the integration scheme.
    xs = np.full((800, 2, 2), 10.0) + currents(6, 2.0, (800, 2, 2)).astype(np.float64)
    with jax.enable_x64(new_val=True), jax.disable_jit():
        spikes, _ = fired(Izhikevich(), SynapticInput(current=jnp.asarray(xs)))
        spikes = np.asarray(spikes)
    expected = reference.izhikevich(xs)
    np.testing.assert_array_equal(spikes, expected)
    assert expected.sum() > 50
    # NEST's order of the quadratic's arithmetic departs from his code here.
    with jax.enable_x64(new_val=True), jax.disable_jit():
        nest, _ = fired(Izhikevich(order="nest"), SynapticInput(current=jnp.asarray(xs)))
    assert (np.asarray(nest) != expected).any()


def test_izhikevich_regular_spiking_fires_tonically_at_the_published_rate():
    # Izhikevich (2003), Fig. 2: a regular-spiking cell under a constant
    # input of 10 adapts, then fires tonically. In his scheme the first
    # interval is 27 ms and the rest stay within 47 to 62 ms (spike times
    # fall on 1 ms steps), about 18 Hz.
    with jax.enable_x64(new_val=True), jax.disable_jit():
        spikes, _ = fired(Izhikevich(), SynapticInput(current=jnp.full((600, 1), 10.0, jnp.float64)))
        times = np.flatnonzero(np.asarray(spikes[:, 0]))
    expected = np.flatnonzero(reference.izhikevich(np.full((600, 1), 10.0))[:, 0])
    np.testing.assert_array_equal(times, expected)
    intervals = np.diff(times)
    assert len(times) >= 10
    assert intervals[0] < 30 and np.all((intervals[1:] >= 45) & (intervals[1:] <= 65))


def test_recurrent_lif_matches_the_reference_loop():
    xs = currents(7, scale=0.6)
    weight = np.random.default_rng(8).normal(0, 0.3, (F, F)).astype(np.float32)
    model = RecurrentCell(LIFCell(0.8), jnp.asarray(weight), jax.lax.Precision.HIGHEST)
    spikes, _ = fired(model, jnp.asarray(xs))
    expected = reference.recurrent_lif(xs.astype(np.float64), weight.astype(np.float64), 0.8)
    np.testing.assert_array_equal(spikes, expected)
    without_feedback = reference.lif(xs.astype(np.float64), 0.8)[0]
    assert (expected != without_feedback).any()


MODELS = {
    "lif": LIFCell(0.8),
    "lif_zero": LIFCell(0.8, reset="zero"),
    "li": LICell(0.8),
    "serial": Serial(LICell(0.5), LIFCell(0.8)),
    "alif": ALIFCell(0.9, 0.95, beta=0.3),
    "izhikevich": Izhikevich(),
    "recurrent_alif": RecurrentCell(
        ALIFCell(0.9, 0.95, beta=0.3),
        jnp.asarray(np.random.default_rng(9).normal(0, 0.3, (F, F)), jnp.float32)),
    "recurrent_rate": RecurrentCell(
        RateCell(0.8, 0.1), jnp.asarray(np.random.default_rng(9).normal(0, 0.5, (F, F)), jnp.float32)),
    "bernoulli": BernoulliCell(0.8, beta=3.0),
    "plastic_rate": PlasticRecurrentCell(
        RateCell(0.0), jnp.asarray(np.random.default_rng(9).normal(0, 0.5, (F, F)), jnp.float32),
        jnp.asarray(np.random.default_rng(10).normal(0, 0.5, (F, F)), jnp.float32),
        ModulatedHebb(jnp.full(F, 0.5), 0.1, jnp.linspace(-2.0, 2.0, F), 0.1)),
    "sparse_rate": SparseRecurrentCell(
        RateCell(jnp.linspace(0.2, 0.9, F), 0.1), jnp.asarray([0, 1, 2, 3, 4, 5, 6, 2, 5]),
        jnp.asarray([1, 2, 3, 4, 5, 6, 0, 0, 3]), jnp.asarray(np.random.default_rng(11).normal(0, 0.8, 9)),
        F),
    "sparse_lif": SparseRecurrentCell(
        LIFCell(0.8), jnp.asarray([0, 1, 2, 3, 4, 5, 6, 2, 5]), jnp.asarray([1, 2, 3, 4, 5, 6, 0, 0, 3]),
        jnp.asarray(np.random.default_rng(12).normal(0, 0.5, 9), jnp.float32), F),
    "plastic_lif": PlasticRecurrentCell(
        LIFCell(0.8), jnp.asarray(np.random.default_rng(9).normal(0, 0.3, (F, F)), jnp.float32),
        jnp.full(F, 0.5), RetroactiveHebb(jnp.full(F, 0.5), -0.2, 0.3)),
}


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("split", [1, 17, 39])
def test_running_in_two_chunks_equals_one_run(name, split):
    model = MODELS[name]
    xs = jnp.asarray(currents(10))
    noise = jax.random.uniform(jax.random.key(4), xs.shape)

    def given(steps):
        if name == "izhikevich":
            return SynapticInput(current=xs[steps] * 10.0)
        if name == "bernoulli":  # each step's noise comes with it, whatever the chunk
            return SynapticInput(jump=xs[steps], noise=noise[steps])
        return xs[steps]

    whole, final = fired(model, given(slice(None)))
    head, middle = fired(model, given(slice(None, split)))
    tail, end = fired(model, given(slice(split, None)), middle)
    np.testing.assert_array_equal(jnp.concatenate([head, tail]), whole)
    for a, b in zip(jax.tree.leaves(end), jax.tree.leaves(final), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("unroll", [2, 8, True])
def test_unrolling_changes_nothing(unroll):
    xs = jnp.asarray(currents(11))
    np.testing.assert_array_equal(fired(LIFCell(0.8), xs, unroll=unroll)[0], fired(LIFCell(0.8), xs)[0])


def test_bf16_inputs_give_bf16_spikes_over_a_float32_membrane():
    xs = jnp.asarray(currents(12), jnp.bfloat16)
    spikes, state = fired(LIFCell(0.8), xs)
    assert spikes.dtype == jnp.bfloat16
    assert state.v.dtype == jnp.float32
    expected, _ = reference.lif(np.asarray(xs, np.float64), 0.8)
    np.testing.assert_array_equal(np.asarray(spikes, np.float64), expected)


def test_recurrent_bf16_carry_keeps_its_dtype():
    xs = jnp.asarray(currents(13), jnp.bfloat16)
    model = RecurrentCell(LIFCell(0.8), jnp.eye(F, dtype=jnp.float32) * 0.2)
    spikes, state = fired(model, xs)
    assert spikes.dtype == jnp.bfloat16
    assert state.output.dtype == jnp.bfloat16


def _two_step_gradient(detach_reset):
    """d s[1] / d x[0] for an LIF that fires at step 0."""
    model = LIFCell(0.8, 1.0, "subtract", Rectangle(width=2.0), detach_reset=detach_reset)

    def second_spike(x0):
        xs = jnp.stack([x0, jnp.asarray(0.9)])[:, None]
        return fired(model, xs)[0][1, 0]

    return jax.grad(second_spike)(jnp.asarray(1.3))


def test_reset_gradient_follows_the_chain_rule():
    # v0 = 1.3 fires; v1 = 0.8 * (1.3 - s0) + 0.9 = 1.14 fires. With a
    # rectangle of width 2 (height 1/2, so both crossings are inside it):
    # ds1/dx0 = g(v1 - 1) * 0.8 * (1 - g(v0 - 1)) = 0.5 * 0.8 * 0.5 = 0.2,
    # and with the reset detached the spike's path drops out:
    # ds1/dx0 = g(v1 - 1) * 0.8 = 0.4.
    # Observed 1.5e-8 relative.
    np.testing.assert_allclose(_two_step_gradient(detach_reset=False), 0.2, rtol=1e-6)
    # Observed 1.5e-8 relative.
    np.testing.assert_allclose(_two_step_gradient(detach_reset=True), 0.4, rtol=1e-6)


def test_gradients_reach_learnable_decays_and_thresholds():
    xs = jnp.asarray(currents(14))

    def loss(decay, threshold):
        spikes, _ = fired(LIFCell(decay, threshold, surrogate=ATan()), xs)
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
    xs = currents(15, scale=1.0)
    spikes, _ = fired(ALIFCell(0.9, 0.97, beta=0.5, threshold=1.0), jnp.asarray(xs))
    expected = reference.eprop_alif(xs.astype(np.float64), 0.9, 0.97, 0.5, 1.0, reset=0.9 * 1.0)
    np.testing.assert_array_equal(spikes, expected)
    assert expected.sum() > 50
    # With their undecayed reset the spike trains differ.
    undecayed = reference.eprop_alif(xs.astype(np.float64), 0.9, 0.97, 0.5, 1.0, reset=1.0)
    assert (undecayed != expected).any()
