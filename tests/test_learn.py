"""Learning rules against the identities that define them."""

from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from sparx.cells import ALIFCell, LIFCell
from sparx.learn import (
    AvgPool,
    ConvLayer,
    DenseLayer,
    EPropParams,
    Flatten,
    Layer,
    MaxPool,
    OTTTLayer,
    bptt_loss,
    eligibility_traces,
    eprop,
    fold_batch_norm,
    normalize,
    ottt,
    relu_forward,
    run_converted,
)
from sparx.learn.events import EventLIF, first_spike_cross_entropy, spike_times
from sparx.surrogate import Sigmoid, Triangle

CELLS = {
    "lif": lambda: LIFCell(decay=float(np.exp(-1 / 20)), threshold=0.6, detach_reset=True,
                           surrogate=Triangle()),
    "alif": lambda: ALIFCell(decay=float(np.exp(-1 / 20)), adapt_decay=float(np.exp(-1 / 200)), beta=0.07,
                             threshold=0.6, detach_reset=True, surrogate=Triangle()),
    # Their numerical verifications run with n_ref = 2.
    "alif_refractory": lambda: ALIFCell(decay=float(np.exp(-1 / 20)), adapt_decay=float(np.exp(-1 / 200)),
                                        beta=0.07, threshold=0.6, detach_reset=True, surrogate=Triangle(),
                                        refractory=2),
}


def problem(seed=0, steps=40, batch=2, n_in=5, n_rec=7, n_out=2):
    rng = np.random.default_rng(seed)
    params = EPropParams(jnp.asarray(rng.normal(0, 0.8, (n_in, n_rec))),
                         jnp.asarray(rng.normal(0, 0.4, (n_rec, n_rec))),
                         jnp.asarray(rng.normal(0, 0.5, (n_rec, n_out))),
                         jnp.asarray(rng.normal(0, 0.1, n_out)))
    inputs = jnp.asarray((rng.random((steps, batch, n_in)) < 0.3).astype(np.float64))
    targets = jnp.asarray(rng.normal(size=(steps, batch, n_out)))
    return params, inputs, targets


def mse(y, target):
    return jnp.mean((y - target) ** 2)


@pytest.mark.parametrize("kind", list(CELLS))
def test_eprop_is_backpropagation_with_the_recurrent_spikes_cut(kind):
    # Bellec et al.'s autodiff check with stop_z_gradients=True: e-prop's
    # online gradients are BPTT's with the gradient stopped at the
    # recurrent spikes, to rounding.
    with jax.enable_x64(new_val=True):
        params, inputs, targets = problem()
        cell, kappa = CELLS[kind](), float(np.exp(-1 / 20))
        total, online = eprop(cell, params, kappa, inputs, targets, mse)
        expected = jax.grad(bptt_loss, argnums=1)(cell, params, kappa, inputs, targets, mse,
                                                  cut_recurrence=True)
        value = bptt_loss(cell, params, kappa, inputs, targets, mse)
    np.testing.assert_allclose(total, value, rtol=1e-12)
    for got, want in zip(online, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
    assert float(jnp.abs(online.w_rec).max()) > 1e-4  # the network spikes and learns


@pytest.mark.parametrize("kind", list(CELLS))
def test_the_eligibility_factorization_is_exactly_backpropagation(kind):
    # Their equation 1: with the true learning signal dE/dz_t (total, through
    # the recurrence), the sum of signal times eligibility trace is BPTT's
    # gradient.
    with jax.enable_x64(new_val=True):
        params, inputs, targets = problem(seed=1)
        cell, kappa = CELLS[kind](), float(np.exp(-1 / 20))

        def loss_of_spikes(zs_shift, params):
            # The loss with an additive probe on every step's spikes, to read dE/dz_t.
            batch, size = inputs.shape[1], params.w_rec.shape[0]

            def step(carry, xs):
                state, z, y = carry
                u, shift = xs
                state, z = cell.step(state, u @ params.w_in + z @ params.w_rec)
                z = z + shift
                y = kappa * y + z @ params.w_out + params.b_out
                return (state, z, y), y

            carry = (cell.init_state((batch, size), inputs.dtype), jnp.zeros((batch, size)),
                     jnp.zeros((batch, params.w_out.shape[1])))
            ys = jax.lax.scan(step, carry, (inputs, zs_shift))[1]
            return jnp.sum(jax.vmap(mse)(ys, targets))

        probe = jnp.zeros((inputs.shape[0], inputs.shape[1], params.w_rec.shape[0]))
        signals = jax.grad(loss_of_spikes)(probe, params)
        traces = eligibility_traces(cell, params, inputs)
        factorized = jnp.einsum("tbn,tbnp->pn", signals, traces)
        bptt = jax.grad(bptt_loss, argnums=1)(cell, params, kappa, inputs, targets, mse)
        eprop_grads = eprop(cell, params, kappa, inputs, targets, mse)[1]
    n_in = inputs.shape[2]
    np.testing.assert_allclose(factorized[:n_in], bptt.w_in, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(factorized[n_in:], bptt.w_rec, rtol=1e-9, atol=1e-12)
    # e-prop is not BPTT once the recurrence carries gradient.
    assert not np.allclose(eprop_grads.w_rec, bptt.w_rec, rtol=1e-3)


OTTT = np.load(Path(__file__).parent / "fixtures" / "ottt.npz")


def test_ottt_matches_xiao_et_als_online_gradients():
    # Their OnlineLIFNode and WrapedSNNOp (tools/make_ottt_fixtures.py): a
    # per-step cross-entropy over 12 steps, gradients accumulated online.
    tau = float(OTTT["tau"])
    names = ("first", "hidden", "readout")
    with jax.enable_x64(new_val=True):
        layers = [OTTTLayer(jnp.asarray(OTTT[f"{n}/weight"]), jnp.asarray(OTTT[f"{n}/bias"])) for n in names]
        cell = LIFCell(decay=1 - 1 / tau, threshold=1.0, detach_reset=True, surrogate=Sigmoid(4.0))
        inputs = jnp.asarray(OTTT["inputs"])
        labels = jnp.broadcast_to(jnp.asarray(OTTT["labels"]), (inputs.shape[0], inputs.shape[1]))
        steps = inputs.shape[0]

        def loss(logits, label):
            return optax.softmax_cross_entropy_with_integer_labels(logits, label).mean() / steps

        _, grads = ottt([cell, cell], layers, 1 - 1 / tau, inputs, labels, loss)
    for name, grad in zip(names, grads, strict=True):
        np.testing.assert_allclose(grad.weight, OTTT[f"{name}/grad_weight"], rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(grad.bias, OTTT[f"{name}/grad_bias"], rtol=1e-10, atol=1e-13)


def test_an_if_neuron_fires_at_its_normalized_drive_within_one_over_t():
    drive = jnp.linspace(-0.3, 1.4, 41)[:, None]
    for steps in (10, 100, 1000):
        (rate,) = run_converted([DenseLayer(jnp.eye(1), jnp.zeros(1))], drive, steps)
        np.testing.assert_array_less(np.abs(rate - jnp.clip(drive, 0, 1)), 1 / steps + 1e-6)


def test_a_converted_network_approaches_its_anns_accuracy():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 1, (4, 10))
    labels = rng.integers(0, 4, 2000)
    points = centers[labels] + rng.normal(0, 0.8, (2000, 10))
    x = jnp.asarray(np.clip(0.5 + 0.25 * points, 0, 1), jnp.float32)
    y = jnp.asarray(labels)
    sizes = (10, 32, 32, 4)
    keys = jax.random.split(jax.random.key(0), 3)
    layers = [DenseLayer(jax.random.normal(k, (a, b)) * np.sqrt(2 / a), jnp.zeros(b))
              for k, a, b in zip(keys, sizes[:-1], sizes[1:], strict=True)]
    optimizer = optax.adam(1e-2)
    state = optimizer.init(layers)

    @jax.jit
    def train(layers, state):
        def loss(layers):
            logits = relu_forward(layers, x[:1500])[-1]
            return optax.softmax_cross_entropy_with_integer_labels(logits, y[:1500]).mean()

        updates, state = optimizer.update(jax.grad(loss)(layers), state)
        return optax.apply_updates(layers, updates), state

    for _ in range(300):
        layers, state = train(layers, state)
    test_x, test_y = x[1500:], y[1500:]
    ann = float(jnp.mean(jnp.argmax(relu_forward(layers, test_x)[-1], -1) == test_y))
    converted = normalize(layers, x[:1500])
    accuracy = {steps: float(jnp.mean(jnp.argmax(run_converted(converted, test_x, steps)[-1], -1) == test_y))
                for steps in (5, 300)}
    assert ann > 0.85
    assert accuracy[300] >= ann - 0.02 and accuracy[5] < accuracy[300]
    # Hidden rates follow the normalized activations.
    rates = run_converted(converted, test_x, 300)
    activations = relu_forward(converted, test_x)
    hidden = np.clip(np.asarray(activations[0]).ravel(), 0, 1)
    assert np.corrcoef(np.asarray(rates[0]).ravel(), hidden)[0, 1] > 0.99


class _CNN(nn.Module):
    """Conv, batch norm, ReLU, max pool, conv, batch norm, ReLU, average pool, dense: every layer kind
    Rueckauer et al. convert."""

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(8, (3, 3))(x)
        x = nn.relu(nn.BatchNorm(use_running_average=not train, momentum=0.9)(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.Conv(16, (3, 3), padding="VALID")(x)
        x = nn.relu(nn.BatchNorm(use_running_average=not train, momentum=0.9)(x))
        x = nn.avg_pool(x, (2, 2), (1, 1))
        return nn.Dense(4)(x.reshape(x.shape[0], -1))


def _converted_layers(variables) -> list[Layer]:
    params, stats = variables["params"], variables["batch_stats"]

    def conv(name, bn, padding):
        layer = ConvLayer(params[name]["kernel"], params[name]["bias"], padding=padding)
        return fold_batch_norm(layer, stats[bn]["mean"], stats[bn]["var"], params[bn]["scale"],
                               params[bn]["bias"])

    return [conv("Conv_0", "BatchNorm_0", "SAME"), MaxPool(), conv("Conv_1", "BatchNorm_1", "VALID"),
            AvgPool((2, 2), (1, 1)), Flatten(),
            DenseLayer(params["Dense_0"]["kernel"], params["Dense_0"]["bias"])]


def _bars(n, seed):
    """8x8 images of one bar, horizontal, vertical or on either diagonal, at a random place, in noise."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, n)
    images = rng.uniform(0, 0.3, (n, 8, 8, 1))
    for image, label, (a, b) in zip(images, labels, rng.integers(0, 6, (n, 2)), strict=True):
        for t in range(3):
            i, j = [(a, b + t), (a + t, b), (a + t, b + t), (a + t, b + 2 - t)][label]
            image[i, j, 0] = rng.uniform(0.7, 1.0)
    return jnp.asarray(images, jnp.float32), jnp.asarray(labels)


def test_folding_batch_norm_keeps_the_networks_outputs():
    # Random statistics, so the fold is exercised away from mean 0, variance 1.
    x, _ = _bars(64, seed=1)
    variables = _CNN().init(jax.random.key(1), x)
    leaves, tree = jax.tree.flatten(variables)
    keys = jax.random.split(jax.random.key(2), len(leaves))
    variables = jax.tree.unflatten(tree, [jax.random.uniform(k, v.shape, v.dtype, 0.2, 1.5)
                                          for k, v in zip(keys, leaves, strict=True)])
    unfolded = _CNN().apply(variables, x)
    folded = relu_forward(_converted_layers(variables), x)[-1]
    np.testing.assert_allclose(folded, unfolded, rtol=1e-5, atol=1e-5)
    assert float(jnp.std(unfolded)) > 0.1


def test_a_converted_cnn_keeps_its_anns_predictions():
    x, y = _bars(1500, seed=0)
    model = _CNN()
    variables = model.init(jax.random.key(0), x[:1])
    optimizer = optax.adam(1e-2)
    state = optimizer.init(variables["params"])

    @jax.jit
    def train(params, stats, state):
        def loss(params):
            logits, updates = model.apply({"params": params, "batch_stats": stats}, x[:1000], train=True,
                                          mutable=["batch_stats"])
            return optax.softmax_cross_entropy_with_integer_labels(logits, y[:1000]).mean(), updates

        grads, updates = jax.grad(loss, has_aux=True)(params)
        steps, state = optimizer.update(grads, state)
        return optax.apply_updates(params, steps), updates["batch_stats"], state

    params, stats = variables["params"], variables["batch_stats"]
    for _ in range(150):
        params, stats, state = train(params, stats, state)
    variables = {"params": params, "batch_stats": stats}
    test_x, test_y = x[1000:], y[1000:]
    ann = model.apply(variables, test_x).argmax(-1)
    converted = normalize(_converted_layers(variables), x[:1000])
    snn = {steps: run_converted(converted, test_x, steps) for steps in (5, 300)}
    agreement = {steps: float(jnp.mean(rates[-1].argmax(-1) == ann)) for steps, rates in snn.items()}
    assert float(jnp.mean(ann == test_y)) > 0.95
    assert agreement[300] > 0.97 and agreement[5] < agreement[300]
    # Hidden rates follow the normalized activations: the first convolution's within the IF bound,
    # the second's after a gated max pool.
    activations = relu_forward(converted, test_x)
    for layer, threshold in ((0, 0.99), (2, 0.95)):
        hidden = np.clip(np.asarray(activations[layer]).ravel(), 0, 1)
        assert np.corrcoef(np.asarray(snn[300][layer]).ravel(), hidden)[0, 1] > threshold


def test_max_pool_gating_passes_the_most_active_inputs_spikes():
    # A 1x1 identity convolution makes each pixel an IF neuron firing at its drive; the gated pool
    # should fire at the largest rate in each 2x2 window, which is neither their sum nor their mean.
    rng = np.random.default_rng(3)
    drive = jnp.asarray(rng.uniform(0, 1, (16, 4, 4, 1)), jnp.float32)
    layers = [ConvLayer(jnp.ones((1, 1, 1, 1)), jnp.zeros(1)), MaxPool()]
    expected = np.asarray(drive).reshape(16, 2, 2, 2, 2, 1).max(axis=(2, 4))
    assert np.abs(expected - np.asarray(drive).reshape(16, 2, 2, 2, 2, 1).mean(axis=(2, 4))).max() > 0.5
    for steps in (200, 2000, 20000):
        _, pooled = run_converted(layers, drive, steps)
        # The gate passes one input's spikes at a time, so it never outfires the most active one.
        np.testing.assert_array_less(np.asarray(pooled), expected + 1 / steps + 1e-6)
        # Until the counts rank the inputs, the gate may follow a slower one, for longer the closer
        # the rates; that costs a fixed number of spikes, so the error falls as 1 / T. The closest
        # pair of rates here differs by 0.003.
        assert np.abs(np.asarray(pooled) - expected).max() < 100 / steps


STB = np.load(Path(__file__).parent / "fixtures" / "snntoolbox.npz")


def test_conversion_matches_rueckauer_et_als_toolbox():
    # Their SNN toolbox on a Keras CNN with batch norm, max and average pooling
    # (tools/make_snntoolbox_fixtures.py), at its defaults: the 99.9th percentile,
    # analog input, reset by subtraction, max pooling gated by spike counts.
    def conv(name, bn, padding):
        weight, bias = jnp.asarray(STB[f"{name}/weight"]), jnp.asarray(STB[f"{name}/bias"])
        layer = ConvLayer(weight, bias, padding=padding)
        return fold_batch_norm(layer, STB[f"{bn}/mean"], STB[f"{bn}/var"], STB[f"{bn}/scale"],
                               STB[f"{bn}/offset"], float(STB[f"{bn}/epsilon"]))

    layers = [conv("conv0", "bn0", "SAME"), MaxPool(), conv("conv1", "bn1", "VALID"), AvgPool((2, 2), (1, 1)),
              Flatten(), DenseLayer(jnp.asarray(STB["dense/weight"]), jnp.asarray(STB["dense/bias"]))]
    converted = normalize(layers, jnp.asarray(STB["x_norm"]))
    # Folding and normalization give the weights their simulator runs.
    for k, layer in enumerate(layer for layer in converted if isinstance(layer, ConvLayer | DenseLayer)):
        np.testing.assert_allclose(layer.weight, STB[f"normalized/{k}/weight"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(layer.bias, STB[f"normalized/{k}/bias"], rtol=1e-5, atol=1e-6)
    steps = int(STB["steps"])
    rates = run_converted(converted, jnp.asarray(STB["x_test"]), steps)
    theirs = [STB[name] / steps for name in sorted((n for n in STB.files if n.startswith("counts/")),
                                                   key=lambda n: int(n.split("/")[-1]))]
    # The first layer fires the same spikes.
    np.testing.assert_array_equal(np.round(np.asarray(rates[0]) * steps), STB["counts/SpikeConv2D/0"])
    # Their gate counts the current step's spikes, so an input can win a tie with the spike it fires
    # now; their pool then outfires its most active input, here by up to 0.26. Ours does not, so the
    # layers after it differ by those spikes, and the predictions agree.
    most_active = theirs[0].reshape(-1, 4, 2, 4, 2, 8).max(axis=(2, 4))
    assert (theirs[1] - most_active).max() > 0.2
    np.testing.assert_array_less(np.asarray(rates[1]), most_active + 1e-6)
    assert np.corrcoef(np.asarray(rates[-1]).ravel(), theirs[-1].ravel())[0, 1] > 0.99
    assert np.array_equal(np.asarray(rates[-1]).argmax(-1), theirs[-1].argmax(-1))


def _event_problem(seed=0, batch=3, inputs=6, outputs=4):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, 20, (batch, inputs, 3)), axis=-1)
    times[rng.random(times.shape) < 0.3] = np.inf
    return jnp.asarray(times), jnp.asarray(rng.normal(1.2, 0.8, (inputs, outputs)))


def test_event_gradients_are_the_exact_derivatives_of_spike_times():
    # EventProp's gradient is the derivative of the event-based dynamics
    # with spike counts fixed: central differences of the exact simulation.
    neuron = EventLIF()
    with jax.enable_x64(new_val=True):
        inputs, weights = _event_problem()

        @jax.jit
        def loss(weights, inputs):
            times, _ = spike_times(inputs, weights, neuron, horizon=60.0, capacity=4)
            return 1e-3 * jnp.sum(jnp.where(jnp.isfinite(times), times, 0.0) ** 2)

        grad_w, grad_in = jax.grad(loss, argnums=(0, 1))(weights, inputs)
        assert np.isfinite(np.asarray(spike_times(inputs, weights, neuron, 60.0, 4)[0])).sum() > 15
        eps = 1e-6
        for index in [(0, 0), (2, 1), (5, 3), (3, 2)]:
            step = jnp.zeros_like(weights).at[index].set(eps)
            numeric = (loss(weights + step, inputs) - loss(weights - step, inputs)) / (2 * eps)
            np.testing.assert_allclose(grad_w[index], numeric, rtol=1e-6, atol=1e-8)
        finite = np.argwhere(np.isfinite(np.asarray(inputs)))[:4]
        for index in map(tuple, finite):
            step = jnp.zeros_like(inputs).at[index].set(eps)
            numeric = (loss(weights, inputs + step) - loss(weights, inputs - step)) / (2 * eps)
            np.testing.assert_allclose(grad_in[index], numeric, rtol=1e-6, atol=1e-8)


def test_event_simulation_is_the_lif_integrated_on_a_fine_grid():
    # The same neurons in sparx.dynamics (exact integration, exponential
    # current synapses) at 1 us fire at the same times, to the grid.
    from sparx.dynamics import LIF, Arrivals, Exponential, PointNeuron, Receptor, integrate

    neuron = EventLIF()
    dt, horizon = 0.001, 60.0
    with jax.enable_x64(new_val=True):
        inputs, weights = _event_problem(seed=3, batch=1)
        exact, _ = spike_times(inputs, weights, neuron, horizon, capacity=6)
        steps = round(horizon / dt)
        arrivals = np.zeros((steps, weights.shape[1]))
        for source, when in np.argwhere(np.isfinite(np.asarray(inputs[0]))):
            step = round(float(inputs[0, source, when]) / dt) - 1  # lands at the end of this step
            arrivals[step] += np.asarray(weights[source])
        lif = LIF(tau_m=neuron.tau_mem, c_m=neuron.tau_mem, e_l=0.0, v_th=1.0, v_reset=0.0, t_ref=0.0)
        cell = PointNeuron(lif, {"syn": Receptor(Exponential(neuron.tau_syn))})
        fired, _ = integrate(cell, Arrivals(0.0, {"syn": jnp.asarray(arrivals)}), dt)
    for n in range(weights.shape[1]):
        grid = (np.flatnonzero(np.asarray(fired.fired[:, n])) + 1) * dt
        want = np.sort(np.asarray(exact[0, n]))
        want = want[np.isfinite(want)]
        assert len(grid) == len(want)
        np.testing.assert_allclose(grid, want, atol=3 * dt)
    assert np.isfinite(np.asarray(exact)).sum() >= 3


def test_a_two_layer_event_network_learns_spike_latencies():
    # Two classes of input spike patterns; the output neuron of the right
    # class must fire first.
    rng = np.random.default_rng(0)
    patterns = rng.uniform(0, 10, (2, 8, 1))
    labels = jnp.asarray(rng.integers(0, 2, 32))
    jitter = rng.normal(0, 0.5, (32, 8, 1))
    inputs = jnp.asarray(np.clip(patterns[np.asarray(labels)] + jitter, 0, None))
    neuron = EventLIF()
    params = (jnp.asarray(rng.normal(1.0, 0.5, (8, 12))), jnp.asarray(rng.normal(1.0, 0.5, (12, 2))))

    def loss(params):
        hidden, _ = spike_times(inputs, params[0], neuron, horizon=40.0, capacity=2)
        out, _ = spike_times(hidden, params[1], neuron, horizon=40.0, capacity=2)
        return first_spike_cross_entropy(out, labels, silent=40.0), out

    optimizer = optax.adam(1e-2)
    state = optimizer.init(params)
    step = jax.jit(jax.value_and_grad(loss, has_aux=True))
    for _ in range(60):
        (value, out), grads = step(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
    first = jnp.min(out, -1)
    accuracy = float(jnp.mean(jnp.argmin(first, -1) == labels))
    assert accuracy >= 0.95 and np.isfinite(float(value))
