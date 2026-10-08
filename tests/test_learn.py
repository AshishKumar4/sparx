"""Learning rules against the identities that define them."""

import functools
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from sparx.dynamics import ALIFCell, BernoulliCell, LICell, LIFCell, Serial, SynapticInput, decay
from sparx.learn import (
    EPropParams,
    EventLIF,
    OTTTLayer,
    ReinforceParams,
    SpikingMaxPool,
    bptt_loss,
    convert,
    eligibility_traces,
    eprop,
    first_spike_cross_entropy,
    fold_batch_norm,
    normalize,
    ottt,
    policy_gradient,
    reinforce,
    run_converted,
    spike_times,
)
from sparx.nn import RATES, STATE, Flatten
from sparx.surrogate import Sigmoid, Triangle

CELLS = {
    "lif": lambda: LIFCell(decay=decay(20.0), threshold=0.6, detach_reset=True,
                           surrogate=Triangle()),
    "alif": lambda: ALIFCell(decay=decay(20.0), adapt_decay=decay(200.0), beta=0.07,
                             threshold=0.6, detach_reset=True, surrogate=Triangle()),
    # Their numerical verifications run with n_ref = 2.
    "alif_refractory": lambda: ALIFCell(decay=decay(20.0), adapt_decay=decay(200.0),
                                        beta=0.07, threshold=0.6, detach_reset=True, surrogate=Triangle(),
                                        refractory=2),
    # With the reset in the gradient the membrane's Jacobian depends on the
    # state, so every state variable keeps an eligibility vector per synapse.
    "lif_reset_gradient": lambda: LIFCell(decay=decay(20.0), threshold=0.6, surrogate=Triangle()),
    # The synaptic current filters the input with a constant decay; the
    # membrane integrates it, so its eligibility reads the filtered input.
    # A decay per neuron: the membrane's filter differs between neurons.
    "lif_per_neuron": lambda: LIFCell(decay=jnp.linspace(0.8, 0.97, 7), threshold=0.6, detach_reset=True,
                                      surrogate=Triangle()),
    "synaptic": lambda: Serial(LICell(decay(5.0)), LIFCell(decay=decay(20.0), threshold=0.3,
                                                           detach_reset=True, surrogate=Triangle())),
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
        cell = CELLS[kind]()
        total, online = eprop(cell, params, inputs, targets, mse, tau=20.0)
        expected = jax.grad(bptt_loss, argnums=1)(cell, params, inputs, targets, mse, tau=20.0,
                                                  cut_recurrence=True)
        value = bptt_loss(cell, params, inputs, targets, mse, tau=20.0)
    np.testing.assert_allclose(total, value, rtol=1e-12)  # observed 2.0e-16 relative
    for got, want in zip(online, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)  # observed 1.4e-12
    assert np.abs(np.asarray(online.w_rec)).max() > 1e-4  # the network spikes and learns


@pytest.mark.parametrize("kind", list(CELLS))
def test_the_eligibility_factorization_is_exactly_backpropagation(kind):
    # Their equation 1: with the true learning signal dE/dz_t (total, through
    # the recurrence), the sum of signal times eligibility trace is BPTT's
    # gradient.
    with jax.enable_x64(new_val=True):
        params, inputs, targets = problem(seed=1)
        cell, kappa = CELLS[kind](), decay(20.0)

        def loss_of_spikes(zs_shift, params):
            # The loss with an additive probe on every step's spikes, to read dE/dz_t.
            batch, size = inputs.shape[1], params.w_rec.shape[0]

            def step(carry, xs):
                state, z, y = carry
                u, shift = xs
                state, spikes = cell.step(state, SynapticInput(jump=u @ params.w_in + z @ params.w_rec), 1.0)
                z = spikes.value
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
        bptt = jax.grad(bptt_loss, argnums=1)(cell, params, inputs, targets, mse, tau=20.0)
        eprop_grads = eprop(cell, params, inputs, targets, mse, tau=20.0)[1]
    n_in = inputs.shape[2]
    np.testing.assert_allclose(factorized[:n_in], bptt.w_in, rtol=1e-9, atol=1e-12)  # observed 1.4e-12
    np.testing.assert_allclose(factorized[n_in:], bptt.w_rec, rtol=1e-9, atol=1e-12)  # observed 2.4e-12
    # e-prop is not BPTT once the recurrence carries gradient.
    assert not np.allclose(eprop_grads.w_rec, bptt.w_rec, rtol=1e-3)


def test_eprop_steps_the_cell_and_the_readout_at_dt():
    # At dt = 2 a decay per unit of time is applied squared, so halving the
    # time constants' steps gives the network at dt = 1: the same gradients.
    with jax.enable_x64(new_val=True):
        params, inputs, targets = problem(seed=2)
        cell = LIFCell(decay=decay(20.0), threshold=0.6, detach_reset=True, surrogate=Triangle())
        halved = LIFCell(decay=decay(40.0), threshold=0.6, detach_reset=True, surrogate=Triangle())
        total, grads = eprop(cell, params, inputs, targets, mse, tau=20.0)
        total_dt, grads_dt = eprop(halved, params, inputs, targets, mse, tau=40.0, dt=2.0)
        bptt_dt = jax.grad(bptt_loss, argnums=1)(halved, params, inputs, targets, mse, tau=40.0, dt=2.0,
                                                 cut_recurrence=True)
    np.testing.assert_allclose(total_dt, total, rtol=1e-12)  # observed 2.0e-15 relative
    for got, want, bptt in zip(grads_dt, grads, bptt_dt, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)  # observed 8.2e-12
        np.testing.assert_allclose(got, bptt, rtol=1e-9, atol=1e-12)  # observed 6.8e-13


def test_eprop_takes_the_cell_as_a_traced_argument():
    # The cell's constants are tracers under jit; the gradients are those of the eager call.
    with jax.enable_x64(new_val=True):
        params, inputs, targets = problem(seed=3)
        cell = CELLS["alif_refractory"]()
        eager = eprop(cell, params, inputs, targets, mse, tau=20.0)[1]
        jitted = jax.jit(lambda cell, params: eprop(cell, params, inputs, targets, mse, tau=20.0))
        jitted = jitted(cell, params)[1]
    for got, want in zip(jitted, eager, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)  # observed 0


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

        # Their trace decays by 1 - 1 / tau, the leak of the time constant below.
        _, grads = ottt([cell, cell], layers, inputs, labels, loss, tau=-1 / np.log(1 - 1 / tau))
    for name, grad in zip(names, grads, strict=True):
        # Observed 1.1e-16.
        np.testing.assert_allclose(grad.weight, OTTT[f"{name}/grad_weight"], rtol=1e-10, atol=1e-13)
        # Observed 5.6e-17.
        np.testing.assert_allclose(grad.bias, OTTT[f"{name}/grad_bias"], rtol=1e-10, atol=1e-13)


def _spiking_rates(snn, variables, x, steps, chunk=50):
    """The converted network's output rates, and each spiking layer's rates by its index in the stack,
    run `chunk` steps at a time as `run_converted` runs."""
    state, out, rates = {}, 0, {}
    for start in range(0, steps, chunk):
        n = min(chunk, steps - start)
        spikes, updated = snn.apply({**variables, STATE: state}, jnp.broadcast_to(x, (n, *x.shape)),
                                    mutable=[STATE, RATES])
        state, out = updated[STATE], out + spikes.sum(0)
        for name, value in updated[RATES].items():
            k = int(name.split("_")[1])
            rates[k] = rates.get(k, 0) + value["rate"][0] * n / steps
    return out / steps, rates


def test_running_in_chunks_is_one_run():
    rng = np.random.default_rng(4)
    model = nn.Sequential([nn.Dense(6), nn.relu, nn.Dense(3)])
    x = jnp.asarray(rng.uniform(0, 1, (8, 4)), jnp.float32)
    snn, variables = convert(model, normalize(model, model.init(jax.random.key(0), x), x))
    whole = snn.apply(variables, jnp.broadcast_to(x, (70, *x.shape))).mean(0)
    np.testing.assert_allclose(run_converted(snn, variables, x, 70, chunk=30), whole, atol=1e-6)  # observed 0
    assert float(whole.max()) > 0


def test_an_if_neuron_fires_at_its_normalized_drive_within_one_over_t():
    drive = jnp.linspace(-0.3, 1.4, 41)[:, None]
    snn, variables = convert(nn.Sequential([nn.Dense(1)]),
                             {"params": {"layers_0": {"kernel": jnp.eye(1), "bias": jnp.zeros(1)}}})
    for steps in (10, 100, 1000):
        rate = run_converted(snn, variables, drive, steps)
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
    model = nn.Sequential([nn.Dense(32), nn.relu, nn.Dense(32), nn.relu, nn.Dense(4)])
    params = {f"layers_{2 * i}": {"kernel": jax.random.normal(k, (a, b)) * np.sqrt(2 / a),
                                  "bias": jnp.zeros(b)}
              for i, (k, a, b) in enumerate(zip(keys, sizes[:-1], sizes[1:], strict=True))}
    optimizer = optax.adam(1e-2)
    state = optimizer.init(params)

    @jax.jit
    def train(params, state):
        def loss(params):
            logits = model.apply({"params": params}, x[:1500])
            return optax.softmax_cross_entropy_with_integer_labels(logits, y[:1500]).mean()

        updates, state = optimizer.update(jax.grad(loss)(params), state)
        return optax.apply_updates(params, updates), state

    for _ in range(300):
        params, state = train(params, state)
    test_x, test_y = x[1500:], y[1500:]
    ann = float(jnp.mean(jnp.argmax(model.apply({"params": params}, test_x), -1) == test_y))
    normalized = normalize(model, {"params": params}, x[:1500])
    snn, variables = convert(model, normalized)
    accuracy = {steps: float(jnp.mean(jnp.argmax(run_converted(snn, variables, test_x, steps), -1) == test_y))
                for steps in (5, 300)}
    assert ann > 0.85
    assert accuracy[300] >= ann - 0.02 and accuracy[5] < accuracy[300]
    # Hidden rates follow the normalized activations.
    _, rates = _spiking_rates(snn, variables, test_x, 300)
    first = normalized["params"]["layers_0"]
    hidden = np.clip(np.asarray(test_x @ first["kernel"] + first["bias"]).ravel(), 0, 1)
    assert np.corrcoef(np.asarray(rates[1]).ravel(), hidden)[0, 1] > 0.99


def _cnn(train=False):
    """Conv, batch norm, ReLU, max pool, conv, batch norm, ReLU, average pool, dense: every layer kind
    Rueckauer et al. convert."""
    return nn.Sequential([
        nn.Conv(8, (3, 3)), nn.BatchNorm(use_running_average=not train, momentum=0.9), nn.relu,
        functools.partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2)),
        nn.Conv(16, (3, 3), padding="VALID"), nn.BatchNorm(use_running_average=not train, momentum=0.9),
        nn.relu,
        functools.partial(nn.avg_pool, window_shape=(2, 2), strides=(1, 1)),
        Flatten(), nn.Dense(4),
    ])


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
    variables = _cnn().init(jax.random.key(1), x)
    leaves, tree = jax.tree.flatten(variables)
    keys = jax.random.split(jax.random.key(2), len(leaves))
    variables = jax.tree.unflatten(tree, [jax.random.uniform(k, v.shape, v.dtype, 0.2, 1.5)
                                          for k, v in zip(keys, leaves, strict=True)])
    unfolded = _cnn().apply(variables, x)
    model, folded = fold_batch_norm(_cnn(), variables)
    assert not any(isinstance(layer, nn.BatchNorm) for layer in model.layers)
    np.testing.assert_allclose(model.apply(folded, x), unfolded, rtol=1e-5, atol=1e-5)  # observed 7.3e-4
    assert float(jnp.std(unfolded)) > 0.1


def test_a_converted_cnn_keeps_its_anns_predictions():
    x, y = _bars(1500, seed=0)
    training = _cnn(train=True)
    variables = training.init(jax.random.key(0), x[:1])
    optimizer = optax.adam(1e-2)
    state = optimizer.init(variables["params"])

    @jax.jit
    def train(params, stats, state):
        def loss(params):
            logits, updates = training.apply({"params": params, "batch_stats": stats}, x[:1000],
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
    ann = _cnn().apply(variables, test_x).argmax(-1)
    model, folded = fold_batch_norm(_cnn(), variables)
    normalized = normalize(model, folded, x[:1000])
    snn, snn_variables = convert(model, normalized)
    snn_rates = {steps: _spiking_rates(snn, snn_variables, test_x, steps) for steps in (5, 300)}
    agreement = {steps: float(jnp.mean(out.argmax(-1) == ann)) for steps, (out, _) in snn_rates.items()}
    assert float(jnp.mean(ann == test_y)) > 0.95
    assert agreement[300] > 0.97 and agreement[5] < agreement[300]
    # Hidden rates follow the normalized activations: the first convolution's within the IF bound,
    # the second's after a gated max pool. The folded stack's convolutions are its layers 0 and 3,
    # and the IF neurons after them the converted stack's layers 1 and 4.
    _, captured = model.apply(normalized, test_x, capture_intermediates=True)
    for conv, neuron, threshold in ((0, 1, 0.99), (3, 4, 0.95)):
        hidden = np.clip(np.asarray(captured["intermediates"][f"layers_{conv}"]["__call__"][0]).ravel(), 0, 1)
        assert np.corrcoef(np.asarray(snn_rates[300][1][neuron]).ravel(), hidden)[0, 1] > threshold


def test_max_pool_gating_passes_the_most_active_inputs_spikes():
    # A 1x1 identity convolution makes each pixel an IF neuron firing at its drive; the gated pool
    # should fire at the largest rate in each 2x2 window, which is neither their sum nor their mean.
    rng = np.random.default_rng(3)
    drive = jnp.asarray(rng.uniform(0, 1, (16, 4, 4, 1)), jnp.float32)
    model = nn.Sequential([nn.Conv(1, (1, 1)), nn.relu,
                           functools.partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2))])
    snn, variables = convert(model, {"params": {"layers_0": {"kernel": jnp.ones((1, 1, 1, 1)),
                                                             "bias": jnp.zeros(1)}}})
    assert isinstance(snn.layers[-1], SpikingMaxPool)
    expected = np.asarray(drive).reshape(16, 2, 2, 2, 2, 1).max(axis=(2, 4))
    assert np.abs(expected - np.asarray(drive).reshape(16, 2, 2, 2, 2, 1).mean(axis=(2, 4))).max() > 0.5
    for steps in (200, 2000, 20000):
        pooled = run_converted(snn, variables, drive, steps)
        # The gate passes one input's spikes at a time, so it never outfires the most active one.
        np.testing.assert_array_less(np.asarray(pooled), expected + 1 / steps + 1e-6)
        # Until the counts rank the inputs, the gate may follow a slower one, for longer the closer
        # the rates; that costs a fixed number of spikes, so the error falls as 1 / T. The closest
        # pair of rates here differs by 0.003.
        assert np.abs(np.asarray(pooled) - expected).max() < 100 / steps


def test_a_converted_mlp_round_trips_through_nir():
    # With NIR's reset to zero, the converted network exports, and the imported network fires the
    # same spikes.
    pytest.importorskip("nir")
    from sparx.nir import from_nir, to_nir

    x = jax.random.uniform(jax.random.key(0), (32, 5))
    model = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(3)])
    normalized = normalize(model, model.init(jax.random.key(1), x), x)
    snn, variables = convert(model, normalized, reset="zero")
    dt = 2.0 ** -10  # seconds; r dt is then 1 exactly, so the weights come back unchanged
    back, back_variables = from_nir(to_nir(snn, variables, dt=dt), dt=dt)
    assert [type(layer) for layer in back.layers] == [type(layer) for layer in snn.layers]
    for name, layer in variables["params"].items():
        for key, value in layer.items():
            np.testing.assert_array_equal(back_variables["params"][name][key], value)
    xs = jnp.broadcast_to(x, (40, *x.shape))
    spikes = snn.apply(variables, xs)
    np.testing.assert_array_equal(back.apply(back_variables, xs), spikes)
    assert float(spikes.mean()) > 0.05


STB = np.load(Path(__file__).parent / "fixtures" / "snntoolbox.npz")


def _channels_first_rows(kernel, shape):
    """A dense kernel after Keras's flatten, whose rows run H, W, C, with its rows in `sparx.nn.Flatten`'s
    order, C, H, W."""
    h, w, c = shape
    return kernel.reshape(h, w, c, -1).transpose(2, 0, 1, 3).reshape(h * w * c, -1)


def test_conversion_matches_rueckauer_et_als_toolbox():
    # Their SNN toolbox on a Keras CNN with batch norm, max and average pooling
    # (tools/make_snntoolbox_fixtures.py), at its defaults: the 99.9th percentile,
    # analog input, reset by subtraction, max pooling gated by spike counts.
    # The dense layer reads Keras's flatten of the average pool's [1, 1, 16] map; on a 1 x 1 map
    # its H, W, C order is sparx.nn.Flatten's C, H, W, and the reordering below is the identity.
    flat = STB["counts/SpikeAveragePooling2D/3"].shape[1:]
    model = nn.Sequential([
        nn.Conv(8, (3, 3), padding="SAME"),
        nn.BatchNorm(use_running_average=True, epsilon=float(STB["bn0/epsilon"])),
        nn.relu, functools.partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2)),
        nn.Conv(16, (3, 3), padding="VALID"),
        nn.BatchNorm(use_running_average=True, epsilon=float(STB["bn1/epsilon"])), nn.relu,
        functools.partial(nn.avg_pool, window_shape=(2, 2), strides=(1, 1)), Flatten(), nn.Dense(4),
    ])

    def batch_norm(name):
        return ({"scale": STB[f"{name}/scale"], "bias": STB[f"{name}/offset"]},
                {"mean": STB[f"{name}/mean"], "var": STB[f"{name}/var"]})

    (bn0, stats0), (bn1, stats1) = batch_norm("bn0"), batch_norm("bn1")
    variables = {
        "params": {"layers_0": {"kernel": STB["conv0/weight"], "bias": STB["conv0/bias"]}, "layers_1": bn0,
                   "layers_4": {"kernel": STB["conv1/weight"], "bias": STB["conv1/bias"]}, "layers_5": bn1,
                   "layers_9": {"kernel": _channels_first_rows(STB["dense/weight"], flat),
                                "bias": STB["dense/bias"]}},
        "batch_stats": {"layers_1": stats0, "layers_5": stats1},
    }
    folded_model, folded = fold_batch_norm(model, variables)
    normalized = normalize(folded_model, folded, jnp.asarray(STB["x_norm"]))
    # Folding and normalization give the weights their simulator runs.
    for k, name in enumerate(("layers_0", "layers_3", "layers_7")):
        want = STB[f"normalized/{k}/weight"]
        if k == 2:
            want = _channels_first_rows(want, flat)
        # Observed 2.4e-7.
        np.testing.assert_allclose(normalized["params"][name]["kernel"], want, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(normalized["params"][name]["bias"], STB[f"normalized/{k}/bias"], rtol=1e-5,
                                   atol=1e-6)  # observed 1.8e-7
    snn, snn_variables = convert(folded_model, normalized)
    steps = int(STB["steps"])
    out, rates = _spiking_rates(snn, snn_variables, jnp.asarray(STB["x_test"]), steps)
    theirs = [STB[name] / steps for name in sorted((n for n in STB.files if n.startswith("counts/")),
                                                   key=lambda n: int(n.split("/")[-1]))]
    # The first layer fires the same spikes: the converted stack's layer 1, the IF after the convolution.
    np.testing.assert_array_equal(np.round(np.asarray(rates[1]) * steps), STB["counts/SpikeConv2D/0"])
    # Their gate counts the current step's spikes, so an input can win a tie with the spike it fires
    # now; their pool then outfires its most active input, here by up to 0.21. Ours does not, so the
    # layers after it differ by those spikes, and the predictions agree.
    most_active = theirs[0].reshape(-1, 4, 2, 4, 2, 8).max(axis=(2, 4))
    assert (theirs[1] - most_active).max() > 0.2
    np.testing.assert_array_less(np.asarray(rates[2]), most_active + 1e-6)
    assert np.corrcoef(np.asarray(out).ravel(), theirs[-1].ravel())[0, 1] > 0.99
    assert np.array_equal(np.asarray(out).argmax(-1), theirs[-1].argmax(-1))


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
            np.testing.assert_allclose(grad_w[index], numeric, rtol=1e-6, atol=1e-8)  # observed 9.7e-10
        finite = np.argwhere(np.isfinite(np.asarray(inputs)))[:4]
        for index in map(tuple, finite):
            step = jnp.zeros_like(inputs).at[index].set(eps)
            numeric = (loss(weights, inputs + step) - loss(weights, inputs - step)) / (2 * eps)
            np.testing.assert_allclose(grad_in[index], numeric, rtol=1e-6, atol=1e-8)  # observed 8.3e-10


def test_event_simulation_is_the_lif_integrated_on_a_fine_grid():
    # The same neurons in sparx.dynamics (exact integration, exponential
    # current synapses) at 1 us fire at the same times, to the grid.
    from sparx.dynamics import Arrivals, Exponential, LeakyIntegrateAndFire, PointNeuron, Receptor, run

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
        lif = LeakyIntegrateAndFire(tau_m=neuron.tau_m, c_m=neuron.tau_m, e_l=0.0, v_th=neuron.v_th,
                                    v_reset=0.0, t_ref=0.0)
        cell = PointNeuron(lif, {"syn": Receptor(Exponential(neuron.tau_syn))})
        fired, _ = run(cell, Arrivals(0.0, {"syn": jnp.asarray(arrivals)}), dt=dt)
    for n in range(weights.shape[1]):
        grid = (np.flatnonzero(np.asarray(fired.value[:, n])) + 1) * dt
        want = np.sort(np.asarray(exact[0, n]))
        want = want[np.isfinite(want)]
        assert len(grid) == len(want)
        np.testing.assert_allclose(grid, want, atol=3 * dt)  # observed 1.2e-3
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


# REINFORCE on a layer of Bernoulli neurons.


def log_probability(cell, params, inputs, spikes):
    """log P(spikes) of one example, `inputs` `[T, in]`, `spikes` `[T, N]`, written from the cell's equations
    with every spike held: its membrane, its sigmoid probability, its reset."""
    v = jnp.zeros(spikes.shape[1])
    last = jnp.zeros(spikes.shape[1])
    total = 0.0
    for u, s in zip(inputs, spikes, strict=True):
        v = cell.decay * v + u @ params.w_in + last @ params.w_rec
        a = cell.beta * (v - cell.threshold)
        total += jnp.sum(s * jax.nn.log_sigmoid(a) + (1 - s) * jax.nn.log_sigmoid(-a))
        v = jnp.where(s > 0, 0.0, v) if cell.reset == "zero" else v - s * cell.threshold
        last = s
    return total


def every_trajectory(steps, size):
    """Every spike train of `size` neurons over `steps` steps, `[2 ** (steps size), steps, size]`."""
    bits = (np.arange(2 ** (steps * size))[:, None] >> np.arange(steps * size)) & 1
    return bits.reshape(-1, steps, size).astype(np.float64)


def small_layer(seed):
    """Two Bernoulli neurons with recurrence over three steps of two inputs, float64."""
    rng = np.random.default_rng(seed)
    params = ReinforceParams(jnp.asarray(rng.normal(0, 1.0, (2, 2))), jnp.asarray(rng.normal(0, 1.0, (2, 2))))
    return params, jnp.asarray(rng.random((3, 2)))


@pytest.mark.parametrize("reset", ["zero", "subtract"])
def test_reinforce_eligibility_is_the_score_of_every_trajectory(reset):
    # Each of the 64 trajectories is forced by its noise: its eligibility is the gradient of its own
    # log-probability, and REINFORCE's expectation over all of them, weighed by their probabilities, is
    # the exact gradient of the expected reward.
    with jax.enable_x64(new_val=True):
        cell = BernoulliCell(0.7, threshold=0.5, beta=2.0, reset=reset)
        params, inputs = small_layer(0)
        spikes = every_trajectory(3, 2)
        count = len(spikes)
        rewards = jnp.asarray(np.random.default_rng(1).normal(size=count))
        batch = jnp.broadcast_to(inputs[:, None], (3, count, 2))
        drawn, eligibility = reinforce(cell, params, batch, jnp.asarray(1 - spikes.transpose(1, 0, 2)))
        np.testing.assert_array_equal(np.asarray(drawn).transpose(1, 0, 2), spikes)

        def log_p(params, s):
            return log_probability(cell, params, inputs, s)

        scores = jax.vmap(jax.grad(log_p), in_axes=(None, 0))(params, jnp.asarray(spikes))
        for mine, theirs in zip(eligibility, scores, strict=True):
            np.testing.assert_allclose(mine, theirs, rtol=1e-12, atol=1e-12)  # observed 8.9e-16
        probability = jnp.exp(jax.vmap(log_p, in_axes=(None, 0))(params, jnp.asarray(spikes)))
        np.testing.assert_allclose(jnp.sum(probability), 1.0, rtol=1e-12)  # observed 2.2e-16

        def expected_reward(params):
            probabilities = jnp.exp(jax.vmap(log_p, in_axes=(None, 0))(params, jnp.asarray(spikes)))
            return jnp.sum(probabilities * rewards)

        exact = jax.grad(expected_reward)(params)
        for e, g in zip(eligibility, exact, strict=True):
            np.testing.assert_allclose(jnp.einsum("b,b...->...", probability * rewards, e), g, rtol=1e-12,
                                       atol=1e-12)  # observed 1.7e-16
            # The score has mean zero, so a baseline adds nothing to the expectation. Observed 3.3e-16.
            np.testing.assert_allclose(jnp.einsum("b,b...->...", probability, e), 0.0, atol=1e-12)
        assert all(np.abs(np.asarray(g)).max() > 1e-3 for g in exact)  # the gradient is not trivially 0


def test_policy_gradient_estimates_the_gradient_of_the_expected_reward():
    # Sampled trajectories: the estimate is within a few standard errors of the exact gradient, and a
    # baseline at the expected reward leaves it so with a smaller spread.
    with jax.enable_x64(new_val=True):
        cell = BernoulliCell(0.7, threshold=0.5, beta=2.0)
        params, inputs = small_layer(2)
        spikes = every_trajectory(3, 2)
        table = jnp.asarray(np.random.default_rng(3).normal(size=len(spikes))) + 3.0

        def log_p(params, s):
            return log_probability(cell, params, inputs, s)

        def expected_reward(params):
            probabilities = jnp.exp(jax.vmap(log_p, in_axes=(None, 0))(params, jnp.asarray(spikes)))
            return jnp.sum(probabilities * table)

        exact, mean = jax.grad(expected_reward)(params), expected_reward(params)
        samples = 20000
        noise = jax.random.uniform(jax.random.key(0), (3, samples, 2), jnp.float64)
        batch = jnp.broadcast_to(inputs[:, None], (3, samples, 2))
        drawn, eligibility = reinforce(cell, params, batch, noise)
        index = jnp.sum(drawn.transpose(1, 0, 2).reshape(samples, 6) * (2 ** jnp.arange(6)), axis=1)
        rewards = table[index.astype(jnp.int32)]
        spreads = []
        for baseline in (0.0, mean):
            estimate = policy_gradient(eligibility, rewards, baseline)
            for e, g, got in zip(eligibility, exact, estimate, strict=True):
                each = (rewards - baseline)[:, None, None] * e
                error = np.asarray(jnp.std(each, axis=0)) / np.sqrt(samples)
                assert np.all(np.abs(np.asarray(got) - np.asarray(g)) < 5 * error)
            each = (rewards - baseline)[:, None, None] * eligibility.w_in
            spreads.append(float(jnp.mean(jnp.std(each, axis=0))))
        assert spreads[1] < spreads[0] / 2  # rewards near 3 make the plain estimate's spread large


def test_ascending_the_policy_gradient_teaches_a_layer_which_neuron_to_fire():
    # The reward counts neuron 0's spikes and takes away neuron 1's; from weights that fire both alike,
    # a hundred REINFORCE steps with a running-mean baseline make neuron 0 fire and neuron 1 fall silent.
    cell = BernoulliCell(0.8, threshold=1.0, beta=3.0)
    params = ReinforceParams(jnp.full((3, 2), 0.4), jnp.zeros((2, 2)))
    inputs = jnp.ones((10, 32, 3))
    baseline = 0.0

    @jax.jit
    def update(params, key, baseline):
        noise = jax.random.uniform(key, (10, 32, 2))
        spikes, eligibility = reinforce(cell, params, inputs, noise)
        rewards = jnp.sum(spikes[..., 0] - spikes[..., 1], axis=0)
        gradient = policy_gradient(eligibility, rewards, baseline)
        ascended = jax.tree.map(lambda w, g: w + 0.05 * g, params, gradient)
        return ascended, jnp.mean(rewards), spikes.mean((0, 1))

    first = None
    for key in jax.random.split(jax.random.key(0), 100):
        params, reward, rates = update(params, key, baseline)
        baseline = 0.9 * baseline + 0.1 * float(reward)
        first = rates if first is None else first
    assert abs(float(first[0] - first[1])) < 0.1  # alike at the start
    assert float(rates[0]) > 0.8 and float(rates[1]) < 0.1
