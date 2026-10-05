"""Learning rules against the identities that define them."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from sparx.cells import ALIFCell, LIFCell
from sparx.learn import EPropParams, OTTTLayer, bptt_loss, eligibility_traces, eprop, ottt
from sparx.surrogate import Sigmoid, Triangle

CELLS = {
    "lif": lambda: LIFCell(decay=float(np.exp(-1 / 20)), threshold=0.6, detach_reset=True,
                           surrogate=Triangle()),
    "alif": lambda: ALIFCell(decay=float(np.exp(-1 / 20)), adapt_decay=float(np.exp(-1 / 200)), beta=0.07,
                             threshold=0.6, detach_reset=True, surrogate=Triangle()),
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
