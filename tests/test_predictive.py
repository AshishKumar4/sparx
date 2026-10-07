from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.learn import PredictiveCoding, residual_mlp, sequential_blocks, squared_error

PCALM = np.load(Path(__file__).parent / "fixtures" / "pcalm.npz")
CASES = {"tanh": (4, 5, 3, 2, "tanh"), "relu": (8, 6, 4, 3, "relu")}  # depth, width, inputs, outputs
METHODS = {"pc": "before", "pcalm": "before", "pcalm_post": "after", "pcalm_inner": "before"}


def case(name):
    return {key.split("/", 1)[1]: PCALM[key] for key in PCALM.files if key.startswith(f"{name}/")}


def stack(name, given):
    """Their residual MLP as sparx builds it, with their weights: its blocks and parameters."""
    depth, width, inputs, outputs, activation = CASES[name]
    model = residual_mlp(width, depth, inputs, outputs, activation)
    params = {f"layers_{layer}": {"kernel": jnp.asarray(given[f"kernel_{layer}"])} for layer in range(depth)}
    return model, *sequential_blocks(model, params)


def rule(given, method):
    return PredictiveCoding(int(given["budget"]), float(given["state_lr"]), float(given["rho"]),
                            float(given[f"{method}/alpha"]), int(given[f"{method}/inner_steps"]),
                            METHODS[method])


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("name", list(CASES))
def test_pc_and_pc_alm_are_seely_and_goulds_inference_and_update(name, method):
    given = case(name)
    depth = CASES[name][0]
    with jax.enable_x64(new_val=True):
        _, blocks, params = stack(name, given)
        x, y = jnp.asarray(given["x"]), jnp.asarray(given["y"])
        settled = rule(given, method).settle(blocks, params, x, y)
        grads = rule(given, method).gradient(blocks, params, x, y)
    batch = x.shape[0]
    # Observed over both networks and all four methods: activity 8.9e-16, multipliers 1.6e-15, and the
    # update 4.8e-14 of its layer's largest entry.
    for layer in range(depth - 1):
        np.testing.assert_allclose(settled.activity[layer], given[f"{method}/free_{layer}"], rtol=0,
                                   atol=1e-12)
        np.testing.assert_allclose(settled.multipliers[layer], given[f"{method}/dual_{layer}"], rtol=0,
                                   atol=1e-12)
    for layer in range(depth):
        # Their update is the batch's mean, ours the sum over its rows.
        expected = given[f"{method}/grad_{layer}"]
        np.testing.assert_allclose(np.asarray(grads[layer]["kernel"]) / batch, expected, rtol=0,
                                   atol=1e-12 * np.abs(expected).max(), err_msg=f"layer {layer}")


def test_pc_alm_aligns_the_update_with_backpropagation_where_pc_does_not():
    # Their finding, on the deeper ReLU network: at the same budget, PC-ALM's update points nearer
    # backpropagation's than PC's.
    given = case("relu")
    with jax.enable_x64(new_val=True):
        model, blocks, params = stack("relu", given)
        x, y = jnp.asarray(given["x"]), jnp.asarray(given["y"])
        variables = {"params": {f"layers_{k}": p for k, p in enumerate(params)}}
        bp = jax.grad(lambda v: jnp.sum(squared_error(model.apply(v, x), y)))(variables)["params"]
        bp = np.concatenate([np.ravel(bp[f"layers_{k}"]["kernel"]) for k in range(8)])

        def cosine(method):
            update = rule(given, method).gradient(blocks, params, x, y)
            update = np.concatenate([np.ravel(g["kernel"]) for g in update])
            return update @ bp / (np.linalg.norm(update) * np.linalg.norm(bp))

        pc, pcalm = cosine("pc"), cosine("pcalm")
    # Observed 0.679 for PC and 0.939 for PC-ALM, as their code gives.
    assert pc < 0.7 < 0.9 < pcalm


def test_the_forward_pass_is_their_network():
    given = case("relu")
    with jax.enable_x64(new_val=True):
        model, _, params = stack("relu", given)
        x, y = jnp.asarray(given["x"]), jnp.asarray(given["y"])
        variables = {"params": {f"layers_{k}": p for k, p in enumerate(params)}}

        def bp_loss(variables):
            return jnp.mean(squared_error(model.apply(variables, x), y))

        grads = jax.grad(bp_loss)(variables)["params"]
    for layer in range(8):
        expected = given[f"bp/grad_{layer}"]
        np.testing.assert_allclose(grads[f"layers_{layer}"]["kernel"], expected, rtol=0,
                                   atol=1e-12 * np.abs(expected).max())


def test_inference_is_each_examples_own():
    # Each example relaxes on its own energy, so it settles the same alone or in a batch.
    given = case("tanh")
    with jax.enable_x64(new_val=True):
        _, blocks, params = stack("tanh", given)
        x, y = jnp.asarray(given["x"]), jnp.asarray(given["y"])
        alone = rule(given, "pcalm").settle(blocks, params, x[:1], y[:1])
        batch = rule(given, "pcalm").settle(blocks, params, x, y)
    for a, b in zip(alone.activity, batch.activity, strict=True):
        np.testing.assert_allclose(a[0], b[0], rtol=0, atol=1e-14)


def test_a_repeated_row_counts_for_nothing():
    given = case("tanh")
    with jax.enable_x64(new_val=True):
        _, blocks, params = stack("tanh", given)
        x, y = jnp.asarray(given["x"]), jnp.asarray(given["y"])
        pcalm = rule(given, "pcalm")
        padded = pcalm.gradient(blocks, params, jnp.concatenate([x, x[:2]]), jnp.concatenate([y, y[:2]]),
                                rows=jnp.concatenate([jnp.ones(7), jnp.zeros(2)]))
        plain = pcalm.gradient(blocks, params, x, y)
    for a, b in zip(jax.tree.leaves(padded), jax.tree.leaves(plain), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_inference_refuses_a_stack_without_a_hidden_layer_and_an_empty_budget():
    with pytest.raises(ValueError, match="at least one step"):
        PredictiveCoding(0, 0.1)
    with pytest.raises(ValueError, match="before or after"):
        PredictiveCoding(2, 0.1, credit="later")
    blocks, params = [lambda p, z: z], [{}]
    with pytest.raises(ValueError, match="hidden layer"):
        PredictiveCoding(2, 0.1).settle(blocks, params, jnp.zeros((1, 2)), jnp.zeros((1, 2)))
