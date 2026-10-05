
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import sparx
from sparx.models import SEWBlock, SEWResNet, connect, sew_resnet18


def frames(seed=0, shape=(4, 2, 16, 16, 3)):
    return (jax.random.uniform(jax.random.key(seed), shape) < 0.3).astype(jnp.float32)


def test_connect_functions_follow_spikingjelly():
    s = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    r = jnp.asarray([0.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(connect(s, r, "add"), [0, 1, 1, 2])
    np.testing.assert_array_equal(connect(s, r, "and"), [0, 0, 0, 1])
    np.testing.assert_array_equal(connect(s, r, "iand"), [0, 0, 1, 0])
    with pytest.raises(ValueError, match="connect"):
        connect(s, r, "or")  # type: ignore[arg-type]


def test_a_silent_residual_branch_passes_the_input_through():
    # The point of SEW: with the residual neurons silent, an ADD block is the identity.
    x = frames(1, (3, 2, 8, 8, 4))
    block = SEWBlock(4, neuron=sparx.nn.LIF(threshold=1e9))
    variables = block.init(jax.random.key(0), x, train=False)
    np.testing.assert_array_equal(block.apply(variables, x, train=False), x)


def test_strided_blocks_downsample_the_shortcut_through_a_neuron():
    x = frames(2, (3, 2, 8, 8, 4))
    block = SEWBlock(8, strides=2)
    variables = block.init(jax.random.key(0), x, train=False)
    assert set(variables["params"]) == {"first_conv", "first_bn", "second_conv", "second_bn",
                                        "downsample_conv", "downsample_bn"}
    out = block.apply(variables, x, train=False)
    assert out.shape == (3, 2, 4, 4, 8)
    assert set(np.unique(np.asarray(out)).tolist()) <= {0.0, 1.0, 2.0}


def test_resnet18_has_the_stage_layout_and_per_step_logits():
    net = sew_resnet18(10, width=8, stem="small")
    x = frames()
    variables = net.init(jax.random.key(0), x, train=False)
    names = [name for name in variables["params"] if name.startswith("stage")]
    assert names == [f"stage{s}_block{b}" for s in range(1, 5) for b in (1, 2)]
    assert net.apply(variables, x, train=False).shape == (4, 2, 10)


def test_a_small_sew_resnet_fits_a_batch():
    net = SEWResNet((1, 1), 2, width=8, stem="small",
                    neuron=sparx.nn.LIF(tau=2.0, detach_reset=True))
    x = frames(3, (4, 8, 8, 8, 1))
    labels = jnp.asarray([0, 1] * 4)
    x = x.at[:, labels == 1, :4].set(1.0)  # class 1 is bright on top
    variables = net.init(jax.random.key(0), x, train=False)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(variables["params"])

    @jax.jit
    def step(variables, opt_state):
        def loss(params):
            logits, updated = net.apply({**variables, "params": params}, x, train=True,
                                        mutable=["batch_stats"])
            ce = optax.softmax_cross_entropy_with_integer_labels(logits.mean(0), labels).mean()
            return ce, updated
        (value, updated), grads = jax.value_and_grad(loss, has_aux=True)(variables["params"])
        updates, opt_state = optimizer.update(grads, opt_state)
        return {**updated, "params": optax.apply_updates(variables["params"], updates)}, opt_state, value

    first = None
    for _ in range(30):
        variables, opt_state, value = step(variables, opt_state)
        first = value if first is None else first
    assert float(value) < 0.2 * float(first)
