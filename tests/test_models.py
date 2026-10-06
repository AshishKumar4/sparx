
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import sparx
from sparx.models import SEWBlock, SEWResNet, SpikingMLP, connect, sew_resnet18
from sparx.nn import DelayedDense


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


def spikes(seed=0, shape=(10, 3, 6)):
    return (jax.random.uniform(jax.random.key(seed), shape) < 0.4).astype(jnp.float32)


def test_an_integer_delay_delays_the_first_synapse_and_a_sequence_each_synapse():
    first = SpikingMLP(hidden=(4, 4), classes=2, delays=3)
    params = first.init(jax.random.key(0), spikes())["params"]
    assert set(params) == {"delayed_0", "dense_1", "readout"} and "delay" not in params["readout"]
    every = SpikingMLP(hidden=(4, 4), classes=2, delays=(3, 0, 2))
    params = every.init(jax.random.key(0), spikes())["params"]
    assert set(params) == {"delayed_0", "dense_1", "readout"}
    assert params["readout"]["delay"].shape == (4, 2) and float(params["readout"]["delay"].max()) <= 2
    with pytest.raises(ValueError, match="3 synapses"):
        SpikingMLP(hidden=(4, 4), classes=2, delays=(3, 3)).init(jax.random.key(0), spikes())


def test_extended_delayed_synapses_lengthen_the_output_by_half_their_range():
    x = spikes()
    net = SpikingMLP(hidden=(4, 4), classes=2, delays=(4, 0, 3), extend=True)
    variables = net.init(jax.random.key(0), x)
    out = net.apply(variables, x, sigma=1.0)
    assert out.shape == (10 + 2 + 1, 3, 2)
    # The appended steps hold zeros, so the first steps are the causal network's.
    causal = SpikingMLP(hidden=(4, 4), classes=2, delays=(4, 0, 3)).apply(variables, x, sigma=1.0)
    np.testing.assert_allclose(out[:10], causal, rtol=1e-6, atol=1e-6)


def test_batch_norm_normalizes_each_hidden_synapse_with_batch_statistics_in_training():
    x = spikes(1)
    net = SpikingMLP(hidden=(4,), classes=2, delays=(2, 2), batch_norm=True)
    variables = net.init(jax.random.key(0), x)
    assert set(variables["batch_stats"]) == {"norm_0"}  # the readout is not normalized
    _, updated = net.apply(variables, x, train=True, sigma=1.0, mutable=["batch_stats"])
    synapse = DelayedDense(4, 2, name="delayed_0").apply({"params": variables["params"]["delayed_0"]},
                                                         x.reshape(10, 3, -1), 1.0)
    np.testing.assert_allclose(updated["batch_stats"]["norm_0"]["mean"], 0.1 * synapse.mean((0, 1)),
                               rtol=1e-5, atol=1e-6)
    # Evaluation reads the running statistics, so it differs from training on the same input.
    trained = net.apply(variables, x, train=True, sigma=1.0, mutable=["batch_stats"])[0]
    assert not np.allclose(net.apply(variables, x, train=False, sigma=1.0), trained)


def test_kaiming_uniform_weights_without_bias():
    net = SpikingMLP(hidden=(512,), classes=2, delays=(2, 0), use_bias=False, weight_init="kaiming_uniform")
    params = net.init(jax.random.key(0), spikes(shape=(4, 2, 300)))["params"]
    assert all("bias" not in layer for layer in params.values())
    kernel = np.asarray(params["delayed_0"]["kernel"])
    # torch's kaiming_uniform_(nonlinearity="relu"): uniform within sqrt(6 / fan_in), variance 2 / fan_in.
    assert np.abs(kernel).max() <= np.sqrt(6 / 300)
    np.testing.assert_allclose(kernel.var(), 2 / 300, rtol=0.02)
    with pytest.raises(ValueError, match="weight_init"):
        SpikingMLP(hidden=(4,), classes=2, weight_init="xavier").init(  # type: ignore[arg-type]
            jax.random.key(0), spikes())


@pytest.mark.parametrize("mask", ["step", "sequence"])
def test_dropout_masks_a_step_or_holds_one_mask_over_the_sequence(mask):
    # Every hidden neuron fires every step, so the dropout's output is its mask.
    neuron = sparx.nn.LIF(threshold=-1e3, reset="none")
    net = SpikingMLP(hidden=(64,), classes=2, neuron=neuron, dropout=0.5, dropout_mask=mask)
    x = spikes(2)
    variables = net.init(jax.random.key(0), x)
    _, state = net.apply(variables, x, train=True, rngs={"dropout": jax.random.key(1)},
                         capture_intermediates=True, mutable=["intermediates"])
    (kept,) = state["intermediates"]["Dropout_0"]["__call__"]
    kept = np.asarray(kept)
    assert set(np.unique(kept)) == {0.0, 2.0}
    held = (kept == kept[:1]).all(axis=0)
    assert held.all() if mask == "sequence" else held.mean() < 0.01


@pytest.mark.parametrize("architecture", ["spiking_mlp", "sew_resnet"])
def test_a_runs_precision_settings_reach_the_synapses(architecture):
    from dew.config import ModelConfig

    fields = ({"hidden": [8], "classes": 3, "delays": [2, 0], "batch_norm": True}
              if architecture == "spiking_mlp" else {"stages": [1, 1, 1, 1], "classes": 3, "width": 4,
                                                     "stem": "small"})
    x = frames(2, (3, 2, 8, 8, 1))
    first = "delayed_0" if architecture == "spiking_mlp" else "Conv_0"
    for dtype in ("float32", "bfloat16"):
        model = ModelConfig(architecture, fields, dtype=dtype, param_dtype="bfloat16",
                            matmul_precision="highest").build()
        assert model.dtype == jnp.dtype(dtype) and model.precision == "highest"
        variables = model.init(jax.random.key(0), x, train=False)
        _, captured = model.apply(variables, x, train=False, capture_intermediates=True,
                                  mutable=["intermediates"])
        # The first synapse computes in the run's dtype, so the setting reached it.
        assert captured["intermediates"][first]["__call__"][0].dtype == jnp.dtype(dtype)
        stored = {jax.tree_util.keystr(path).split("'")[-2]: leaf.dtype
                  for path, leaf in jax.tree_util.tree_leaves_with_path(variables["params"])}
        assert stored["kernel"] == jnp.bfloat16
        # The delays stay float32 whatever the synapses store.
        assert stored.get("delay", jnp.float32) == jnp.float32
        # The run's record reads the settings back from the built model.
        recorded = ModelConfig.from_model(model)
        settings = (recorded.dtype, recorded.param_dtype, recorded.matmul_precision)
        assert settings == (dtype, "bfloat16", "highest")
