from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from dew import Field, Supervised, Trainer
from dew.data import Dataset, Loading
from dew.inputs import InputSpec

import sparx.nn as snn
from sparx.graph.connectome import FLYNN, Connectome, spectral_radius

GIVEN = np.load(Path(__file__).parent / "fixtures" / "flynn.npz")


def network(given=GIVEN, **fields):
    graph = Connectome(np.arange(given["types"].size), given["pre"], given["post"], given["synapses"])
    return FLYNN(connectome=graph, types=tuple(given["types"].tolist()),
                 input_neurons=tuple(given["inputs"].tolist()),
                 output_neurons=tuple(given["outputs"].tolist()), **fields)


def test_flynn_is_wang_and_chens_cell_step_by_step_and_in_its_gradient():
    with jax.enable_x64(new_val=True):
        net = network()
        params = {"weight": jnp.asarray(GIVEN["weight"]), "bias": jnp.asarray(GIVEN["bias"]),
                  "update_logits": jnp.asarray(GIVEN["alpha_logits"])}
        drive, coefficients = jnp.asarray(GIVEN["drive"]), jnp.asarray(GIVEN["coefficients"])

        def loss(params):
            out = net.apply({"params": params}, drive)
            return jnp.sum(out * coefficients), out

        (value, out), grads = jax.value_and_grad(loss, has_aux=True)(params)
        alone = net.apply({"params": {**params, "weight": jnp.zeros_like(params["weight"])}}, drive)
    # Observed: activity 7.8e-16, loss 8.0e-16, gradients 4.3e-16 of their largest entries.
    np.testing.assert_allclose(out, GIVEN["activity"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(value, GIVEN["loss"], rtol=1e-12)
    theirs = {"weight": "grad_weight", "bias": "grad_bias", "update_logits": "grad_alpha_logits"}
    for name, key in theirs.items():
        expected = GIVEN[key]
        np.testing.assert_allclose(np.asarray(grads[name]), expected, rtol=0,
                                   atol=1e-12 * np.abs(expected).max(), err_msg=name)
    # The recurrence shapes the output: without the synapses it differs.
    assert np.abs(np.asarray(alone) - GIVEN["activity"]).max() > 0.1


def radius(given, weight):
    matrix = np.zeros((given["types"].size,) * 2)
    np.add.at(matrix, (given["post"], given["pre"]), np.asarray(weight, np.float64))
    return np.abs(np.linalg.eigvals(matrix)).max()


def test_the_weights_start_at_the_synapse_counts_scaled_to_the_radius():
    # The fixture's signed connectome, whose dominant eigenvalues are a complex pair: their power
    # iteration scaled it to a radius of 2.2; the radius here is the one asked for.
    variables = network(radius=0.9).init(jax.random.key(0), jnp.zeros((2, 1, GIVEN["inputs"].size)))
    weight = variables["params"]["weight"]
    np.testing.assert_allclose(radius(GIVEN, weight), 0.9, rtol=1e-5)  # observed 1.9e-8
    scale = np.asarray(weight) / GIVEN["synapses"]
    np.testing.assert_allclose(scale, scale[0], rtol=1e-6)
    assert radius(GIVEN, GIVEN["weight"]) > 2  # theirs, from their estimate: 2.15


def test_the_radius_of_a_large_connectome_is_arnoldis():
    # Above 2,000 neurons the radius comes from ARPACK; the dense eigenvalues agree.
    rng = np.random.default_rng(4)
    size, edges = 2100, 20000
    pairs = rng.choice(size * size, edges, replace=False)
    synapses = rng.integers(5, 30, edges) * rng.choice([-1, 1], edges, p=[0.3, 0.7])
    graph = Connectome(np.arange(size), pairs % size, pairs // size, synapses)
    given = {"types": np.zeros(size), "pre": pairs % size, "post": pairs // size}
    np.testing.assert_allclose(spectral_radius(graph), radius(given, synapses), rtol=1e-6)  # observed 2.3e-15


def test_each_class_starts_at_their_update_fraction():
    # Their logit starts at log(0.2 / 0.8), so the update fraction is 0.99 * 0.2 + 0.01.
    net = network()
    params = net.init(jax.random.key(0), jnp.zeros((2, 1, GIVEN["inputs"].size)))["params"]
    np.testing.assert_allclose(0.99 * jax.nn.sigmoid(params["update_logits"]) + 0.01, 0.208, rtol=1e-6)
    assert params["update_logits"].shape == (int(GIVEN["types"].max()) + 1,)


def test_flynn_fed_in_chunks_carries_its_activity():
    net = network()
    drive = jnp.asarray(GIVEN["drive"], jnp.float32)
    variables = net.init(jax.random.key(0), drive)
    whole = net.apply(variables, drive)
    head, state = net.apply(variables, drive[:6], mutable=["state"])
    tail, _ = net.apply({**variables, **state}, drive[6:], mutable=["state"])
    np.testing.assert_allclose(jnp.concatenate([head, tail]), whole, rtol=0, atol=1e-6)  # observed 0


def test_flynn_trains_through_dews_supervised_objective():
    rng = np.random.default_rng(5)
    records = {"drive": rng.normal(size=(16, 8, GIVEN["inputs"].size)).astype(np.float32),
               "target": rng.normal(size=(16, GIVEN["outputs"].size)).astype(np.float32)}

    def last_step_error(outputs, batch):
        return jnp.sum((outputs[:, -1] - batch["target"]) ** 2, axis=-1)

    objective = Supervised(snn.BatchMajor(network()), last_step_error,
                           inputs=InputSpec(Field("drive", records["drive"].shape[1:])))
    data = Dataset.from_records(records, batch=8, loading=Loading(workers=0, threads=1, read_buffer=1))
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0))
    start = trainer.fit(data, steps=1, log_every=1).variables["params"]["layer"]
    end = trainer.fit(data, steps=4, log_every=4).variables["params"]["layer"]
    for name in ("weight", "bias", "update_logits"):
        assert np.any(np.asarray(start[name]) != np.asarray(end[name])), name
