"""Write a CNN converted and simulated by Rueckauer et al.'s SNN toolbox for sparx's parity test.

Trains a small Keras CNN (convolution, batch norm, ReLU, max pool,
convolution, batch norm, ReLU, average pool, flatten, dense) on 8x8 images
of bars, then runs the toolbox's own pipeline with its defaults: parse,
which folds batch norm into the convolutions; normalize at the 99.9th
percentile; convert; and simulate with the INI simulator (analog input,
reset by subtraction, max pooling gated by the input spike counts). The
last layer is a ReLU so the toolbox's percentile and sparx's are taken over
the same values. Saves the Keras weights and batch-norm statistics, the
normalized weights the toolbox simulates, and every layer's spike counts
over the test batch to `tests/fixtures/snntoolbox.npz`.

snntoolbox 0.6 predates Keras 3 and numpy 2. It runs with tf-keras
(`TF_USE_LEGACY_KERAS=1`, set below) after two shims: `np.product`, and
writing its numpy-scalar scale factors to json.

    uv pip sync tools/environments/snntoolbox.txt
    python tools/make_snntoolbox_fixtures.py
"""

import configparser
import json
import os
import tempfile
import types
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from importlib.metadata import version

import numpy as np
from references import require

np.product = np.prod  # removed in numpy 2, still called by snntoolbox 0.6
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "snntoolbox.npz"
STEPS, NORM, TEST = 100, 300, 50


def bars(n, seed):
    """8x8 images of one bar, horizontal, vertical or on either diagonal, at a random place, in noise."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, n)
    images = rng.uniform(0, 0.3, (n, 8, 8, 1))
    for image, label, (a, b) in zip(images, labels, rng.integers(0, 6, (n, 2)), strict=True):
        for t in range(3):
            i, j = [(a, b + t), (a + t, b), (a + t, b + t), (a + t, b + 2 - t)][label]
            image[i, j, 0] = rng.uniform(0.7, 1.0)
    return images.astype("float32"), labels


def cnn(keras, last):
    return keras.Sequential([
        keras.Input((8, 8, 1)),
        keras.layers.Conv2D(8, 3, padding="same", name="conv0"),
        keras.layers.BatchNormalization(name="bn0"),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(16, 3, padding="valid", name="conv1"),
        keras.layers.BatchNormalization(name="bn1"),
        keras.layers.Activation("relu"),
        keras.layers.AveragePooling2D(2, strides=1),
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation=last, name="dense"),
    ])


def main():
    require("tensorflow", "snntoolbox")
    import tensorflow as tf
    from tensorflow import keras

    # Python's, numpy's and TensorFlow's seeds: tf-keras draws its initial weights from all three.
    keras.utils.set_random_seed(0)
    x, y = bars(1000 + TEST, seed=0)
    trained = cnn(keras, "linear")
    trained.compile(optimizer=keras.optimizers.Adam(1e-2),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    trained.fit(x[:1000], y[:1000], epochs=20, batch_size=100, verbose=0)
    model = cnn(keras, "relu")
    model.set_weights(trained.get_weights())
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

    import snntoolbox.conversion.utils as conversion_utils
    from snntoolbox.bin.utils import import_target_sim, update_setup
    from snntoolbox.conversion.utils import normalize_parameters
    from snntoolbox.datasets.utils import get_dataset
    from snntoolbox.parsing.model_libs import keras_input_lib

    # numpy 2 percentiles are numpy scalars, which json cannot write.
    conversion_utils.json = types.SimpleNamespace(
        dump=lambda o, f, **k: json.dump(o, f, default=float, **k), load=json.load)

    x_norm, x_test = x[:NORM], x[1000:]
    with tempfile.TemporaryDirectory() as work:
        model.save(os.path.join(work, "cnn.h5"))
        np.savez_compressed(os.path.join(work, "x_norm"), x_norm)
        np.savez_compressed(os.path.join(work, "x_test"), x_test)
        np.savez_compressed(os.path.join(work, "y_test"), keras.utils.to_categorical(y[1000:], 4))
        config = configparser.ConfigParser()
        config["paths"] = {"path_wd": work, "dataset_path": work, "filename_ann": "cnn"}
        config["simulation"] = {"duration": STEPS, "batch_size": TEST, "num_to_test": TEST}
        with open(os.path.join(work, "config"), "w") as f:
            config.write(f)
        config = update_setup(os.path.join(work, "config"))

        normset, testset = get_dataset(config)
        parser = keras_input_lib.ModelParser(keras_input_lib.load(work, "cnn")["model"], config)
        parser.parse()
        parsed = parser.build_parsed_model()
        normalize_parameters(parsed, config, **normset)
        snn = import_target_sim(config).SNN(config)
        snn.build(parsed, **testset)

        layers = snn.snn.layers[1:]
        probe = keras.Model(snn.snn.input, [layer.output for layer in layers])
        counts = [np.zeros(layer.output_shape[1:], "float64") for layer in layers]
        counts = [np.zeros((TEST, *c.shape)) for c in counts]
        for t in range(STEPS):
            snn.set_time(t + 1)
            for c, out in zip(counts, probe.predict_on_batch(x_test), strict=True):
                c += out > 0

    cases = {"x_norm": x_norm, "x_test": x_test, "steps": np.array(STEPS),
             "meta/snntoolbox": np.array(version("snntoolbox")),
             "meta/tensorflow": np.array(tf.__version__)}
    for name in ("conv0", "conv1", "dense"):
        weight, bias = model.get_layer(name).get_weights()
        cases[f"{name}/weight"], cases[f"{name}/bias"] = weight, bias
    for name in ("bn0", "bn1"):
        bn = model.get_layer(name)
        gamma, beta, mean, var = bn.get_weights()
        cases.update({f"{name}/scale": gamma, f"{name}/offset": beta, f"{name}/mean": mean,
                      f"{name}/var": var, f"{name}/epsilon": np.array(bn.epsilon)})
    for k, layer in enumerate(layer for layer in parsed.layers if layer.get_weights()):
        weight, bias = layer.get_weights()
        cases[f"normalized/{k}/weight"], cases[f"normalized/{k}/bias"] = weight, bias
    for layer, c in zip(layers, counts, strict=True):
        cases[f"counts/{layer.__class__.__name__}/{layers.index(layer)}"] = c.astype("int32")
    np.savez_compressed(OUT, **cases)
    print({k: v.shape for k, v in cases.items()})


if __name__ == "__main__":
    main()
