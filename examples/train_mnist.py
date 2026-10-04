"""Train a spiking MNIST classifier with a plain JAX and optax loop, no trainer.

    python examples/train_mnist.py --epochs 2

Downloads MNIST (11 MB) into ~/.cache/sparx on first use. The images are
encoded as Bernoulli spike trains, a fresh draw every step, and a two-layer
LIF network with a leaky integrator readout is trained on the cross entropy
of its time-averaged membrane. Prints the test accuracy after each epoch.
"""

import argparse
import gzip
import time
import urllib.request
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import sparx

MNIST = "https://storage.googleapis.com/cvdf-datasets/mnist/{name}.gz"


def load(split):
    """MNIST `split` as uint8 images `[N, 28, 28]` and int32 labels `[N]`, cached in ~/.cache/sparx."""
    cache = Path.home() / ".cache" / "sparx"
    prefix = "train" if split == "train" else "t10k"
    arrays = []
    for name, offset in ((f"{prefix}-images-idx3-ubyte", 16), (f"{prefix}-labels-idx1-ubyte", 8)):
        path = cache / f"{name}.gz"
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MNIST.format(name=name), path)
        with gzip.open(path) as file:
            arrays.append(np.frombuffer(file.read(), np.uint8, offset=offset))
    images, labels = arrays
    return images.reshape(-1, 28, 28), labels.astype(np.int32)


class Net(nn.Module):
    hidden: int = 512

    @nn.compact
    def __call__(self, spikes):  # [T, B, 28, 28]
        x = spikes.reshape(*spikes.shape[:2], -1)
        x = sparx.nn.LIF(tau=2.0, detach_reset=True)(nn.Dense(self.hidden)(x))
        x = sparx.nn.LIF(tau=2.0, detach_reset=True)(nn.Dense(self.hidden)(x))
        return sparx.nn.LI(tau=2.0)(nn.Dense(10)(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8, help="time steps per image")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    train_x, train_y = load("train")
    test_x, test_y = load("test")
    net = Net()
    key = jax.random.key(0)
    params = net.init(key, jnp.zeros((args.steps, 1, 28, 28)))
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)

    def logits(params, key, images):
        spikes = sparx.encode.rate(key, images.astype(jnp.float32) / 255, args.steps)
        return jnp.mean(net.apply(params, spikes), axis=0)

    @jax.jit
    def train_step(params, opt_state, key, images, labels):
        def loss(params):
            return optax.softmax_cross_entropy_with_integer_labels(logits(params, key, images), labels).mean()

        value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, value

    @jax.jit
    def correct(params, key, images, labels):
        return jnp.sum(jnp.argmax(logits(params, key, images), -1) == labels)

    batches = len(train_y) // args.batch
    for epoch in range(args.epochs):
        start = time.perf_counter()
        order = np.random.default_rng(epoch).permutation(len(train_y))
        for i in range(batches):
            rows = order[i * args.batch:(i + 1) * args.batch]
            key, step_key = jax.random.split(key)
            params, opt_state, loss = train_step(params, opt_state, step_key, train_x[rows], train_y[rows])
        loss.block_until_ready()
        elapsed = time.perf_counter() - start
        hits = sum(int(correct(params, jax.random.fold_in(key, i), test_x[i:i + 1000], test_y[i:i + 1000]))
                   for i in range(0, len(test_y), 1000))
        print(f"epoch {epoch + 1}: loss {float(loss):.4f}, test accuracy {hits / len(test_y):.4f}, "
              f"{batches} steps in {elapsed:.0f} s on {jax.devices()[0].device_kind}")


if __name__ == "__main__":
    main()
