"""Train a recurrent spiking network on the Spiking Heidelberg Digits with e-prop, online.

    python examples/train_shd_eprop.py --rule eprop --epochs 5
    python examples/train_shd_eprop.py --rule random --epochs 5      # random feedback weights
    python examples/train_shd_eprop.py --rule bptt --epochs 5        # the same network by BPTT

A layer of recurrent adaptive LIF neurons (Bellec et al. 2020) reads SHD,
binned into 100 steps of 14 ms with adjacent channels pooled to 140, and a
leaky readout scores the 20 classes. e-prop (`sparx.learn.eprop`) computes
the gradients as the recording runs: each synapse keeps an eligibility
trace, and each step's cross entropy weights it through the readout
(`--rule eprop`) or through fixed random weights (`--rule random`). Its
memory does not grow with the recording's length. `--rule bptt` trains the
same network by backpropagation through time, for comparison. The class
is the argmax of the readout averaged over time. Prints the test accuracy
over all 2264 test recordings after each epoch.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sparx.datasets import shd
from sparx.dynamics import ALIFCell, decay
from sparx.learn import EPropParams, bptt_loss, eprop, eprop_forward
from sparx.surrogate import Triangle

DT = 14.0
"""One step of the binned recordings, in ms."""
TAU_READOUT = 20.0
"""The readout's time constant, in ms."""


def pooled(split, channels):
    data = shd(split)
    spikes = data["spikes"].reshape(*data["spikes"].shape[:2], channels, -1).sum(-1)
    return spikes.astype(np.float32), data["label"]


def accuracy(cell, params, test):
    """The share of `test` recordings whose readout, averaged over time, peaks at their class."""

    @jax.jit
    def predict(spikes):
        outputs, _ = eprop_forward(cell, params, jnp.swapaxes(spikes, 0, 1), tau=TAU_READOUT, dt=DT)
        return jnp.argmax(outputs.mean(0), -1)

    spikes, labels = test
    hits = sum(int((predict(jnp.asarray(spikes[i:i + 256])) == labels[i:i + 256]).sum())
               for i in range(0, len(labels), 256))
    return hits / len(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rule", choices=["eprop", "random", "bptt"], default="eprop")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--channels", type=int, default=140)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train, test = pooled("train", args.channels), pooled("test", args.channels)
    # Time in ms, in steps of 14 ms: membrane 20 ms, adaptation 200 ms and
    # readout 20 ms, as Bellec et al. take them for speech (TIMIT), and two
    # steps of refractoriness.
    cell = ALIFCell(decay=decay(20.0), adapt_decay=decay(200.0), beta=0.2, detach_reset=True,
                    surrogate=Triangle(scale=0.3), refractory=2 * DT)
    rng = np.random.default_rng(args.seed)
    n, h = args.channels, args.hidden
    w_rec = rng.normal(0, 1 / np.sqrt(h), (h, h))
    np.fill_diagonal(w_rec, 0)
    params = EPropParams(jnp.asarray(rng.normal(0, 1 / np.sqrt(n), (n, h)), jnp.float32),
                         jnp.asarray(w_rec, jnp.float32),
                         jnp.asarray(rng.normal(0, 1 / np.sqrt(h), (h, 20)), jnp.float32),
                         jnp.zeros(20, jnp.float32))
    feedback = None
    if args.rule == "random":
        feedback = jnp.asarray(rng.normal(0, 1 / np.sqrt(h), (h, 20)), jnp.float32)
    no_self = 1 - jnp.eye(h)

    def loss(y, label):
        return optax.softmax_cross_entropy_with_integer_labels(y, label).mean() / 100

    @jax.jit
    def gradients(params, spikes, labels):
        inputs = jnp.swapaxes(spikes, 0, 1)  # [T, B, channels]
        targets = jnp.broadcast_to(labels, (inputs.shape[0], *labels.shape))
        if args.rule == "bptt":
            return jax.value_and_grad(
                lambda p: bptt_loss(cell, p, inputs, targets, loss, tau=TAU_READOUT, dt=DT))(params)
        return eprop(cell, params, inputs, targets, loss, tau=TAU_READOUT, dt=DT, feedback=feedback)

    optimizer = optax.adam(args.learning_rate)
    state = optimizer.init(params)

    @jax.jit
    def update(params, state, grads):
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params._replace(w_rec=params.w_rec * no_self), state

    start = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train[1]))
        total = 0.0
        for i in range(0, len(order) - args.batch + 1, args.batch):
            index = order[i:i + args.batch]
            value, grads = gradients(params, jnp.asarray(train[0][index]), jnp.asarray(train[1][index]))
            params, state = update(params, state, grads)
            total += float(value)
        score = accuracy(cell, params, test)
        print(f"epoch {epoch + 1}: train loss {total / (len(order) // args.batch):.4f}, "
              f"test accuracy {score:.2%}, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
