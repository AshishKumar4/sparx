"""Time e-prop's online gradients against BPTT's for a recurrent layer and leaky readout, at SHD's shapes.

    python benchmarks/bench_eprop.py [--steps 100] [--batch 64] [--inputs 140] [--hidden 128]

The network is `examples/train_shd_eprop.py`'s: 140 pooled channels, 128
recurrent neurons, 20 readout units, 100 steps, batch 64, with ALIF
(three state variables: membrane, adaptation, refractory count) and with
LIF. Prints the median of `--repeats` timed calls after two warm-up
calls, each blocking on its result, and the scratch memory XLA's compiled
program reserves (`memory_analysis().temp_size_in_bytes`), which is where
the eligibility vectors live.
"""

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sparx.dynamics import ALIFCell, LIFCell, decay
from sparx.learn import EPropParams, bptt_loss, eprop
from sparx.surrogate import Triangle

CELLS = {
    "ALIF": ALIFCell(decay=decay(20.0), adapt_decay=decay(200.0), beta=0.2, detach_reset=True,
                     surrogate=Triangle(scale=0.3), refractory=2),
    "LIF": LIFCell(decay=decay(20.0), detach_reset=True, surrogate=Triangle(scale=0.3)),
}


def timed(fn, *args, repeats):
    for _ in range(2):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--inputs", type=int, default=140)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    print(jax.devices())
    rng = np.random.default_rng(0)
    n, h = args.inputs, args.hidden
    params = EPropParams(jnp.asarray(rng.normal(0, 1 / np.sqrt(n), (n, h)), jnp.float32),
                         jnp.asarray(rng.normal(0, 1 / np.sqrt(h), (h, h)), jnp.float32),
                         jnp.asarray(rng.normal(0, 1 / np.sqrt(h), (h, 20)), jnp.float32),
                         jnp.zeros(20, jnp.float32))
    inputs = jnp.asarray(rng.poisson(0.3, (args.steps, args.batch, n)), jnp.float32)
    labels = jnp.asarray(rng.integers(0, 20, args.batch))
    targets = jnp.broadcast_to(labels, (args.steps, args.batch))

    def loss(y, label):
        return optax.softmax_cross_entropy_with_integer_labels(y, label).mean() / args.steps

    print(f"T={args.steps} B={args.batch} in={n} N={h}")
    for name, cell in CELLS.items():
        rules = {
            "eprop": lambda p, cell=cell: eprop(cell, p, inputs, targets, loss, tau=20.0),
            "bptt": lambda p, cell=cell: jax.value_and_grad(
                lambda p: bptt_loss(cell, p, inputs, targets, loss, tau=20.0))(p),
        }
        for rule, fn in rules.items():
            compiled = jax.jit(fn).lower(params).compile()
            memory = compiled.memory_analysis()
            scratch = memory.temp_size_in_bytes / 2 ** 20 if memory is not None else float("nan")
            seconds = timed(compiled, params, repeats=args.repeats)
            print(f"  {name:<5} {rule:<6} {seconds * 1e3:9.1f} ms   scratch {scratch:7.1f} MiB")


if __name__ == "__main__":
    main()
