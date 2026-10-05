"""Time an LIF layer's forward and backward pass under the scan's unroll settings,
and a two-layer network with its synapses folded over time or applied per step.

    python benchmarks/bench_lif.py [--steps 100] [--batch 64] [--features 512]

Prints the median of `--repeats` timed calls after two warm-up calls; each
timed call blocks on its result. The numbers in the README came from this
script on the hardware they name.
"""

import argparse
import statistics
import time

import flax.linen as nn
import jax
import jax.numpy as jnp

import sparx
from sparx.dynamics import LIFCell, SynapticInput, decay, run


def timed(fn, *args, repeats):
    for _ in range(2):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def lif_unroll(args):
    x = jax.random.normal(jax.random.key(0), (args.steps, args.batch, args.features))
    print(f"LIF forward+backward, T={args.steps} B={args.batch} F={args.features}")
    for unroll in (1, 2, 4, 8, 16):
        def loss(x, unroll=unroll):
            return jnp.sum(run(LIFCell(0.8), x, unroll=unroll)[0].fired)
        fn = jax.jit(jax.grad(loss))
        print(f"  unroll={unroll:<3} {timed(fn, x, repeats=args.repeats) * 1e3:8.2f} ms")


class Folded(nn.Module):
    """Synapses applied to all time steps at once, neurons scanned."""
    features: int

    @nn.compact
    def __call__(self, x):
        x = sparx.nn.LIF()(nn.Dense(self.features)(x))
        return sparx.nn.LIF()(nn.Dense(self.features)(x))


def per_step(params, x):
    """`Folded` with the same parameters, stepped one time step at a time:
    both synaptic products run inside the loop, on `B` rows each."""
    cell = LIFCell(decay(2.0))
    first, second = params["params"]["Dense_0"], params["params"]["Dense_1"]

    def step(states, x_t):
        a, s = cell.step(states[0], SynapticInput(jump=x_t @ first["kernel"] + first["bias"]), 1.0)
        b, s = cell.step(states[1], SynapticInput(jump=s.fired @ second["kernel"] + second["bias"]), 1.0)
        return (a, b), s.fired

    features = first["kernel"].shape[1]
    states = (cell.init_state((*x.shape[1:-1], features), x.dtype),) * 2
    return jax.lax.scan(step, states, x)[1]


def folding(args):
    x = (jax.random.uniform(jax.random.key(0), (args.steps, args.batch, args.features)) < 0.2)
    x = x.astype(jnp.float32)
    print(f"Two Dense+LIF layers forward+backward, T={args.steps} B={args.batch} F={args.features}")
    net = Folded(args.features)
    params = net.init(jax.random.key(1), x)
    for name, forward in (("folded over time", net.apply), ("per step in the scan", per_step)):
        fn = jax.jit(jax.grad(lambda p, x, forward=forward: jnp.sum(forward(p, x))))
        print(f"  {name:<22} {timed(fn, params, x, repeats=args.repeats) * 1e3:8.2f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    print(jax.devices())
    lif_unroll(args)
    folding(args)


if __name__ == "__main__":
    main()
