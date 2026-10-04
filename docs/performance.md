# Performance

What was measured, on which hardware, and what was decided from it. Every number here comes from a command in this repository; rerun it on your hardware before relying on it.

All numbers below are from a 4-core x86 CPU container (`jax.devices()` reports `cpu`), JAX 0.11.2, Flax 0.12.10, float32, median of 10 timed calls after 2 warm-up calls. No GPU or TPU was available, so nothing here describes accelerator behavior.

## Where the time goes

A spiking layer is a synaptic product followed by neuron dynamics. Sparx applies the synapses to every time step at once (a Flax `Dense` over `[T, B, F]` is one `[T*B, F] x [F, H]` product) and scans only the elementwise neuron recurrence over time.

`python benchmarks/bench_lif.py` (T=100, B=64, F=512):

| Forward and backward | Time |
| --- | --- |
| One LIF layer's scan, `unroll=1` | 11.7 ms |
| Two Dense+LIF layers, synapses folded over time | 161.7 ms |
| The same two layers, synapses inside the time loop | 189.2 ms |

On this CPU the matrix products dominate. The neuron scan costs about 7% of the two-layer step, and folding the synapses over time saves 15% against computing them step by step. Folding turns `T` products over `B` rows into one product over `T*B` rows, which should matter more on matrix units, but that has not been measured.

## The scan's unroll

`run(cell, xs, unroll=k)` places `k` time steps in each iteration of the compiled loop. Same benchmark, one LIF layer:

| unroll | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| Time | 11.7 ms | 20.3 ms | 18.4 ms | 29.3 ms | 52.0 ms |

Unrolling is slower on this CPU, so the default is 1. Every layer takes `unroll=` for hardware where the loop's per-iteration cost is higher.

## Tried and not adopted

An associative (parallel prefix) scan for the linear leaky integrator, `jax.lax.associative_scan` over `(decay, x)` pairs, against the sequential scan, forward and backward:

| T, neurons | Sequential | Associative |
| --- | --- | --- |
| 100, 32768 | 12.4 ms | 49.3 ms |
| 1000, 4096 | 15.4 ms | 75.8 ms |
| 4000, 1024 | 16.5 ms | 101.9 ms |

The parallel scan does about twice the work and is 4 to 6 times slower here. It may win on a GPU or TPU for long sequences, where the sequential loop's per-step cost is higher; that needs measuring on one before it ships.

A hand-written backward pass for the LIF scan (`jax.custom_vjp` storing only the pre-spike membrane, then a reverse scan) matched autodiff's gradients to 1e-5 but took 16.8 ms against autodiff's 18.0 ms at T=100 with 32768 neurons, and 16.9 ms against 16.0 ms at T=1000 with 4096. XLA's autodiff of the scan already keeps one residual per step, so there was nothing to save, and it did not ship.

## Training throughput

Steady-state steps per second as dew's display or the script reports them, on the 4-core CPU:

| Run | Throughput |
| --- | --- |
| `examples/train_mnist.py`, batch 128, 8 steps, 784-512-512-10 | 33 steps/s (468 steps in 14 s) |
| `examples/train_shd.py`, batch 64, 100 steps x 700 channels, 256 ALIF | 11.3 steps/s |
| `examples/train_shd.py --recurrent`, the same with a 256 x 256 recurrent matrix | 6.6 steps/s |

The recurrent run differs in two ways, the `[64, 256] x [256, 256]` feedback product inside the time loop and the surrogate, and the cost of each has not been separated.
