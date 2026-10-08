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
| `examples/train_shd.py --channels 140 --hidden 128` | 31.9 steps/s |
| the same with `--delays 15`: 16 lagged products in the input layer | 12.6 steps/s |

The recurrent run differs in two ways, the `[64, 256] x [256, 256]` feedback product inside the time loop and the surrogate, and the cost of each has not been separated.

### A recurrence over a sparse wiring

A `RecurrentCell` over a `Sparse` wiring gathers each edge's source and sums at its target with `segment_sum`, and backpropagating through a gather scatters its gradient back. Gathering one value per edge and example along the last axis made that a scatter of single values. Putting the units first, so each edge gathers a row of the batch, makes it a scatter of rows. On the continual core of `research/continual` (256 units, 20,480 edges with delays of 1 to 8 steps, batch 32, 24 steps), sending through the wiring and back took 146 ms per sequence with single values and 80 ms with rows. A training step of the core went from 5.6 to 9.5 per second, and of the core with fast weights on 4,096 of its edges, whose rule reads each edge's units the same way, from 2.8 to 3.6 (`research/continual/train.py --doors 4`, measured 8 October 2026). Dense matrices, one per delay, took 32 to 40 ms for the same send: at this size and density (7.8% of the pairs of each delay) they would be faster still, and are not built.


## e-prop

`python benchmarks/bench_eprop.py` times one batch of `examples/train_shd_eprop.py`'s network (T=100, B=64, 140 inputs, 128 recurrent neurons, 20 readout units) and reports the scratch memory of XLA's compiled program, where the eligibility traces live. The CPU was shared with other jobs (load average 5 to 8 on its 4 cores), so the times carry about 30% of noise.

| Gradient of one batch | Time | Scratch memory |
| --- | --- | --- |
| BPTT, ALIF | 24.4 ms | 11.4 MiB |
| e-prop, ALIF, a vector per state variable and synapse | 4887 ms | 84.2 MiB |
| e-prop, ALIF, as shipped | 349 ms | 25.7 MiB |
| BPTT, LIF | 19.7 ms | 10.5 MiB |
| e-prop, LIF, a vector per state variable and synapse | 419 ms | 25.5 MiB |
| e-prop, LIF, as shipped | 131 ms | 8.8 MiB |

The first version (commit b11bb08, timed with the benchmark's two calls written in its API) advanced an eligibility vector `[B, N, in + N, d]` for the `d` state variables (three for ALIF: membrane, adaptation, refractory count) through each neuron's `d x d` Jacobian with an einsum every step, and took the Jacobian by `d + 1` forward derivatives. Three changes made it 14 times faster for ALIF and 3 times for LIF, with the same gradients (the identity tests hold at 1e-9):

- The structure of the step's Jacobian is read once from its jaxpr. A state variable whose input enters with constant coefficients shared by all neurons, the membrane under a detached reset, keeps Bellec et al.'s filtered presynaptic trace `[B, in + N]` instead of a vector per synapse; one no gradient reaches, the refractory count, keeps nothing. ALIF keeps one vector per synapse, for the adaptation, where it kept three.
- The remaining products are elementwise and fuse into one loop, and the Jacobian's columns come from one linearization of the step.
- The weight gradient sums `signal[b, n] * filtered[b, n, p]` over the batch elementwise. XLA on this CPU ran the einsum `"bn,bnp->pn"` as a batched matrix product, three times slower (284 ms against 97 ms for 100 such steps alone).

The readout's leak still needs one filtered trace per synapse, `B x N x (in + N)` numbers, since the learning signal of each step weights the traces of every earlier step. Training SHD for five epochs (`examples/train_shd_eprop.py --rule eprop`) went from 39 min to 3 min 30 s, against 30 s for BPTT. On 8 October 2026 the same runs took 4 min 16 s to 4 min 53 s and 34 to 38 s on this machine, where the previous commit's e-prop run, whose evaluation was not compiled, took 4 min 54 s.

## Against NEST and Brian2

`python benchmarks/bench_networks.py` times sparx, and `python tools/bench_reference_simulators.py` (in the reference environment of HANDOFF.md) times NEST 3.10 and Brian2 2.10, on the same networks at `dt = 0.1` ms: Brunel's (2000) network at the paper's size in its asynchronous irregular regime, Brette et al.'s (2007) CUBA and COBA, and Potjans and Diesmann's (2014) cortical microcircuit at a fifth of its neurons and inputs, which NEST builds with the reference's own PyNEST code. The machine is a 4-core Intel Xeon at 2.8 GHz with 15 GB, otherwise idle; NEST runs 4 threads, Brian2's C++ standalone 4 OpenMP threads, and sparx JAX 0.11.2 on CPU in float32. Measured on 8 October 2026. Wall time per simulated second, after building and compiling:

| Network | Neurons, synapses | Excitatory rate | sparx | NEST | Brian2 standalone | Brian2 Cython |
| --- | --- | --- | --- | --- | --- | --- |
| Brunel | 12,500, 15.6M | 37.3 to 37.6 Hz | 9.6 s | 7.5 s | 11.8 s | 16.1 s |
| CUBA | 4,000, 320,000 | 5.4 to 5.6 Hz | 0.71 s | 0.42 s | 0.33 s | 0.59 s |
| COBA | 4,000, 320,000 | 18 to 20 Hz | 1.01 s | 3.85 s | 0.54 s | 0.98 s |
| Microcircuit, a fifth | 15,435, 12.0M | 0.6 to 0.7 Hz (layer 2/3) | 8.76 s | 2.89 s | | |

Each simulator's rate is within the spread of the others', so they run the same networks. The microcircuit row was measured the same day, after its event delivery gained delays per edge: building it took sparx 8.8 s and compiling a chunk 8.8 s, against NEST's 4.5 s. A step of it carries about 4 spikes over 55 projections, and sparx takes 0.88 ms for it. Two changes took it there from 21.0 s per simulated second:

- A step writes each receptor's arrivals before it reads them. Read first, the row being read made XLA copy the whole buffer of arrivals, 51 rows of 15,435 neurons, every step: 0.36 ms against 0.012 ms in isolation, and 12.4 s against 21.0 s for the network.
- Its projections deliver in passes of 4 spiking neurons, since each carries few events a step: 8.8 s, against 9.7 s for passes of 2, 9.9 s for 8 and 12.8 s for 16. The other networks keep 16: CUBA took 0.69, 0.69 and 0.81 s for passes of 4, 8 and 16, COBA 1.25, 1.05 and 1.02 s, Brunel 21.6, 13.7 and 9.9 s. NEST integrates `iaf_cond_exp` with adaptive Runge-Kutta per neuron, which is more accurate and costs it the COBA row; sparx and Brian2 hold the conductance over the step. Building Brunel's network and drawing its synapses took sparx 12.5 s and compiling a chunk 2.7 s, against NEST's 2.8 s to build, and Brian2's 10.2 s (standalone, its C++ compilation included) and 26.7 s (Cython).

Until 8 October 2026 a projection under `format="auto"` was a dense matrix or an edge list, and every step paid for every synapse: Brunel took 457 s per simulated second, CUBA 15.5 s and COBA 15.3 s. The changes that brought them to the table's times, each measured on these three networks:

- Event delivery for every spiking projection with one delay and fixed weights. With its then fixed capacity of 4,096 spiking neurons a step, Brunel took 39.9 s, CUBA 6.3 s and COBA 8.7 s, most of a step in a binary search sized for that capacity.
- Padded rows of out-edges when the out-degrees are near even or the rows few, read without a search, and steps sized for their spikes: CUBA 1.8 s.
- `jax.lax.top_k` in place of `jnp.nonzero` to find the spiking neurons: 18 us against 80 us for 16 of 10,000 neurons.
- The noise drawn from a Threefry-4x32 key. JAX's default Threefry-2x32 compiles to a loop on CPU: Brunel took 12.4 s with it.
- Passes of 16 spiking neurons, as many as a step needs, in place of tiers of fixed size, so a run of many trials under `jax.vmap` pays for its busiest trial's spikes and not for every tier, and no spike is dropped.

## Networks and connectomes

`sparx.graph` on the same 4-core CPU, float32, `dt = 0.1` ms, wall time per simulated second after compilation, measured before event delivery changed on 8 October 2026 (the connectome tables are not in that day's container, so these rows were not measured again):

| Network | Neurons | Connections | Delivery | Time per simulated s | Peak memory |
| --- | --- | --- | --- | --- | --- |
| Shiu et al. (2024) on FlyWire v630, 21 sugar neurons at 100 Hz (about 9,600 spikes/s) | 127,400 | 14.7M | events | 27 to 31 s | 1.6 GB |
| The same model on the male CNS v0.9, both giant fibres at 200 Hz (about 860,000 spikes/s) | 165,899 | 25.6M | events | 56 s | 1.8 GB |

Measured choices behind these:

- Edge delivery gathers and sums over every edge each step (`jax.ops.segment_sum`): 2.2 ms for 500,000 edges. A dense projection of the same density multiplies a spike vector by a matrix in 0.87 ms, so a projection with one delay and fixed weights is stored dense when it has at most 2^25 entries and a density of at least 2%.
- Event delivery visits only the out-edges of neurons that spiked. A connectome's out-degrees are uneven, so its spiking neurons' edges are laid end to end in blocks of 4,096, as many blocks as a pass needs. Its first version laid every step's edges into a fixed 2^20 slots: 63 ms a step on FlyWire, against 7 ms for blocks. Finding the spiking neurons (`jnp.nonzero` over 127,400) cost 1.4 ms of that; `jax.lax.top_k` finds 16 of them in 0.24 ms.
- Poisson inputs with a static mean invert a precomputed CDF: `jax.random.poisson` loops per draw and took 1.8 ms a step for 2,500 neurons, 40% of a Brunel step.

No GPU or TPU numbers yet; Phase 7 measures them.
