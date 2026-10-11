# Spiking against graded networks: latency at batch 1

The racer study (`research/racer`) found that a spiking network drives as well as graded ones and would
need about a quarter of their work on event-driven hardware, with no saving on a GPU as XLA computes it.
This study asks whether the spikes' sparsity can be turned into lower latency on a GPU at batch 1, the
way a controller runs: one input in, one readout out, every 20 ms. The win condition is the wall time
from a step's input to its readout on the host, median and 99th percentile, at task performance within
the arms' seed-to-seed spread.

The arms are Phase 2's networks, each the seed that drove farthest on its held-out tracks: A spiking,
B graded on events, sigma-delta and dendritic, and a homeostatic arm still to train, A with intrinsic
plasticity holding its neurons' rates lower (`sparx.nn.Homeostatic`). Every run is on the workstation's
RTX 4080, one at a time through the GPU queue.

## First result: dense XLA

`latency.py` loads each arm's weights and feeds it the inputs of its own drive on an unseen track. One
step is compiled alone and called once per input, its state carried between calls, 2,000 calls after
200 of warm-up; a stream is 1,500 steps in one compiled scan. `results/latency-rtx4080.json`, with XLA's
CUDA command buffers on (its default), and `results/latency-rtx4080-no-command-buffers.json`, off:

| Arm | Units active a step | One step, median | One step, 99th percentile | Per step in a stream |
| --- | --- | --- | --- | --- |
| A, spiking | 9.0% | 0.40 ms | 0.76 ms | 0.029 ms |
| B, graded, events | 45.5% | 0.40 ms | 0.77 ms | 0.028 ms |
| sigma-delta | 7.7% | 0.43 ms | 0.85 ms | 0.028 ms |
| dendritic | 11.2% | 0.42 ms | 0.87 ms | 0.031 ms |

Every arm takes the same time. A step called alone costs 0.40 to 0.43 ms, which is the cost of
dispatching a compiled function from Python and reading its readout back, not of its 1.46 million
multiply-adds: in a stream the same step takes 0.03 ms. Command buffers off changed no number by more
than 0.03 ms. As XLA computes these networks, the spiking arm's sparsity buys no latency on this GPU.

## Second result: Pallas with zero-block skipping

`kernel.py` steps both trained arms, A's run 7 and B's run 29, at batch 1 through the same
Pallas/Triton implementation. There are four compute launches: conv0 with its membrane update,
conv1 with its membrane update, 64 parallel partial sums of the dense drive, and the final dense
reduction with its membrane update and readout. The standalone compiled return also has one
XLA-generated copy kernel because it returns the readout both in the state and on its own.

The convolutions gather their stride-2, SAME-padded windows and reduce products. They are not
tensor-core GEMMs. Each convolution tests a window's channel block before reading its weights;
the dense layer tests contiguous 16-input blocks and keeps their weight loads inside the nonzero
branch. `skip=False` removes those tests from the same implementation. LIF units sum their drive,
fire at 1 and reset to zero; B's graded units average their drive and send ReLU of their membranes.
Weights and carried membranes are float32. Products, sums and membrane arithmetic are float64,
with one rounding for the stored membrane. This is not a comparison at matched accumulator
precision: the XLA baseline is the original `stack.act` in float32.

A multi-CTA, single-launch step needs a grid-wide barrier between populations, which this
Triton backend does not expose. A three-launch version put the whole dense/readout in one CTA
and serialized its weight reads. The fourth launch distributes that reduction over the GPU.
Mosaic GPU's sm90 requirement excludes this sm89 device.

`measure.py` records each arm's own 1,500-step drive on track seed 1000, including the speed before
the car is advanced, as `racer.inputs` reads it. After 200 warm-up steps, it measures 2,000 calls
with state carried and the readout transferred to the host, then ten synchronized 1,500-step scans.
Inputs are sliced and made ready before timing. The first result sliced immediately before each
timed call, so its interval also waited for any pending input-slice work. The new host numbers do
not include that work. `results/kernel-rtx4080.json` records the full precision of every measurement,
the model and input hashes, the first 32 recorded event images and speeds, and the fixed test bounds.

RTX 4080, JAX 0.11.2.post3, CUDA command buffers on. The wrapper supplied
`XLA_FLAGS=--xla_allow_excess_precision=false --xla_gpu_enable_allocator_spatial_partitioning=false`.
All times below are milliseconds. Device active time is the sum of CUPTI kernel durations divided
by 100 standalone calls over the first 100 recorded inputs, not an end-to-end device timer; it
excludes gaps and the copy kernel.

| Arm | Step | Host median | Host p99 | Per step in scan | Device active |
| --- | --- | --- | --- | --- | --- |
| A, spiking | Pallas, skipping | 0.197 | 2.049 | 0.0453 | 0.0424 |
| A, spiking | Pallas, no skipping | 0.156 | 1.516 | 0.0401 | 0.0367 |
| A, spiking | XLA float32 | 0.179 | 2.019 | 0.0272 | 0.0289 |
| B, graded | Pallas, skipping | 0.162 | 0.312 | 0.0451 | 0.0370 |
| B, graded | Pallas, no skipping | 0.132 | 0.262 | 0.0427 | 0.0341 |
| B, graded | XLA float32 | 0.197 | 0.417 | 0.0282 | 0.0250 |

Skipping does not meet the latency win condition. It is slower than leaving it off in both
streams and both device traces, and A's skipping median and p99 are above XLA's. A has 61.5% all-zero
dense input blocks, B 0.0281%; the saved reads do not repay the tests and branches in this
implementation. B's host median is lower than XLA's, but its streamed step and device active time
are higher. The host p99 includes the workstation's dispatch and scheduling tails.

The independent NumPy reference sums in float64, in the browser implementation's tap-row,
tap-column, channel order, with the bias added after the sum. Relative RMS is
`L2(error) / L2(reference)` over all 1,500 steps; max-relative is
`Linf(error) / Linf(reference)`, not an elementwise division by near-zero membranes. Skipping on
and off have identical measured distances. The XLA float32 yardstick uses the same inputs and
starts at rest too.

| Arm | Membranes | Pallas RMS-rel | Pallas max-rel | XLA RMS-rel | XLA max-rel |
| --- | --- | --- | --- | --- | --- |
| A | conv0 | 3.64e-8 | 6.92e-8 | 1.21e-7 | 2.37e-7 |
| A | conv1 | 3.78e-8 | 8.49e-8 | 1.18e-7 | 2.62e-7 |
| A | hidden | 3.59e-8 | 7.35e-8 | 1.01e-7 | 1.56e-7 |
| A | readout | 3.69e-8 | 9.08e-8 | 6.88e-8 | 1.90e-7 |
| B | conv0 | 3.62e-8 | 1.10e-7 | 1.17e-7 | 2.62e-7 |
| B | conv1 | 3.94e-8 | 1.06e-7 | 1.97e-7 | 3.20e-7 |
| B | hidden | 3.61e-8 | 7.07e-8 | 2.81e-7 | 3.94e-7 |
| B | readout | 4.58e-8 | 1.19e-7 | 2.42e-7 | 3.55e-7 |

Every population has zero spike mismatches against both float64 and XLA float32 across the drive,
with skipping on or off. For B these are active-unit, not binary-spike, mismatches. XLA also has
zero mismatches against float64. `test_kernel.py` runs the first 32 recorded steps at these real
shapes in Pallas CPU interpret mode and on the GPU, compiles the GPU call, and checks those counts.
Its fixed RMS-relative bound is 5e-8 for every output; max-relative is 1e-7 for A and B's hidden
population, and 1.5e-7 for B's other outputs. Every bound is below XLA's measured distance both
on the 32-step prefix and the complete drive. It also checks that skipping a zero drive still
leaks the stored negative graded membranes. All 16 tests pass. The Triton backend emits its
JAX deprecation warning; it remains the backend supported on this card.

Run `measure.py` through the racer GPU wrapper with `--minutes 20`, `env PYTHONPATH=src` and the
Dew Python environment. `--mode profile` adds the CUPTI measurements to the existing result;
`--mode verify` records the queued test result with `JAX_PLATFORMS=cuda,cpu` for interpretation.
`--runs` overrides the Phase 2 artifacts directory. Arm A can use the checked-in racer; arm B
needs run 29's trained weights.

## Next

- Reduce reduction, branching and launch costs while keeping the fixed correctness bounds.
- Train and measure the homeostatic arm through the same step and input method.
