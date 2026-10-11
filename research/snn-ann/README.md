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
| A, spiking | 9.0% | 0.53 / 0.39 ms | 3.52 / 0.75 ms | 0.028 / 0.032 ms |
| B, graded, events | 45.5% | 0.41 / 0.39 ms | 1.85 / 0.83 ms | 0.027 / 0.034 ms |
| sigma-delta | 7.7% | 0.41 / 0.57 ms | 0.86 / 1.30 ms | 0.029 / 0.033 ms |
| dendritic | 11.2% | 0.41 / 0.53 ms | 0.81 / 1.09 ms | 0.031 / 0.034 ms |

Each cell gives command buffers on, then off. Every arm takes about the same time. A step called alone
costs 0.39 to 0.57 ms, which is the cost of dispatching a compiled function from Python and reading its
readout back, not of its 1.46 million multiply-adds: in a stream the same step takes 0.027 to 0.034 ms.
The two runs of each arm differ by up to 0.17 ms in their medians and 2.8 ms in their 99th percentiles,
in both directions, so command buffers make no measurable difference here; the spread is the host's,
shared with other jobs. As XLA computes these networks, the spiking arm's sparsity buys no latency on this
GPU.

## Next

- A fused Pallas kernel for one step of A: both convolutions as implicit GEMMs and the dense layer, the
  dense layer's 4,097 × 128 weights resident in L2 across steps, and input blocks that are all zero
  skipped, which at 9% density removes about nine tenths of the weight reads, the step's real cost at
  batch 1 (DeafMoth reviews). The same kernel with skipping off is the dense baseline it must beat.
- Its distance from a float64 reference, RMS-relative and max-relative, per output, at the real shapes on
  the RTX 4080, with a fixed bound in its test at or below XLA's own measured distance.
- The step's latency with the kernel, called alone and in a stream, for A against B through the same
  kernel; the homeostatic arm, trained and measured the same way.
