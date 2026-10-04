# Design

How sparx represents spiking networks, how models are built from them, and how they are trained. Each section says what the code does today, the choices behind it, and what is still open. Measurements behind performance choices are in [performance.md](performance.md).

## Time and layout

Every array inside a network is time-major, `[T, B, ...]`. Flax's synaptic layers (`Dense`, `Conv`, `BatchNorm`, pooling) treat leading axes as batch axes, so a synapse applies to every time step in one call, as one large matrix product. Only the neuron recurrence runs step by step.

The alternatives were batch-major `[B, T, ...]`, which Flax's `nn.RNN` uses, and single-step modules that a training loop calls once per step, as snnTorch does. Batch-major would put a transpose before every scan. Single-step modules move the time loop outside the network, so every synapse runs `T` times on `B` rows instead of once on `T * B` rows (15% slower on CPU in `benchmarks/bench_lif.py`, with larger differences expected on matrix units). Data stays batch-major on disk and in dew's loaders; encoders produce the time-major layout at the network's input.

Open: layouts for very long sequences where `T * B` rows do not fit in memory. Chunked execution through the `state` collection already gives exact results; what is missing is an automatic chunking policy and rematerialization across chunks.

## Neurons

A neuron is a cell, a Flax struct dataclass with `init_state(shape, dtype)` and `step(state, x)`. Numerical constants are PyTree leaves, so they can be learned, traced through `vmap` or sharded; choices such as the reset rule and the surrogate are static fields. `sparx.run` scans a cell over time. Flax layers in `sparx.nn` only build a cell from attributes and parameters, then run it.

This split keeps the dynamics usable from plain JAX, and it makes every neuron testable against a NumPy loop of its equations (`tests/reference.py`). A new neuron is a cell, a reference loop and a thin layer.

Conventions shared by the LIF family:

- `dt = 1` and decay `exp(-1 / tau)` per step; the input enters unscaled.
- The reset reads the spike of the same step.
- The membrane runs in float32 whatever the input dtype; spikes return in the input dtype.
- Learned decays are the sigmoid of a parameter, one per feature.

Two families need no loop over time:

- Parallel spiking neurons (`PSN`, `MaskedPSN`, `SlidingPSN`) replace the recurrence with a learned mixing matrix over time.
- `DelayedDense` moves the temporal structure into the synapse: each synapse learns a delay.

Open questions:

- Neurons whose reset makes the recurrence nonlinear could still run in parallel over time with fixed-point or associative formulations. These only pay off if they beat the sequential scan on an accelerator, which has not been measured.
- Resonate-and-fire and other oscillatory neurons, as complex-valued cells.
- Learnable thresholds and per-neuron surrogate widths.

## Gradients

Spikes are Heaviside steps with a surrogate derivative, defined once as a `jax.custom_jvp`, so reverse mode, forward mode and `vmap` share one rule. Gradients flow through time by ordinary backpropagation through the scan.

The choice of surrogate interacts with recurrence. With ATan, whose tails fall as `1 / x^2`, every neuron passes gradient, and a recurrent layer's backward Jacobian grew past 1 as training enlarged the recurrent matrix (gradient norm past 1e8 within 300 SHD steps). `FastSigmoid(100)` kept it below 10. The default stays ATan, which trained the feedforward networks fastest, and the `Recurrent` docstring records the finding.

Open: training without backpropagation through time. Online rules (e-prop, OTTT) keep eligibility traces per synapse and update at every step, which bounds memory by the network rather than the sequence length and is how a neuromorphic chip would learn. They fit as alternative objectives that read the same cells, since a cell's step exposes everything a trace needs.

## Models

Models are plain Flax modules. Building blocks so far:

| Block | Use |
| --- | --- |
| `Dense` or `Conv` followed by a neuron layer | feedforward spiking layers |
| `Recurrent(neuron)` | recurrent layers; the input projection stays outside the loop |
| `DelayedDense` | synapses with learned delays, for temporal tasks |
| `PSN`, `SlidingPSN` | temporal mixing without a loop |
| `LI` | the non-spiking readout whose membrane becomes the logits |
| `models.SEWResNet` | deep convolutional networks; spike-element-wise residuals keep the identity map |

`SEWBlock` and `SEWResNet` take a `neuron` factory, so a whole network changes neuron type, time constant or surrogate in one argument.

Open: spiking self-attention (Spikformer's attention over spike matrices, without softmax) and spiking language models. These are the main architectures missing for sequence modeling beyond classification.

## Objectives and losses

A loss reads the outputs over time, `[T, B, C]`. sparx keeps the reduction separate from the loss:

| Readout | When |
| --- | --- |
| mean over time | rate coding; the default for static images |
| max over time of an `LI` membrane | temporal tasks where the decision peaks once (SHD) |
| spike count | output neurons that spike |
| every step (`per_step_cross_entropy`) | asks each step to classify alone (TET, Deng et al. 2022) |

Regularization reads the `spike_rates` collection: `rate_penalty` keeps each neuron's rate in a band, which prevents silent layers (no gradient) and saturated ones (no information, more energy).

`sparx.dew.SpikingClassifier` packages encoder, readout, rate band and scheduled model arguments (`call`) as a dew `Objective`, so dew's `Trainer` runs it. Schedules reach the model through `call`, which maps the step counter to keyword arguments: the delay width of `DelayedDense`, the masking of `MaskedPSN`.

Open:

- Objectives beyond classification. Sequence prediction over spike trains, regression of continuous signals, and self-supervised objectives (JEPA over spiking encoders) would reuse dew's objectives with a spiking model.
- Energy as a training signal. A synaptic-operation count (spikes times fan-out) is computable from the sown rates and could join the loss as a measured cost.
- Conversion from trained ANNs, as an alternative to surrogate training for deep networks.

## Performance

Measured on CPU only so far. Synapses dominate the cost, the neuron scan is a few percent, and `unroll=1` is fastest. Accelerator work comes next, in this order:

1. Measure the scan, synapse folding and the PSNs on a GPU and a TPU.
2. Fuse the time loop into one kernel (Pallas) where the scan's per-step overhead dominates.
3. Revisit the associative scan for linear dynamics on hardware where it may win.

Event-driven sparse execution is deferred. Dense products on binary inputs are what accelerators run fastest at the sparsities measured (6 to 15% in the SHD runs).
