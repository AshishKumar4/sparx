# Sparx

Spiking neural networks in JAX and Flax.

Sparx builds spiking networks out of ordinary Flax linen layers and a small set of neuron layers that run over time. Neuron dynamics are pure JAX, gradients pass through spikes by surrogate derivatives, and networks train with optax, with your own loop or with [dew](https://github.com/AshishKumar4/dew)'s `Trainer`. Everything is a JAX PyTree, so `jit`, `grad`, `vmap`, forward-mode `jvp` and sharding work as they do for any Flax model.

APIs can change before 1.0.

## Contents

- [A first network](#a-first-network)
- [How a network runs over time](#how-a-network-runs-over-time)
- [Neurons](#neurons)
- [Surrogate gradients](#surrogate-gradients)
- [Encoding, losses and firing rates](#encoding-losses-and-firing-rates)
- [Streaming](#streaming)
- [Pure JAX cells](#pure-jax-cells)
- [Training with dew](#training-with-dew)
- [Results](#results)
- [Performance](#performance)
- [Correctness](#correctness)
- [Installation](#installation)
- [Roadmap](#roadmap)

## A first network

```python
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

import sparx


class Net(nn.Module):
    @nn.compact
    def __call__(self, spikes):                     # [T, B, 784]
        x = sparx.nn.LIF(tau=2.0)(nn.Dense(256)(spikes))
        return sparx.nn.LI(tau=2.0)(nn.Dense(10)(x))  # membrane [T, B, 10]


net = Net()
images = jax.random.uniform(jax.random.key(0), (32, 784))  # intensities in [0, 1]
labels = jnp.zeros(32, jnp.int32)
spikes = sparx.encode.rate(jax.random.key(1), images, steps=8)  # [8, 32, 784]
params = net.init(jax.random.key(2), spikes)


def loss(params):
    logits = jnp.mean(net.apply(params, spikes), axis=0)  # mean membrane over time
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


grads = jax.grad(loss)(params)
```

`LIF` turns input currents into spikes, exactly 0 or 1, and `LI` is a leaky integrator whose membrane is the readout. The rest is Flax and optax. [`examples/train_mnist.py`](examples/train_mnist.py) is this network trained to completion with a plain JAX loop.

## How a network runs over time

Time is the leading axis of every array inside a network: `[T, B, ...]`. Flax's `Dense`, `Conv`, `BatchNorm` and pooling treat every leading axis as a batch axis, so a synaptic layer applies to all time steps in one call, as one large matrix product. Only the neurons' elementwise recurrence runs step by step, as a `jax.lax.scan` inside each neuron layer. A convolutional network needs nothing extra:

```python
class ConvNet(nn.Module):
    @nn.compact
    def __call__(self, x):                          # [T, B, H, W, C]
        x = sparx.nn.LIF()(nn.BatchNorm(use_running_average=False)(nn.Conv(32, (3, 3))(x)))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = x.reshape(*x.shape[:2], -1)             # keep [T, B], flatten the rest
        return sparx.nn.LI()(nn.Dense(10)(x))
```

Each neuron layer keeps its membrane in float32 whatever its input dtype, and returns spikes in the input's dtype, which holds 0 and 1 exactly even in bfloat16.

## Neurons

The LIF family (`LIF`, `IF`, `LI`, `Synaptic`, `ALIF`) shares one discrete-time convention with step `dt = 1` and per-step decay `exp(-1 / tau)`: `v[t] = decay * v[t-1] + x[t]`, a spike where `v[t] >= threshold`, then a reset. The input enters unscaled, as in snnTorch's `Leaky`.

| Layer | Dynamics | Learnable |
| --- | --- | --- |
| `LIF(tau, threshold, reset, surrogate, detach_reset)` | leaky integrate-and-fire | `learn_tau=True`: a decay per feature |
| `IF(threshold, reset, ...)` | integrate-and-fire, no leak | |
| `LI(tau)` | leaky integrator, never fires, returns its membrane | `learn_tau` |
| `Synaptic(tau, tau_synapse, ...)` | current-based LIF: a decaying synaptic current charges the membrane | `learn_tau` (both) |
| `ALIF(tau, tau_adapt, beta, ...)` | adaptive threshold that rises by `beta` per spike (Bellec et al. 2020) | `learn_tau` (both) |
| `Izhikevich(a, b, c, d, dt)` | Izhikevich's two-variable neuron (2003) | |
| `Recurrent(neuron)` | feeds any neuron's spikes back to its input through a learned `[F, F]` matrix | the matrix |
| `PSN()` | parallel spiking neuron: `H = W X + b` over all `T x T` step pairs (Fang et al. 2023) | `W`, `b` |
| `MaskedPSN(k)` | the PSN restricted to the `k` most recent steps | `W`, `b` |
| `SlidingPSN(k)` | `k` weights slid over time, any `T`, causal | weights, `b` |

`reset` is `"subtract"` (soft reset, the default), `"zero"` (hard reset) or `"none"`. `detach_reset=True` stops the gradient through the reset, as SpyTorch's tutorials and SpikingJelly's `detach_reset` do. Learned decays are the sigmoid of a parameter, so training cannot push them outside (0, 1).

The PSNs have no loop over time at all. Each is one `[T, T] x [T, N]` product followed by a threshold, so no step waits for the one before it; Fang et al. report that this also learns longer dependencies than the LIF. `Recurrent(ALIF())` is the recurrent adaptive network (LSNN) of Bellec et al.

## Surrogate gradients

A spike is the Heaviside step of `v - threshold`. Its derivative is zero almost everywhere, so the backward pass uses a surrogate's derivative instead, while the forward pass stays binary. `sparx.spike` is a `jax.custom_jvp`, so the same rule serves `jax.grad`, `jax.jvp` (forward-mode and forward-gradient training) and `vmap`.

| Surrogate | Derivative at `x = v - threshold` | Source |
| --- | --- | --- |
| `ATan(alpha=2)` (default) | `alpha / 2 / (1 + (pi / 2 * alpha * x)^2)` | Fang et al. 2021, SpikingJelly, snnTorch |
| `Sigmoid(alpha=4)` | `alpha * s * (1 - s)`, `s = sigmoid(alpha x)` | SpikingJelly |
| `FastSigmoid(slope=25)` | `1 / (slope * abs(x) + 1)^2` | SuperSpike (Zenke and Ganguli 2018) |
| `Triangle(width, scale)` | `scale * max(0, 1 - abs(x) / width)` | Bellec et al. 2018 |
| `Rectangle(width)` | `1 / width` inside `abs(x) < width / 2` | Wu et al. 2018 |
| `Gaussian(sigma)` | the normal density | Wu et al. 2018 |
| `StraightThrough()` | 1 | |

Pass one to any neuron: `sparx.nn.LIF(surrogate=sparx.surrogate.FastSigmoid(100.0))`.

## Encoding, losses and firing rates

`sparx.encode` turns data into time-major spike trains: `rate` (Bernoulli spikes at the value's probability), `latency` (one spike, earlier for larger values), `delta` (spikes on changes of a signal) and `repeat` (the values as a constant input current, direct encoding).

`sparx.losses` has losses over the whole output sequence, one value per example: `per_step_cross_entropy` asks every time step to classify (Deng et al. 2022) and `rate_mse` pulls each output neuron's firing rate toward a target. For a loss on one readout, reduce time first and use optax: `jnp.max(v, axis=0)` of an `LI` membrane, its mean, or the spike count.

Spiking layers report their firing rates when the `spike_rates` collection is mutable:

```python
outputs, sown = net.apply(params, spikes, mutable=["spike_rates"])
sparx.firing_rates(sown)                          # {"LIF_0": mean rate, ...}
penalty = sparx.rate_penalty(sown, lower=0.01, upper=0.3)  # differentiable
```

A plain `apply` sows nothing and costs nothing.

## Streaming

When the `state` collection is mutable, each neuron layer starts from the state it holds and writes back its final state. A sequence fed in chunks, down to one step at a time, gives exactly the output of one call over the whole sequence:

```python
carried = {}
for chunk in chunks:                               # each [t, B, ...]
    out, carried = net.apply({**params, **carried}, chunk, mutable=["state"])
```

A call without `mutable=["state"]` starts every neuron at rest. `SlidingPSN` streams the same way; `PSN` and `MaskedPSN` read every step of a fixed `T` and refuse to.

## Pure JAX cells

The layers build cells from `sparx.cells`, which need no Flax. A cell holds its constants as PyTree leaves and its choices as static fields; `sparx.run` scans one over time:

```python
from sparx.cells import ALIFCell, LIFCell, RecurrentCell

cell = LIFCell(decay=0.9, threshold=1.0, reset="subtract")
spikes, state = sparx.run(cell, currents)          # currents [T, ...]
more, state = sparx.run(cell, next_currents, state)  # continues where it stopped

lsnn = RecurrentCell(ALIFCell(decay=0.95, adapt_decay=0.995, beta=1.8), weight)  # weight [F, F]
```

Decays, thresholds and weights can be traced arrays, so they can be learned, swept with `vmap`, or sharded.

## Training with dew

`sparx.dew.SpikingClassifier` is a dew objective. It encodes a batch field into spikes, runs the network, and scores its outputs against the labels, so a spiking network trains under dew's `Trainer` with its checkpoints, EMA, evaluation and display. The tests run it on one CPU device; multi-device meshes have not been tried yet.

```python
import optax
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset

from sparx.datasets import shd
from sparx.dew import Events, RateBand, SpikingClassifier, accuracy

train, test = shd("train"), shd("test")           # {"spikes": [N, 100, 700], "label": [N]}
objective = SpikingClassifier(
    net, Field("spikes", (100, 700)), Events(),   # the records already hold spikes
    readout="max",                                # each class's peak membrane
    rates=RateBand(lower=0.01, upper=0.3),        # keep neurons in a firing band
)
trainer = Trainer(objective, optax.adamw(2e-3), key=0, checkpoints=Checkpoints("runs/shd"))
state = trainer.fit(Dataset.from_records(train, batch=64, validation=test),
                    steps=3000, eval_every=500, metrics=[accuracy])
```

Encoders for static data are `Direct(steps)`, `Rate(steps)` and `Latency(steps)`; uint8 fields are read as `x / 255`. The readout is `"mean"`, `"max"`, `"sum"` or `"per_step"`. The objective logs the batch accuracy and every spiking layer's firing rate (`rate/<layer>`), updates BatchNorm statistics, passes `train` and dropout keys to a model that takes them, and evaluates to dew's `TokenScores`, which `sparx.dew.accuracy` reads. [`examples/train_shd.py`](examples/train_shd.py) is the full script.

## Results

All runs below are the example scripts as committed, on a 4-core x86 CPU with JAX 0.11.2, float32, seed 0. They are short runs that show the library training real data end to end, not tuned results.

| Task | Command | Network | Test accuracy | Time |
| --- | --- | --- | --- | --- |
| MNIST, rate-coded, 8 steps | `python examples/train_mnist.py --epochs 2` | 784-512-512 LIF, LI readout, plain JAX loop | 97.46% after 2 epochs | 15 s per epoch |
| SHD, 100 steps of 14 ms | `python examples/train_shd.py --steps 3000` | 700-256 ALIF, LI readout (max), dew `Trainer` | 53.00% | 4 min 45 s |
| SHD | `python examples/train_shd.py --steps 3000 --recurrent --surrogate superspike` | 256 recurrent ALIF | 45.23%, still rising at the last evaluation | 8 min |

For scale, Cramer et al. (2020) report about 71% for recurrent and under 50% for feedforward LIF networks on SHD, and Hammouamri et al. (ICLR 2024) reach 95% with learned synaptic delays.

The recurrent SHD run needs the steep SuperSpike surrogate. With ATan, backpropagation through the recurrence exploded once training grew the recurrent matrix's spectral radius from 1 to 5: the gradient norm passed 1e8 within 300 steps and test accuracy fell below 15%. `FastSigmoid(100)` kept the gradient norm below 10. The `Recurrent` docstring records this.

## Performance

The design keeps the sequential part of a spiking network small: synapses run over all time steps at once, and only the elementwise neuron update is scanned. [docs/performance.md](docs/performance.md) has the measurements behind the defaults, on a 4-core CPU, including two faster-sounding paths that were measured slower and not shipped. No GPU or TPU numbers have been taken yet.

## Correctness

- Every cell is compared step by step with a float64 NumPy loop written from its docstring's equations (`tests/reference.py`), spikes exactly and membranes to float32 rounding.
- LIF (soft, hard and detached reset) and the three PSNs match SpikingJelly's own modules: spikes exactly, gradients within 1.2e-6 (`tools/make_reference_fixtures.py`, `tests/test_reference.py`).
- `latency`, `delta`, `per_step_cross_entropy` and `rate_mse` match snnTorch 1.0.0 (`tools/make_snntorch_fixtures.py`).
- Each surrogate's gradient and forward-mode tangent match its published formula, and its area matches its stated normalization.
- Invariants are tested directly: a run in chunks equals one run for every cell and layer, a call without the state collection starts at rest, `init` creates only parameters, bfloat16 inputs keep exact spikes over a float32 membrane.
- `SpikingClassifier` trains through dew's real `Trainer`, and its loss is checked against a manual computation.

`pytest -q` runs all of it on CPU in about 90 seconds.

## Installation

Sparx needs Python 3.12 or later. It has been tested with JAX 0.11.2, Flax 0.12.10 and optax 0.2.8 on CPU.

```bash
git clone https://github.com/AshishKumar4/sparx.git
cd sparx
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .                    # the library
uv pip install -e ".[dew,datasets]"    # plus dew's Trainer and the SHD loader
uv pip install -e ".[test]" && pytest -q
```

For a GPU or TPU, install the matching JAX build first (`jax[cuda12]` or `jax[tpu]`).

## Roadmap

- Accelerator measurements of the scan, the PSNs and synapse folding, then fused time-loop kernels (Pallas) where they pay.
- An associative-scan path for linear dynamics, if it wins on accelerators.
- Learnable synaptic delays, spiking self-attention, and more neuromorphic datasets (SSC, N-MNIST, DVS Gesture).
- Online learning rules (e-prop, OTTT) that train without backpropagation through time.

## License

MIT
