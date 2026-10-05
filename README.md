# Sparx

Spiking neural networks in JAX and Flax.

Sparx builds spiking networks out of ordinary Flax linen layers and a small set of neuron layers that run over time. Neuron dynamics are pure JAX, gradients pass through spikes by surrogate derivatives, and networks train with optax, with your own loop or with [dew](https://github.com/AshishKumar4/dew)'s `Trainer`. Everything is a JAX PyTree, so `jit`, `grad`, `vmap`, forward-mode `jvp` and sharding work as they do for any Flax model.

Sparx also simulates circuits as neuroscience states them: neuron models in physical units (LIF, AdEx, Izhikevich, Hodgkin-Huxley), receptor kinetics and plasticity (`sparx.dynamics`), wired into populations and projections with delays (`sparx.graph`). These match NEST and Brian2, spike for spike where the models are deterministic and statistically where they are chaotic.

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
- [Simulating circuits](#simulating-circuits)
- [Learning beyond backpropagation through time](#learning-beyond-backpropagation-through-time)
- [Connectomes, serving and exchange](#connectomes-serving-and-exchange)
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
| `DelayedDense(features, max_delay)` | a dense synapse where every connection has its own delay of 0 to `max_delay` steps (Hammouamri et al. 2024) | weights, delays |

`reset` is `"subtract"` (soft reset, the default), `"zero"` (hard reset) or `"none"`. `detach_reset=True` stops the gradient through the reset, as SpyTorch's tutorials and SpikingJelly's `detach_reset` do. Learned decays are the sigmoid of a parameter, so training cannot push them outside (0, 1).

`DelayedDense` learns each delay by spreading the synapse over a Gaussian centered at it; the width is a call argument that a schedule shrinks during training, and `sigma=0` reads exactly the rounded delay, the network to deploy.

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

## Models

`sparx.models` builds architectures from these layers. `SEWResNet` is the spike-element-wise residual network of Fang et al. (2021), laid out as in SpikingJelly, with `sew_resnet18` and `sew_resnet34` presets, and `SpikingMLP` is a dense network for event data (stacked, optionally recurrent or delayed, with a leaky integrator readout). Their `neuron` argument is the template every neuron of the network copies:

```python
net = sparx.models.sew_resnet18(10, width=32, stem="small",
                                neuron=sparx.nn.LIF(tau=2.0, detach_reset=True))
logits = net.apply(variables, frames, train=False)   # frames [T, B, 32, 32, 3] -> [T, B, 10]
```

[docs/design.md](docs/design.md) explains how neurons, models and objectives fit together, and what is planned.

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

Encoders for static data are `Direct(steps)`, `Rate(steps)` and `Latency(steps)`; uint8 fields are read as `x / 255`. The readout is `"mean"`, `"max"`, `"sum"` or `"per_step"`. `schedules` names model keyword arguments that follow one of dew's schedules over `schedule_steps`, such as `schedules={"sigma": Linear(peak=7.5, end=0.5)}` for a `DelayedDense`. The objective logs the batch accuracy and every spiking layer's firing rate (`rate/<layer>`), updates BatchNorm statistics, passes `train` and dropout keys to a model that takes them, and evaluates to dew's `TokenScores`, which `sparx.dew.accuracy` reads. [`examples/train_shd.py`](examples/train_shd.py) is the full script.

Sparx is a dew plugin. Its models, neurons, surrogates, encoders, objective and datasets are registered in dew's registry, so a run's `run.json` records a spiking model the way it records a transformer, and `dew.pipeline(run_dir)` loads a trained classifier back in a fresh process as a `SpikingClassification`:

```python
import dew

classifier = dew.pipeline("runs/shd")
predictions = classifier(test_spikes)              # [B]
```

[`recipes/snn/train.py`](recipes/snn/train.py) is a dew recipe: every setting is a typed flag, and the model, encoder and schedules are `{"name": ..., "fields": {...}}` records of registered members:

```bash
python recipes/snn/train.py data:shd --data.channels 140 --trainer.batch-size 64 --trainer.steps 3000 \
    --trainer.checkpoint-dir runs --trainer.name shd \
    --model.config '{"hidden": [128], "classes": 20, "delays": 15, "neuron": {"name": "alif", "fields": {"tau": 5.0}}}' \
    --schedules '{"sigma": {"name": "linear", "fields": {"peak": 7.5, "end": 0.5}}}'
```

Training on several devices is dew's: `Trainer(..., mesh=MeshSpec(fsdp=2))` places the run, and a test checks that eight simulated CPU devices train the same parameters as one, within 1.8e-7.

## Simulating circuits

`sparx.dynamics` holds neuron, synapse and plasticity models in ms, mV, pA, nS and pF; `sparx.graph` wires them into a `Network`, a Flax module whose variables hold the connectome, trainable weights and the simulation state. This is Vogels and Abbott's network with conductance-based synapses, one of the benchmarks simulators are compared on:

```python
import jax
from sparx.dynamics import LIF, Exponential, Receptor
from sparx.graph import FixedProbability, Network, Population, PopulationRate, Projection, Spikes, simulate
from sparx.graph.analysis import cv_isi, firing_rates

neuron = LIF(tau_m=20.0, c_m=200.0, e_l=-60.0, v_th=-50.0, v_reset=-60.0, t_ref=5.0,
             reversal={"ex": 0.0, "in": -80.0})
receptors = {"ex": Receptor(Exponential(5.0), "conductance"), "in": Receptor(Exponential(10.0), "conductance")}

def kick(rng, state):  # start from random voltages and conductances
    v = rng.uniform(-60.0, -50.0, state.neuron.v.shape).astype("float32")
    g = {"ex": rng.normal(40.0, 15.0, v.shape), "in": rng.normal(200.0, 120.0, v.shape)}
    return state._replace(neuron=state.neuron._replace(v=v),
                          synapses={k: x.astype("float32") for k, x in g.items()})

network = Network(
    populations=(Population("e", 3200, neuron, receptors, initial=kick),
                 Population("i", 800, neuron, receptors, initial=kick)),
    projections=tuple(Projection(pre, post, FixedProbability(0.02), weight=6.0 if pre == "e" else 67.0,
                                 delay=0.0, receptor="ex" if pre == "e" else "in")
                      for pre in ("e", "i") for post in ("e", "i")),
    dt=0.1,
)
result = simulate(network, network.init(jax.random.key(0)), duration=300.0,
                  monitors=(Spikes("e"), PopulationRate("e")))
spikes = result.records[0][1000:]  # after the first 100 ms
print(firing_rates(spikes, 0.1).mean(), cv_isi(spikes).mean())  # about 17 Hz, CV about 0.8
```

A step runs in NEST's order: synapses deliver what is due, membranes integrate (exactly where the equations are linear) and spike, spikes enter per-population ring buffers, kinetic synapses receive what arrives at the end of the step, plasticity updates, monitors record. `simulate` compiles one chunk of steps and carries the state between chunks, so a long run needs memory for one chunk of records, and a run continued from `result.variables` is the run it would have been unbroken. `sparx.graph.models` builds Brunel's (2000) network and the CUBA and COBA benchmarks; `Projection`s take per-edge weights and delays, pair and triplet STDP, and short-term plasticity.

## Learning beyond backpropagation through time

`sparx.learn` holds the rules design.md section 7 names, each checked against what defines it (`tests/test_learn.py`):

- `eprop`: e-prop (Bellec et al. 2020) for a recurrent layer of any sparx cell and a leaky readout, computed online in memory independent of the sequence length. It equals backpropagation with the recurrent spikes' gradient cut, and its eligibility traces with the true learning signal equal backpropagation, the two identities their own code verifies.
- `ottt`: online training through time (Xiao et al. 2022), matching their PyTorch modules' gradients to 1e-10.
- `events.spike_times`: exact spike times of LIF networks with current synapses in continuous time, differentiable: the exact gradient EventProp (Wunderlich and Pehle 2021) computes, checked against finite differences.
- `convert`: ReLU networks, CNNs with batch norm, average and max pooling included, to integrate-and-fire networks by robust threshold balancing (Rueckauer et al. 2017); against their toolbox (snntoolbox), the same weights, the same first-layer spikes and the same predictions.
- `sparx.dew.ActivityFit` fits a network's spikes to recorded ones by van Rossum distance (`sparx.losses.van_rossum`, exact on the grid) or smoothed rates.

## Connectomes, serving and exchange

- `sparx.graph.connectome` reads FlyWire (Shiu et al.'s tables) and the male CNS release into a `Connectome` and builds Shiu et al.'s (2024) whole-brain model; on FlyWire v630 it reproduces their published runs (rate correlation 0.999, MN9 at 67.1 Hz against their 67.0 +- 6.6) at about 30 s per simulated second on 4 CPU cores. On the male CNS, whose neurons receive about 1.7 times FlyWire's synapses, `matched_w_syn` rescales their weight (0.275 to 0.163 mV): sugar neurons then recruit about 670 neurons and drive MN9 at 81 Hz, against FlyWire's 400 and 67 Hz.
- `simulate(trials=..., mesh=...)` spreads trials, or one network's neurons, over devices, with one device's results.
- `sparx.serve.StreamServer` serves streaming models to many sessions at once, each with its own neuron state in a slot of one batch; a session's outputs equal a direct call over its stream.
- `sparx.nir` exchanges networks through NIR: dense and 2-d convolutional layers, `Flatten`, hard-reset `LIF` and `Recurrent(LIF)`. Dense, convolutional and recurrent networks exported by snnTorch run in sparx spike for spike and export back with the same parameters.

## Results

All runs below are the example scripts as committed, on a 4-core x86 CPU with JAX 0.11.2, float32, seed 0. They are short runs that show the library training real data end to end, not tuned results.

| Task | Command | Network | Test accuracy | Time |
| --- | --- | --- | --- | --- |
| MNIST, rate-coded, 8 steps | `python examples/train_mnist.py --epochs 2` | 784-512-512 LIF, LI readout, plain JAX loop | 97.46% after 2 epochs | 15 s per epoch |
| SHD, 100 steps of 14 ms | `python examples/train_shd.py --steps 3000` | 700-256 ALIF, LI readout (max), dew `Trainer` | 53.00% | 4 min 45 s |
| SHD | `python examples/train_shd.py --steps 3000 --recurrent --surrogate superspike` | 256 recurrent ALIF | 45.23%, still rising at the last evaluation | 8 min |
| SHD, channels pooled to 140 | `python examples/train_shd.py --steps 3000 --channels 140 --hidden 128` | 140-128 ALIF | 64.53% | 1 min 42 s |
| SHD, channels pooled to 140 | `... --channels 140 --hidden 128 --delays 15` | the same, with a learned delay of 0 to 15 steps per input synapse | 74.56%, with every delay rounded to a whole step | 4 min 19 s |

The last two rows differ only in the delays, which add 10 points. For scale, Cramer et al. (2020) report about 71% for recurrent and under 50% for feedforward LIF networks on SHD, and Hammouamri et al. (ICLR 2024) reach 95% with learned synaptic delays.

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
- The physical models match NEST 3.10 and Brian2 2.10 (`tools/make_nest_fixtures.py`, `tools/make_brian2_fixtures.py`, `tests/test_simulators.py`): current-based LIF with exponential, alpha and delta synapses to 1e-11 mV and spike for spike; conductance-based LIF, AdEx, Izhikevich (bit for bit, op by op) and Hodgkin-Huxley spike for spike or within a stated step; Izhikevich's (2004) twenty firing patterns (`izhikevich_2004`) spike for spike against his own `figure1.m` run in Octave; STDP, triplet STDP and Tsodyks-Markram synapses to every transmitted weight.
- Recurrent networks with per-edge delays fire with NEST spike for spike; Brunel's four regimes and the CUBA and COBA benchmarks match NEST's and Brian2's rates, irregularity and synchrony within their spread over seeds (`tests/test_graph.py`).

[docs/fidelity.md](docs/fidelity.md) lists, for every model, its references, what was checked and each difference found between them. `pytest -q` runs all of it on CPU in about eight minutes; the whole-brain comparison runs when Shiu et al.'s repository is next to sparx (`SPARX_SHIU_REPO`).

## Installation

Sparx needs Python 3.12 or later and installs dew, which it trains, distributes, checkpoints and serves through; until dew's plugin registry reaches its main branch, the dependency pins the integration commit that carries it. It has been tested with JAX 0.11.2, Flax 0.12.10 and optax 0.2.8 on CPU.

```bash
git clone https://github.com/AshishKumar4/sparx.git
cd sparx
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .                    # the library, with dew
uv pip install -e ".[datasets]"        # plus the SHD reader
uv pip install -e ".[test]" && pytest -q
```

For a GPU or TPU, install the matching JAX build first (`jax[cuda12]` or `jax[tpu]`).

## Roadmap

- Accelerator measurements of the scan, the PSNs and synapse folding, then fused time-loop kernels (Pallas) where they pay.
- An associative-scan path for linear dynamics, if it wins on accelerators.
- Spiking self-attention and spiking sequence models, and more neuromorphic datasets (SSC, N-MNIST, DVS Gesture).
- GPU and TPU measurements of networks and connectomes, and fused kernels where they pay (design.md phase 7).
- Validating whole-brain models on the male CNS beyond one behaviour (its weight is calibrated to FlyWire's synapse counts, `matched_w_syn`).
- Stateful serving in dew itself (AshishKumar4/dew#30), with `sparx.serve` as its first user.

## License

MIT
