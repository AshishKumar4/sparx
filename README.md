# Sparx

Spiking neural networks in JAX and Flax.

Sparx builds spiking networks out of ordinary Flax linen layers and a small set of neuron layers that run over time. Neuron dynamics are pure JAX, gradients pass through spikes by surrogate derivatives, and networks train with optax, with your own loop or with [dew](https://github.com/AshishKumar4/dew)'s `Trainer`. Everything is a JAX PyTree, so `jit`, `grad`, `vmap`, forward-mode `jvp` and sharding work as they do for any Flax model.

Sparx also simulates circuits as neuroscience states them: neuron models in physical units (LIF, AdEx, Izhikevich, Hodgkin-Huxley), receptor kinetics and plasticity (`sparx.dynamics`), wired into populations and projections with delays (`sparx.graph`). These match NEST and Brian2, spike for spike where the models are deterministic and statistically where they are chaotic.

`import sparx` reaches every part: `sparx.nn`, `sparx.models`, `sparx.dynamics` and the other modules a network is built from load with it, and `sparx.graph`, `sparx.learn`, `sparx.objectives`, `sparx.metrics`, `sparx.tasks`, `sparx.optim`, `sparx.datasets`, `sparx.serve` and `sparx.nir` load the first time they are used.

APIs can change before 1.0.

## Contents

- [A first network](#a-first-network)
- [How a network runs over time](#how-a-network-runs-over-time)
- [Neurons](#neurons)
- [Surrogate gradients](#surrogate-gradients)
- [Encoding, losses and firing rates](#encoding-losses-and-firing-rates)
- [Streaming](#streaming)
- [Pure JAX models](#pure-jax-models)
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
spikes = sparx.encode.Rate(steps=8)(jax.random.key(1), images)  # [8, 32, 784]
params = net.init(jax.random.key(2), spikes)


def loss(params):
    logits = jnp.mean(net.apply(params, spikes), axis=0)  # mean membrane over time
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


grads = jax.grad(loss)(params)
```

`LIF` turns input currents into spikes, exactly 0 or 1, and `LI` is a leaky integrator whose membrane is the readout. The rest is Flax and optax. [`examples/train_mnist.py`](examples/train_mnist.py) trains a network like it, with a second hidden layer, to completion under dew's `Trainer` ([Training with dew](#training-with-dew)).

## How a network runs over time

Time is the leading axis of every array inside a network: `[T, B, ...]`. Flax's `Dense`, `Conv`, `BatchNorm` and pooling treat every leading axis as a batch axis, so a synaptic layer applies to all time steps in one call, as one large matrix product. Only the neurons' elementwise recurrence runs step by step, as a `jax.lax.scan` inside each neuron layer. A neuron layer cannot tell time from batch, so `LIF()(x)` on a `[B, F]` array runs `B` as time steps without an error; give it `[T, B, F]`. A convolutional network needs nothing extra:

```python
class ConvNet(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = False):     # [T, B, H, W, C]
        x = sparx.nn.LIF()(nn.BatchNorm(use_running_average=not train)(nn.Conv(32, (3, 3))(x)))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = x.reshape(*x.shape[:2], -1)             # keep [T, B], flatten the rest
        return sparx.nn.LI()(nn.Dense(10)(x))
```

Each neuron layer keeps its membrane in float32 whatever its input dtype, and returns spikes in the input's dtype, which holds 0 and 1 exactly even in bfloat16.

## Neurons

The LIF family (`LIF`, `IF`, `LI`, `Synaptic`, `ALIF`) shares one discrete-time convention with step `dt = 1` and per-step decay `exp(-1 / tau)`: `v[t] = decay * v[t-1] + x[t]`, a spike where `v[t] >= threshold`, then a reset. The input enters unscaled, as in snnTorch's `Leaky`. Every layer takes `dt`, the step in the unit of its time constants, so the default counts `tau` in steps; a step of `dt` decays by `exp(-dt / tau)`.

| Layer | Dynamics | Learnable |
| --- | --- | --- |
| `LIF(tau, threshold, reset, surrogate, detach_reset)` | leaky integrate-and-fire | `learn_tau=True`: a decay per feature |
| `IF(threshold, reset, ...)` | integrate-and-fire, no leak | |
| `LI(tau)` | leaky integrator, never fires, returns its membrane | `learn_tau` |
| `Synaptic(tau, tau_synapse, ...)` | current-based LIF: a decaying synaptic current charges the membrane | `learn_tau` (both) |
| `ALIF(tau, tau_adapt, beta, ...)` | adaptive threshold that rises by `beta` per spike (Bellec et al. 2020) | `learn_tau` (both) |
| `Izhikevich(a, b, c, d, dt)` | Izhikevich's two-variable neuron (2003) on input currents, `dt` in ms | |
| `Dynamics(model, dt=dt)` | any neuron model of `sparx.dynamics`, such as `AdEx`, on input currents | |
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

A physical model (`sparx.nn.Dynamics(AdEx(), dt=0.1)`) takes its surrogate as a field of the model, `AdEx(surrogate=...)`, and the surrogate reads `v - threshold` in mV, so its slope is per mV. Backpropagation also runs through the membrane equation, and AdEx's exponential upswing multiplies the gradient at every step a neuron spends near its peak. Behind a dense layer, over 2000 steps of 0.1 ms at 40 Hz, the gradient norm reaching that layer was 7.9e8 with the default `ATan()` and 0.42 with `FastSigmoid(100)`; a wider surrogate made it larger. Train physical models with a steep `FastSigmoid` (the `Dynamics` docstring has the measurements).

## Models

`sparx.models` builds architectures from these layers. `SEWResNet` is the spike-element-wise residual network of Fang et al. (2021), laid out as in SpikingJelly, with `sew_resnet18` and `sew_resnet34` presets, and `SpikingMLP` is a dense network for event data (stacked, optionally recurrent, with a leaky integrator readout). `SpikingMLP(delays=K)` delays its first synapse by up to `K` steps, and `delays=(24, 24, 24)` delays every synapse, the readout's too; with `extend=True`, `batch_norm=True`, `use_bias=False`, `weight_init="kaiming_uniform"` and `dropout_mask="sequence"` it is the network of Hammouamri et al.'s SNN-delays, checked against their code (`docs/fidelity.md`). Their `neuron` argument is the template every neuron of the network copies:

```python
net = sparx.models.sew_resnet18(10, width=32, stem="small",
                                neuron=sparx.nn.LIF(tau=2.0, detach_reset=True))
logits = net.apply(variables, frames, train=False)   # frames [T, B, 32, 32, 3] -> [T, B, 10]
```

[docs/design.md](docs/design.md) explains how neurons, models and objectives fit together, and what is planned.

## Encoding, losses and firing rates

`sparx.encode` turns a batch field `[B, ...]` into time-major input `[T, B, ...]`. An encoder is a registered frozen dataclass called as `encoder(key, x)`, where `key` is a JAX PRNG key such as `jax.random.key(0)`, never an int seed; the entry points called from outside JAX (dew's `Trainer`, `SpikingClassification.logits`) take an int seed, as dew's do. The encoders are `Rate(steps)` (Bernoulli spikes at the value's probability), `Latency(steps)` (one spike, earlier for larger values), `Direct(steps)` (the values as a constant input current, direct encoding), `Delta(threshold)` (spikes on changes of a signal over each record's time axis) and `Events()` (records that already hold spikes over time). The first four read uint8 fields as `x / 255`; `Events` passes spike counts unscaled.

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

## Pure JAX models

The layers build neuron models from `sparx.dynamics`, which run without Flax modules. Every model, from the dimensionless `LIFCell` to the physical `AdEx`, has `init_state(shape, dtype)` and `step(state, SynapticInput, dt) -> (state, Spikes)`, and `sparx.run` scans one over time. An array input is a jump of the membrane each step, the dimensionless family's input:

```python
from sparx.dynamics import ALIFCell, LICell, LIFCell, RecurrentCell, Serial, decay

cell = LIFCell(decay=decay(tau=10.0), threshold=1.0, reset="subtract")
spikes, state = sparx.run(cell, currents)          # currents [T, ...]; spikes.fired [T, ...]
more, state = sparx.run(cell, next_currents, state)  # continues where it stopped

synaptic = Serial(LICell(decay(5.0)), LIFCell(decay(10.0)))  # a synaptic current, then the membrane
lsnn = RecurrentCell(ALIFCell(decay=0.95, adapt_decay=0.995, beta=1.8), weight)  # weight [F, F]
```

A model stores its decay per unit of time and a step of `dt` (`sparx.run(..., dt=...)`, 1 by default) applies `decay ** dt`. Decays, thresholds and weights can be traced arrays, so they can be learned, swept with `vmap`, or sharded.

## Training with dew

`sparx.objectives.SpikingClassifierObjective` is a dew objective. It encodes a batch field into spikes, runs the network, and scores its outputs against the labels, so a spiking network trains under dew's `Trainer` with its checkpoints, EMA, evaluation and display. The model takes `train`, as dew's models do. The tests train it on one CPU device and on eight simulated CPU devices; no real multi-device mesh has been tried.

```python
import optax
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset

from sparx.datasets import evaluation_pass, shd
from sparx.encode import Events
from sparx.metrics import Accuracy
from sparx.models import SpikingMLP
from sparx.nn import ALIF
from sparx.objectives import RateBand, SpikingClassifierObjective

train, test = shd("train"), shd("test")           # {"spikes": [N, 100, 700], "label": [N]}
net = SpikingMLP(hidden=(256,), classes=20, neuron=ALIF(tau=5.0, tau_adapt=20.0, learn_tau=True))
objective = SpikingClassifierObjective(
    net, Field("spikes", (100, 700)), Events(),   # the records already hold spikes
    readout="max",                                # each class's peak membrane
    rates=RateBand(lower=0.01, upper=0.3),        # keep neurons in a firing band
)
trainer = Trainer(objective, optax.adamw(2e-3), key=0, checkpoints=Checkpoints("runs/shd"))
state = trainer.fit(Dataset.from_records(train, batch=64), steps=3000, eval_every=500,
                    metrics=[Accuracy()], validation={"test": evaluation_pass(test, 64)})
classifier = objective.pipeline(state)            # the trained classifier, as dew.pipeline loads it
```

The encoder is any of `sparx.encode`'s, the same objects a plain JAX loop calls. The readout is one of `sparx.losses.Readout`: `"mean"`, `"max"`, `"sum"`, `"softmax_sum"` (the softmax of every step summed over time, SNN-delays' loss) or `"per_step"`. `schedules` names model keyword arguments that follow one of dew's schedules over `schedule_steps`, such as `schedules={"sigma": Linear(peak=7.5, end=0.5)}` for a `DelayedDense`; `schedule_every` advances them once every that many steps, as a torch scheduler stepped once an epoch does, and `deployed={"sigma": 0}` evaluates with every delay rounded. `groups` gives parameters their own optimizers by path pattern, each a `sparx.optim.GroupAdam` with its own schedules, L2 weight decay and bounds, under `optax.multi_transform`; the trainer's optimizer updates the rest. `sparx.optim` holds these until dew's own schedules and parameter groups cover them:

```python
from dew.training.optim import Cosine
from sparx.optim import GroupAdam, OneCycle

groups = {"delays": GroupAdam(("*/delay",), Cosine(peak=0.1, warmup_steps=0), bounds=(0.0, 24.0)),
          "weights": GroupAdam(("*",), OneCycle(peak=5e-3, start=2e-4, end=2e-8), weight_decay=1e-5)}
```

`evaluation_pass` scores every record of a split, filling the last batch with copies that weigh nothing; a split passed as `Dataset.from_records(..., validation=test)` is scored in whole batches only, which leaves out the last partial one. SHD has no validation split, and `sparx.datasets.holdout(train, 0.1)` holds out part of the training set to select on. The objective logs the batch accuracy and every spiking layer's firing rate (`rate/<layer>`), updates BatchNorm statistics, passes dropout keys, and evaluates to dew's `TokenScores`, which `sparx.metrics.Accuracy` reads. `Accuracy` is registered in dew's metrics table as `spike_accuracy`, and it works with dew's `Best` to keep the checkpoint of best validation accuracy. [`examples/train_shd.py`](examples/train_shd.py) is the full script; `--recipe snn-delays` runs Hammouamri et al.'s SHD recipe, and its docstring lists what still differs from their code. `sparx.objectives.EPropObjective` trains a recurrent layer with e-prop's gradients under the same trainer ([`examples/train_shd_eprop.py`](examples/train_shd_eprop.py)). Every example takes `--smoke`, which trains a small network for a few steps on synthetic data and downloads nothing.

Sparx is a dew plugin. Its models, neurons, surrogates, encoders, objective and datasets are registered in dew's registry, so a run's `run.json` records a spiking model the way it records a transformer, and `dew.pipeline(run_dir)` loads a trained classifier back in a fresh process as a `SpikingClassification`. A model of your own trains without registering, but reloads only once its class carries `@dew.registry.models("name")`, as `SpikingMLP` does:

```python
import dew

classifier = dew.pipeline("runs/shd")
predictions = classifier(test_spikes)              # [B]
```

[`recipes/snn/train.py`](recipes/snn/train.py) is a dew recipe: every setting is a typed flag, the dataset and the encoder are subcommands over their registries (`data:shd`, `encoder:rate`), the model's settings and the schedules are `{"name": ..., "fields": {...}}` records of registered members, and `--model.dtype` reaches the synapses. `--smoke` runs it for a few seconds on synthetic recordings:

```bash
python recipes/snn/train.py data:shd --data.channels 140 --trainer.batch-size 64 --trainer.steps 3000 \
    --trainer.checkpoint-dir runs --trainer.name shd \
    --model.config '{"hidden": [128], "classes": 20, "delays": 15, "neuron": {"name": "alif", "fields": {"tau": 5.0}}}' \
    --schedules '{"sigma": {"name": "linear", "fields": {"peak": 7.5, "end": 0.5}}}'
JAX_PLATFORMS=cpu python recipes/snn/train.py --smoke --trainer.checkpoint-dir /tmp/snn-smoke
```

Training on several devices is dew's: `Trainer(..., mesh=MeshSpec(fsdp=2))` places the run, and a test checks that eight simulated CPU devices train the same parameters as one, within 1.8e-7.

## Simulating circuits

`sparx.dynamics` holds neuron, synapse and plasticity models in ms, mV, pA, nS and pF; `sparx.graph` wires them into a `Network`, a Flax module whose variables hold the connectome, trainable weights and the simulation state. This is Vogels and Abbott's network with conductance-based synapses, one of the benchmarks simulators are compared on:

```python
import jax
from sparx.dynamics import LIF, Exponential, Receptor
from sparx.graph import (FixedProbability, Network, Population, PopulationRate, Projection, SpikeRaster,
                         StateMonitor, simulate)
from sparx.spiketrains import cv_isi, rates_hz

neuron = LIF(tau_m=20.0, c_m=200.0, e_l=-60.0, v_th=-50.0, v_reset=-60.0, t_ref=5.0)
receptors = {"ampa": Receptor(Exponential(5.0), "conductance"),  # LIF reverses ampa at 0 mV
             "gaba_a": Receptor(Exponential(10.0), "conductance")}  # and gaba_a at -80 mV
initial = {"v": lambda rng, n: rng.uniform(-60.0, -50.0, n),  # random voltages (mV)
           "ampa": lambda rng, n: rng.normal(40.0, 15.0, n),  # and conductances (nS)
           "gaba_a": lambda rng, n: rng.normal(200.0, 120.0, n)}

network = Network(
    populations=(Population("e", 3200, neuron, receptors, initial=initial),
                 Population("i", 800, neuron, receptors, initial=initial)),
    projections=tuple(Projection(pre, post, FixedProbability(0.02), weight=6.0 if pre == "e" else 67.0,
                                 delay=0.0, receptor="ampa" if pre == "e" else "gaba_a")
                      for pre in ("e", "i") for post in ("e", "i")),
    dt=0.1,
)
monitors = {"spikes": SpikeRaster("e"), "rate": PopulationRate("e"),
            "v": StateMonitor("e", neurons=(0, 1, 2))}  # three voltage traces
result = simulate(network, network.init(jax.random.key(0)), duration=300.0, monitors=monitors)
spikes = result.records["spikes"][1000:]  # [steps, 3200] after the first 100 ms
print(rates_hz(spikes, 0.1).mean(), cv_isi(spikes).mean())  # about 17 Hz, CV about 0.8
print(result.records["rate"][1000:].mean(), result.records["v"].shape)  # the same rate in Hz; (3000, 3)
```

The network is checked when it is built: a projection or input onto a receptor its population lacks, a conductance receptor without a reversal potential in the neuron model, or a kinetic synapse onto a dimensionless model (`ALIFCell`, which takes voltage jumps through `Delta` receptors) raises a `ValueError` that names the population and receptor. Every projection and input names its receptor, which sets the weight's unit. Delays, `duration` and `chunk` must be whole numbers of steps, and a time between steps raises a `ValueError` too.

Records come back under the names the monitors were given, and every rate is in Hz, as `PoissonInput(rate=...)` and `rates_hz` are. After a run, `network.connections(result.variables)` reads each projection's synapses as NumPy arrays, the way NEST's `GetConnections` does: `pre, post, weight, delay = network.connections(result.variables)["e->e:ampa"]`, with STDP's weights as learned.

A step runs in NEST's order: synapses deliver what is due, membranes integrate (exactly where the equations are linear) and spike, spikes enter per-population ring buffers, kinetic synapses receive what arrives at the end of the step, plasticity updates, monitors record. `simulate` compiles one chunk of steps and carries the state between chunks, so a long run needs memory for one chunk of records, and a run continued from `result.variables` is the run it would have been unbroken. With `checkpoints=dew.Checkpoints(directory)` it writes the state after every chunk, and a run started again on the same directory continues from the last one. `sparx.graph` builds Brunel's (2000) network and the CUBA and COBA benchmarks (`brunel`, `cuba`, `coba`); `Projection`s take per-edge weights and delays, pair and triplet STDP (any `sparx.dynamics.Plasticity` rule), and short-term plasticity.

A population holds any neuron model of `sparx.dynamics`, and its synapses reach the model through the neuron protocol: each model says where it is refractory (`is_refractory`) and how a voltage jump that lands after the threshold test changes it (`after_threshold`), and each synapse model says where its arrivals land (`lands`: into its own state, or as a jump before or after the threshold test).

The builders are registered in `sparx.registry.networks`, so a network is a record that rebuilds in another process: `sparx.graph.from_record({"name": "brunel", "fields": {"order": 2500, "g": 5.0}})`, or for a model on a connectome, `{"name": "shiu2024", "fields": {"connectome": {"name": "flywire", "fields": {"completeness": ..., "connectivity": ...}}, "stimuli": ...}}`, which names the reader of its tables (`sparx.registry.connectomes`).

## Learning beyond backpropagation through time

`sparx.learn` holds the rules design.md section 7 names, each checked against what defines it (`tests/test_learn.py`):

- `eprop`: e-prop (Bellec et al. 2020) for a recurrent layer of any elementwise sparx model and a leaky readout, `eprop_forward`'s network (a `RecurrentCell` and an `LICell`), computed online in memory independent of the sequence length. It equals backpropagation with the recurrent spikes' gradient cut, and its eligibility traces with the true learning signal equal backpropagation, the two identities their own code verifies. Time constants are `tau` in the unit of `dt`.
- `ottt`: online training through time (Xiao et al. 2022), matching their PyTorch modules' gradients to 1e-10.
- `spike_times` with `EventLIF`: exact spike times of LIF networks with current synapses in continuous time (ms), differentiable: the exact gradient EventProp (Wunderlich and Pehle 2021) computes, checked against finite differences.
- `convert`: a ReLU network written as a flax `nn.Sequential` (dense and convolutional layers, batch norm, average and max pooling, `sparx.nn.Flatten`) to a `sparx.nn` stack of the same layers with `IF` neurons and a gated `SpikingMaxPool`, balancing thresholds at a percentile of the activations (Rueckauer et al. 2017); against their toolbox (snntoolbox), the same weights, the same first-layer spikes and the same predictions. The result runs, records rates and trains like any `sparx.nn` stack, and with `reset="zero"` a dense one exports to NIR.

```python
from functools import partial

import flax.linen as nn
import jax

from sparx.learn import convert, fold_batch_norm, normalize, run_converted
from sparx.nn import Flatten

# A ReLU network as a flax nn.Sequential; in practice, a trained one.
ann = nn.Sequential([nn.Conv(8, (3, 3)), nn.BatchNorm(use_running_average=True), nn.relu,
                     partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2)), Flatten(), nn.Dense(10)])
images = jax.random.uniform(jax.random.key(0), (16, 16, 16, 1))  # [B, H, W, C], values in [0, 1]
variables = ann.init(jax.random.key(1), images)
ann, variables = fold_batch_norm(ann, variables)
variables = normalize(ann, variables, images)  # threshold balancing on calibration images
snn, snn_variables = convert(ann, variables)   # Conv, IF, SpikingMaxPool, Flatten, Dense, IF
rates = run_converted(snn, snn_variables, images, steps=100)  # output firing rates, [16, 10]
```
- `sparx.objectives.ActivityFitObjective` fits a network's spikes to recorded ones by van Rossum distance (`sparx.losses.van_rossum`, exact on the grid) or smoothed rates.

## Connectomes, serving and exchange

- `sparx.graph.connectome` reads FlyWire (Shiu et al.'s tables) and the male CNS release into a `Connectome` and builds Shiu et al.'s (2024) whole-brain model; on FlyWire v630 it reproduces their published runs (rate correlation 0.999, MN9 at 67.1 Hz against their 67.0 +- 6.6) at about 30 s per simulated second on 4 CPU cores. On the male CNS, whose neurons receive about 1.7 times FlyWire's synapses, `matched_w_syn` rescales their weight (0.275 to 0.163 mV): sugar neurons then recruit about 670 neurons and drive MN9 at 81 Hz, against FlyWire's 400 and 67 Hz.
- `simulate(trials=..., mesh=dew.MeshSpec(...))` spreads trials, or one network's neurons, over devices, with one device's results. The mesh is built as dew's `Trainer` builds it, and `sparx.graph.RULES` places the logical axes `trials` (on the data axis) and `neurons` (on fsdp, or on data when there is no trial axis): `MeshSpec()` partitions one network's neurons over every device, and `MeshSpec(fsdp=2)` runs trials over the data axis with each trial's neurons split in two.
- `sparx.serve.StreamServer` serves streaming models to many sessions at once, each with its own neuron state in a slot of one batch; a session's outputs equal a direct call over its stream. A reloaded run serves as `StreamServer(classifier.model, classifier.variables, slots=8, frame=10, sample_shape=(700,))`, fed frames of its encoder's output; any model whose outputs are time-major can be served, and one that cannot stream (a `PSN`, or a readout averaged over time) is refused when the server is built.
- `sparx.nir` exchanges networks through NIR: dense and 2-d convolutional layers, `Flatten`, hard-reset `LIF` and `IF`, and `Recurrent(LIF)`. Dense, convolutional and recurrent networks exported by snnTorch run in sparx spike for spike and export back with the same parameters.

## Results

All runs below are the example scripts on a 4-core x86 CPU with JAX 0.11.2, float32, seed 0. They are short runs that show the library training real data end to end, not tuned results. They were measured at commit 6f5ed31, before the MNIST and e-prop examples moved from their own loops to dew's `Trainer`; the networks, losses and optimizers did not change, and the runs have not been repeated since.

| Task | Command | Network | Test accuracy | Time |
| --- | --- | --- | --- | --- |
| MNIST, rate-coded, 8 steps | `python examples/train_mnist.py --epochs 2` | 784-512-512 LIF, LI readout, measured with a plain JAX loop | 97.46% after 2 epochs | 15 s per epoch |
| SHD, 100 steps of 14 ms | `python examples/train_shd.py --steps 3000` | 700-256 ALIF, LI readout (max), dew `Trainer` | 53.00% | 4 min 45 s |
| SHD | `python examples/train_shd.py --steps 3000 --recurrent --surrogate superspike` | 256 recurrent ALIF | 45.23%, still rising at the last evaluation | 8 min |
| SHD, channels pooled to 140 | `python examples/train_shd.py --steps 3000 --channels 140 --hidden 128` | 140-128 ALIF | 64.53% | 1 min 42 s |
| SHD, channels pooled to 140 | `... --channels 140 --hidden 128 --delays 15` | the same, with a learned delay of 0 to 15 steps per input synapse | 74.56%, with every delay rounded to a whole step | 4 min 19 s |
| SHD, Hammouamri et al.'s recipe, 20 of their 150 epochs | `python examples/train_shd.py --recipe snn-delays --epochs 20 --validation 0` | 140-256-256 LIF, a learned delay on every synapse (readout included), batch norm, summed-softmax readout | 91.87% at the last epoch, also the best (their code on the same machine and schedule: 93.59% last, 94.03% best) | 48 min |
| SHD, channels pooled to 140 | `python examples/train_shd_eprop.py --rule eprop --epochs 5` | 140-128 recurrent ALIF (refractory 2 steps), leaky readout, trained online by e-prop | 53.36% (56.93% at epoch 3) | 3 min 30 s |
| SHD, channels pooled to 140 | `... --rule bptt --epochs 5` | the same network by BPTT | 46.38% (53.80% at epoch 3) | 30 s |

The last two rows of the first five differ only in the delays, which add 10 points. The e-prop rows train the same network with the same optimizer: after five epochs e-prop scores 53% and BPTT 46%, and both move by several points from one epoch to the next (neither is tuned). e-prop's memory does not grow with the recording, but on a CPU it is about 7 times slower here: it advances an eligibility trace for every synapse every step, `B x N x (in + N)` numbers for the readout's filter and as many for the adaptive threshold, where BPTT does one backward pass ([performance](docs/performance.md#e-prop)). The SNN-delays row is a matched comparison: their official code (with the 2023 SpikingJelly it was written for) and sparx ran the same recipe and 20-epoch schedule on this machine, one seed each, every training recording used, test accuracy after each epoch. Sparx ends 1.7 points below their last epoch and 2.2 below their best; the gradients of one training step agree with theirs to 6e-7 (`docs/fidelity.md`), so the gap lies in what the step sees, the remaining differences the example's docstring lists (every recording padded to 124 steps where they pad to each batch's longest, 31 steps an epoch where they take 32, and different random streams), not in the arithmetic. One seed of each does not separate those from run-to-run spread. Sparx's epochs took 146 s against about 210 s for theirs on the same CPU. Their reported 95% is the best of 150 epochs, selected on the test set. For scale, Cramer et al. (2020) report about 71% for recurrent and under 50% for feedforward LIF networks on SHD, and Hammouamri et al. (ICLR 2024) reach 95% with learned synaptic delays.

The recurrent SHD run needs the steep SuperSpike surrogate. With ATan, backpropagation through the recurrence exploded once training grew the recurrent matrix's spectral radius from 1 to 5: the gradient norm passed 1e8 within 300 steps and test accuracy fell below 15%. `FastSigmoid(100)` kept the gradient norm below 10. The `Recurrent` docstring records this.

## Performance

The design keeps the sequential part of a spiking network small: synapses run over all time steps at once, and only the elementwise neuron update is scanned. [docs/performance.md](docs/performance.md) has the measurements behind the defaults, on a 4-core CPU, including two faster-sounding paths that were measured slower and not shipped. No GPU or TPU numbers have been taken yet.

## Correctness

- Every cell is compared step by step with a float64 NumPy loop written from its docstring's equations (`tests/reference.py`), spikes exactly and membranes to float32 rounding.
- LIF (soft, hard and detached reset) and the three PSNs match SpikingJelly's own modules: spikes exactly, gradients within 1.2e-6 (`tools/make_reference_fixtures.py`, `tests/test_reference.py`).
- `latency`, `delta`, `per_step_cross_entropy` and `rate_mse` match snnTorch 1.0.0 (`tools/make_snntorch_fixtures.py`).
- Each surrogate's gradient and forward-mode tangent match its published formula, and its area matches its stated normalization.
- Invariants are tested directly: a run in chunks equals one run for every cell and layer, a call without the state collection starts at rest, `init` creates only parameters, bfloat16 inputs keep exact spikes over a float32 membrane.
- `SpikingClassifierObjective` trains through dew's real `Trainer`, and its loss is checked against a manual computation. `EPropObjective`'s gradient through the trainer equals `sparx.learn.eprop`'s.
- A `SpikingMLP` with every synapse delayed matches Hammouamri et al.'s SNN-delays network run from their code on DCLS: outputs and loss within 1.2e-7 and every gradient within 5.7e-7 in training, outputs within 2.4e-7 in evaluation. Their learning-rate, momentum and width schedules match over all 150 epochs, and `shd(binning="events")` reproduces their binned SHD exactly (`tools/make_snn_delays_fixtures.py`).
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
