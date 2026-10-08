# sparx guide

The README shows what sparx does. This guide covers how to use each part. [units.md](units.md) lists the units of time, rates and every physical quantity, [fidelity.md](fidelity.md) what every model is checked against, [design.md](design.md) the architecture, and [performance.md](performance.md) the measurements behind the defaults.

## Contents

- [Time](#time)
- [Neuron layers](#neuron-layers)
- [Recurrent layers and fast weights](#recurrent-layers-and-fast-weights)
- [Surrogate gradients](#surrogate-gradients)
- [Models](#models)
- [Encoders, losses and firing rates](#encoders-losses-and-firing-rates)
- [Streaming](#streaming)
- [Models without Flax](#models-without-flax)
- [Training on dew](#training-on-dew)
- [Simulating circuits](#simulating-circuits)
- [Graded signalling and neuromodulation](#graded-signalling-and-neuromodulation)
- [Learning rules](#learning-rules)
- [Connectomes, serving and exchange](#connectomes-serving-and-exchange)
- [Results in detail](#results-in-detail)

## Time

Every array inside a network is time-major, `[T, B, ...]`. Flax's `Dense`, `Conv`, `BatchNorm` and pooling treat leading axes as batch axes, so a synaptic layer runs over all time steps in one call, as one large matrix product. Only the neurons step through time, each layer with a `jax.lax.scan`.

A neuron layer can't tell time from batch. `LIF()(x)` on a `[B, F]` array treats `B` as time steps and raises nothing, so give it `[T, B, F]`. A convolutional network needs nothing extra:

```python
class ConvNet(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = False):     # [T, B, H, W, C]
        x = sparx.nn.LIF()(nn.BatchNorm(use_running_average=not train)(nn.Conv(32, (3, 3))(x)))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = x.reshape(*x.shape[:2], -1)             # keep [T, B], flatten the rest
        return sparx.nn.LI()(nn.Dense(10)(x))
```

Membranes stay in float32 whatever the input dtype. Spikes come back in the input's dtype, which holds 0 and 1 exactly, bfloat16 included. Time constants are in the unit of a layer's `dt`, 1 step by default; [units.md](units.md) covers steps of real time and the physical units.

## Neuron layers

The LIF family (`LIF`, `IF`, `LI`, `Synaptic`, `ALIF`) shares one discrete-time convention. With step `dt = 1` and decay `exp(-1 / tau)` per step, `v[t] = decay * v[t-1] + x[t]`, a spike where `v[t] >= threshold`, then a reset. The input enters unscaled, as in snnTorch's `Leaky`. Each layer takes `dt` in the unit of its time constants, and a step of `dt` decays by `exp(-dt / tau)`.

| Layer | Dynamics | Learnable |
| --- | --- | --- |
| `LIF(tau, threshold, reset, surrogate, detach_reset)` | leaky integrate-and-fire | `learn_tau=True`, a decay per feature |
| `IF(threshold, reset, ...)` | integrate-and-fire without leak | |
| `LI(tau)` | leaky integrator; it never fires and returns its membrane | `learn_tau` |
| `Rate(tau, activation)` | FLYNN's leaky rate unit, `h <- alpha h + (1 - alpha) f(x + b)`; `tau=0` keeps no memory | `b`, `learn_tau` |
| `Synaptic(tau, tau_synapse, ...)` | current-based LIF, a decaying synaptic current charging the membrane | `learn_tau` for both |
| `ALIF(tau, tau_adapt, beta, ...)` | a threshold that rises by `beta` per spike (Bellec et al. 2020) | `learn_tau` for both |
| `Dynamics(model, dt=dt)` | any model of `sparx.dynamics`, such as `AdEx` or `Izhikevich`, on input currents | |
| `Recurrent(neuron, rule=None)` | any neuron fed back through a learned `[F, F]` matrix, optionally with fast weights | the matrix, `alpha`, the rule's parameters |
| `PSN()` | parallel spiking neuron, `H = W X + b` over all `T x T` step pairs (Fang et al. 2023) | `W`, `b` |
| `MaskedPSN(k)` | the PSN restricted to the `k` most recent steps | `W`, `b` |
| `SlidingPSN(k)` | `k` weights slid over time, causal, any `T` | weights, `b` |
| `DelayedDense(features, max_delay)` | a dense synapse whose every connection has its own delay of 0 to `max_delay` steps (Hammouamri et al. 2024) | weights, delays |

`reset` is `"subtract"` (the default soft reset), `"zero"` or `"none"`. `detach_reset=True` stops the gradient through the reset, as SpikingJelly's option of that name does. A learned decay is the sigmoid of a parameter, so it stays inside (0, 1).

`DelayedDense` spreads each synapse over a Gaussian centered at its delay, so the delay gets a gradient. A schedule shrinks the Gaussian's width during training, and at `sigma=0` each synapse reads exactly its rounded delay, which is the network to deploy.

The PSNs have no loop over time. Each is one `[T, T] x [T, N]` product and a threshold, so no step waits for the one before it, and Fang et al. report that they learn longer dependencies than LIF.

## Recurrent layers and fast weights

`sparx.dynamics.RecurrentCell` feeds any model's output back to its input through a wiring. `Dense` connects every unit to every unit. `Sparse` takes an edge list, such as a connectome's, and gathers and sums along the edges in memory proportional to them; a sparse wiring that lists every pair computes exactly the dense layer. A `Sparse` wiring can give each edge its own delay in steps (`delay`, up to `longest_delay`). Each message is weighted when it leaves, and the cell's state holds the messages still on their way.

`FastWeights(alpha, rule)` adds differentiable plasticity (Miconi et al. 2018). The effective weight is `recurrent + alpha * hebb`, where the Hebbian trace `hebb` starts at zero for each sequence and follows `rule`, so it holds what that sequence showed the network. Backpropagation through the traces learns the weights, each connection's `alpha` and the rule's parameters. `DecayingHebb` and `OjaHebb` are Miconi et al.'s decaying trace and Oja's rule. `ModulatedHebb` and `RetroactiveHebb` are Backpropamine's (Miconi et al. 2019): a neuromodulator read off the units sets each unit's plasticity, or writes an eligibility trace of recent coactivity into the weights. The rules read connections through the wiring, so the same rule runs on a dense layer and on a connectome. On a `Sparse` wiring whose delays differ, a connection's trace pairs the new output with what the connection delivers that step, its source's output `delay` steps before, and a message leaves with the fast weights of the step that sent it.

In `sparx.nn`, `Recurrent(neuron, rule=...)` takes the rule as a module that declares its parameters: `DecayingTrace`, `OjaTrace`, `ModulatedTrace`, `RetroactiveTrace`, or your own `HebbianTrace` around a `HebbianRule`. `Recurrent(Rate(tau=0), rule=...)` is Miconi et al.'s tanh network; against their four networks in PyTorch its activity, traces and gradients agree within 5e-14. `Recurrent(ALIF())` is Bellec et al.'s recurrent adaptive network (LSNN), and `Recurrent(Rate())` is FLYNN's recurrence with a dense matrix. Backpropagating through a plastic layer keeps one trace per example per step.

```python
import sparx.nn as snn

layer = snn.Recurrent(snn.Rate(tau=0), rule=snn.ModulatedTrace())     # Backpropamine's network
spiking = snn.Recurrent(snn.LIF(), rule=snn.RetroactiveTrace(eta=0.1))  # fast weights between spikes
```

## Surrogate gradients

A spike is the Heaviside step of `v - threshold`, whose derivative is zero almost everywhere. The forward pass stays binary and the backward pass uses a surrogate's derivative instead. `sparx.spike` is a `jax.custom_jvp`, so the same rule serves `jax.grad`, `jax.jvp` and `vmap`.

| Surrogate | Derivative at `x = v - threshold` | Source |
| --- | --- | --- |
| `ATan(alpha=2)`, the default | `alpha / 2 / (1 + (pi / 2 * alpha * x)^2)` | Fang et al. 2021, SpikingJelly, snnTorch |
| `Sigmoid(alpha=4)` | `alpha * s * (1 - s)`, `s = sigmoid(alpha x)` | SpikingJelly |
| `FastSigmoid(slope=25)` | `1 / (slope * abs(x) + 1)^2` | SuperSpike (Zenke and Ganguli 2018) |
| `Triangle(width, scale)` | `scale * max(0, 1 - abs(x) / width)` | Bellec et al. 2018 |
| `Rectangle(width)` | `1 / width` inside `abs(x) < width / 2` | Wu et al. 2018 |
| `Gaussian(sigma)` | the normal density | Wu et al. 2018 |
| `StraightThrough()` | 1 | |

Pass one to any neuron, as in `sparx.nn.LIF(surrogate=sparx.surrogate.FastSigmoid(100.0))`.

A physical model takes its surrogate as a field, `AdEx(surrogate=...)`, and the surrogate reads `v - threshold` in mV. Backpropagation also runs through the membrane equation, and AdEx's exponential upswing multiplies the gradient at each step a neuron spends near its peak. Behind a dense layer, over 2000 steps of 0.1 ms at 40 Hz, the gradient reaching that layer had norm 7.9e8 with `ATan()` and 0.42 with `FastSigmoid(100)`. Train physical models with a steep `FastSigmoid`.

## Models

`sparx.models` builds architectures from these layers. `SEWResNet` is Fang et al.'s (2021) spike-element-wise residual network laid out as in SpikingJelly, with `sew_resnet18` and `sew_resnet34` presets. `SpikingMLP` is a dense network for event data, optionally recurrent, with a leaky integrator readout. `SpikingMLP(delays=K)` delays its first synapse by up to `K` steps, and `delays=(24, 24, 24)` delays every synapse, the readout's included. With `extend=True`, `batch_norm=True`, `use_bias=False`, `weight_init="kaiming_uniform"` and `dropout_mask="sequence"` it is Hammouamri et al.'s SNN-delays network, checked against their code. The `neuron` argument is the template every neuron of the network copies:

```python
net = sparx.models.sew_resnet18(10, width=32, stem="small",
                                neuron=sparx.nn.LIF(tau=2.0, detach_reset=True))
logits = net.apply(variables, frames, train=False)   # frames [T, B, 32, 32, 3] -> [T, B, 10]
```

## Encoders, losses and firing rates

`sparx.encode` turns a batch field `[B, ...]` into time-major input `[T, B, ...]`. An encoder is a frozen dataclass called as `encoder(key, x)` with a JAX PRNG key. Entry points called from outside JAX, such as dew's `Trainer`, take an int seed instead.

- `RateEncoder(steps)` draws Bernoulli spikes with the value as probability.
- `LatencyEncoder(steps)` fires one spike per value, earlier for larger values.
- `DirectEncoder(steps)` feeds the values as a constant input.
- `DeltaEncoder(threshold)` fires on changes of a signal along each record's time axis.
- `EventsEncoder()` passes records that already hold spikes over time.

The first four read a uint8 field as `x / 255`; `EventsEncoder` passes spike counts unscaled.

`sparx.losses` has losses over the whole output sequence. `per_step_cross_entropy` asks every step to classify (Deng et al. 2022), and `rate_mse` pulls each output neuron's rate toward a target. For a loss on one readout, reduce time first and use optax, with `jnp.max(v, axis=0)` of an `LI` membrane, its mean, or the spike count.

Spiking layers report their firing rates when the `spike_rates` collection is mutable:

```python
outputs, sown = net.apply(params, spikes, mutable=["spike_rates"])
sparx.firing_rates(sown)                          # {"LIF_0": mean rate, ...}
penalty = sparx.rate_penalty(sown, lower=0.01, upper=0.3)  # differentiable
```

A plain `apply` sows nothing and costs nothing.

## Streaming

With the `state` collection mutable, each neuron layer starts from the state it holds and writes back its final state. A sequence fed in chunks, down to one step at a time, gives exactly the output of one call over the whole sequence:

```python
carried = {}
for chunk in chunks:                               # each [t, B, ...]
    out, carried = net.apply({**params, **carried}, chunk, mutable=["state"])
```

Without `mutable=["state"]` every neuron starts at rest. `SlidingPSN` streams the same way; `PSN` and `MaskedPSN` need every step of a fixed `T` and refuse to.

## Models without Flax

The layers build neuron models from `sparx.dynamics`, and the models run without Flax. Every model, from the dimensionless `LIFCell` to the physical `AdEx`, has `init_state(shape, dtype)` and `step(state, SynapticInput, dt) -> (state, Output)`, and `sparx.run` scans one over time. `Output.value` holds spikes, or a real value for a model whose `graded` is True. An array input is a jump of the membrane each step:

```python
from sparx.dynamics import ALIFCell, Dense, LICell, LIFCell, RecurrentCell, Serial, decay

cell = LIFCell(decay=decay(tau=10.0), threshold=1.0, reset="subtract")
spikes, state = sparx.run(cell, inputs)            # inputs [T, ...]; spikes.value [T, ...]
more, state = sparx.run(cell, next_inputs, state)  # continues where it stopped

synaptic = Serial(LICell(decay(5.0)), LIFCell(decay(10.0)))  # a synaptic current, then the membrane
lsnn = RecurrentCell(ALIFCell(decay=0.95, adapt_decay=0.995, beta=1.8), Dense(weight))  # weight [F, F]
adaptive, _ = sparx.run(lsnn, inputs)
```

A model stores its decay per unit of time, and a step of `dt` (`sparx.run(..., dt=...)`, 1 by default) applies `decay ** dt`. Decays, thresholds and weights can be traced arrays, so you can learn them, sweep them with `vmap` or shard them.

## Training on dew

`sparx.objectives.SpikingClassifierObjective` is a [dew](https://github.com/AshishKumar4/dew) objective. It encodes a batch field into spikes, runs the network and scores the outputs against the labels, so a spiking network trains under dew's `Trainer` with its checkpoints, EMA, evaluation and display. The tests train it on one CPU device and on eight simulated CPU devices.

```python
import optax
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset

from sparx.datasets import shd
from sparx.encode import EventsEncoder
from sparx.metrics import Accuracy
from sparx.models import SpikingMLP
from sparx.nn import ALIF
from sparx.objectives import RateBand, SpikingClassifierObjective

train, test = shd("train"), shd("test")           # {"spikes": [N, 100, 700], "label": [N]}
net = SpikingMLP(hidden=(256,), classes=20, neuron=ALIF(tau=5.0, tau_adapt=20.0, learn_tau=True))
objective = SpikingClassifierObjective(
    net, Field("spikes", (100, 700)), EventsEncoder(),   # the records already hold spikes
    readout="max",                                # each class's peak membrane
    rates=RateBand(lower=0.01, upper=0.3),        # keep neurons in a firing band
)
data = Dataset.from_records(train, batch=64, validation=test)
trainer = Trainer(objective, optax.adamw(2e-3), key=0, checkpoints=Checkpoints("runs/shd"))
state = trainer.fit(data, steps=3000, eval_every=500, metrics=[Accuracy()], validation={"test": data.val})
classifier = objective.pipeline(state)            # the trained classifier, as dew.pipeline loads it
```

The readout is one of `"mean"`, `"max"`, `"sum"`, `"softmax_sum"` (SNN-delays' loss, the softmax of each step summed over time) or `"per_step"`. `schedules` lets model keyword arguments follow dew's schedules, such as `schedules={"sigma": Linear(peak=7.5, end=0.5)}` for a `DelayedDense`, and `deployed={"sigma": 0}` evaluates with every delay rounded. Parameter groups get their own optimizers through dew's `OptimConfig`, matched by path pattern, each with its schedule, weight decay and bounds:

```python
from dew.config import OptimConfig
from dew.training.optim import Cosine, OneCycle, ParamGroup

optimizer = OptimConfig(optimizer="adam", param_groups=(
    ParamGroup("delays", ("*/delay",), schedule=Cosine(peak=0.1, warmup_steps=0), bounds=(0.0, 24.0)),
    ParamGroup("weights", ("*",), schedule=OneCycle(peak=5e-3), weight_decay=1e-5)))
```

Dew's validation pass scores every record of a split. It pads the last batch with repeats, which the losses and metrics give no weight. SHD has no validation split, so `sparx.datasets.holdout(train, 0.1)` holds out part of the training set to select on. `sparx.datasets.mnist(split, fashion=False)` reads MNIST or Fashion-MNIST, downloaded once to `~/.cache/sparx`. The objective logs batch accuracy and each spiking layer's rate, updates BatchNorm statistics, passes dropout keys, and evaluates to dew's `TokenScores` for `sparx.metrics.Accuracy`.

Other objectives:

- `EPropObjective` trains a `SpikingMLP` with one recurrent layer by e-prop's gradients, handed to the trainer as the loss's own through `Objective.with_gradients` ([`examples/train_shd_eprop.py`](../examples/train_shd_eprop.py)). The run loads back as that `SpikingMLP`, so a network trained online streams, serves and exports as one trained by backpropagation does.
- `PredictiveCodingObjective` trains a stack of layers by predictive coding or PC-ALM ([`examples/train_pcalm.py`](../examples/train_pcalm.py)).
- `RNeuralNetObjective` trains an `RNeuralNet` by reward diffusion, by diffusion with two of AGREL's changes (`rule="gated"`), by AGREL's update (`rule="agrel"`) or by REINFORCE ([`examples/reward_diffusion.py`](../examples/reward_diffusion.py)).
- `ActivityFitObjective` fits a network's spikes to recorded ones by van Rossum distance or smoothed rates.
- `sparx.nn.BatchMajor` runs a time-major stack on dew's batch-major records, so any sparx stack also trains under dew's generic `Supervised`. [`examples/pattern_completion.py`](../examples/pattern_completion.py) trains a plastic `Recurrent` network that way on Miconi et al.'s pattern completion.

Every example takes `--smoke`, which trains a small network for a few steps on synthetic data and downloads nothing.

A run's record names each sparx class by import path, so `run.json` holds a model as `{"class": "sparx.models:SpikingMLP", "fields": {...}}` and nothing is registered. `dew.pipeline(run_dir, trust=("sparx",))` loads a trained classifier in a new process; `trust` lets the record import sparx, as `trust_remote_code` does in transformers:

```python
import dew

classifier = dew.pipeline("runs/shd", trust=("sparx",))
predictions = classifier(test_spikes)              # [B]
```

[`recipes/snn/train.py`](../recipes/snn/train.py) runs `sparx.config.SNNRunConfig`, dew's `RunConfig` for a spiking classifier on SHD. Every setting is a typed flag: each model field (`--model.hidden 128`), each objective argument (`--objective.readout max`), the encoder as a subcommand (`encoder:rate --encoder.steps 8`), and a nested record as JSON. `dew train runs/shd/run.json --trust sparx --set trainer.steps=6000` rebuilds a run from its record and trains on from its last checkpoint.

```bash
python recipes/snn/train.py --data.channels 140 --trainer.batch-size 64 --trainer.steps 3000 \
    --trainer.checkpoint-dir runs --trainer.name shd --model.hidden 128 --model.delays 15 \
    --model.neuron '{"class": "sparx.nn.neurons:ALIF", "fields": {"tau": 5.0}}' \
    --objective.schedules '{"sigma": {"class": "linear", "fields": {"peak": 7.5, "end": 0.5}}}'
JAX_PLATFORMS=cpu python recipes/snn/train.py --smoke --trainer.checkpoint-dir /tmp/snn-smoke
```

`Trainer(..., mesh=MeshSpec(fsdp=2))` trains on several devices; a test checks that eight simulated CPU devices train the same parameters as one, within 1.8e-7.

## Simulating circuits

`sparx.dynamics` holds neuron, synapse and plasticity models in ms, mV, pA, nS and pF. `sparx.graph` wires them into a `Network`, a Flax module whose variables hold the connectome, trainable weights and simulation state. Vogels and Abbott's network with conductance-based synapses, a standard simulator benchmark:

```python
import jax
from sparx.dynamics import Exponential, LeakyIntegrateAndFire, Receptor
from sparx.graph import (FixedProbability, Network, Population, PopulationRate, Projection, SpikeRaster,
                         StateMonitor, simulate)
from sparx.spiketrains import cv_isi, rates_hz

neuron = LeakyIntegrateAndFire(tau_m=20.0, c_m=200.0, e_l=-60.0, v_th=-50.0, v_reset=-60.0, t_ref=5.0)
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

`Network` checks itself when built. A projection or input onto a receptor the population lacks, a conductance receptor without a reversal potential, or a kinetic synapse onto a dimensionless model raises a `ValueError` naming the population and receptor. Each projection and input names its receptor, which sets the weight's unit. Delays, `duration` and `chunk` must be whole numbers of steps.

Records come back under the monitors' names, and every rate is in Hz. After a run, `network.connections(result.variables)` reads each projection's synapses as NumPy arrays, as NEST's `GetConnections` does, with STDP's weights as learned.

A step runs in NEST's order. Synapses deliver what is due, membranes integrate (exactly where the equations are linear) and spike, spikes enter per-population ring buffers, kinetic synapses receive what arrives, plasticity updates, and monitors record. `simulate` compiles one chunk of steps and carries the state between chunks, so a long run holds one chunk of records in memory. With `checkpoints=dew.Checkpoints(directory)` it writes the state after each chunk, and a run started again on that directory continues from the last one.

`sparx.graph.models` builds Brunel's (2000) network and the CUBA and COBA benchmarks (`brunel`, `cuba`, `coba`). `Projection`s take per-edge weights and delays, pair, triplet and reward-modulated STDP, and short-term plasticity. A population holds any model of `sparx.dynamics`; each model says where it is refractory (`is_refractory`) and how a jump that lands after the threshold test changes it (`after_threshold`). The builders have short names in `sparx.registry.networks`, so `sparx.graph.from_record({"class": "brunel", "fields": {"order": 2500, "g": 5.0}})` rebuilds one in another process.

## Graded signalling and neuromodulation

Many neurons never spike. Much of the fly's visual system releases transmitter continuously as a function of voltage. A model's output is `Output(value, offset)`, and a `graded` model sends a real value each step. `GradedPotential` is a passive membrane in mV whose output is a sigmoidal release of its voltage (Prinz et al. 2004), and a `Graded` synapse follows the weighted release with its own time constant. A graded population projects through edge or dense delivery; event delivery is for spikes and refuses it.

`StochasticRelease(p, quantal)` makes each synapse release with probability `p` at each presynaptic spike. `GapJunction` couples two populations' membranes with `I = g (v_partner - v)` in both directions, solved to second order in `dt`. A `Modulator` turns a population's spikes into a volume-transmitted concentration, which each `Plasticity` rule reads as a third factor:

```python
import jax
import numpy as np
from sparx.dynamics import Exponential, Graded, GradedPotential, LeakyIntegrateAndFire, Receptor, StochasticRelease
from sparx.graph import (CurrentInput, FixedProbability, GapJunction, Modulator, ModulatorTrace, Network,
                         OutputTrace, Population, Projection, SpikeRaster, simulate)

receptors = {"ampa": Receptor(Graded(tau=5.0), "conductance"),  # follows the graded release it receives
             "gaba_a": Receptor(Exponential(10.0), "conductance")}
network = Network(
    populations=(Population("graded", 20, GradedPotential()),  # never spikes, sends its release (0 to 1)
                 Population("relay", 50, LeakyIntegrateAndFire(), receptors)),
    projections=(Projection("graded", "relay", FixedProbability(0.3), weight=3.0, delay=0.0, receptor="ampa"),
                 Projection("relay", "relay", FixedProbability(0.2), weight=10.0, delay=1.0, receptor="gaba_a",
                            release=StochasticRelease(p=0.4))),  # each synapse releases with p = 0.4
    inputs=(CurrentInput("graded", "light"),),
    junctions=(GapJunction("graded", "graded", FixedProbability(0.2), weight=2.0),),  # nS, both ways
    modulators=(Modulator("dopamine", "relay", tau=200.0, release=0.01),),  # each relay spike adds 0.01
    dt=0.1,
)
light = np.random.default_rng(0).uniform(100.0, 400.0, (3000, 20))  # pA, per step and graded neuron
result = simulate(network, network.init(jax.random.key(0)), duration=300.0, key=jax.random.key(1),
                  drive={"light": light}, monitors={"release": OutputTrace("graded"),
                                                    "spikes": SpikeRaster("relay"),
                                                    "dopamine": ModulatorTrace("dopamine")})
```

`DopamineSTDP` is Izhikevich's (2007) reward-modulated STDP, NEST's `stdp_dopamine_synapse`. Pair STDP writes an eligibility trace on each synapse, and the weight integrates that trace times the dopamine above a baseline, so a pairing changes the weight only if dopamine arrives within about a second. It reads the modulator named `"dopamine"`, whose `tau` must be the rule's `tau_n` (200 ms), with `release=1 / 200` for NEST's increment per spike.

## Learning rules

`sparx.learn` holds the rules beyond surrogate backpropagation, each checked against what defines it.

- `eprop` is e-prop (Bellec et al. 2020) for a recurrent layer of any elementwise sparx model and a leaky readout, computed online in memory that does not grow with the sequence. It equals backpropagation with the recurrent spikes' gradient cut, and its traces with the true learning signal equal backpropagation, the two identities their own code verifies.
- `ottt` is online training through time (Xiao et al. 2022), matching their PyTorch modules' gradients to 1e-10.
- `reinforce` is REINFORCE (Williams 1992) for a recurrent layer of `BernoulliCell`s, LIF neurons that fire with probability `sigmoid(beta (v - threshold))`. Each synapse accumulates `d log P(spikes) / dw` as the layer runs, and `policy_gradient(eligibility, rewards, baseline)` turns rewards into the gradient estimate. On a layer small enough to enumerate every trajectory, the expectation is the exact gradient.
- `PredictiveCoding` is predictive coding and PC-ALM (Seely and Gould 2026) for any stack of layers. Hidden activity relaxes on each example's energy, the output's loss plus every layer's prediction error, and each weight's update reads only its own layer's error. PC-ALM adds a multiplier per layer that accumulates the error between activity steps. Against their JAX reference, activity, multipliers and update agree within 5e-14 in float64.
- `spike_times` with `EventLIF` gives exact, differentiable spike times of LIF networks with current synapses in continuous time, the gradient EventProp (Wunderlich and Pehle 2021) computes, checked against finite differences.
- `RNeuralNet` and `reward_diffusion` rebuild RNeuralNet-Research (2018) deterministically, a random recurrent network of graded neurons (`PulseCell`) whose connections deliver weighted messages after their own delays, and its learning rule. A reward spreads backward from a feeder neuron, shared among each unit's inputs by a softmax of their absolute activity. Compiled and run single-threaded in a fixed order, the original C++ and sparx agree within 7.2e-7. `paths="first"` is the original's depth-first spread; `paths="all"` counts every path, discounted. The share ignores signs, so the rule follows no gradient, and sparx keeps it as a baseline.
- `convert` turns a ReLU network written as a flax `nn.Sequential` (dense and convolutional layers, batch norm, pooling, `Flatten`) into a `sparx.nn` stack with `IF` neurons and a gated `SpikingMaxPool`, balancing thresholds at a percentile of the activations (Rueckauer et al. 2017). Against their toolbox it gives the same weights, first-layer spikes and predictions.

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

## Connectomes, serving and exchange

`sparx.graph.connectome` reads FlyWire (Shiu et al.'s tables) and the male CNS release into a `Connectome` and builds Shiu et al.'s (2024) whole-brain model. On FlyWire v630 it reproduces their published runs, a rate correlation of 0.999 and MN9 at 67.1 Hz against their 67.0 ± 6.6, at about 30 s per simulated second on 4 CPU cores. The male CNS gives its neurons about 1.7 times FlyWire's synapses, so `matched_w_syn` rescales their weight from 0.275 to 0.163 mV; sugar neurons then recruit about 670 neurons and drive MN9 at 81 Hz.

`FLYNN` is Wang and Chen's trainable fly connectome (arXiv 2607.00025): a leaky tanh unit per neuron, recurrent through the connectome's synapses on a `Sparse` wiring, with a weight per synapse, a bias per neuron and a leak per cell class. Against their PyTorch cell its activity and gradients agree within 1e-15 in float64. The weights start at the signed synapse counts scaled to a spectral radius of 0.9, computed exactly; their code estimates the radius by power iteration, which depends on its random start when the dominant eigenvalues are a complex pair. Through `nn.BatchMajor` it trains under dew's `Supervised`, and with a `rule` it learns fast weights on its synapses.

`simulate(trials=..., mesh=dew.MeshSpec(...))` spreads trials, or one network's neurons, over devices. `MeshSpec()` partitions one network's neurons over every device, and `MeshSpec(fsdp=2)` runs trials on the data axis with each trial's neurons split in two.

`sparx.serve.StreamServer` serves streaming models to many sessions at once, each with its own neuron state in a slot of one batch, and a session's outputs equal a direct call over its stream. A reloaded run is served with `StreamServer(classifier.model, classifier.variables, slots=8, frame=10, sample_shape=(700,))`. A model that cannot stream, a `PSN` or a readout averaged over time, is refused when the server is built.

`sparx.nir` exchanges networks through NIR: dense and 2-D convolutional layers, `Flatten`, hard-reset `LIF` and `IF`, and `Recurrent(LIF)`. Networks exported by snnTorch run in sparx spike for spike and export back with the same parameters.

## Results in detail

All runs are the example scripts on a 4-core x86 CPU with JAX 0.11.2, float32, seed 0 unless stated. They are short, untuned runs. The MNIST and SHD rows were measured at commit 6f5ed31, before those examples moved to dew's `Trainer`; the networks, losses and optimizers did not change. The e-prop rows were measured on 8 October 2026, after e-prop's network became a `SpikingMLP`.

| Task | Command | Network | Test accuracy | Time |
| --- | --- | --- | --- | --- |
| MNIST, rate-coded, 8 steps | `python examples/train_mnist.py --epochs 2` | 784-512-512 LIF, LI readout | 97.46% after 2 epochs | 15 s per epoch |
| SHD, 100 steps of 14 ms | `python examples/train_shd.py --steps 3000` | 700-256 ALIF, LI readout (max) | 53.00% | 4 min 45 s |
| SHD | `... --steps 3000 --recurrent --surrogate superspike` | 256 recurrent ALIF | 45.23%, still rising | 8 min |
| SHD, channels pooled to 140 | `... --channels 140 --hidden 128` | 140-128 ALIF | 64.53% | 1 min 42 s |
| SHD, channels pooled to 140 | `... --channels 140 --hidden 128 --delays 15` | the same, a learned delay of 0 to 15 steps per input synapse | 74.56%, delays rounded | 4 min 19 s |
| SHD, Hammouamri et al.'s recipe, 20 of 150 epochs | `python examples/train_shd.py --recipe snn-delays --epochs 20 --validation 0` | 140-256-256 LIF, a learned delay on every synapse | 91.87% (their code: 93.59% last, 94.03% best) | 48 min |
| SHD, channels pooled to 140 | `python examples/train_shd_eprop.py --rule eprop --epochs 5`, seeds 0 to 2 | 140-128 recurrent ALIF, trained online by e-prop | 49.2, 48.6 and 43.2% (best epochs 54.8, 57.0 and 55.7%) | 4 min 32 s to 4 min 53 s |
| SHD, channels pooled to 140 | `... --rule bptt --epochs 5`, seeds 0 to 2 | the same network by BPTT | 51.6, 45.7 and 46.9% (best epochs 51.6, 51.8 and 48.2%) | 37 to 38 s |
| Fashion-MNIST, Seely and Gould's headline cell | `python examples/train_pcalm.py` (`--method pc`, `--method bp`) | 784-32-...-32-10 ReLU residual MLP, depth 32, one epoch | PC-ALM 75.1%, PC 62.2%, BP 77.8% | 83 s, 82 s, 41 s |
| Delayed cue order, RNeuralNet | `python examples/reward_diffusion.py --rule first` (`all`, `gated`, `agrel`, `reinforce`, `none`), seeds 0 to 4 | 256-neuron `RNeuralNet`, delays of 1 to 21 ticks, 19,200 rewarded trials | diffusion unchanged from 44.5 to 50.2%; gated diffusion no better, 0% on one seed; AGREL's update and REINFORCE 100% on 4 of 5 seeds; linear readout 100% | 98 s, 39 s, 40 s, 22 s, 20 s |

The two pooled SHD rows with and without delays differ only in the delays, which add 10 points. The e-prop and BPTT rows train the same network with the same optimizer. Both swing by up to 10 points between epochs, and neither is tuned. Before the network became a `SpikingMLP`, it had no input bias and took Bellec et al.'s initialization (normal weights over the square root of the fan-in); over the same seeds e-prop ended at 57.8, 56.3 and 47.4% (best epochs 57.8, 56.3 and 53.2%). With that initialization and an input bias it ended at 54.6, 49.9 and 44.4%, so the bias may cost a few points; three seeds cannot separate that from the swing. e-prop's memory does not grow with the recording, but on a CPU it runs about 7 times slower here, since it advances an eligibility trace for every synapse every step ([performance](performance.md#e-prop)).

For the SNN-delays row, their official code (with the 2023 SpikingJelly it was written for) and sparx ran the same recipe and schedule on this machine, one seed each. sparx ends 1.7 points below their last epoch and 2.2 below their best. One training step's gradients agree with theirs to 6e-7, so the gap comes from what the steps see. The example's docstring lists those differences: every recording padded to 124 steps where theirs pads to each batch's longest, 31 steps an epoch where they take 32, and different random streams. One seed each does not separate these from run-to-run spread. sparx's epochs took 146 s against about 210 s for theirs. Their reported 95% is the best of 150 epochs, selected on the test set.

The Fashion-MNIST row trains Seely and Gould's (2026) headline network three ways through dew's trainer, measured on 7 October 2026. Their reference code on the same machine, seeds 0 to 2, scores 77.73, 76.49 and 76.34% by PC-ALM, 68.17, 64.49 and 66.54% by PC, and 78.65, 76.85 and 77.31% by BP. sparx's seeds 0 to 2 score 75.1, 76.1 and 76.5% by PC-ALM, 62.2, 65.5 and 65.6% by PC, and 77.8, 76.9 and 76.7% by BP, the same ranking, averaging 1.0, 2.0 and 0.5 points lower. Part of that is the two codes' different draws of weights and batches, which three seeds cannot separate from the rest.

The cue-order row runs the experiment the RNeuralNet notes propose, in which the network reports which of two cues came first after 10 ticks of distractors and each choice earns 1 or -1 (measured 8 October 2026 at commit cd8b492). A least-squares readout of every neuron at the last tick is right on every test trial, so the network keeps the order. REINFORCE through the network teaches its output neurons on four seeds; on the fifth the cues barely reach them at the start (they move the outputs' mean by at most 1.2e-7) and it stays at chance. Reward diffusion never changes which output wins, because its share ignores signs and which output was chosen. With every connection at the shortest delay (`--myelin 0`) it moves the choice in either direction, from 54.3 to 62.5% on seed 0 and from 52.3 to 17.8% on seed 3.

Two of the changes attention-gated reinforcement learning (AGREL) makes do not rescue it. Spreading a signed reward prediction error from the chosen output instead of the reward from the feeder (`--rule gated`) takes the networks as drawn, at 48.8, 50.2, 46.7, 47.7 and 44.5%, to 51.2, 49.8, 46.7, 0.0 and 44.5%. On seed 3 it learns the reverse of the task. Its share still ignores the signs of the activity and the weights, so the credit it passes does not say which way a weight should move. Sending the error back through the weights and the neurons' slopes along every delayed path as well (`--rule agrel`, measured 8 October 2026) learns the task on the same four seeds as REINFORCE, at 100%, and leaves seed 1 at chance. The times are from that date's runs, whose evaluation is compiled.

The recurrent SHD run needs the steep SuperSpike surrogate. With `ATan`, backpropagation through the recurrence exploded once training grew the recurrent matrix's spectral radius from 1 to 5, the gradient norm passing 1e8 within 300 steps. `FastSigmoid(100)` kept it below 10.
