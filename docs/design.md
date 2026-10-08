# Sparx design

How sparx is built, as of 8 October 2026: what each part is for, the contracts between the parts, and why they are shaped as they are. [HANDOFF.md](../HANDOFF.md) lists the open work, [fidelity.md](fidelity.md) what each model is checked against, [performance.md](performance.md) the measurements behind the defaults, and [units.md](units.md) the units.

## 1. What sparx is for

Sparx is dew's spiking and biophysical modality. It serves two kinds of work with one representation:

1. **Spiking networks as machine learning.** Spiking networks trained on event and static data by surrogate gradients, online rules or local learning rules, served as streams and exchanged with other libraries.
2. **Brains as networks.** Building, simulating and fitting biological circuits, from Brunel's balanced network to whole connectomes such as FlyWire, with neuron and synapse models as close to the biology as the question needs.

One neuron protocol serves both. A trainable layer runs a neuron model over time; a simulated population runs the same protocol on one clock with others. Training, distribution, checkpoints, run records and serving come from dew.

Sparx does not add a second trainer, checkpoint format or runtime unit system, and it does not model dendrites in morphological detail.

## 2. Principles

Sparx keeps dew's six rules (`dew/docs/design/api.md`):

1. A record names a class or function by its import path; nothing is registered, and sparx keeps short names only where its own code reads them (`sparx.registry`).
2. Values that cross `jit` are `flax.struct.dataclass`es; configuration is a frozen dataclass; interchangeable implementations are a `Protocol`.
3. Randomness is a `key` argument.
4. Effects live in capabilities handed to the trainer or simulator (`Checkpoints`, `Tracker`).
5. An unknown name or field raises.
6. A schedule is a value.

It adds six of its own:

7. **dew is the platform.** Sparx adds dynamics, structure, learning rules and objectives. It never adds a trainer, a mesh, a checkpoint format, a data loader framework, a launcher or a server of its own design; where dew lacks an extension point, the change goes to dew (section 9.2).
8. **The science decides, references are evidence.** Each model is checked against ground truth where one exists (an analytic result, or a float64 integration at a finer step), then against its reference implementation. Where a reference departs from the science, sparx follows the science and records the departure in a test and in [fidelity.md](fidelity.md). An optimized path must agree with the plain one to a stated tolerance.
9. **Time is explicit.** Every dynamical object takes its step `dt`, and its time constants are in the unit of `dt`. The deep-learning convention, time constants in steps, is the same code at `dt = 1` ([units.md](units.md)).
10. **Structure is a graph.** Populations and projections are first-class; a stack of layers is the dense special case and keeps its own path.
11. **Parameters, structure and state are separate collections.** What trains, what is fixed wiring and what evolves in time live in different Flax collections, so the optimizer, the sharding layout and the checkpoint each see their part.
12. **A fast path is chosen by measurement.** A delivery format, a kernel or a parallel scan is the default only when it is faster on the hardware it targets and matches the plain path; [performance.md](performance.md) records what was measured.

## 3. Architecture

```
dew (platform)    Trainer . Objective . MeshSpec/Layout . Checkpoints . Dataset . records . RunConfig/CLI . pipeline
                    ^ objectives, models, encoders, networks and tasks, recorded by import path
sparx.objectives  dew objectives: classification, activity fitting, e-prop, predictive coding, rewards
sparx.learn       exact spike times (EventProp) . e-prop, OTTT . REINFORCE . predictive coding, PC-ALM
                  . reward diffusion (RNeuralNet) . ANN-to-SNN conversion
sparx.graph       Population . Projection . Connectivity . Network (a Flax module) . monitors . simulate
                  . canonical networks . connectomes
sparx.nn          time-major Flax layers over the models below; sparx.models builds architectures from them
sparx.dynamics    neuron models . synapses . plasticity . wirings, recurrence and fast weights . run
sparx.surrogate   the spike and its surrogate gradients
```

Around them: `sparx.encode` turns a batch field into spikes over time, `sparx.losses`, `sparx.rates` and `sparx.spiketrains` read outputs and spike trains, `sparx.tasks` and `sparx.serve` serve trained networks, `sparx.nir` exchanges them through NIR, `sparx.datasets` reads SHD and MNIST, and `sparx.config` is the run class a recipe trains.

Each layer depends only on the ones below it. `sparx.dynamics` builds its models as `flax.struct` dataclasses and uses no Flax module, so its models run from plain JAX; the package's `__init__` loads `sparx.nn` with it.

## 4. Dynamics

### 4.1 The model contract

A model is a `flax.struct` dataclass of its parameters, which are leaves that can be learned, swept with `vmap` and sharded, and of its choices, which are static fields. It meets one protocol:

```python
class NeuronModel(Protocol[State]):
    graded: bool                                  # a value every step, or spikes
    def init_state(self, shape, dtype) -> State: ...
    def step(self, state: State, inputs: SynapticInput, dt: float) -> tuple[State, Output]: ...
    def is_refractory(self, state: State, dt: float) -> jax.Array: ...
    def after_threshold(self, state: State, jump: jax.Array, fired: jax.Array) -> State: ...

@struct.dataclass
class SynapticInput:
    current: jax.Array            # pA, held over the step
    currents: tuple[Term, ...]    # current waveforms over the step, integrated exactly by linear models
    conductance: Mapping[str, jax.Array]   # nS per receptor, against the model's reversal potentials
    jump: jax.Array               # added to the voltage at the end of the step, before the threshold test
    gap: Gap | None               # gap-junction coupling to other neurons' voltages
    noise: jax.Array | None       # a uniform draw per neuron, for escape noise
```

`sparx.run(model, inputs, state, dt=...)` scans a model over time; an array input is a jump. `is_refractory` and `after_threshold` let a network apply input that lands after the threshold test (a stimulus in Shiu et al.'s model) the way each model's own step would. `Output(value, offset)` holds spikes, or a real value for a `graded` model (a graded neuron's release, a rate unit's activity), and the offset is where in the step a spike crossed threshold. A dimensionless model refuses a current or a conductance, and a network refuses, when it is built, any input a model cannot take.

### 4.2 Integration

Each model states how it is integrated, and its tests pin the choice:

| Model | Integration |
| --- | --- |
| `LeakyIntegrateAndFire`, `GradedPotential` | the exact solution of the linear membrane over the step, with current waveforms integrated exactly and conductances held, as NEST's `iaf_psc_exp` |
| conductances under `PointNeuron` | held at their exact mean over the step (`hold="mean"`, second order), or at their start-of-step value (`hold="start"`, Brian2's `exponential_euler`) |
| synapse kinetics (`Exponential`, `Alpha`, `BiExponential`, `Graded`) | exact |
| `AdEx` | RK4 in substeps of at most 0.01 ms, with the spike found and reset within its substep |
| `Izhikevich` | `scheme="published"`: two half-steps of `v`, then `u`, as the 2003 code; `"euler"` as NEST; `"semi_implicit"` as the 2004 code |
| `HodgkinHuxley` | `scheme="strang"`: Rush-Larsen half-steps of the gates around an exact voltage step, in substeps of 0.01 ms; `"rk4"` and `"exponential_euler"` |
| dimensionless cells (`sparx.dynamics.ml`) | `v = decay ** dt * v + x`, the exact decay of the leak |

A spiking model reports where in the step its membrane crossed threshold, by linear interpolation of the voltage (Hansel et al. 1998): `LeakyIntegrateAndFire`, `Izhikevich` and `AdEx` do, `HodgkinHuxley` stamps the end of the step. The network keeps spikes on the step grid; no delay or plasticity rule reads the offset yet. A refractory neuron holds its reset for `round(t_ref / dt)` steps after the step it fired in, as NEST counts.

### 4.3 Catalogue

| Kind | Models |
| --- | --- |
| Dimensionless cells | `LIFCell`, `ALIFCell` (Bellec et al. 2020), `LICell`, `RateCell` (FLYNN's leaky unit), `PulseCell` (RNeuralNet's), `BernoulliCell` (escape noise), `Serial` (a synaptic current, then a membrane) |
| Physical neurons | `LeakyIntegrateAndFire`, `AdEx` (Brette and Gerstner 2005), `Izhikevich` (2003, and the twenty patterns of 2004), `HodgkinHuxley`, `GradedPotential` (Prinz et al. 2004) |
| Synapses | `Delta`, `Exponential`, `Alpha`, `BiExponential`, `Graded`, each a current or a conductance by its `Receptor`; `MgBlock` (Jahr and Stevens 1990); `StochasticRelease` |
| Plasticity | `PairSTDP`, `TripletSTDP` (Pfister and Gerstner 2006), `DopamineSTDP` (Izhikevich 2007), `TsodyksMarkram` |
| Recurrence | `RecurrentCell` over a `Dense` or `Sparse` wiring, a `Sparse` one with a delay per edge, with optional `FastWeights` on all its connections or a chosen few, and a `HebbianRule`: `DecayingHebb`, `OjaHebb`, `ModulatedHebb`, `RetroactiveHebb` |

Linear synapses sum, so each receptor of a population holds one state per neuron, not one per synapse: `N` states in place of `E`, which is what makes a connectome affordable. Plastic projections keep per-edge weights and traces; short-term plasticity keeps one state per presynaptic neuron.

### 4.4 Units

Units are a convention, not a runtime system: ms, mV, pA, nS and pF in the physical models, steps in the dimensionless ones. [units.md](units.md) lists them, the two units of rates, and where the halves meet. A runtime unit system such as Brian2's does not trace through `jit`.

## 5. Structure

### 5.1 The graph

```python
Population(name, size, neuron, receptors={}, hold="mean", initial={}, reset_synapses=False,
           freeze_synapses=False)
Projection(pre, post, connectivity, weight=1.0, delay=1.0, *, receptor, plasticity=None,
           short_term=None, release=None, trainable=False, name=None, format="auto", per_pass=16)
Network(populations, projections=(), inputs=(), dt=0.1, dtype=float32, junctions=(), modulators=())
```

A projection's receptor sets the unit of its weight; its delay is in ms and a whole number of steps; weights and delays may be per edge. Inputs are `PoissonInput` (Hz), `CurrentInput` (pA, from a drive) and `ArrivalInput` (weights arriving each step); `GapJunction` couples two populations' membranes; a `Modulator` turns a population's spikes into a concentration that plasticity reads. A `Network` is a Flax module whose variables split by role:

| Collection | Holds | Seen by |
| --- | --- | --- |
| `params` | the weights of trainable projections | the optimizer |
| `connectome` | each projection's edges and delays in its delivery format, and its weights unless trainable | the layout and checkpoints, never the optimizer |
| `state` | per population, neuron and synapse states and a ring buffer of recent outputs; plastic weights and traces; short-term release; modulators; the step count | carried from call to call |

Weights that a local rule changes during a run are state, not parameters. `Network.connections(variables)` reads every projection back as edges, weights and delays.

### 5.2 Connectivity

`Connectivity.edges(rng, pre, post, same)` draws a projection's edges once, when the network is initialized: `AllToAll`, `OneToOne`, `FixedProbability`, `FixedInDegree`, `FixedOutDegree` (NEST's rules, with or without autapses and multapses) and `FromEdges`, a table. `sparx.graph.connectome.Connectome` reads Shiu et al.'s FlyWire tables (`from_shiu`) and the male CNS v0.9 release (`from_malecns`), where each neuron's transmitter gives its synapses' sign (`SIGNS`: acetylcholine, dopamine, serotonin and octopamine excite; GABA, glutamate and histamine inhibit).

### 5.3 Canonical networks

Builders, also validation targets, with short names a record can use (`sparx.registry.networks`): `brunel` (Brunel 2000, model A, in any of his regimes), `cuba` and `coba` (Vogels and Abbott 2005, Brette et al.'s 2007 benchmarks) and `shiu2024` (Shiu et al.'s whole-brain LIF on a connectome). `FLYNN` (Wang and Chen 2026) is the trainable connectome, a `sparx.nn` layer over a `RecurrentCell` on the connectome's `Sparse` wiring.

## 6. Execution

### 6.1 One step

Every network advances in NEST's order, which the single-neuron and network tests pin against NEST and Brian2. A step covers `(t, t + dt]`:

1. Delta synapses deliver the spikes due at the end of the step as voltage jumps.
2. Each population advances its membranes on its synapses' output and its gap junctions, and emits its output (spikes, or graded values).
3. The outputs enter each population's ring buffer, and each modulator takes up the spikes of its source.
4. Every other synapse receives what is due at the end of the step, which shapes the membrane from the next step on.
5. Plasticity updates traces and weights, reading the modulators.
6. Monitors record.

A spike sent in step `m` over a delay of `D` steps is due at the end of step `m + D`. Kinetic synapses take `D = 0`; delta synapses need `D >= 1`, since a jump due in its own step would feed back into that step's threshold test.

### 6.2 Delivery

A projection is stored and delivered in one of three formats, all giving the same input up to the order of summation:

| Format | Cost per step | Taken by `"auto"` |
| --- | --- | --- |
| `"events"` | the spiking neurons' out-edges | whenever it can: a spiking population, one delay, fixed weights |
| `"dense"` | `pre x post` | otherwise, with at most `DENSE_LIMIT` entries and a density of at least 2% |
| `"edges"` | every edge, summed per target with `segment_sum` | otherwise |

Event delivery takes a step's spiking neurons in passes of `per_pass`, found by `jax.lax.top_k`, as many passes as the step has spikes. Near-even or small out-degrees keep each neuron's out-edges as one padded row; uneven ones (a connectome's) are laid end to end in blocks and found by binary search. A `while_loop` of passes keeps shapes static, drops no spike, and under `vmap` costs the busiest trial's spikes. [performance.md](performance.md) has the measurements behind each choice.

Each population keeps a ring buffer of its last outputs, in the network's dtype, as long as its longest outgoing delay; delays are int32 steps, and a projection with one delay reads one row of the ring.

### 6.3 Scale

`simulate(mesh=MeshSpec(...))` places a run on dew's mesh by logical axes, `trials` on the data axis and `neurons` on the fsdp or data axis: a leaf whose last dimension is a population's size is split over neurons, and the compiler partitions the step's gathers and the exchange of spikes. Per-edge plastic weights stay whole. The male CNS, 166,000 neurons and 25.6M synapse pairs, simulates on one CPU in 1.8 GB. A hand-written exchange of spike bitmasks between devices, which would cut that traffic, waits for measurements on multi-device hardware.

### 6.4 `simulate`

```python
simulate(network, variables, *, duration, key=None, drive=None, monitors=None, chunk=100.0,
         trials=None, mesh=None, layout=LAYOUT, checkpoints=None) -> Simulation
```

Simulation is a function, not a runner beside dew's `Trainer`. It compiles one chunk of steps, carries the state between chunks, and brings each chunk's records to the host, so a long run holds one chunk of records in memory. `trials` runs independent trials under `vmap`, each with its own noise. With `checkpoints` (dew's) it saves the state after every chunk, and a run started again on the directory continues from the last one as if it had not stopped. Monitors are a mapping of names to `SpikeRaster`, `SpikeCounts`, `SpikeTimes`, `PopulationRate`, `OutputTrace`, `ModulatorTrace` and `StateMonitor`.

## 7. Learning

Every learning rule works on the same models, and each is checked against what defines it ([fidelity.md](fidelity.md)).

| Regime | How | Lives in |
| --- | --- | --- |
| Surrogate-gradient BPTT | the spike is a `jax.custom_jvp` Heaviside step whose derivative is a surrogate's | `sparx.surrogate` |
| Exact spike-time gradients | EventProp's gradient (Wunderlich and Pehle 2021) for LIF networks with current synapses, by implicit differentiation of exact spike times | `sparx.learn.events` |
| Online rules | e-prop (Bellec et al. 2020) and OTTT (Xiao et al. 2022), traces carried forward, memory independent of the sequence's length; e-prop's eligibility structure read from the step's jaxpr | `sparx.learn.online`, `EPropObjective` |
| Local plasticity | pair, triplet and reward-modulated STDP (Izhikevich 2007, NEST's `stdp_dopamine_synapse`) and Tsodyks-Markram, updated during simulation | `sparx.dynamics.plasticity` |
| Fast weights | differentiable plasticity and Backpropamine (Miconi et al. 2018, 2019): Hebbian traces each sequence writes on a dense or sparse wiring, their plasticity learned by BPTT | `sparx.dynamics.FastWeights`, `sparx.nn.Recurrent(rule=...)` |
| Reward-driven learning | REINFORCE (Williams 1992) for escape-noise neurons; reward diffusion (RNeuralNet-Research 2018), its AGREL-style variants (Roelfsema and van Ooyen 2005) and REINFORCE through the same network | `sparx.learn.reinforce`, `sparx.learn.diffusion`, `RNeuralNetObjective` |
| Local energy minimization | predictive coding and PC-ALM (Seely and Gould 2026): activity relaxed on a layered energy, each weight's update read from its own layer's error | `sparx.learn.predictive`, `PredictiveCodingObjective` |
| Trainable connectomes | FLYNN (Wang and Chen 2026): a rate unit per neuron, recurrent through the connectome's synapses, every weight, bias and class leak trained by BPTT | `sparx.graph.connectome.FLYNN` |
| Fitting to recordings | gradient descent against recorded spikes or rates, by van Rossum distance or smoothed PSTHs | `ActivityFitObjective` |
| Conversion | a trained ReLU network's weights on IF neurons, thresholds balanced at a percentile of activations (Rueckauer et al. 2017) | `sparx.learn.convert` |

A rule whose update is not a loss's gradient hands it to dew's trainer as one (`Objective.with_gradients`), so every rule uses the trainer's one gradient path, with its accumulation, sharding, logging and checkpoints.

Surrogate BPTT through recurrence can explode: training a recurrent network on SHD with `ATan` grew the gradient norm past 1e8 within 300 steps, where `FastSigmoid(100)` kept it below 10. EventProp's exact gradients and the online rules avoid backpropagation through time.

## 8. Models

Models are Flax modules that a run's record names by import path.

- `sparx.nn` layers over time-major arrays `[T, B, ...]`: `LIF`, `IF`, `LI`, `Rate`, `Synaptic`, `ALIF`, `Dynamics` (any model of `sparx.dynamics` as a layer), `Recurrent` (with or without fast weights), the parallel spiking neurons `PSN`, `MaskedPSN` and `SlidingPSN` (Fang et al. 2023), `DelayedDense` with learned delays (Hammouamri et al. 2024), `Flatten`, and `BatchMajor`, which runs a stack on dew's batch-major records.
- `sparx.models`: `SEWResNet` (Fang et al. 2021, as SpikingJelly lays it out) and `SpikingMLP`, a dense network for event data, recurrent or delayed, that both `SpikingClassifierObjective` and `EPropObjective` train.
- `Network` graphs, configured by record (`{"class": "shiu2024", "fields": {"connectome": {...}}}`), and `RNeuralNet`, the graded network of RNeuralNet-Research.

A neuron layer's `dt` is the step in the unit of its time constants, its state streams through the `state` collection, so a sequence fed in chunks gives the output of one call, and it sows its firing rate into `spike_rates` when that collection is mutable.

## 9. dew integration

### 9.1 What sparx uses

| dew piece | Use in sparx |
| --- | --- |
| `Trainer`, `Objective`, `Step`, `Aux`, `Ratio`, `Objective.with_gradients` | every gradient-trained objective, and every learning rule's update handed to the trainer |
| records by import path, `to_record`, `trust=` | a run's record names sparx's models, encoders, objectives and networks, and `dew.pipeline(run_dir, trust=("sparx",))` rebuilds them in a fresh process |
| `RunConfig` run classes, the `dew` CLI | `sparx.config.SNNRunConfig` and `recipes/snn/train.py`; `dew train run.json --trust sparx` continues a run |
| `MeshSpec`, `Layout` | data and fsdp parallel training; trials and neurons of a simulation spread over devices |
| `Checkpoints` | training runs, and a long simulation's state after each chunk |
| `Dataset`, `VALID_ROWS`, `Objective.row_mean` | in-memory records, and validation passes that count every record once |
| `TokenScores`, metrics, `pipeline`, `SavedTask` | evaluation, and `sparx.tasks.SpikingClassification` as the task a run loads as |

### 9.2 Changes dew needs

dew's `main` has every change sparx's earlier stopgaps waited for: records by import path, run classes, schedules and parameter groups, mapping checkpoints, whole validation passes, nested records, the gradient hook, and dew's prose checker with SLOP010. HANDOFF.md lists what is still open in dew for sparx: stateful serving (dew#30), which would make `sparx.serve.StreamServer` dew's server; two exports sparx imports from outside `__all__`; a validation reader on its own; a gradient hook that leaves the rule out of the loss's value; and `Supervised`'s metrics in the validation pass. Changes to dew go through its own pull requests, which the owner merges.

## 10. Data

`sparx.datasets.SHD` is a dew `DatasetSpec`: `shd()` bins the recordings into dense spike counts on a grid of steps, or keeps every event, and both splits are held in memory and streamed by dew's `Dataset`. `mnist()` reads MNIST and Fashion-MNIST as arrays. `holdout` splits off a validation set, and `write_synthetic_shd` writes files in SHD's layout for smoke runs that download nothing. Event datasets beyond SHD, and event augmentations as dew transforms, are not built.

## 11. Fidelity program

Every model, encoder, loss, learning rule and network carries tests at the highest tier available:

1. **Ground truth.** An analytic result (a LIF's f-I curve, a subthreshold response) or a float64 integration at a finer step; `tests/reference.py` writes each dimensionless cell as a float64 NumPy loop from its docstring's equations.
2. **Reference implementation.** The defining simulator or the authors' code, run by a fixture tool under `tools/` at a recorded version or commit, with committed fixtures: NEST, Brian2, snnTorch, SpikingJelly, OTTT's and SNN-delays' PyTorch, PC-ALM's JAX, Miconi et al.'s PyTorch, FLYNN's PyTorch cell, snntoolbox, NIR, Izhikevich's MATLAB, RNeuralNet-Research's C++.
3. **Published behavior.** Values or results a paper reports: Izhikevich's twenty firing patterns, Brunel's regimes, Shiu et al.'s responses to taste stimulation.

Each tolerance sits beside the difference observed when it was set, and each new check ships with a mutation that breaks it. [fidelity.md](fidelity.md) lists every model's references, the check, the observed error and every known difference, with the choice sparx made. Chaotic networks are compared in statistics (rate, irregularity, synchrony) over seeds, since they cannot match spike for spike.

## 12. What is not built

Directions the design leaves room for, which no code holds yet:

- Logical axes on `sparx.nn` layers and in the network's step, so `MeshSpec(tensor=...)` splits wide layers; GPU and TPU measurements, and kernels where they pay (event delivery, bit-packed spikes).
- More neuron families (exponential and quadratic integrate-and-fire, two-variable and conductance-based models beyond these, resonate-and-fire, few-compartment models), homeostatic plasticity, and distance-dependent connectivity.
- Delays and plasticity that read a spike's offset within the step, for timing finer than `dt`.
- Spiking backbones behind dew's other objectives (language models, JEPA, reinforcement learning), which need an adapter from tokens to currents over an inner time axis.
- Rematerializing the time loop in chunks, to train long sequences in less memory.
- Neuromorphic datasets beyond SHD, as dataset specs with event augmentations.

[HANDOFF.md](../HANDOFF.md) orders the open work.

## 13. Decisions taken

1. Sparx takes dew as a required dependency, with Python 3.12 and dew's JAX pin.
2. Spiking objectives, dataset specs and tasks live in sparx, and dew records them by import path; dew's core stays modality-neutral.
3. Changes sparx needs from dew go to dew as its own pull requests.
4. Units are a documented convention ([units.md](units.md)), not a runtime unit system.
5. FlyWire came first, to validate against Shiu et al.'s published runs; the male CNS uses the same builder, its weight calibrated to FlyWire's synapse counts (`matched_w_syn`).
