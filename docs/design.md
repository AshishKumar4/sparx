# Sparx design

Status: proposal, 2026-10-04. This document describes the framework sparx is meant to become and how it gets there. Sections marked "today" describe code that exists; everything else is design. Claims about other projects that have not yet been checked against their code are marked "to verify".

## 1. What sparx is for

Sparx is dew's spiking and biophysical modality. It serves two workloads with one representation:

1. **Spiking networks as machine learning.** Deep spiking networks trained on event and static data, spiking backbones inside any dew objective (classification, sequence modeling, JEPA, diffusion, RL), and deployment to streaming inference and neuromorphic hardware.
2. **Brains as networks.** Building, simulating and fitting biological circuits: randomly connected cortical models, region models, and whole connectomes such as FlyWire (Dorkenwald et al. 2024) and MaleCNS v1.0 (Janelia FlyEM and Google Research, 2026; about 166,700 neurons and 125 million synapses across brain, optic lobes and ventral nerve cord), with neuron and synapse models as close to the biology as the question needs.

The same neuron, synapse and connectivity objects serve both. A deep network is a graph whose projections happen to be dense and stacked; a connectome is a graph whose projections are sparse, recurrent and delayed. Training, distribution, checkpoints, tracking and serving come from dew.

Non-goals: a second trainer, a second checkpoint format, a runtime unit system, morphologically detailed compartmental modeling at NEURON's level of detail (multi-compartment point models are in scope; full cable reconstructions from SWC files are not, at first).

## 2. Principles

Sparx inherits dew's six rules unchanged (`dew/docs/design/api.md`):

1. Everything extensible has one `Registry`; sparx registers into dew's registries rather than keeping its own.
2. Values that cross `jit` are `flax.struct.dataclass`es; configuration is a frozen dataclass; interchangeable implementations are a `Protocol`.
3. Randomness is a `key` argument.
4. Effects live in capabilities handed to the trainer or simulator (`Checkpoints`, `Tracker`).
5. An unknown name or field raises.
6. A schedule is a value.

It adds six of its own:

7. **dew is the platform.** Sparx adds dynamics, structure and objectives. It never adds a trainer, a mesh, a checkpoint format, a data loader framework, a launcher or a server; where dew lacks an extension point sparx needs, the change goes into dew (section 9.2).
8. **The science decides, references are evidence.** Each model is checked against ground truth where one exists (an analytic solution, or a float64 integration at a much finer step), then against its reference implementation. Where a reference departs from the science, sparx follows the science and records the departure in a test and in the fidelity ledger (section 11). Optimized implementations must agree with the plain one to stated tolerance.
9. **Time is explicit.** Every dynamical object knows its step `dt` and its time constants in physical units. The deep-learning convention (`dt = 1`, time constants in steps) is the same code with `dt = 1`.
10. **Structure is a graph.** Populations and projections are first-class. Layer stacks are the dense special case and keep a direct fast path.
11. **Parameters, structure and state are separate collections.** What trains, what is fixed wiring, and what evolves in time live in different Flax collections, so the optimizer, the sharding layout and the checkpoint each see exactly their part.
12. **Fast paths are implementations of a protocol, chosen by measurement.** A sparse kernel, a fused time loop or a parallel scan is admitted when it is faster on the hardware it targets and matches the reference path; [performance.md](performance.md) records what was measured.

## 3. Architecture

```
dew (platform)    Trainer . MeshSpec/Layout . Checkpoints . Dataset/Grain . Tracker . Profiler . registry . recipes/CLI/launch . pipeline . Server
                    ^ objectives, datasets, models, tasks register here
sparx.dew         objectives (classification, sequence, activity fitting, online learning), dataset specs, tasks, recipes
sparx.learn       surrogate gradients . exact event gradients (EventProp) . online rules (e-prop, OTTT) . local plasticity
sparx.sim         simulate() . monitors . step order . delay buffers . connectivity kernels . sharding rules
sparx.graph       Population . Projection . Connectivity . Network (a Flax module) . builders (random, spatial, connectome)
sparx.nn          layers over time-major tensors: the dense fast path (today)
sparx.dynamics    neuron models . synapse models . plasticity models . integrators . spike detection (today: sparx.cells)
```

Each layer depends only on the ones below it. `sparx.dynamics` has no Flax dependency and can be used from plain JAX; everything above it is Flax and dew.

## 4. Dynamics

### 4.1 The model contract

A dynamical model is a struct dataclass of its parameters (leaves, so they can be learned, swept with `vmap`, and sharded) and its choices (static fields). It declares:

```python
class NeuronModel(Protocol[State]):
    def init_state(self, shape, dtype) -> State: ...
    def step(self, state: State, inputs: SynapticInput, dt: float) -> tuple[State, Spikes]: ...

@struct.dataclass
class SynapticInput:
    current: jax.Array                 # pA, summed current-based input
    conductance: Mapping[str, jax.Array]   # nS per receptor ("ampa", "nmda", "gaba_a", ...)
```

Neuron models read conductances together with their own reversal potentials, so conductance-based input is computed against the neuron's voltage at the step, not approximated as a current. A deep-learning layer passes only `current`.

Today's cells (`LIFCell`, `ALIFCell`, ...) become these models with `dt` explicit. Their numerics are already the exact exponential solution of the leak for `dt = 1`, so existing behavior is preserved.

### 4.2 Integration

Each model names how it is integrated, and the choice is part of its tested contract:

| Method | Used for | Why |
| --- | --- | --- |
| Exact (matrix exponential) | linear subthreshold dynamics: LIF, current-based exponential and alpha synapses | No integration error; the reference simulators NEST (Rotter and Diesmann 1999) and Brian2 (exact for linear equations) do this. Euler at `dt = 0.1 ms` is not exact, and several ML libraries use it. |
| Exponential Euler | conductance-based LIF, AdEx | Stable for stiff leak terms |
| Rush-Larsen | Hodgkin-Huxley gating variables | Exact for each gate given the voltage, stable at larger `dt` than Euler |
| RK4 | small nonlinear models where accuracy matters more than cost | Ground-truth runs |
| Float64 at a fine step | test oracles only | The ground truth every method is compared with |

Spike detection is a threshold crossing within the step. The exact crossing time is recovered by linear interpolation (Hansel et al. 1998; Morrison et al. 2007) and carried with the spike when a downstream consumer needs it (event-based gradients, STDP timing, precise delays). A model with refractoriness holds its state for `t_ref` from that time.

### 4.3 Catalogue

Each entry has a reference: equations from the paper, and an implementation to compare with.

| Family | Models | Reference implementation |
| --- | --- | --- |
| Integrate-and-fire | IF, LIF (current and conductance based), ALIF / GLIF adaptations, QIF, EIF, AdEx (Brette and Gerstner 2005) | NEST, Brian2; SpikingJelly and snnTorch for the ML variants |
| Two-variable | Izhikevich 2003 (the 20 firing patterns of Izhikevich 2004 as a test set), FitzHugh-Nagumo, Morris-Lecar | Izhikevich's published code; Brian2 examples |
| Conductance-based | Hodgkin-Huxley (squid axon), Wang-Buzsaki, Traub-Miles | Brian2, NEURON for single compartments |
| Resonant | resonate-and-fire, balanced resonate-and-fire (Higuchi et al. 2024) | the authors' code |
| Parallel (ML) | PSN family (Fang et al. 2023) | SpikingJelly (parity tested today) |
| Multi-compartment | two- and few-compartment models (soma plus dendrite, Pinsky-Rinzel) | Brian2 |

Synapse models are separate from neuron models (today `SynapticCell` merges them; it splits):

| Kind | Models | Where its state lives |
| --- | --- | --- |
| Linear | delta, exponential, alpha, bi-exponential, per receptor | Aggregated on the postsynaptic neuron, one state per receptor: linear synapses sum, so `N` states replace `E`. This is what makes connectome scale affordable. |
| Nonlinear in the postsynaptic voltage | NMDA with magnesium block (Jahr and Stevens 1990) | Aggregated conductance, nonlinearity applied at the neuron |
| Presynaptic dynamics | short-term plasticity (Tsodyks and Markram 1997) | Per presynaptic neuron and projection |
| Plastic | pair and triplet STDP (Pfister and Gerstner 2006), three-factor rules with eligibility traces, homeostatic scaling | Per edge (weights) plus per-neuron traces |
| Gap junctions | electrical coupling | Per edge, symmetric |

### 4.4 Units

Units are a documented convention, not a runtime system: time in ms, voltage in mV, current in pA, conductance in nS, capacitance in pF. Model fields carry the unit in their docstring and in a `units` table that a builder checks when it reads values from a dataset (a connectome table, a parameter file). A runtime unit system such as Brian2's does not trace through `jit`, and its checks belong at construction, where the builder already validates.

## 5. Structure

### 5.1 The graph

```python
@dataclass(frozen=True)
class Population:
    name: str
    size: int
    model: NeuronModel          # one model per population; heterogeneity lives in its parameter leaves

@dataclass(frozen=True)
class Projection:
    pre: str
    post: str
    connectivity: Connectivity  # which pairs connect
    weight: WeightSpec          # initial values, sign, trainable or fixed
    delay: DelaySpec            # per edge, in ms, quantized to dt; learnable per edge when asked
    receptor: str = "current"   # or "ampa", "gaba_a", ...
    synapse: SynapseModel = Exponential(tau=5.0)
    plasticity: Plasticity | None = None

class Network(nn.Module):       # a Flax module, so dew trains, shards and checkpoints it
    populations: Sequence[Population]
    projections: Sequence[Projection]
    inputs: Mapping[str, str]   # external input name -> population
    outputs: Sequence[str]      # populations whose spikes or voltages are returned
```

A `Network`'s variables are split into collections:

| Collection | Holds | Seen by |
| --- | --- | --- |
| `params` | trainable weights, delays, time constants | the optimizer |
| `connectome` | fixed structure: edge indices, signs, receptor types, fixed weights | the layout and checkpoints, never the optimizer |
| `state` | membranes, synaptic states, delay buffers, plasticity traces, plastic weights | carried across calls (today's streaming contract) |
| `spike_rates`, `monitors` | sown observations | trackers and losses |

Plastic weights that change during a simulation by a local rule are state, not params: they evolve with the dynamics. A run can still train their initial values or the rule's coefficients by gradient.

### 5.2 Connectivity

```python
class Connectivity(Protocol):
    def edges(self, key, pre_size, post_size) -> EdgeList: ...   # used once, at init
```

Generators: all-to-all, fixed in-degree, fixed probability (Erdős-Rényi), distance-dependent on population positions, and `FromTable`, which reads edge lists. Connectome builders (`sparx.graph.connectome`) read neuron and connection tables (FlyWire Codex exports, neuPrint for MaleCNS) and map neurotransmitter predictions to signs and receptors. Shiu et al. (2024) used acetylcholine as excitatory, GABA and glutamate as inhibitory, and the modulators as excitatory; that mapping is the default and a field, since the glutamate assignment is known to be receptor dependent in the fly.

At init, edges are compiled into an execution format (section 6.2). Dense and convolutional projections keep their own fast path, and the existing `sparx.nn` layers are those paths with a layer interface.

### 5.3 Canonical networks

Shipped as builders and as validation targets:

- Brunel (2000) balanced excitatory-inhibitory network, with its asynchronous-irregular, synchronous-regular and oscillatory regimes.
- The COBA and CUBA networks of Vogels and Abbott (2005), the benchmarks of Brette et al. (2007) on which simulators are compared.
- The Shiu et al. (2024) whole-brain LIF model on FlyWire, and the same construction on MaleCNS v1.0.

## 6. Execution

### 6.1 One step

Every network advances in the same order, the order NEST uses, which the single-neuron and network tests pin against NEST and Brian2 (`tests/test_simulators.py`, `tests/test_graph.py`). A step covers `(t, t + dt]`:

1. Delta synapses deliver the spikes due at the end of the step as voltage jumps.
2. Each population advances its membranes on its synapses' output (current waveforms integrated exactly, conductances held) and detects spikes.
3. The new spikes enter each population's ring buffer.
4. Every other synapse receives the spikes due at the end of the step, which shape the membrane from the next step on.
5. Plasticity updates traces and weights.
6. Monitors record.

A spike sent in step `m` over `D` steps is due at the end of step `m + D`: NEST's timing for a delay of `D dt`, Brian2's for `(D - 1) dt`. Kinetic synapses take `D = 0` (Brian2's default); delta synapses need `D >= 1`, since a jump due in its own step would feed back into that step's threshold test.

### 6.2 Connectivity kernels

Projections compile to one of these execution formats, chosen per projection by size and density, all producing the same input to fp32 tolerance:

| Format | Cost per step | Fits |
| --- | --- | --- |
| Dense matmul | `pre x post` | layers, small dense projections, accelerators' matrix units |
| Convolution | per kernel | spatially structured layers |
| Edge list with `segment_sum` by postsynaptic index | `E` | sparse recurrent graphs, connectomes |
| Event list, capacity-bounded | `active x fan-out` | very sparse firing; a fixed capacity keeps shapes static, and overflow is detected and raises |

Delays use a ring buffer of the last `D` steps of spikes per population, stored as bits. Each edge reads its presynaptic neuron at its own delay. Per-edge delays cost one gather per edge per step; projections with one delay per projection read one slice.

### 6.3 Scale

MaleCNS at full size has about 166,700 neurons and 125 million synapses, which collapse into fewer unique neuron pairs with a synapse count each (Shiu et al. use the count as the weight). An edge needs a presynaptic index (int32), a weight (float32 or bfloat16) and a delay (uint8): about 9 bytes, so the full synapse list is about 1.1 GB and the pair list smaller. One accelerator holds it. Larger graphs and many simultaneous trials shard:

- Neurons are partitioned over a mesh axis; each edge lives with its postsynaptic neuron's shard.
- Each step, shards exchange their new spikes as a bitmask (166,700 neurons is 21 KB), then compute their own inputs locally.
- Independent trials (stimuli, parameter sweeps) run over the data axis.

Sparx declares logical axis names (`neurons`, `edges`, `trials`) and adds their rules to dew's `Layout`, so dew's `MeshSpec` places a simulation the same way it places a model.

### 6.4 `simulate`

Simulation without gradient training is a function, not a new runner, so it adds no noun beside dew's `Trainer`:

```python
result = sparx.simulate(
    network, variables, inputs,        # inputs: external drive, [T, trials, ...] or a function of time
    duration=1000.0, dt=0.1, key=key,
    monitors=[Spikes("mn9"), Rate("all", window=50.0), Voltage("gustatory", every=10)],
    mesh=MeshSpec(fsdp=8), checkpoints=Checkpoints("runs/fly"),   # dew capabilities
    chunk=1000.0,                      # ms per compiled chunk; state and monitors stream to the host between chunks
)
```

It compiles one chunk of the time loop, carries state between chunks, streams monitors to the host so memory stays bounded for long runs, and writes dew checkpoints of the state so a long run resumes. It shares the step function with training.

## 7. Learning

| Regime | How | Lives in |
| --- | --- | --- |
| Surrogate-gradient BPTT | `custom_jvp` spikes (today), with dew's remat policies over time chunks for memory | `sparx.learn.surrogate` |
| Exact event-based gradients | EventProp (Wunderlich and Pehle 2021): adjoint dynamics over spike times; needs the in-step spike times of 4.2 | `sparx.learn.eventprop`, a custom VJP of the time loop |
| Forward and online learning | forward-mode gradients through the `custom_jvp`; e-prop (Bellec et al. 2020) and OTTT (Xiao et al. 2022) as eligibility-trace updates every step, memory independent of `T` | `sparx.learn.online`, with a dew objective that updates every chunk |
| Local plasticity | STDP and three-factor rules as state updates during simulation | `sparx.dynamics.plasticity` |
| Fitting to recordings | gradient descent on network parameters against recorded spikes, rates or voltages, with spike-train distances (van Rossum 2001, Victor-Purpura 1996) and PSTH losses | `sparx.dew.ActivityFit` |
| Conversion | trained ANN weights mapped to an IF network with threshold balancing | `sparx.learn.convert` |

The recurrent gradient explosion measured on SHD (the `Recurrent` docstring) is a property of surrogate BPTT through recurrence. EventProp gives exact gradients and the online rules avoid backpropagation through time, so they are the principled alternatives to tuning the surrogate.

## 8. Models

Models are Flax modules registered in dew's model registry:

- `sparx.nn` layer stacks: MLP and convolutional SNNs, SEW-ResNet (today), spiking self-attention (Spikformer, Zhou et al. 2023), recurrent and delayed networks for temporal data (today).
- `Network` graphs: canonical circuits and connectomes, configured by record (`{"connectome": "malecns-1.0", "model": "shiu2024", ...}`) so a run rebuilds them.
- Spiking backbones for dew's objectives. A dew objective expects a model interface, for example a decoder from tokens to logits `[B, S, V]`. An adapter embeds inputs as currents over an inner time axis, runs the spiking network time-major, and reads the output back into the interface's layout. `LMObjective`, `JepaObjective` and the RL objectives then train spiking models unchanged.

## 9. dew integration

### 9.1 What sparx uses

| dew piece | Use in sparx |
| --- | --- |
| `Trainer`, `Objective`, `Step`, `Aux`, `Ratio` | all gradient training; sparx objectives subclass `Objective` (today: `SpikingClassifier`) |
| `MeshSpec`, `Layout`, logical axes | data, fsdp and tensor parallel training of layers; neuron-partitioned simulation |
| `Checkpoints` (Orbax), preemption handling | training and long simulation runs |
| `Dataset`, Grain sources and transforms | neuromorphic datasets as dew dataset specs, with event-level augmentation as Grain transforms and resumable, sharded reading |
| registry, recipes, `RunConfig`, `dew` CLI, `dew launch` | `recipes/snn/*.py`, launched on GPUs and TPU pods the same way as dew's |
| trackers, `Profiler`, telemetry | firing rates and sparsity as metrics; XProf for kernel work |
| `dew.pipeline`, `Server` | loading trained spiking models and serving streaming sessions |

Sparx takes dew as a required dependency. The optional `sparx[dew]` extra goes away, and `sparx.dew` becomes the place where objectives, dataset specs, tasks and registrations live.

### 9.2 Changes dew needs

These are small, general extension points, each useful to dew beyond sparx:

1. **Registry plugins.** dew's registry finds members by scanning dew's own sources, so a `run.json` naming a sparx model cannot be rebuilt in a process that has not imported sparx. Add discovery through a `dew.plugins` entry-point group: a package lists its registering modules, and a lookup miss imports them.
2. **Open artifact types.** `Artifact` is a closed union. Spiking evaluation needs activity artifacts (rasters, rates, traces) that metrics read. Make it a registered protocol.
3. **Open inference tasks and stateful serving.** dew's `Server` keeps per-slot KV caches for token generation. A spiking session keeps per-slot neuron state and advances by input chunks. Generalize the slot to "state the task declares", with KV caches as one implementation, so sparx's streaming task reuses admission, batching and mesh placement.
4. **Layout rules from plugins.** Let a plugin contribute logical-axis rules (`neurons`, `edges`, `trials`) to `DEFAULT_RULES`, or confirm that passing `Layout(rules=...)` covers it (to verify).

### 9.3 Distributed training

- Data parallel over batch rows is the default; time-major activations keep the batch on axis 1, and sparx layers constrain it to dew's batch axes.
- Weights of wide layers and large projections declare logical axes, so `MeshSpec(fsdp=..., tensor=...)` splits them as it does dew's transformers.
- Time cannot be split across devices for recurrent dynamics, since each step needs the last. The PSN family and linear layers can split time, and sequence parallelism applies to them.
- Memory for long sequences comes from rematerializing the time loop in chunks (a checkpointed scan), configured like dew's remat policies.

### 9.4 Serving and deployment

- `objective.pipeline(state)` returns a sparx task: classification over a whole input, or a streaming session that advances by chunks and keeps neuron state between calls (the `state` collection, which already makes chunked runs exact).
- The served form runs on dew's generalized `Server` (9.2.3), one slot per session.
- Neuromorphic hardware and other libraries: export and import through NIR, the Neuromorphic Intermediate Representation (Pedersen et al. 2024), which snnTorch, Norse, SpikingJelly, Lava, SpiNNaker and others read and write (to verify per target). NIR's primitives map onto populations, projections and the LIF and CUBA models.

## 10. Data

Neuromorphic datasets become dew dataset specs over Grain: SHD and SSC (today, in memory: SHD), N-MNIST, DVS Gesture, CIFAR10-DVS, plus recorded neural activity for fitting. Events are stored sparse and binned by a transform, so the bin size is a training choice rather than a preprocessing decision. Augmentations act on events (time jitter, channel shift, event drop), as Grain random transforms keyed per record.

## 11. Fidelity program

### 11.1 Tiers

Every model, encoder, loss and network carries tests at the highest tier available:

1. **Ground truth.** An analytic result (LIF's f-I curve; subthreshold response to a step) or a float64 integration at a step ten times finer.
2. **Reference implementation.** The defining library or the authors' code, run by a fixture tool under `tools/`, with committed fixtures (today: SpikingJelly and snnTorch).
3. **Published behavior.** Values or qualitative results a paper reports: the Izhikevich firing patterns, Brunel's regime diagram, Shiu et al.'s predicted responses to taste stimulation.

### 11.2 The ledger

`docs/fidelity.md` lists, for each model, its references, every known difference between a reference and the science or another reference, the choice sparx made, and the test that pins it. Differences already found or suspected:

| Item | Reference behavior | Sparx choice | State |
| --- | --- | --- | --- |
| LIF integration | several ML libraries step the leak with Euler, `v += dt / tau * (-v + x)` | exact exponential decay, as NEST and Brian2 do | today, to document |
| `Synaptic` reset timing | snnTorch may apply the reset with the previous step's spike | reset on the step that fires | to verify against snnTorch |
| ALIF input scaling and reset | Bellec et al.'s code scales input by `1 - alpha` and resets against the baseline threshold | to be decided against the paper's equations and their code | to verify |
| Izhikevich integration | the 2003 code takes two half-steps of 0.5 ms for `v` and one step for `u`, and clips the spike peak at 30 mV | reproduce the published code as one model, and offer an RK4-integrated variant labeled as such | to verify |
| Delays | many ML libraries have none | per-edge delays in every projection | design |
| Spike timing | clock-driven simulators place spikes on the grid | in-step crossing times when a consumer needs them | design |

### 11.3 Network validation

Sparx's networks are compared with Brian2 or NEST on the Brette et al. (2007) benchmarks and on the Brunel network: population rates, coefficient-of-variation and correlation statistics agree within stated bounds, since chaotic networks cannot match spike for spike. The whole-brain model is compared with the Shiu et al. code on their published stimulations.

## 12. What changes in today's code

| Today | Becomes |
| --- | --- |
| `sparx.cells` with `dt = 1` | `sparx.dynamics` with explicit `dt` and units; the `dt = 1` behavior is unchanged and tested equal |
| `SynapticCell` (synapse and neuron in one) | an exponential synapse model plus an LIF neuron |
| `sparx.dew` optional, `SpikingClassifier` only | required; registered objectives, dataset specs and tasks |
| SHD loaded into memory | a Grain dataset spec |
| layers without logical axes | layers with logical axes for dew's `Layout` |
| `docs/performance.md` CPU numbers | extended with accelerator numbers before any accelerator-specific path ships |

## 13. Phases

Each phase ends with its acceptance tests passing, on the hardware they name.

| Phase | Work | Accepted when |
| --- | --- | --- |
| 0. Platform | dew registry plugins and layout rules from plugins (dew PRs); sparx takes dew as a required dependency; registrations; SHD as a Grain dataset spec; `recipes/snn/train.py` | a sparx run written by the recipe is rebuilt by `dew.pipeline(run_dir)` in a fresh process, and trains on a simulated 8-device mesh with parity to one device |
| 1. Fidelity of what exists | `docs/fidelity.md`; fixtures for snnTorch `Synaptic` and `Leaky`, Bellec's ALIF, Izhikevich's code, DCLS-Delays, SpikingJelly's SEW-ResNet weights | every row of the ledger has a test; differences fixed or named |
| 2. Dynamics | `sparx.dynamics` with `dt`, units and integrators; synapses split from neurons; AdEx, HH, conductance-based LIF, STP, STDP | ground-truth tests per model; Izhikevich 2004 pattern set; Brian2 parity on single neurons |
| 3. Graph and simulation | `Population`, `Projection`, `Network`, connectivity kernels, delay buffers, `simulate`, monitors | Brunel regimes and the COBA and CUBA benchmarks agree with Brian2; layer stacks give the same result through `Network` and `sparx.nn` |
| 4. Connectome | FlyWire and MaleCNS builders; Shiu et al. model | their published predictions reproduced, compared with their code; memory and time per simulated second measured on one GPU |
| 5. Learning | EventProp, e-prop, OTTT, activity fitting | gradients against finite differences of the exact dynamics where defined; published task results reproduced |
| 6. Scale and serving | neuron-partitioned simulation over a mesh; dew `Server` generalization; streaming task; NIR export | multi-device simulation equals one device; a served session equals a direct streaming call; NIR round trip with one other library |
| 7. Accelerator performance | GPU and TPU measurements; Pallas time-loop kernels where measurement justifies | numbers in `docs/performance.md`, each path equal to the reference path |

Phases 0 and 1 come first because every later phase builds on dew's extension points and on knowing which existing models are right.

## 14. Decisions to confirm

1. Sparx takes dew as a required dependency, with Python 3.12 and dew's JAX pin. (Recommended.)
2. Spiking objectives, dataset specs and tasks live in sparx and register into dew, rather than living in dew. (Recommended: dew stays modality-neutral at its core, as with its own objectives per modality.)
3. The four dew changes in 9.2 go into dew as their own PRs.
4. Units are a convention checked at construction, not a runtime unit system.
5. The first connectome target is MaleCNS v1.0 or FlyWire. FlyWire has the Shiu et al. model to validate against; MaleCNS is newer and larger. (Recommended: FlyWire first for validation, MaleCNS second with the same builder.)
