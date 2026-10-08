# Status

What sparx supports, what it does not, and what is open: the one ledger of the project's state. [fidelity.md](fidelity.md) holds each model's reference and observed error, [performance.md](performance.md) the measurements, and the [guide](guide.md#results-in-detail) the results with their commands.

## Capabilities and limits

| Area | Supported and tested | Not supported, or not established |
| --- | --- | --- |
| Reloading a run (`dew.pipeline(run, trust=("sparx",))`) | `SpikingClassifierObjective` and `EPropObjective` runs load as `sparx.tasks.SpikingClassification`, in a fresh process (`tests/test_objectives.py`, `tests/test_recipe.py`) | `ActivityFitObjective`, `PredictiveCodingObjective` and `RNeuralNetObjective` train and checkpoint but declare no saved task, so dew refuses to load them as one |
| Serving (`sparx.serve.StreamServer`) | any stack whose layers carry the `state` collection: `SpikingMLP` (plain, recurrent, delayed), `SEWResNet`, `nn.Sequential` stacks; the model runs with `train=False` and a loaded classifier's `call`; cancelled frames are dropped, a failing step fails its frames (`tests/test_serve.py`) | `PSN`, `MaskedPSN`, readouts averaged over time and `SpikingMLP(extend=True)`, which change or mix the time axis, are refused when the server is built; one host, no mesh |
| NIR export and import (`sparx.nir`) | `nn.Sequential` stacks of dense, convolution and flatten layers, hard-reset `LIF` and `IF` with one threshold and time constant, `LI`, and dense recurrent `LIF`, against snnTorch (`tests/test_nir.py`) | soft reset, adaptive and physical neurons, sparse or plastic recurrence, learned delays; `SpikingMLP` and `SEWResNet` as models |
| Stochastic neurons (`BernoulliCell`) | noise from the caller: `SynapticInput.noise` through `run`, `Serial` and `RecurrentCell`, `Arrivals.noise` through a `PointNeuron`, and `sparx.learn.reinforce` (`tests/test_ml.py`, `tests/test_learn.py`) | a `sparx.nn` layer draws no noise; a `Network` refuses a stochastic population when it is built |
| Exact spike times (`sparx.learn.events`) | feedforward LIF with exponential current synapses; every spike up to `capacity`, the neurons past it counted; gradients against central differences (`tests/test_learn.py`) | recurrence, other neuron models, a gradient that changes spike counts |
| Precision | float32 by default; float64 parity tests under `jax.enable_x64`; bfloat16 synapses with float32 membranes (`dtype`, `param_dtype` and `precision` reach `SpikingMLP`'s and `SEWResNet`'s synapses); e-prop in float64 (`tests/test_models.py`, `tests/test_objectives.py`) | e-prop below float32 is refused; `Recurrent`'s matrix and every neuron parameter stay float32 whatever `param_dtype` is |
| Sharding | training over data and fsdp axes on 8 virtual CPU devices, parameters placed by the axes `SpikingMLP` and `SEWResNet` declare (`tests/test_distributed.py`); a simulation's trials over devices and one network's neurons partitioned (`tests/test_graph_distributed.py`) | tensor parallelism and multi-host runs are untested; GPU and TPU are unmeasured |
| Simulator equivalence | spike for spike with NEST and Brian2, in float64: current-based LIF (to 1e-11 mV), Brian2's `exponential_euler` conductances (`hold="start"`), Izhikevich run op by op (`jax.disable_jit`, to NEST's last bit), and the cortical microcircuit on a network both simulators run edge for edge | conductance LIF at `hold="mean"` is within 2e-3 mV of NEST's RK45; AdEx's spikes within 0.8 ms at ten substeps; compiled Izhikevich within 1 ms a spike; Hodgkin-Huxley within 2e-2 mV with `rk4`; chaotic networks (Brunel, CUBA, COBA, the microcircuit's own draws) agree in statistics over seeds, not spikes ([fidelity.md](fidelity.md)) |
| Connectomes | FlyWire with Shiu et al.'s model reproduces their published rates (`tests/test_connectome.py`, when the tables are present) | the male CNS is calibrated to FlyWire's synapse counts (`matched_w_syn`), not validated; FLYNN is checked on a small connectome, never trained on a whole brain |
| Reference fixtures | every fixture's tool checks the package versions, repository commits, programs and files it reads, in a locked environment (`tools/environments`); `tests/fixtures/SHA256SUMS` holds the committed fixtures, and the References workflow regenerates all 26: on GitHub's runners 20 exactly and 5 to rounding (at most 54 units in the last place, Miconi et al.'s modulated network); the two references that are not deterministic across CPUs (Brian2's Hodgkin-Huxley voltage, snntoolbox's trained network) are held by their parity tests when they differ | regeneration runs on x86-64 Linux only (the torch lock names CPU wheels for it) |
| Speed | CPU comparisons with NEST and Brian2 ([performance.md](performance.md)) | the merged event loop is not shipped (its prototype was lost); the microcircuit takes 8.8 s per simulated second against NEST's 2.9 s; no accelerator numbers |
| Results | short CPU runs, each beside its reference's number ([README](../README.md#results)) | the full-length, multi-seed SHD comparison with the official SNN-delays code is not run |

## Open work, in order

1. **A reference-quality SHD result.** Hammouamri et al.'s recipe at its full 150 epochs over several seeds, beside their official code on the same hardware, with train, validation and test kept apart, raw curves, configuration, commits and versions recorded. It needs GPU time.
2. **Speed.** The merged event loop below; the connectome rows of [performance.md](performance.md) predate the event delivery of 8 October 2026; drawing Brunel's 15.6M synapses takes 12.5 s against NEST's 2.8 s; a step of CUBA costs twice Brian2's.
3. **GPU and TPU measurements**, then kernels where profiling shows they pay: event delivery and bit-packed spikes. A plastic layer's step is memory traffic on CPU (1.6 s per episode of 106 steps at F = 1001 on 4 cores); a remat of the step would trade compute for it.
4. **Research on evidence** (`research/continual`): the notes' size of 16 modules of 256 units, more seeds, a task that changes without announcement, replay and consolidation; FLYNN on the whole FlyWire connectome with its navigation task.
5. **Smaller items:** stochastic release depleting Tsodyks-Markram resources by actual releases; voltage-dependent graded synapses; trainable gap junctions; gradients through stochastic release; a backward pass for event delivery over the edges of neurons near threshold; `Recurrent` and neuron parameters following `param_dtype`; the male CNS's unexplained left/right MN9 asymmetry (8 Hz against 81 Hz); tutorials on connectomes and on mixing the halves, and generated API pages.

New learning rules and neuron families wait until item 1 is done.

### The merged event loop

The microcircuit at a fifth carries about 4 spikes a step over 55 projections, each running its own event loop (`sparx.graph.delivery.deliver_events`). One loop per step for every event projection without stochastic release or short-term plasticity was prototyped and lost:

- **Which projections.** Those `_events(p, pre)` takes with `p.release is None` and `p.short_term is None`. A frozen `_Merged` dataclass, built from the projections and populations alone, lists them, gives each source population its first neuron in one range laid end to end and each `population:receptor` it feeds its first column, and takes as its pass width the widest `per_pass` among them, capped by the neuron count.
- **Storage.** In `_build`, draw every projection as now (the draws must stay in order, or the NEST fixtures break), then concatenate the merged ones' edges: neuron plus source offset, post plus column offset, a row lag of `delay` (`delay - 1` onto a delta receptor), the weight, and the projection's index. Sort them with `event_edges` (give it the per-edge index to sort along) and keep them in the connectome as `events`; the merged projections get no `edges` entry and no `weight:` variable. Store the lag as one scalar when all lags are equal, and let `_landing` take a scalar delay.
- **State.** One buffer `pending["events"]`, `[longest lag + 1, columns]`, and `ready["events"]`, its row, when any receptor is a delta one. The key holds no colon, so no `population:receptor` key can equal it.
- **Step.** In `send`, clear row `t - 1` of the buffer, concatenate the step's outputs of the source populations, and deliver with `deliver_events(..., t=t)`. At the end of the step every merged receptor reads row `t`, a kinetic receptor as its arrivals and a delta receptor as next step's `ready`. Skip merged projections in `_lags`, `_pending`, `gather`'s per-projection delivery, the weights and `_Stepper.ahead`.
- **Reading back.** `Network.connections` selects each projection's edges from `events` by the stored index and subtracts the offsets; a delta receptor's delay is its lag plus one.
- **Tests.** A network of two populations with five event projections onto kinetic and delta receptors, with one delay and delays per edge, fired spike for spike with all projections as `format="edges"`, in one chunk and in chunks of 30 ms, and read back the same connections. Dropping the delta lag shift, reading `ready` a row late, or storing every edge under projection 0 each failed it.
- **Measured** with the prototype (4-core CPU, float32, `dt = 0.1` ms, 1 simulated s after 300 ms of warm-up, single runs about 5% apart), in s per simulated second; every pair fired at the same rate:

  | Network | Per projection | Merged, by pass width |
  | --- | --- | --- |
  | Microcircuit, a fifth | 9.84 (width 4) | 3.78 (4), 3.53 (8), 4.11 (16) |
  | CUBA | 0.89 (16) | 0.60 (8), 0.62 (16), 0.78 (32) |
  | COBA | 1.11 (16) | 0.87 (8), 0.77 (16), 0.84 (32) |
  | Brunel, 12,500 neurons | 10.01 (16) | 10.60 (16), 9.65 (32), 9.95 (64) |

  After it lands, re-run `benchmarks/bench_networks.py` and update [performance.md](performance.md), the widths in `Projection.per_pass` and the microcircuit's `per_pass=4` in `sparx.graph.models`.

## dew

sparx uses dew wherever dew has the concept; changes dew needs go to dew as pull requests the owner merges ([design.md](design.md#9-dew-integration)). Open for sparx:

| What | Where it helps |
| --- | --- |
| `Objective.row_weights`, a batch's real rows as weights | a learning rule weights rows itself (`sparx.objectives._row_weights`) |
| `Supervised`'s metrics in the validation pass | `examples/pattern_completion.py` scores its holdout's zeroed bits itself after `fit` |
| Logical-axis declarations scoped to the class that declares them | sparx's module names (`readout`, `dense_0`, ...) declare axes for any model in the process with a module of that name |

Landed in dew `b255a88d` and taken here: `Objective.with_gradients` leaves the rule out of a value-only pass, `Dataset.validation` reads a held-out split alone, and `OMITTED`, `Omitted`, `Artifact` and `logical_axes` are exported.

Stateful serving in dew (dew#30) was declined and the issue closed: dew's server overlaps dispatch and token draws, which synchronous spiking frames do not need, so `sparx.serve.StreamServer` stays sparx's.
