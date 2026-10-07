# Handoff

The state of sparx and its dew work as of 7 October 2026: what exists, what is open, and how to pick it up. `README.md` describes the library, `docs/design.md` its architecture and plan, `docs/fidelity.md` every model's reference and check, and `docs/performance.md` the measurements.

## State

- The suite passes: 551 tests, plus the two whole-brain tests, which run when their data is present. ruff, pyright and dew's prose checker (`tools/lint_slop.py`, dew's file verbatim) are clean. CI runs the same gate and checks that the prose checker is still dew's.
- sparx pins dew at `6329435` on dew's `main` (`pyproject.toml`).
- Every commit is authored by Ashish Kumar Singh <ashishkmr472@gmail.com>.

### What sparx does

- **Training spiking networks.**
  - Flax layers: LIF, ALIF, synaptic, rate, PSN, learned delays, and recurrent layers, fixed or with fast weights (`Recurrent(neuron, rule=...)`).
  - Surrogate gradients.
  - Objectives on dew's `Trainer`: classifier, activity fit, e-prop and predictive coding; any sparx stack also trains under dew's generic `Supervised` through `nn.BatchMajor`.
  - Online rules: e-prop and OTTT.
  - REINFORCE for a recurrent layer of escape-noise (Bernoulli) LIF neurons.
  - One recurrence for every algorithm: `sparx.dynamics.RecurrentCell(inner, wiring, fast_weights)` runs any neuron model over a `Dense` or `Sparse` (connectome) wiring, fixed or with `FastWeights`. The Hebbian rules (differentiable plasticity's decaying trace and Oja's rule, Backpropamine's simple and retroactive neuromodulation) read each connection's units through the wiring, so each runs on a dense layer and on a connectome alike, and a rule of your own is one dataclass and one module.
  - Predictive coding and PC-ALM for any stack of layers, with a dew objective that hands the local update to the trainer.
  - FLYNN, a whole connectome trained as a rate network: a learned weight per synapse, bias per neuron and leak per cell class.
  - EventProp-style exact gradients.
  - ANN-to-SNN conversion of CNNs, checked against snntoolbox.
  - NIR exchange with snnTorch: dense, conv and recurrent.
  - Streaming serving.
- **Simulating biology.**
  - Neurons: LIF, AdEx, Izhikevich (2003 classes and all twenty 2004 patterns), Hodgkin-Huxley, graded-potential neurons and rate units.
  - Synapses: current, conductance and graded synapses; stochastic release; gap junctions; neuromodulators.
  - Plasticity: STDP, triplet STDP, reward-modulated STDP (NEST's `stdp_dopamine_synapse`) and Tsodyks-Markram.
  - Networks with delays and event delivery; `simulate` on dew's mesh and checkpoints; records by name.
  - Connectomes: Shiu et al.'s whole fly brain on FlyWire (reproduced) and on the male CNS (weight calibrated with `matched_w_syn`).
- **One neuron protocol for both halves.** `step(state, SynapticInput, dt) -> (state, Output)` covers ML cells, physical models and graded models. `nn.Dynamics` makes any of them a layer, and any of them can be a `Population`.

### Results worth knowing

- **SHD, Hammouamri et al.'s recipe, 20 epochs, matched on one machine:** sparx 91.87%, against their official code's 93.59% at the last epoch and 94.03% at the best. A training step's gradients agree with theirs to 6e-7. The README lists the differences that remain.
- **FlyWire whole brain:** 1.9 ms per 0.1 ms step on a 4-core CPU.
- **Fast weights:** Miconi et al.'s four plastic networks, run in PyTorch, agree with sparx's in activity, traces and gradients within 5e-14 in float64. On their full pattern completion (1000 bits, five patterns, 2000 episodes, 59 minutes on 4 CPU cores) the decaying trace leaves 0.3% of the zeroed bits wrong and the same network without fast weights 50.1%, chance. A sparse wiring of every pair computes the dense layer under every rule.
- **PC-ALM:** Seely and Gould's JAX reference and sparx agree in settled activity, multipliers and weight updates within 5e-14 in float64. On their headline Fashion-MNIST cell (width and depth 32, one epoch, through dew's trainer) sparx scores 75.1, 76.1 and 76.5% by PC-ALM, 62.2, 65.5 and 65.6% by PC and 77.8, 76.9 and 76.7% by BP at seeds 0 to 2; their code on the same machine scores 77.73, 76.49 and 76.34%, 68.17, 64.49 and 66.54%, and 78.65, 76.85 and 77.31%. The ranking is theirs, and sparx averages 0.5 to 2 points lower in all three, backpropagation included.
- **FLYNN:** Wang and Chen's PyTorch cell and sparx's agree in activity and gradients within 1e-15 in float64. Their code scales the weights by a power iteration whose estimate depends on its random start when the dominant eigenvalues are a complex pair, as a signed connectome's often are (on the parity test's connectome: 1.8 to 46 by start, against an exact 48.1); sparx scales by the exact radius.
- **No GPU or TPU numbers exist yet.**

## dew: the bedrock

The owner's rule: dew is the foundation wherever it has the concept, and sparx dogfoods it. Changes to dew go only through issues and pull requests from branches. Never merge them, and never push to dew's shared branches; the owner merges.

On 7 October 2026 sparx moved from `integration/all` at `306b2bf` to dew's `main` at `6329435`. Every change sparx's stopgaps waited for had landed there, some in the owner's own form, and every stopgap is gone:

- Records name classes by import path, and dew has no plugin registry (`6ac0bac2`). Sparx registers nothing; `sparx.registry` keeps two `Aliases` tables its own code reads (`spike_encoders`, `networks`), and a run loads with `trust=("sparx",)`.
- Recipes are run classes: `sparx.config.SNNRunConfig` builds the run in `prepare`, and `dew train run.json --trust sparx` rebuilds it.
- #37's schedules and parameter groups (`OneCycle`, `Exponential`, `every`, `ParamGroup` with schedule, `b1` and bounds): `sparx.optim` and the objective's `optimizer` override are deleted.
- #38's mapping checkpoints: `simulate` saves `{"state", "key"}`.
- #40's whole validation pass with `VALID_ROWS`: `WEIGHT`, `whole_batches` and `evaluation_pass` are deleted, and the objectives count rows through `Objective.row_mean`.
- #41's nested records and #34's public record writer (`dew.registry.to_record`).
- #43's gradient hook: `EPropObjective` hands e-prop's gradient over through `Objective.with_gradients`, with the memory the `custom_vjp` had (0.56 MB at T=100, 1.20 MB at T=1000, against 7.3 MB for BPTT at T=1000).
- #44's exports and #45's one JSON flag for a mapping of records (`--objective.schedules`).
- #46: dew's `tools/lint_slop.py` takes `--root` and `--package`, and adds SLOP010; sparx's copy is dew's verbatim.
- #36's extra validation splits (`RunConfig.train(..., validation=...)`) and a validation loss that drops the repeats filling the last batch.

This session had no GitHub API access to dew (git reads only), so the issues' and pull requests' own states were not read; the list above is what dew `main`'s code and commit messages show.

Open in dew, for sparx:

| What | Where it helps |
|---|---|
| #30, stateful serving | `sparx.serve.StreamServer` becomes dew's server |
| Export `OMITTED` and `Omitted` from `dew.objectives.base` | an objective's `build_task` override needs the sentinel; sparx imports it outside `__all__` |
| Export `Artifact` from `dew.artifacts` | a metric's `__call__` takes it, as `dew.objectives.base.Metric` declares; sparx imports it outside `__all__` |
| A constructor for a validation reader alone | `Dataset.from_records(records, batch=..., validation=split).val` builds a training stream too, and refuses a source smaller than one batch, so the SNN-delays example pairs its holdout with the training records |
| `Objective.with_gradients` without the rule in the loss's value | `value + vdot(rule, params - stop_gradient(params))` keeps the rule live in the value, so a validation loss computes an update it never applies (PC-ALM: 46 ms a validation batch against 0.3 ms without the rule); a `custom_vjp` whose primal ignores the rule would let XLA drop it |
| `Supervised`'s metrics in the validation pass | the validation pass reports `Supervised`'s loss alone, so `examples/pattern_completion.py` scores its holdout's zeroed bits itself after `fit` |

## Open work, in suggested order

1. **Learning rules from the owner's research notes** (bio-inspired continual learning), each checked against a reference. Done: REINFORCE with Bernoulli neurons (enumerated trajectories), reward-modulated STDP (NEST), fast weights (Miconi et al.'s four networks), PC-ALM (Sakana AI's JAX reference), and the FLYNN trainable connectome (their PyTorch cell). Open:
   - a deterministic reconstruction of RNeuralNet with its reward-diffusion rule as a baseline. No public source by that name was found; the owner's notes should say which paper or code it is.
   - FLYNN on the whole FlyWire connectome: `sparx.graph.connectome.FLYNN` takes `Connectome.from_shiu`'s tables and their cell classes, sensory and descending neurons, but no full-brain training has been run, and their navigation task (MuJoCo) is not ported.
2. **`research/continual/`**, built only on sparx and dew's public API.
   - Start with the small modular core (16 x 256 units), selective fast plasticity (`sparx.nn.Recurrent` with a neuromodulated trace; on a connectome's sparse wiring as well), a BPTT reference and a switch-and-door adaptation task, as the notes recommend.
   - Anything awkward to express there is a gap to fix in sparx or dew.
3. **Remaining DX review items** (the review's numbering):
   - item 6, the name clashes: `LIF` in four places, `Delta`, `Izhikevich`;
   - item 7, a "Time, units and rates" reference page;
   - item 11, e-prop weights into an `nn` model;
   - the documentation plan: tutorials from training to export, a cortical circuit with a Brian2/NEST lookup table, connectomes, mixing the halves, and generated API pages.
4. **Speed benchmarks against Brian2 and NEST** on an idle machine. There are none today; the parity tests only check agreement.
5. **A `docs/design.md` rewrite** to describe the code as it is. Sections 5 and 6 still sketch an older monitor and receptor API.
6. **GPU and TPU measurements** (design phase 7), then kernels where profiling shows they pay: event delivery and bit-packed spikes. A plastic layer's step reads and writes several `[B, F, F]` arrays and backpropagating keeps one trace per step, so its CPU time is memory traffic (1.6 s per episode of 106 steps at F = 1001 on 4 cores); a remat of the step would trade compute for that memory.
7. **Smaller deferred items:**
   - stochastic release should deplete Tsodyks-Markram resources by actual releases;
   - voltage-dependent graded-synapse kinetics;
   - trainable gap-junction weights;
   - logical axes on layers and in the network step;
   - a `voltage()` method on the neuron protocol;
   - fixture regeneration tests;
   - the male CNS left/right MN9 asymmetry (left sugar neurons drive MN9 at 8 Hz, right at 81 Hz), unexplained.

## Working on it

- **Environment.** A container starts without these; recreate them:
  - `/home/user/.venv` is sparx's, with dew installed editable from `/home/user/dew` (`--no-deps -e` after sparx, since dew's git pin would otherwise conflict). `constraints.txt` pins jax to an archive of AshishKumar4/jax at `19a48d1d`; where the network proxy refuses GitHub archives (403), clone that commit and point a local constraint at the checkout (`jax @ file:///path/to/jax`).
  - `/home/user/.venv-ref` holds the simulators: NEST 3.10 and Brian2 2.10, plus torch 2.14.1+cpu (`--index-url https://download.pytorch.org/whl/cpu`; PyPI's wheel pulls CUDA). It regenerates `nest.npz` bit for bit.
  - Neither venv holds elephant, neo and quantities, which `tools/make_elephant_fixtures.py` needs; the elephant fixture was not regenerated in this container.
  - Their PC-ALM code is plain JAX and runs in `/home/user/.venv` (it needs PyYAML, which dew brings): `PYTHONPATH=<pc-alm checkout> python scripts/run_headline_grid.py ... --data-dir <dir with FashionMNIST/raw/*.gz>`, the IDX files `sparx.datasets.mnist` downloads.
  - `/home/user/.venv-torch` holds the PyTorch tools' references: torch 2.14.1+cpu, snnTorch 1.0.0, nir 1.0.8, nirtorch 2.6, dcls 0.1.1, torchvision and SpikingJelly's imports. Every torch-based fixture tool reproduces its fixture bit for bit there, except the order of edges in the `.nir` files (`PYTHONHASHSEED`). snntoolbox needs tensorflow 2.21 and tf-keras in a venv of its own.
  - Reference checkouts live in `/home/user/refs` (differentiable-plasticity, backpropamine, spikingjelly, OTTT-SNN, SNN-delays, pc-alm, fly-gym); each tool's docstring names the commit and takes the path.
  - Octave runs Izhikevich's `figure1.m` for `tools/make_izhikevich_2004_fixtures.py`.
- **Keep `/home/user/dew` detached at the pinned commit.** sparx's tests import dew from that checkout, so a branch checked out there changes what they test. Do dew work in separate worktrees (`git worktree add ... /home/user/dew-wt/<topic>`).
- **Data, which tests skip when absent:**
  - FlyWire tables in `../ref-shiu` (github.com/philshiu/Drosophila_brain_model);
  - the male CNS v0.9 release tables in `../data/malecns`, or `$SPARX_MALECNS`;
  - SHD and MNIST download to `~/.cache/sparx`.
- **Gate before committing:**

  ```
  uvx ruff@0.14.3 check src tests tools benchmarks examples recipes
  python tools/lint_slop.py --package sparx src/sparx tests tools recipes examples benchmarks
  uvx pyright@1.1.406 src/sparx
  JAX_PLATFORMS=cpu pytest tests -q
  ```

  The whole-brain tests take about 9 GB, and the full suite about 17 minutes on 4 CPU cores. On a 15 GB machine, run one suite at a time.
- **Conventions:** dew's `CONTRIBUTING.md` and `AGENTS.md` apply, mirrored in sparx's:
  - one path for each thing;
  - parity tests against the reference implementation, with the observed difference written beside each tolerance, and a mutation shown to break each new check;
  - comments explain why;
  - no `Any`;
  - dew's prose rules.
- **Commits:** as Ashish Kumar Singh <ashishkmr472@gmail.com>, conventional messages, no co-author or tool trailers.
