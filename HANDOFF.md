# Handoff

The state of sparx and its dew work as of 6 October 2026: what exists, what is open, and how to pick it up. `README.md` describes the library, `docs/design.md` its architecture and plan, `docs/fidelity.md` every model's reference and check, and `docs/performance.md` the measurements.

## State

- The suite passes: 479 tests, plus the two whole-brain tests, which run when their data is present. ruff, pyright and the prose checker (`tools/lint_slop.py`) are clean. CI runs the same gate.
- sparx pins dew at `6329435` on dew's `main` (`pyproject.toml`), which holds everything `integration/all` had.
- Every commit is authored by Ashish Kumar Singh <ashishkmr472@gmail.com>.

### What sparx does

- **Training spiking networks.**
  - Flax layers: LIF, ALIF, synaptic, rate, PSN, learned delays and recurrent layers.
  - Surrogate gradients.
  - Objectives on dew's `Trainer`: classifier, activity fit and e-prop.
  - Online rules: e-prop (14x faster than before) and OTTT.
  - EventProp-style exact gradients.
  - ANN-to-SNN conversion of CNNs, checked against snntoolbox.
  - NIR exchange with snnTorch: dense, conv and recurrent.
  - Streaming serving.
- **Simulating biology.**
  - Neurons: LIF, AdEx, Izhikevich (2003 classes and all twenty 2004 patterns), Hodgkin-Huxley, graded-potential neurons and rate units.
  - Synapses: current, conductance and graded synapses; stochastic release; gap junctions; neuromodulators.
  - Plasticity: STDP, triplet STDP and Tsodyks-Markram.
  - Networks with delays and event delivery; `simulate` on dew's mesh and checkpoints; records by name.
  - Connectomes: Shiu et al.'s whole fly brain on FlyWire (reproduced) and on the male CNS (weight calibrated with `matched_w_syn`).
- **One neuron protocol for both halves.** `step(state, SynapticInput, dt) -> (state, Output)` covers ML cells, physical models and graded models. `nn.Dynamics` makes any of them a layer, and any of them can be a `Population`.

### Results worth knowing

- **SHD, Hammouamri et al.'s recipe, 20 epochs, matched on one machine:** sparx 91.87%, against their official code's 93.59% at the last epoch and 94.03% at the best. A training step's gradients agree with theirs to 6e-7. The README lists the differences that remain.
- **FlyWire whole brain:** 1.9 ms per 0.1 ms step on a 4-core CPU.
- **No GPU or TPU n## dew: the bedrock

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
- #46: dew's `tools/lint_slop.py` takes `--root` and `--package`, and adds SLOP010.

This session had no GitHub API access to dew (read-only git), so the issues' and pull requests' own states were not read; the list above is what dew `main`'s code and commit messages show.

Open in dew, for sparx:

| What | Where it helps |
|---|---|
| #30, stateful serving | `sparx.serve.StreamServer` becomes dew's server |
| Export `OMITTED` and `Omitted` from `dew.objectives.base` | an objective's `build_task` override needs the sentinel; sparx imports it outside `__all__` |
| Export `Artifact` from `dew.artifacts` | a metric's `__call__` takes it, as `dew.objectives.base.Metric` declares; sparx imports it outside `__all__` |
| A constructor for a validation reader alone | `Dataset.from_records(records, batch=..., validation=split).val` builds a training stream too, and refuses a source smaller than one batch, so the SNN-delays example pairs its holdout with the training records |

ich #40 leaves out.

## Open work, in suggested order

1. **Learning rules from the owner's research notes** (bio-inspired continual learning), each checked against a reference:
   - REINFORCE eligibility with Bernoulli neurons, verified by enumerating trajectories;
   - reward-modulated STDP (Izhikevich 2007) on the new neuromodulator hook;
   - fast weights (differentiable plasticity and Backpropamine, against Miconi's code);
   - PC-ALM (against Sakana's JAX reference);
   - a FLYNN-style trainable connectome builder;
   - a deterministic reconstruction of RNeuralNet with its reward-diffusion rule as a baseline.
2. **`research/continual/`**, built only on sparx and dew's public API.
   - Start with the small modular core (16 x 256 units), selective fast plasticity, a BPTT reference and a switch-and-door adaptation task, as the notes recommend.
   - Anything awkward to express there is a gap to fix in sparx or dew.
3. **Remaining DX review items** (the review's numbering):
   - item 6, the name clashes: `LIF` in four places, `Delta`, `Izhikevich`;
   - item 7, a "Time, units and rates" reference page;
   - item 11, e-prop weights into an `nn` model;
   - the documentation plan: tutorials from training to export, a cortical circuit with a Brian2/NEST lookup table, connectomes, mixing the halves, and generated API pages.
4. **Speed benchmarks against Brian2 and NEST** on an idle machine. There are none today; the parity tests only check agreement.
5. **A `docs/design.md` rewrite** to describe the code as it is. Sections 5 and 6 still sketch an older monitor and receptor API.
6. **GPU and TPU measurements** (design phase 7), then kernels where profiling shows they pay: event delivery and bit-packed spikes.
7. **Smaller deferred items:**
   - stochastic release should deplete Tsodyks-Markram resources by actual releases;
   - voltage-dependent graded-synapse kinetics;
   - trainable gap-junction weights;
   - logical axes on layers and in the network step;
   - a `voltage()` method on the neuron protocol;
   - fixture regeneration tests;
   - the male CNS left/right MN9 asymmetry (left sugar neurons drive MN9 at 8 Hz, right at 81 Hz), unexplained.

## Working on it

- **Environment:**
  - `/home/user/.venv` is sparx's, with dew installed editable from `/home/user/dew`.
  - `/home/user/.venv-ref` holds the reference tools: NEST 3.10, Brian2 2.10, snnTorch, torch, nir, elephant, DCLS, snntoolbox and tensorflow.
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

  The whole-brain tests take about 9 GB, and the full suite about 12 minutes on 4 CPU cores. On a 15 GB machine, run one suite at a time.
- **Conventions:** dew's `CONTRIBUTING.md` and `AGENTS.md` apply, mirrored in sparx's:
  - one path for each thing;
  - parity tests against the reference implementation, with the observed difference written beside each tolerance;
  - comments explain why;
  - no `Any`;
  - dew's prose rules.
- **Commits:** as Ashish Kumar Singh <ashishkmr472@gmail.com>, conventional messages, no co-author or tool trailers.
