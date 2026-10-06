# Handoff

The state of sparx and its dew work as of 6 October 2026: what exists, what is open, and how to pick it up. `README.md` describes the library, `docs/design.md` its architecture and plan, `docs/fidelity.md` every model's reference and check, and `docs/performance.md` the measurements.

## State

- `main` passes the full suite: 478 tests, plus the two whole-brain tests, which run when their data is present. ruff, pyright and dew's prose checker (`tools/lint_slop.py`) are clean. CI runs the same gate.
- sparx pins dew at `306b2bf` on dew's `integration/all` (`pyproject.toml`).
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
- **No GPU or TPU numbers exist yet.**

## dew: the bedrock

The owner's rule: dew is the foundation wherever it has the concept, and sparx dogfoods it. Changes to dew go only through issues and pull requests from branches. Never merge them, and never push to dew's shared branches; the owner merges.

| dew | Status | What sparx does when it merges |
|---|---|---|
| #30 | open issue | Extension points for plugins; its comment proposes a stateful `Server` (`sparx.serve.StreamServer` would become dew's server) |
| #32 | merged | `Objective.pipeline -> Task \| SavedTask`; sparx already uses it |
| #34 | open PR | Public `to_json`: switch `_to_json` imports in `objectives`, the recipe and `test_encode` |
| #37 | open PR | Schedules, per-epoch `every`, param-group schedules, momentum, bounds, coupled Adam decay (#36): delete `sparx.optim` and the objective's `optimizer` override; the recipe and example use `OptimConfig` param groups |
| #38 | open PR | `Checkpoints.save_tree`/`restore_tree` (#35): replace the placeholder `TrainState` in `graph/simulate.py` |
| #39 | open PR | Exported names, `Objective.pipeline_variables`: rename the `_pipeline_weights` call |
| #40 | open PR | Scoring the last partial validation batch: delete `WEIGHT`, `whole_batches` and `evaluation_pass` in `sparx.datasets` |
| #41 | open PR | Nested records through function members: delete the connectome special case in `graph/models.py` |
| #42 | open PR | Refusing a precision setting the model has no field for: sparx models already declare the fields |
| #43 | open issue | A gradient hook on `Objective`, which `EPropObjective` would use instead of its `custom_vjp` |
| #44 | open issue | Export `dtype_name` and `ScheduleSpec` |
| #45 | open issue | A CLI form for a mapping of records (the recipe's `schedules`) |
| #46 | open issue | `lint_slop.py` taking a package argument, so sparx can drop its copy; NumPy object dtype false positive |

Every stopgap in sparx carries a comment `Remove when AshishKumar4/dew#NN merges: ...`. Grep for `dew#` to find them. After a merge, move the pin in `pyproject.toml` to the new `integration/all` commit, delete the stopgaps, and run the full suite.

A second batch of open work in dew's issue #36: item 5 (extra validation splits in `RunConfig.train`), and `val/loss` over the filled last batch, which #40 leaves out.

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
  python tools/lint_slop.py
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
