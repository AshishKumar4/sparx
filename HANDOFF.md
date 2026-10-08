# Handoff

How to pick sparx up. [docs/status.md](docs/status.md) is the one ledger of what sparx supports, what it does not and what is open, in order. `README.md` introduces the library, `docs/guide.md` covers each part with code, `docs/design.md` its architecture, `docs/fidelity.md` every model's reference and check, and `docs/performance.md` the measurements.

## State

- sparx pins dew at `8e184b83` on dew's `main` (`pyproject.toml`), and `tools/lint_slop.py` is dew's file at that commit.
- CI (`.github/workflows/ci.yml`) runs ruff, dew's prose checker, pyright and the CPU suite on every push to `main` and to `ci/**` branches; a head lands on `main` by fast-forward once its run is green.
- dew is the foundation wherever it has the concept. Changes dew needs go to dew as pull requests the owner merges; `docs/status.md` lists the open ones.
- Every commit is authored by Ashish Kumar Singh <ashishkmr472@gmail.com>.

## Working on it

- **Not in the repository.** The owner's research notes, "Complete Conversation: Bio-Inspired Continual Learning" (a .docx they uploaded to the session), which the RNeuralNet work and `research/continual` follow. They are the owner's document and were never committed; ask the owner for them. What was built from them: the RNeuralNet rebuild and its cue-order task, reward diffusion with the all-paths repair, the AGREL variants, the modular core with selective fast weights, and the switch-and-door sessions.
- **Merging.** The owner chose to have finished, tested batches pushed straight onto `main` (`git push origin <branch>:main`, a fast-forward) besides the working branch, with no pull request. dew is the exception: its changes go through pull requests the owner merges.
- **Measurements without a script in the repository.** `docs/tutorials/fit-a-circuit.md` quotes two runs made with scratch scripts. To measure them again:
  - gradient growth through recurrence: populations `in` (50), `e` (400) and `i` (100) of `LeakyIntegrateAndFire(tau_m=20, c_m=250, e_l=-65, v_th=-50, v_reset=-65, t_ref=2, detach_reset=True)` with `ampa` `Exponential(5)` and `gaba_a` `Exponential(10)`, `e` and `i` starting uniform in -65 to -50 mV; `in` to `e` trainable and `in` to `i` fixed, `FixedProbability(0.2)`, 300 pA, 1 ms; `e` and `i` onto both with `FixedInDegree(40)` at 100 pA and `FixedInDegree(10)` at -400 pA, 1 ms; currents `normal(450, 200)` pA into `in` and 250 pA into `e` and `i`; the gradient of the last step's mean `e` membrane with respect to the trainable weights, over 5 to 80 ms, with `ATan()` and `FastSigmoid(100.0)`, seed 0;
  - the voltage fit with firing outputs: the tutorial's circuit with `weights + 100.0`, its outputs' membranes fit from zero weights by the same L-BFGS for 100 iterations.

- **Environment.** A container starts without these; recreate them:
  - `/home/user/.venv` is sparx's, with dew installed editable from `/home/user/dew` (`--no-deps -e` after sparx, since dew's git pin would otherwise conflict). `constraints.txt` pins jax to an archive of AshishKumar4/jax at `19a48d1d`; where the network proxy refuses GitHub archives (403), clone that commit and point a local constraint at the checkout (`jax @ file:///path/to/jax`).
  - The reference environments are locked in `tools/environments` (`tools/references.py lock` writes them): pip lock files with hashes for `torch` (torch 2.14.1+cpu, snnTorch 1.0.0, nir 1.0.8, nirtorch 2.6, dcls 0.1.1 and SpikingJelly's imports), `brian2` (Brian2 2.10.1), `modelfitting` (brian2modelfitting 0.4), `elephant` (1.2.1) and `snntoolbox` (0.6.0 on tensorflow 2.21.0), each installed with `uv pip sync tools/environments/<name>.txt`, and `nest.yml` for conda (NEST 3.10.0). Every fixture tool checks the versions and the reference commits it runs before it computes anything (`tools/references.py`).
  - `tools/regenerate.py <tool> [arguments]` regenerates a tool's fixtures in a copy of the repository and compares them with `tests/fixtures/SHA256SUMS`, which `tests/test_fixtures.py` holds the committed fixtures to; `--update` keeps a deliberate change. The `References` workflow (`.github/workflows/references.yml`, run by hand) does this for every tool whose environment GitHub's runners can install.
  - Reference checkouts live in `/home/user/refs` (differentiable-plasticity, backpropamine, spikingjelly, OTTT-SNN, SNN-delays, pc-alm, fly-gym, RNeuralNet-Research); `tools/references.py` names each one's commit, and each tool's docstring how it takes the path. `tools/make_rneuralnet_fixtures.py` needs g++, and Octave runs Izhikevich's `figure1.m` for `tools/make_izhikevich_2004_fixtures.py`.
  - Their PC-ALM code is plain JAX and runs in `/home/user/.venv` (it needs PyYAML, which dew brings): `PYTHONPATH=<pc-alm checkout> python scripts/run_headline_grid.py ... --data-dir <dir with FashionMNIST/raw/*.gz>`, the IDX files `sparx.datasets.mnist` downloads.
  - The figures and clips (`tools/make_figures.py`, `tools/make_clips.py [--mp4]`, drawn by `tools/drawing.py`) need fontTools and brotli in `/home/user/.venv`, apt's `fonts-inter`, JetBrains Mono from `npm pack @fontsource/jetbrains-mono@5.1.1` unpacked under `/home/user/refs/fonts`, a global Node playwright (1.56.1) with its Chromium, and ffmpeg with libwebp (6.1.1). The training clip downloads MNIST. Simulation and training results are cached in `.cache/figures`; delete it to recompute them. Re-render both after changing the palette or a figure, and look at the light and dark versions at GitHub's width (about 880 px) before committing, each on the other theme's page too: `<picture>` picks a version by the reader's system theme, which can differ from GitHub's, so every figure sits on its own opaque surface (`Canvas.surface`).
- **Keep `/home/user/dew` detached at the pinned commit.** sparx's tests import dew from that checkout, so a branch checked out there changes what they test. Do dew work in separate worktrees (`git worktree add ... /home/user/dew-wt/<topic>`).
- **Data, which tests skip when absent:**
  - FlyWire tables in `../ref-shiu` (github.com/philshiu/Drosophila_brain_model);
  - the male CNS v0.9 release tables in `../data/malecns`, or `$SPARX_MALECNS`;
  - SHD and MNIST download to `~/.cache/sparx`.
- **Gate before committing:**

  ```
  uvx ruff@0.14.3 check src tests tools benchmarks examples recipes research
  python tools/lint_slop.py --package sparx src/sparx tests tools recipes examples benchmarks research
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
