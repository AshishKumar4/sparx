# Contributing

Sparx follows the same contract as [dew](https://github.com/AshishKumar4/dew/blob/main/CONTRIBUTING.md): a small library where every line earns its place. The rules below are dew's, narrowed to what a spiking library meets.

## Design

- Compose before you write. Look for the primitive first: `jax.lax.scan`, `jax.custom_jvp`, `flax.linen` layers (which already treat leading axes as batch axes), `optax` losses. A reimplementation needs a reason a reader can check, written where the code is.
- One path. A capability has one implementation and one config field. No fallbacks or flags without a demonstrated need.
- The seams are the contract. Neuron dynamics are pure JAX models (`sparx.dynamics`) with one contract, `init_state` and `step(state, SynapticInput, dt)`, that know nothing about Flax layers. Layers (`sparx.nn`) build models from attributes and parameters and run them over time with `sparx.dynamics.run`. Synapses are ordinary Flax layers. Objectives (`sparx.objectives`) own encoding, loss and evaluation; dew's `Trainer` owns everything else. A run's record names every class by import path, so nothing is registered; a class a record names lives at the top level of an importable module.
- Time is the leading axis, `[T, ...]`, everywhere inside a network.
- Frozen at 1.0: parameter names and shapes, the `state` and `spike_rates` collections, cell field order, and the `SpikingClassifierObjective` metric keys. Before 1.0 these change outright, with no compatibility path.

## Reference parity

A neuron, surrogate, encoder or loss that a paper or another library defines is a port, and a port must reproduce the reference.

- The reference is the authors' code or the library that defines it (SpikingJelly, snnTorch), or the equation when no code exists. Match its parameter layout, initialization and operation order.
- A parity test ships with the port. `tools/make_reference_fixtures.py` and `tools/make_snntorch_fixtures.py` run the reference and write small fixtures under `tests/fixtures/`; the tests compare spikes exactly and gradients to a stated tolerance, with the largest observed difference written beside it. A fixture keeps every membrane away from its threshold, so rounding cannot flip a spike.
- A docstring that says "this is library X's Y" is a claim a fixture checks.

## Code

- Keep the smallest correct implementation. Remove dead parameters and unreachable branches.
- Fix the cause. Do not suppress warnings, special-case inputs or fill in zeros as a fallback.
- Types are narrow and true. No `Any`, and no casts to quiet a checker. Narrow Flax's union returns with an `isinstance` assertion that says why it holds.
- Comments say why, never what or what changed. Docstrings describe the code as it is.
- One lint gate, dew's, run from the repository root: `uvx ruff@0.14.3 check src tests tools benchmarks examples recipes && python tools/lint_slop.py --package sparx src/sparx tests tools recipes examples benchmarks && uvx pyright@1.1.406 --pythonpath .venv/bin/python src/sparx`. `tools/lint_slop.py` is dew's checker for what ruff and a type checker cannot state, the file as it is at the dew commit sparx pins (CI compares them); its docstring names every rule and the roots each runs over. Its SLOP010 asks a model for a capability instead of its class: `sparx.nn.Modelled` (a layer's `model`) and `sparx.nn.Flattens` are sparx's. A rule that is wrong for a real reason becomes an ignore in `pyproject.toml` with that reason beside it, never a `# noqa` in `src/`.
- Install with dew's jax: `pip install -e '.[datasets,test]' -c constraints.txt`.
- Measure performance claims. A change that claims to be faster ships with the number, the command and the hardware. A path that is not faster where it can be measured does not ship; [docs/performance.md](docs/performance.md) records what was tried.
- Performance never costs anything else. An optimization matches the outputs and gradients it replaces to fp32 tolerance.

## Tests

A test is worth keeping only if it would fail on a plausible bug in the thing it names.

- Compare dynamics against an independent implementation: `tests/reference.py` writes each cell as a float64 NumPy loop from its docstring's equations.
- Prove the test can fail. A new invariant ships with a mutation that breaks it and the observation that the test then fails.
- Assert the invariants a spiking network promises: running in chunks equals one run, a call without the state collection starts at rest, spikes are exactly binary in every dtype, the membrane stays float32.
- `importorskip` only for an optional dependency (dew, h5py), never for the code under test.
- Run the suite with `pytest -q` on CPU. Device-specific behavior is validated on the device.

## Writing

Plain sentences, short, in the register of someone explaining their own work to a colleague. The README and docs describe what the code does today; a claim without code behind it is a bug. Dew's list of banned constructions applies here unchanged: no colon reveals, no "not X but Y", no puffery, no selling words, no em dashes, no decorative formatting.
