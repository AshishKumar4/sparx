# Working on Sparx

Read `CONTRIBUTING.md` for design, reference parity, code, tests and writing standards. This checklist adds agent workflow rules.

- Read the affected module, its tests and the cell or layer it builds on before editing. Check JAX, Flax and optax for an existing primitive.
- New dynamics go into `sparx.dynamics` first (`ml.py` for the dimensionless family, `neurons.py` for physical models), as a struct dataclass with `init_state` and `step(state, SynapticInput, dt)`, with a reference in `tests/reference.py` or a fixture, and the chunked-run invariant in `tests/test_ml.py`. The Flax layer in `sparx.nn` only builds the model.
- A port of another library's neuron, encoder or loss extends the fixture tools under `tools/` and is checked by a parity test. Record the reference commit or version, the observed error and the tolerance.
- Commit as `Ashish Kumar Singh <ashishkmr472@gmail.com>` with a concise conventional commit message. Never force-push.
- Run the affected test files, then `pytest -q`, ruff and pyright before a commit. A pipeline's last command succeeding does not prove the tests passed.
- Report benchmark conditions and commands. Do not claim GPU or TPU behavior that was not run on one.
