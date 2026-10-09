#!/bin/sh
# As the user, once per environment: sparx's main with its test extras on dew's jax (constraints.txt), and
# bun for the site's engines. run.sh checks out the commit a job names before it runs anything.
set -eu
git clone -q https://github.com/AshishKumar4/sparx.git "$HOME/sparx"
cd "$HOME/sparx"
uv venv -q --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -q -e ".[test]" -c constraints.txt
.venv/bin/python -c "import sparx, nir; print(sparx.__version__)"
curl -fsSL https://bun.sh/install | bash -s "bun-v1.3.2" > /dev/null
"$HOME/.bun/bin/bun" --version
