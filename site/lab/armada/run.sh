#!/bin/sh
# Run a site/lab script at a commit on armada, off the workstation:
#
#     site/lab/armada/run.sh <commit> <label> <script and arguments...>
#
# The script's stdout comes back; a script that writes --out <file> gets armada's {out}.
set -eu
commit=$1; label=$2; shift 2
here=$(dirname "$0")
exec bun "$HOME/armada/src/cli.ts" map --connection="$HOME/.config/armada/armada-dew.json" \
  --env="$here/recipe.json" --size=small --pool=1 --times=1 --output --json --label="$label" -- \
  sh -c "cd \$HOME/sparx && git fetch -q origin $commit && git checkout -q --detach $commit && JAX_PLATFORMS=cpu .venv/bin/python $*"
