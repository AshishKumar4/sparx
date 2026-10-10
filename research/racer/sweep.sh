#!/bin/sh
# Train each configuration of a sweep on armada, at the current commit (pushed), one container each:
#
#     research/racer/sweep.sh <config.json> <label> <out dir> [pool]
#
# <config.json> lists site/lab/racer.py arguments, one run each. A run writes its model and evaluations to
# <out dir>/artifacts/<index>/, which research/racer/summarize.py reads into a summary for results/.
set -eu
config=$1 label=$2 out=$3 pool=${4:-4}
commit=$(git rev-parse HEAD)
git branch -r --contains "$commit" | grep -q . || { echo "push $commit first" >&2; exit 1; }
mkdir -p "$out"
echo "$commit" > "$out/commit"
cp "$config" "$out/config.json"
lab=$(cd "$(dirname "$0")/../../site/lab" && pwd)
exec bun /mnt/local/dew/tmp/armada/src/cli.ts map --connection="$HOME/.config/armada/armada-dew2.json" \
  --env="$lab/armada/recipe.json" --size=medium --pool="$pool" --timeout=14400 --items="$config" \
  --artifacts="$out/artifacts" --json --label="$label" -- \
  sh -c "cd \$HOME/sparx && git fetch -q origin $commit && git checkout -q --detach $commit && JAX_PLATFORMS=cpu .venv/bin/python site/lab/racer.py {item} --out {artifacts}" \
  > "$out/outcomes.jsonl"
