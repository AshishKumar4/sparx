#!/bin/sh
# Run a TypeScript file against the site's engines at a commit on armada, off the workstation:
#
#     site/lab/armada/bun.sh <commit> <label> <file.ts>
#
# The file travels with the job; imports of this worktree's site/ resolve to the commit's. Its stdout comes back.
set -eu
commit=$(git rev-parse "$1"); label=$2; file=$3
here=$(dirname "$0")
site=$(cd "$here/../.." && pwd)
items=$(mktemp)
printf '[{"script": "%s"}]' "$(sed "s|$site|SITE|g" "$file" | base64 -w0)" > "$items"
exec bun /mnt/local/dew/tmp/armada/src/cli.ts map --connection="$HOME/.config/armada/armada-dew2.json" \
  --env="$here/recipe.json" --size=small --pool=1 --items="$items" --output --json --label="$label" -- \
  sh -c "cd \$HOME/sparx && git fetch -q origin $commit && git checkout -q --detach $commit && echo {script} | base64 -d | sed \"s|SITE|\$HOME/sparx/site|g\" > /tmp/job.ts && \$HOME/.bun/bin/bun /tmp/job.ts"
