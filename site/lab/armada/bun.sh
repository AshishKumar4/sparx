#!/bin/sh
# Run a TypeScript file against the site's engines at a commit on armada, off the workstation:
#
#     site/lab/armada/bun.sh <commit> <label> <file.ts>
#
# A file inside site/ runs where it is in the commit's checkout, so its relative imports resolve there;
# any other file travels with the job, and its imports of this worktree's site/ resolve to the commit's.
# Its stdout comes back.
set -eu
commit=$(git rev-parse "$1"); label=$2; file=$3
here=$(dirname "$0")
site=$(cd "$here/../.." && pwd)
path=$(cd "$(dirname "$file")" && pwd)/$(basename "$file")
case $path in
  "$site"/*)
    exec bun /mnt/local/dew/tmp/armada/src/cli.ts map --connection="$HOME/.config/armada/armada-dew2.json" \
      --env="$here/recipe.json" --size=small --pool=1 --times=1 --output --json --label="$label" -- \
      sh -c "cd \$HOME/sparx && git fetch -q origin $commit && git checkout -q --detach $commit && cd site && \$HOME/.bun/bin/bun ${path#"$site"/}" ;;
esac
items=$(mktemp)
printf '[{"script": "%s"}]' "$(sed "s|$site|SITE|g" "$file" | base64 -w0)" > "$items"
exec bun /mnt/local/dew/tmp/armada/src/cli.ts map --connection="$HOME/.config/armada/armada-dew2.json" \
  --env="$here/recipe.json" --size=small --pool=1 --items="$items" --output --json --label="$label" -- \
  sh -c "cd \$HOME/sparx && git fetch -q origin $commit && git checkout -q --detach $commit && echo {script} | base64 -d | sed \"s|SITE|\$HOME/sparx/site|g\" > /tmp/job.ts && \$HOME/.bun/bin/bun /tmp/job.ts"
