#!/bin/sh
# Every run of the stream study, one after another on the workstation's GPU through dew-gpu-run as the
# racer lane, stopping at the first that fails. Runs already written are skipped.
#
#     research/continual/queue.sh <out dir>
set -eu
out=$1
python=${PYTHON:-/home/mrwhite0racle/Desktop/dew/.venv/bin/python}
for seed in 0 1 2; do
  for stream in classes permuted; do
    for learner in joint bptt eprop eprop-random ewc:1000 ewc:10000 ewc:100000; do
      rule=${learner%%:*}; lambda=${learner#*:}
      name=$stream-$rule-$seed; extra=""
      if [ "$rule" = ewc ]; then name=$stream-ewc$lambda-$seed; extra="--ewc $lambda"; fi
      [ -f "$out/$name.json" ] && continue
      # shellcheck disable=SC2086
      DEW_GPU_LANE=racer "$HOME/.cache/dew/dew-gpu-run" --minutes 20 env PYTHONPATH=src "$python" \
        research/continual/stream.py --stream "$stream" --learner "$rule" --seed "$seed" $extra --out "$out"
      [ -f "$out/$name.json" ] || { echo "$name wrote no result" >&2; exit 1; }
    done
  done
done
