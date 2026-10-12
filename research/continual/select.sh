#!/bin/sh
# The learning-rate selection (stream.py --select): BPTT, e-prop and random e-prop at each rate on the first
# task of each stream at seed 0, one after another on the workstation's GPU through dew-gpu-run as the racer
# lane, stopping at the first that fails. EWC is BPTT until its first task ends and joint training is BPTT,
# so both take BPTT's rate. Runs already written are skipped.
#
#     research/continual/select.sh <out dir> [stream.py options for every run, such as --epochs 20]
set -eu
out=$1; shift
python=${PYTHON:-/home/mrwhite0racle/Desktop/dew/.venv/bin/python}
for stream in classes permuted; do
  for learner in bptt eprop eprop-random; do
    for lr in 0.0001 0.0003 0.001 0.003; do
      name=$stream-$learner-0-lr$lr
      [ -f "$out/$name.json" ] && continue
      DEW_GPU_LANE=racer "$HOME/.cache/dew/dew-gpu-run" --minutes 20 env PYTHONPATH=src "$python" \
        research/continual/stream.py --stream "$stream" --learner "$learner" --lr "$lr" --select "$@" \
        --out "$out"
      [ -f "$out/$name.json" ] || { echo "$name wrote no result" >&2; exit 1; }
    done
  done
done
