#!/bin/sh
# Train a sweep's runs one after another on the workstation's GPU, each through dew-gpu-run as the racer
# lane, and after each run commit the sweep's summary in a second checkout, on that checkout's branch:
#
#     research/racer/queue.sh <config.json> <out dir> <results checkout> <first> <last>
#
# Run it from a checkout of the code at a fixed commit, since run.py continues a sweep only under the
# commit it began with. It stops at the first run that fails, leaves no evaluation, or logs a loss or
# gradient that is not finite, and retries nothing.
set -eu
config=$1 out=$2 results=$3 first=$4 last=$5
python=${PYTHON:-/home/mrwhite0racle/Desktop/dew/.venv/bin/python}
summary=research/racer/results/$(basename "$config")
i=$first
while [ "$i" -le "$last" ]; do
  DEW_GPU_LANE=racer "$HOME/.cache/dew/dew-gpu-run" --minutes 40 \
    env PYTHONPATH=src "$python" research/racer/run.py "$config" "$out" --items "$i" "$i"
  python3 - "$out/artifacts/$i" <<'EOF'
import json, math, sys
from pathlib import Path
run = Path(sys.argv[1])
history = json.loads((run / "racer.json").read_text())["meta"]["history"]
summary = json.loads((run / "evaluation.json").read_text())["summary"]
bad = [row["step"] for row in history if not (math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]))]
if bad or not isinstance(summary.get("finished"), float):
    sys.exit(f"{run}: a loss or gradient norm not finite at steps {bad[:5]}, or no score")
print(f"{run}: finished {summary['finished']:.3f} of {summary['tracks']} tracks", flush=True)
EOF
  python3 research/racer/summarize.py "$out" "$results/$summary"
  git -C "$results" add "$summary"
  git -C "$results" -c user.name="Ashish Kumar Singh" -c user.email="ashishkmr472@gmail.com" \
    commit -q -m "research(racer): $(basename "$config" .json), run $i"
  git -C "$results" push -q origin HEAD
  i=$((i + 1))
done
