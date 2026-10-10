"""A racer sweep's runs as one small summary for results/: each run's arguments, its evaluation on 200
unseen tracks, its results on the harder sets by bend and by tightest bend, its training curve every
250 steps, and the median and largest gradient norm over every step it logged.

    python research/racer/summarize.py <out dir> research/racer/results/<name>.json

<out dir> is research/racer/sweep.sh's: config.json, commit, and artifacts/<index>/ with racer.json,
evaluation.json and difficulty.json as each run wrote them.
"""

import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    config = json.loads((out / "config.json").read_text())
    runs = []
    for index, arguments in enumerate(config):
        run = out / "artifacts" / str(index)
        row: dict = {"arguments": arguments}
        if (run / "racer.json").exists():
            history = json.loads((run / "racer.json").read_text())["meta"]["history"]
            row["history"] = [h for h in history if h["step"] == 1 or h["step"] % 250 == 0]
            norms = [h["grad_norm"] for h in history]
            row["grad_norm"] = {"logged": len(norms), "median": statistics.median(norms), "max": max(norms)}
        if (run / "evaluation.json").exists():
            row["evaluation"] = json.loads((run / "evaluation.json").read_text())["summary"]
        if (run / "difficulty.json").exists():
            harder = json.loads((run / "difficulty.json").read_text())
            row["difficulty"] = {"by_bend": harder["by_bend"], "by_radius": harder["by_radius"]}
        runs.append(row)
    commit = (out / "commit").read_text().strip() if (out / "commit").exists() else None
    target.write_text(json.dumps({"commit": commit, "runs": runs}, indent=1) + "\n")


if __name__ == "__main__":
    main()
