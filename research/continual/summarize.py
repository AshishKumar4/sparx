"""The stream study's runs as one table per stream: each learner's mean and sample deviation over seeds of
its final accuracy over all 20 classes, over each task's own classes, its mean forgetting, and its
accuracy on each task right after learning it. A directory of `stream.py --select` runs gives instead each
learner's held-out accuracy at each learning rate.

    python3 research/continual/summarize.py research/continual/results/unclipped

Standard library only.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ORDER = ("joint", "bptt", "eprop", "eprop-random", "ewc1000", "ewc10000", "ewc100000")


def spread(values: list[float]) -> str:
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{100 * statistics.fmean(values):.1f} ± {100 * deviation:.1f}"


def selection(runs: list[dict]) -> None:
    rates = sorted({d["lr"] for d in runs})
    for stream in ("classes", "permuted"):
        print(f"\n{stream}\n")
        print("| Learner | " + " | ".join(f"{lr:g}" for lr in rates) + " |")
        print("| --- |" + " --- |" * len(rates))
        for learner in ORDER:
            held = {d["lr"]: d["held_out"] for d in runs if (d["stream"], d["learner"]) == (stream, learner)}
            if held:
                print(f"| {learner} | " + " | ".join(f"{100 * held[lr]:.1f}" if lr in held else ""
                                                      for lr in rates) + " |")


def main() -> None:
    every = {path: json.loads(path.read_text()) for path in sorted(Path(sys.argv[1]).glob("*.json"))}
    if all("held_out" in d for d in every.values()):
        selection(list(every.values()))
        return
    runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path, d in every.items():
        stream, rest = path.stem.split("-", 1)
        runs[stream, rest.rsplit("-", 1)[0]].append(d)
    for stream in ("classes", "permuted"):
        print(f"\n{stream}\n")
        print("| Learner | Seeds | All 20 classes, % | Its own classes, % | Forgetting, points "
              "| Just learned, % |")
        print("| --- | --- | --- | --- | --- | --- |")
        for learner in ORDER:
            done = runs.get((stream, learner), [])
            if not done:
                continue
            forgetting = [d["mean_forgetting"] for d in done if d["mean_forgetting"] is not None]
            learned = [statistics.fmean(d["accuracy"][k][k] for k in range(len(d["accuracy"]))) for d in done]
            print(f"| {learner} | {len(done)} | {spread([d['final_mean'] for d in done])} "
                  f"| {spread([d['own_classes_final_mean'] for d in done])} "
                  f"| {spread(forgetting) if forgetting else 'none'} | {spread(learned)} |")


if __name__ == "__main__":
    main()
