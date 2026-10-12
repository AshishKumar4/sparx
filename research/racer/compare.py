"""A sweep's summary (summarize.py's output) as a comparison of its arms: for each arm, every seed's result
and the mean and sample standard deviation over seeds, as JSON and as a Markdown table.

    python research/racer/compare.py results/phase2c.json results/phase2c-arms.json   # in research/racer

An arm is a run's arguments without its seed. Each seed's numbers are its evaluation on 200 unseen tracks,
its harder sets by bend, its median and largest gradient norm, and its first step.
"""

import json
import shlex
import statistics
import sys
from pathlib import Path

NAMES = {
    ("lif", 1, "events"): "A, spiking",
    ("relu", 1, "events"): "B, graded, events",
    ("relu", 1, "frames"): "C, graded, frames",
    ("lif", 2, "events"): "spikes of 2 bits",
    ("lif", 4, "events"): "spikes of 4 bits",
    ("sigma-delta", 1, "events"): "sigma-delta",
    ("dendritic", 1, "events"): "dendritic",
    ("homeostatic", 1, "events"): "homeostatic",
}
"""Each arm's name by its units, the bits its spikes carry, and what it sees."""


def option(arguments: list[str], name: str, default: str) -> str:
    return arguments[arguments.index(name) + 1] if name in arguments else default


def spread(values: list[float]) -> dict:
    """The values, their mean and their sample standard deviation (0 for one value)."""
    return {"values": values, "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def show(values: dict, scale: float = 1.0, digits: int = 1) -> str:
    return f"{values['mean'] * scale:.{digits}f} ± {values['sd'] * scale:.{digits}f}"


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    runs = json.loads(source.read_text())["runs"]
    arms: dict[str, list[dict]] = {}
    for run in runs:
        if "evaluation" not in run:
            continue
        arguments = shlex.split(run["arguments"])
        key = (option(arguments, "--neuron", "lif"), int(option(arguments, "--bits", "1")),
               option(arguments, "--sees", "events"))
        name = NAMES.get(key, " ".join(map(str, key)))
        if "--normalize-levels" in arguments:
            name += ", level-scaled"
        arms.setdefault(name, []).append(
            {"seed": int(option(arguments, "--seed", "0")), **run})
    table = {}
    for name, seeds in arms.items():
        seeds.sort(key=lambda run: run["seed"])
        evaluations = [run["evaluation"] for run in seeds]
        laps = [e["median_lap_time"] for e in evaluations if e["median_lap_time"] is not None]
        responses = [e["response_ms"] for e in evaluations if e.get("response_ms") is not None]
        table[name] = {
            "seeds": [run["seed"] for run in seeds],
            "finished": spread([e["finished"] for e in evaluations]),
            "every_track": sum(e["finished"] == 1.0 for e in evaluations),
            "harder": {bend: spread([run["difficulty"]["by_bend"][bend]["finished"] for run in seeds])
                       for bend in seeds[0]["difficulty"]["by_bend"]},
            "median_lap_time": spread(laps) if laps else None,
            "response_ms": spread(responses) if responses else None,
            "events_per_step": spread([e["events_per_step"] for e in evaluations]),
            "active_units_per_step": spread([e["spikes_per_step"] for e in evaluations]),
            "triggered_macs_per_step": spread([e["triggered_macs_per_step"] for e in evaluations]),
            "dense_macs_per_step": evaluations[0]["dense_macs_per_step"],
            "parameters": evaluations[0]["parameters"],
            "units": evaluations[0]["units"],
            "grad_norm_median": [run["grad_norm"]["median"] for run in seeds],
            "grad_norm_max": [run["grad_norm"]["max"] for run in seeds],
            "start": [run["start"] for run in seeds],
        }
    target.write_text(json.dumps(table, indent=1) + "\n")
    print("| Arm | Seeds | Every track | Unseen tracks finished, % | Bend 1.8, % | Lap, s | Response, ms "
          "| Events a step | Active units a step | Triggered MACs a step | Dense MACs a step |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, row in table.items():
        lap = show(row["median_lap_time"]) if row["median_lap_time"] else "none"
        response = show(row["response_ms"], digits=0) if row["response_ms"] else "none"
        events, active = show(row["events_per_step"], digits=0), show(row["active_units_per_step"], digits=0)
        print(f"| {name} | {len(row['seeds'])} | {row['every_track']} | {show(row['finished'], 100)} "
              f"| {show(row['harder']['1.8'], 100)} | {lap} | {response} | {events} | {active} "
              f"| {show(row['triggered_macs_per_step'], digits=0)} | {row['dense_macs_per_step']:,} |")


if __name__ == "__main__":
    main()
