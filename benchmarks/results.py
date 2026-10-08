"""A benchmark's measurements as JSON lines, each with the conditions it was taken under.

    python benchmarks/bench_networks.py --results runs/bench.jsonl

Each line is one measurement: the benchmark's name, what it measured, and
`conditions`, which hold the commit (and whether the tree had changes), the
time, the machine (platform, processor count, the devices JAX sees) and the
versions of the packages the benchmark ran. A file collects the lines of
every run that wrote to it, so results from different machines and commits
sit side by side.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPARX = ("jax", "jaxlib", "flax", "optax", "dewml", "sparxml")
"""The packages a benchmark of sparx records."""

type Value = str | int | float | bool | None
type Measurement = Mapping[str, Value]


def _git(*arguments: str) -> str:
    done = subprocess.run(["git", "-C", str(ROOT), *arguments], capture_output=True, text=True)
    return done.stdout.strip()


def _version(distribution: importlib.metadata.Distribution) -> str:
    """The distribution's version, and the commit it was installed from when pip took it from git (dew's)."""
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = source.get("vcs_info", {}).get("commit_id")
    return distribution.version if commit is None else f"{distribution.version}@{commit}"


def conditions(packages: Iterable[str], devices: Iterable[str] = (),
               known: Mapping[str, str] | None = None) -> dict[str, Value | dict[str, str]]:
    """Where and with what a measurement was taken: `packages` by distribution name, beside the versions
    `known` already, and `devices` as their kinds."""
    versions = dict(known or {})
    for name in packages:
        try:
            versions[name] = _version(importlib.metadata.distribution(name))
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    kinds = list(devices)
    changed = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"commit": _git("rev-parse", "HEAD"), "changed": changed,
            "time": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "platform": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count(),
            "devices": ", ".join(sorted(set(kinds))), "device_count": len(kinds),
            "packages": versions}


def write(path: Path | None, benchmark: str, measured: Iterable[Measurement],
          taken: Mapping[str, Value | dict[str, str]]) -> None:
    """Append each measurement of `benchmark` to `path`, one JSON object a line; nothing without a path."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        for row in measured:
            out.write(json.dumps({"benchmark": benchmark, **row, "conditions": taken}) + "\n")
