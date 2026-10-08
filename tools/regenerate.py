"""Regenerate reference fixtures in a copy of the repository and compare them with the committed ones.

    python tools/regenerate.py make_nest_fixtures.py
    python tools/regenerate.py make_ottt_fixtures.py ../ref-ottt
    python tools/regenerate.py make_snntorch_fixtures.py --update     # keep what it wrote

The tool runs with its arguments in a temporary copy of `src`, `tools` and
`tests/fixtures`, in the environment this script runs in (its lock file,
`tools/references.py`), so the committed fixtures stay as they are. Each
fixture the tool makes (`references.MADE_BY`) is then compared with
`tests/fixtures/SHA256SUMS`. One that differs is compared array by array,
which names the arrays that changed; arrays that vary from run to run, the
wall-clock times a fixture keeps (`VARIES`), are left out of that
comparison, and a NIR graph is compared as nodes and edges, whose order in
the file follows Python's string hashing. A float array within `ULPS` units
in the last place of its largest value is reproduced to rounding, and the
line says the worst case: a reduction rounds differently on another
instruction set, by an amount that follows the magnitude of what it sums.
The largest seen is 3 (DCLS's outputs on two x86 CPUs), so 16 leaves a
margin of five that a drift would show in the reports before it crossed.
Integer, boolean and string arrays match exactly. With `--update`, a fixture that
changed replaces the committed one and its checksum. Exits with 1 when a
fixture changed and `--update` is not given.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from references import FIXTURES, MADE_BY, ROOT, digest, recorded, write_checksums

VARIES: dict[str, str] = {"microcircuit.npz": "/seconds"}
"""Fixtures with arrays that differ between identical runs, by the suffix of their names."""

ULPS = 16


def ulps(a: np.ndarray, b: np.ndarray) -> float | None:
    """How far float arrays `a` and `b` are apart, in units in the last place of `b`'s largest value;
    None for arrays that are not floats of one dtype and shape, or that disagree on where a NaN is."""
    if a.dtype != b.dtype or a.dtype.kind != "f" or a.shape != b.shape:
        return None
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return None
    scale = float(np.max(np.abs(b[np.isfinite(b)]), initial=0.0))
    unit = float(np.spacing(np.asarray(scale, b.dtype)))
    real = ~np.isnan(b)
    return float(np.max(np.abs(a[real] - b[real]), initial=0.0)) / unit


def changed_arrays(made: Path, committed: Path) -> dict[str, float | None]:
    """The arrays of two NPZ files that differ, by name, each with how many units in the last place apart
    (`ulps`), beside those `VARIES` leaves out."""
    varying = VARIES.get(committed.name)

    def same(a: np.ndarray, b: np.ndarray) -> bool:
        return np.array_equal(a, b, equal_nan=a.dtype.kind in "fc" and b.dtype.kind in "fc")

    with np.load(made) as new, np.load(committed) as old:
        names = sorted(set(new.files) | set(old.files))
        changed = [name for name in names if not (varying and name.endswith(varying))
                   and (name not in new.files or name not in old.files or not same(new[name], old[name]))]
        return {name: ulps(new[name], old[name]) if name in new.files and name in old.files else None
                for name in changed}


def same_graph(made: Path, committed: Path) -> bool:
    """Whether two NIR files hold the same nodes and edges, in whatever order."""
    import nir

    new, old = nir.read(made), nir.read(committed)
    if sorted(new.edges) != sorted(old.edges) or set(new.nodes) != set(old.nodes):
        return False
    for name, node in new.nodes.items():
        fields, kept = vars(node), vars(old.nodes[name])
        if set(fields) != set(kept) or not all(np.array_equal(np.asarray(fields[k]), np.asarray(kept[k]))
                                              for k in fields if isinstance(fields[k], np.ndarray)):
            return False
    return True


def compare(made: Path, committed: Path) -> tuple[str, bool]:
    """How `made` compares with the committed fixture, and whether that counts as reproduced."""
    if digest(made) == recorded()[committed.name]:
        return "reproduced", True
    if committed.suffix == ".nir":
        graphs = same_graph(made, committed)
        return ("reproduced", True) if graphs else ("differs in its nodes or edges", False)
    changed = changed_arrays(made, committed)
    beyond = [name for name, apart in changed.items() if apart is None or apart > ULPS]
    if beyond:
        return f"differs in arrays {', '.join(beyond)}", False
    if not changed:
        return "reproduced", True
    worst = max(apart for apart in changed.values() if apart is not None)
    return f"reproduced to rounding, at most {worst:.1f} ulps, in arrays {', '.join(changed)}", True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tool", choices=sorted({tool for tool, _ in MADE_BY.values()}))
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="the tool's own arguments")
    parser.add_argument("--update", action="store_true", help="keep the fixtures that changed")
    args = parser.parse_args()
    made = [name for name, (tool, _) in MADE_BY.items() if tool == args.tool]
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch)
        for part in ("src", "tools", "tests/fixtures"):
            shutil.copytree(ROOT / part, copy / part)
        environment = {**os.environ, "PYTHONHASHSEED": "0"}
        # The tool writes beside its own copy; paths among its arguments stay the caller's.
        subprocess.run([sys.executable, str(copy / "tools" / args.tool), *args.arguments], env=environment,
                       check=True)
        failed = []
        for name in made:
            outcome, reproduced = compare(copy / "tests" / "fixtures" / name, FIXTURES / name)
            print(f"{name}: {outcome}")
            if not reproduced:
                failed.append(name)
                if args.update:
                    shutil.copyfile(copy / "tests" / "fixtures" / name, FIXTURES / name)
    if failed and args.update:
        write_checksums()
    elif failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
