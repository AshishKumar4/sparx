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
the file follows Python's string hashing. With `--update`, a fixture that
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


def changed_arrays(made: Path, committed: Path) -> list[str]:
    """The arrays of two NPZ files that differ, by name, beside those `VARIES` leaves out."""
    varying = VARIES.get(committed.name)
    with np.load(made) as new, np.load(committed) as old:
        names = sorted(set(new.files) | set(old.files))
        return [name for name in names if not (varying and name.endswith(varying))
                and (name not in new.files or name not in old.files
                     or not np.array_equal(new[name], old[name], equal_nan=True))]


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


def compare(made: Path, committed: Path) -> str | None:
    """None when `made` is the committed fixture, else what differs."""
    if digest(made) == recorded()[committed.name]:
        return None
    if committed.suffix == ".nir":
        return None if same_graph(made, committed) else "its nodes or edges"
    arrays = changed_arrays(made, committed)
    return f"arrays {', '.join(arrays)}" if arrays else None


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
        differing = {}
        for name in made:
            difference = compare(copy / "tests" / "fixtures" / name, FIXTURES / name)
            if difference is not None:
                differing[name] = difference
                if args.update:
                    shutil.copyfile(copy / "tests" / "fixtures" / name, FIXTURES / name)
    for name in made:
        print(f"{name}: {'differs in ' + differing[name] if name in differing else 'reproduced'}")
    if differing and args.update:
        write_checksums()
    elif differing:
        sys.exit(1)


if __name__ == "__main__":
    main()
