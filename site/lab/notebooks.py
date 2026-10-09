"""The site's Colab notebooks, written from the snippets its pages show (site/snippets), and the check that
every snippet runs:

    python site/lab/notebooks.py write        # site/notebooks/<name>.ipynb
    python site/lab/notebooks.py run          # each snippet in a fresh interpreter; prints their output

A notebook installs sparx from GitHub, then runs its snippets in order, each as a cell.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
SNIPPETS = SITE / "snippets"
INSTALL = "%pip install -q 'sparxml[nir] @ git+https://github.com/AshishKumar4/sparx'"

NOTEBOOKS = {
    "neuron": ("Neurons", "neuron", ["neuron", "physical", "izhikevich"]),
    "train": ("A first spiking network", "surrogate-gradients", ["train"]),
    "teach": ("Teach a neuron when to fire", "surrogate-gradients", ["teach"]),
    "delays": ("Learned delays", "delays", ["delays"]),
    "stdp": ("STDP in a network", "plasticity", ["stdp"]),
    "brunel": ("Brunel's balanced network", "networks", ["brunel"]),
    "nir": ("NIR export", "nir", ["nir"]),
}


def cell(kind: str, text: str) -> dict:
    lines = text.strip("\n").split("\n")
    body = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": body}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": body}


def write(_: argparse.Namespace) -> None:
    out = SITE / "notebooks"
    out.mkdir(exist_ok=True)
    for name, (title, page, snippets) in NOTEBOOKS.items():
        cells = [
            cell(
                "markdown",
                f"# {title}\n\nThe code of [sparxml.dev/learn/{page}](https://sparxml.dev/learn/{page}/), "
                "run on a CPU runtime. The first cell installs sparx.",
            ),
            cell("code", INSTALL),
        ]
        cells += [cell("code", (SNIPPETS / f"{snippet}.py").read_text()) for snippet in snippets]
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (out / f"{name}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"notebooks: {len(NOTEBOOKS)}")


def run(args: argparse.Namespace) -> None:
    failed = []
    for snippet in sorted(SNIPPETS.glob("*.py")):
        start = time.time()
        done = subprocess.run(
            [sys.executable, snippet.name], cwd=SNIPPETS, capture_output=True, text=True, check=False
        )
        seconds = time.time() - start
        status = "ok" if done.returncode == 0 else f"exit {done.returncode}"
        print(f"== {snippet.name}: {status} in {seconds:.1f} s\n{done.stdout.strip()}", flush=True)
        if done.returncode:
            print(done.stderr[-3000:], flush=True)
            failed.append(snippet.name)
    for leftover in SNIPPETS.glob("*.nir"):
        leftover.unlink()
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(required=True)
    commands.add_parser("write").set_defaults(func=write)
    commands.add_parser("run").set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
