"""Train a sweep's runs one after another on this machine, as research/racer/sweep.sh does on armada: each
run of <config.json> (a list of site/lab/racer.py arguments) writes to <out dir>/artifacts/<index>/, its
log beside it, and <out dir>/outcomes.jsonl gets a line per run. research/racer/summarize.py reads the
directory. A directory already begun continues only under the same configuration and commit. The runner
exits 1 when any run failed.

    python research/racer/run.py <config.json> <out dir> [--items 0 14]

Run from the repository's root, with the Python whose JAX sees the accelerator.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("out")
    parser.add_argument("--items", type=int, nargs=2, metavar=("FIRST", "LAST"), help="inclusive")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (("config.json", json.dumps(config, indent=1) + "\n"), ("commit", commit)):
        recorded = out / name
        if recorded.exists() and recorded.read_text() != value:
            sys.exit(f"{recorded} records another sweep; continue it with its {name}, or start a new one")
        recorded.write_text(value)
    first, last = args.items or (0, len(config) - 1)
    failed = 0
    for index in range(first, last + 1):
        run = out / "artifacts" / str(index)
        run.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with (run / "log.txt").open("w") as log:
            command = [sys.executable, "site/lab/racer.py", *shlex.split(config[index]), "--out", str(run)]
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        seconds = round(time.time() - start, 1)
        row = {"index": index, "item": config[index], "exitCode": code, "seconds": seconds}
        with (out / "outcomes.jsonl").open("a") as outcomes:
            outcomes.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        failed += code != 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
