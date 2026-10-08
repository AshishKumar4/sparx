"""Run sparx's and the official SNN-delays code's SHD recipe over seeds, and compare them as data.

Hammouamri et al.'s recipe ("Learning Delays in Spiking Neural Networks
using Dilated Convolutions with Learnable Spacings", ICLR 2024) at its full
150 epochs, on one GPU, each seed a run of its own:

    python tools/shd_comparison.py sparx --seed 0 --validation 0 --out runs/cmp/sparx-all-0
    python tools/shd_comparison.py sparx --seed 0 --validation 0.1 --out runs/cmp/sparx-held-0
    python tools/shd_comparison.py official --seed 0 --checkout SNN-delays --spikingjelly spikingjelly \\
        --data data/SHD --out runs/cmp/official-0
    python tools/shd_comparison.py summarize runs/cmp

`--validation 0` trains on every training recording as their script does,
and scores the test set after every epoch, as their script scores it as
its validation set; that pair of runs compares the two codes on one
protocol. `--validation 0.1` holds a tenth of the training set out, and its
number is the test accuracy of the epoch the holdout chose, with test never
used to choose; their code has no such protocol. Each run writes
`result.json`: every epoch's accuracies, the last epoch's, the best test
accuracy over epochs (their reported number), the test accuracy at the best
holdout epoch where there is one, seconds per epoch, and the conditions
(commits, package versions, GPU). `summarize` gives the mean and standard
deviation over seeds of each code and protocol, in `summary.json`.

Environments, on a CUDA 12 machine:

    # sparx, at the commit compared, on dew's jax with its CUDA plugin
    pip install -e ".[datasets]" -c constraints.txt "jax-cuda12-plugin[with-cuda]==0.11.2"
    # theirs: SNN-delays d169b4e3 and the SpikingJelly of 2023 it runs on
    git clone https://github.com/Thvnvtos/SNN-delays && git -C SNN-delays checkout d169b4e3
    git clone https://github.com/fangwei123456/spikingjelly
    git -C spikingjelly checkout 6fbee6ed34ed5a65187f4721d1a412f6a526ca6a
    pip install torch torchvision torchaudio dcls==0.1.1 h5py tqdm wandb

Their `config.py` is their `best_config_SHD.py`; the seed, the epochs and
the dataset's directory are set on it before their `main.py` runs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPOCH = re.compile(r"=====> Epoch (\d+) : \nLoss Train = [\d.]+  \|  Acc Train = ([\d.]+)%\n"
                   r"Loss Valid = [\d.]+  \|  Acc Valid = ([\d.]+)%")


def _run(command: list[str], out: Path, cwd: Path) -> float:
    """Run `command` in `cwd` with its output in `out/stdout.log`; the seconds it took."""
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (out / "stdout.log").open("w") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def _commit(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def _conditions(packages: tuple[str, ...], checkouts: dict[str, Path]) -> dict[str, object]:
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"commits": {name: _commit(path) for name, path in checkouts.items()}, "packages": versions,
            "gpu": gpu or "none", "python": sys.version.split()[0]}


def sparx(args: argparse.Namespace) -> None:
    command = [sys.executable, str(ROOT / "examples" / "train_shd.py"), "--recipe", "snn-delays",
               "--epochs", str(args.epochs), "--validation", str(args.validation), "--seed", str(args.seed),
               "--out", str(args.out.resolve())]
    seconds = _run(command, args.out, ROOT)
    epochs: dict[int, dict[str, float]] = {}
    for line in (args.out / "tracking" / "scalars.jsonl").read_text().splitlines():
        row = json.loads(line)
        epochs.setdefault(row["step"], {}).update(row["scalars"])
    scored = [scalars for _, scalars in sorted(epochs.items()) if "test/accuracy" in scalars]
    test = [scalars["test/accuracy"] for scalars in scored]
    held = [scalars["val/accuracy"] for scalars in scored if "val/accuracy" in scalars]
    result = {"code": "sparx", "protocol": "holdout" if args.validation else "all", "seed": args.seed,
              "test": test, "holdout": held, "last": test[-1], "best": max(test),
              "at_best_holdout": test[held.index(max(held))] if held else None,
              "seconds_per_epoch": seconds / len(test),
              "conditions": _conditions(("sparxml", "dewml", "jax", "jaxlib", "flax", "optax"),
                                        {"sparx": ROOT})}
    (args.out / "result.json").write_text(json.dumps(result, indent=1))


def official(args: argparse.Namespace) -> None:
    checkout, data = args.checkout.resolve(), args.data.resolve()
    patch = ("import sys, runpy; sys.path[:0] = [{code!r}, {sj!r}]; import config; "
             "config.Config.seed = {seed}; config.Config.epochs = {epochs}; "
             "config.Config.datasets_path = {data!r}; runpy.run_path({main!r}, run_name='__main__')").format(
        code=str(checkout), sj=str(args.spikingjelly.resolve()), seed=args.seed, epochs=args.epochs,
        data=str(data), main=str(checkout / "main.py"))
    seconds = _run([sys.executable, "-c", patch], args.out, args.out)
    found = EPOCH.findall((args.out / "stdout.log").read_text())
    test = [float(valid) / 100 for _, _, valid in found]
    result = {"code": "official", "protocol": "all", "seed": args.seed, "test": test, "holdout": [],
              "train": [float(train) / 100 for _, train, _ in found], "last": test[-1], "best": max(test),
              "at_best_holdout": None, "seconds_per_epoch": seconds / len(test),
              "conditions": _conditions(("torch", "dcls"), {"SNN-delays": checkout,
                                                            "spikingjelly": args.spikingjelly.resolve()})}
    (args.out / "result.json").write_text(json.dumps(result, indent=1))


def summarize(args: argparse.Namespace) -> None:
    runs = [json.loads(path.read_text()) for path in sorted(args.directory.glob("*/result.json"))]
    summary = {}
    for code, protocol in sorted({(run["code"], run["protocol"]) for run in runs}):
        group = [run for run in runs if (run["code"], run["protocol"]) == (code, protocol)]
        measures = {}
        for measure in ("last", "best", "at_best_holdout", "seconds_per_epoch"):
            values = [run[measure] for run in group if run[measure] is not None]
            if values:
                measures[measure] = {"mean": statistics.fmean(values),
                                     "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                                     "values": values}
        summary[f"{code}/{protocol}"] = {"seeds": [run["seed"] for run in group], **measures}
        shown = "  ".join(f"{name} {m['mean'] * 100:.2f} ± {m['std'] * 100:.2f}%"
                          for name, m in measures.items() if name != "seconds_per_epoch")
        print(f"{code:8s} {protocol:8s} seeds {len(group)}  {shown}")
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("sparx", "official"):
        run = commands.add_parser(name)
        run.add_argument("--seed", type=int, required=True)
        run.add_argument("--epochs", type=int, default=150)
        run.add_argument("--out", type=Path, required=True)
        if name == "sparx":
            run.add_argument("--validation", type=float, default=0.1)
        else:
            run.add_argument("--checkout", type=Path, required=True, help="SNN-delays at d169b4e3")
            run.add_argument("--spikingjelly", type=Path, required=True, help="SpikingJelly at 6fbee6ed")
            run.add_argument("--data", type=Path, required=True, help="their SHD directory")
            run.epilog = ("their schedules' horizons are set from 150 epochs when their config is built, so "
                          "--epochs below 150 is a short check of the pipeline, not a shorter recipe")
    commands.add_parser("summarize").add_argument("directory", type=Path)
    args = parser.parse_args()
    {"sparx": sparx, "official": official, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
