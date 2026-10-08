"""The environments and checkouts sparx's reference fixtures come from, and the checks each fixture tool runs.

    python tools/references.py lock              # write tools/environments/*.txt from PINS, with hashes
    python tools/references.py checksums         # write tests/fixtures/SHA256SUMS from the fixtures

A fixture tool calls `require` with the packages it runs and
`require_checkout` with each reference repository it reads, before it
computes anything, so a fixture is regenerated only with the versions and
commits it records. Each environment is a lock file under
`tools/environments`: pip's, with every transitive pin and its hashes,
installed with `uv pip sync tools/environments/<name>.txt`, or conda's for
NEST, which PyPI does not carry. `tools/regenerate.py` runs a tool in a
copy of the repository and compares what it writes with `SHA256SUMS`;
`tests/test_fixtures.py` checks that every committed fixture matches its
checksum and names its tool and environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENTS = ROOT / "tools" / "environments"
FIXTURES = ROOT / "tests" / "fixtures"
CHECKSUMS = FIXTURES / "SHA256SUMS"

PINS: dict[str, tuple[str, str]] = {
    "nest-simulator": ("nest", "3.10.0"),
    "brian2": ("brian2", "2.10.1"),
    "torch": ("torch", "2.14.1+cpu"),
    "snntorch": ("snntorch", "1.0.0"),
    "nir": ("nir", "1.0.8"),
    "nirtorch": ("nirtorch", "2.6"),
    "dcls": ("DCLS", "0.1.1"),
    "tensorflow": ("tensorflow", "2.21.0"),
    "snntoolbox": ("snntoolbox", "0.6.0"),
    "brian2modelfitting": ("brian2modelfitting", "0.4"),
    "elephant": ("elephant", "1.2.1"),
    "jax": ("jax", "0.11.2.post3"),
    "pandas": ("pandas", "3.0.6"),
    "pyarrow": ("pyarrow", "25.0.1"),
}
"""Each reference package's module and version, by distribution: the version its fixtures record."""

CHECKOUTS: dict[str, str] = {
    "spikingjelly": "c6cb8e46738bf6010cb94904fe5995e66d6451fc",
    "spikingjelly-2023": "6fbee6ed34ed5a65187f4721d1a412f6a526ca6a",
    "SNN-delays": "d169b4e3",
    "OTTT-SNN": "c15d5da05eea0c48e1fa837a4f8073297006d93f",
    "differentiable-plasticity": "5bd29a18",
    "backpropamine": "180c9101",
    "fly-gym": "8d964599119b420de67ee259434d273651a1a76d",
    "pc-alm": "660747f61a8a7e547c0ecd2c48c8883380a7d1f6",
    "RNeuralNet-Research": "d4b7803a5bbe87747d27a7137cc05a756bef42f7",
    "microcircuit-PD14-model": "f79f8ac",
    "Drosophila_brain_model": "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960",
}
"""Every reference repository's commit, as a prefix of its full hash."""

PROGRAMS: dict[str, str] = {"octave": "8.4.0", "g++": "13.3.0"}
"""The version of each program a fixture tool runs, as `<program> --version` prints it."""

FILES: dict[str, str] = {"figure1.m": "52a9aac93f7b3a5f0fd61d5f1250deb57d88deda048367b71b3933382aaf4ead"}
"""The SHA-256 of each reference file a tool reads that no repository holds."""

MADE_BY: dict[str, tuple[str, str]] = {
    "nest.npz": ("make_nest_fixtures.py", "nest"),
    "nest_gap.npz": ("make_nest_gap_fixtures.py", "nest"),
    "brunel.npz": ("make_brunel_fixtures.py", "nest"),
    "microcircuit.npz": ("make_microcircuit_fixtures.py", "nest"),
    "brian2.npz": ("make_brian2_fixtures.py", "brian2"),
    "benchmarks.npz": ("make_brian2_benchmarks.py", "brian2"),
    "shiu.npz": ("make_shiu_fixtures.py", "brian2"),
    "gamma.npz": ("make_gamma_fixtures.py", "modelfitting"),
    "elephant.npz": ("make_elephant_fixtures.py", "elephant"),
    "snntoolbox.npz": ("make_snntoolbox_fixtures.py", "snntoolbox"),
    "snntorch.npz": ("make_snntorch_fixtures.py", "torch"),
    "spikingjelly.npz": ("make_reference_fixtures.py", "torch"),
    "snn_delays.npz": ("make_snn_delays_fixtures.py", "torch"),
    "dcls.npz": ("make_dcls_fixtures.py", "torch"),
    "ottt.npz": ("make_ottt_fixtures.py", "torch"),
    "miconi.npz": ("make_miconi_fixtures.py", "torch"),
    "flynn.npz": ("make_flynn_fixtures.py", "torch"),
    "nir.npz": ("make_nir_fixtures.py", "torch"),
    "nir_conv.npz": ("make_nir_fixtures.py", "torch"),
    "nir_rleaky.npz": ("make_nir_fixtures.py", "torch"),
    "snntorch.nir": ("make_nir_fixtures.py", "torch"),
    "snntorch_conv.nir": ("make_nir_fixtures.py", "torch"),
    "snntorch_rleaky.nir": ("make_nir_fixtures.py", "torch"),
    "pcalm.npz": ("make_pcalm_fixtures.py", "sparx"),
    "izhikevich_2004.npz": ("make_izhikevich_2004_fixtures.py", "brian2"),
    "rneuralnet.npz": ("make_rneuralnet_fixtures.py", "brian2"),
}
"""Each committed fixture's tool and the environment it runs in. `sparx` is the project's own environment
(`constraints.txt`); a tool that runs a program (Octave, g++) checks its version (`PROGRAMS`)."""

LOCKED: dict[str, tuple[str, ...]] = {
    # environment: the pins its lock file is compiled from, beyond PINS' own entries
    "brian2": ("brian2", "numpy", "scipy", "pandas", "pyarrow"),
    "torch": ("torch", "snntorch", "nir", "nirtorch", "dcls", "torchvision", "loguru", "packaging", "scipy",
              "matplotlib", "tqdm", "h5py"),
    "modelfitting": ("brian2", "brian2modelfitting", "scipy<1.14", "bayesian-optimization<2"),
    "elephant": ("elephant", "neo", "quantities"),
    "snntoolbox": ("snntoolbox", "tensorflow", "tf-keras"),
}
"""The pip environments, by the requirements their lock files resolve; NEST's is `nest.yml`, for conda."""

OVERRIDES: dict[str, tuple[str, ...]] = {"modelfitting": ("numpy<2",)}
"""Pins that replace what a package declares: brian2 2.10.1 declares numpy>=2, which brian2modelfitting 0.4
cannot import (`numpy.NaN`), and gamma.npz was made with both on numpy 1."""

WHEELS: dict[str, str] = {
    name: f"{name} @ https://download.pytorch.org/whl/cpu/{name}-{version}%2Bcpu-cp312-cp312-"
          f"manylinux_2_28_x86_64.whl#sha256={digest}"
    for name, version, digest in (
        ("torch", "2.14.1", "5a6363570c753812540a05eb82380e329469cbe668643e88111414c12627711f"),
        ("torchvision", "0.29.1", "ef220d4b21878c58947851beb7337382897159105f012f0c9c07bf7e84174d7e"))}
"""PyTorch's CPU builds by their wheels, for Linux on x86-64, so the rest of the torch environment resolves
from PyPI alone: an index beside PyPI's offers its own builds of other packages, with other hashes. Each
wheel carries its own hash, which uv otherwise replaces with those of every build of the version."""


def installed(distribution: str) -> str:
    """The running version of `distribution`, from its metadata, or from its module where conda installed
    it without any (NEST)."""
    module, _ = PINS[distribution]
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(importlib.import_module(module).__version__)


def require(*distributions: str) -> None:
    """Exit unless each of `distributions` runs at its pinned version."""
    wrong = [f"{name} {installed(name)} (pinned {PINS[name][1]})" for name in distributions
             if installed(name) != PINS[name][1]]
    if wrong:
        sys.exit(f"this fixture is made with the pinned versions, and the environment has "
                 f"{', '.join(wrong)}; install its lock file from {ENVIRONMENTS.relative_to(ROOT)}")


def require_checkout(path: Path, name: str) -> None:
    """Exit unless the git checkout at `path` is at the commit `CHECKOUTS[name]` names."""
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if not head.startswith(CHECKOUTS[name]):
        sys.exit(f"{name} at {path} is at {head}; this fixture is made from {CHECKOUTS[name]}: "
                 f"git -C {path} checkout {CHECKOUTS[name]}")


def require_program(name: str) -> None:
    """Exit unless the program `name` on the path is the version `PROGRAMS` names."""
    printed = subprocess.run([name, "--version"], capture_output=True, text=True, check=True).stdout
    found = re.search(r"\d+\.\d+\.\d+", printed.splitlines()[0])
    if found is None or found.group() != PROGRAMS[name]:
        sys.exit(f"this fixture is made with {name} {PROGRAMS[name]}, and the path has "
                 f"{printed.splitlines()[0]}")


def require_file(path: Path, name: str) -> None:
    """Exit unless the file at `path` is the reference file `FILES[name]` names, byte for byte."""
    if digest(path) != FILES[name]:
        sys.exit(f"{path} is not the {name} this fixture is made from (SHA-256 {FILES[name]})")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded() -> dict[str, str]:
    """Each fixture's checksum as `SHA256SUMS` records it."""
    lines = CHECKSUMS.read_text().splitlines()
    return {name: checksum for checksum, name in (line.split("  ", 1) for line in lines)}


def write_checksums() -> None:
    CHECKSUMS.write_text("".join(f"{digest(FIXTURES / name)}  {name}\n" for name in sorted(MADE_BY)))


def lock() -> None:
    """Compile each pip environment's lock file, pinned by `PINS` where it names a version, with hashes."""
    ENVIRONMENTS.mkdir(exist_ok=True)
    for environment, requirements in LOCKED.items():
        pinned = [WHEELS.get(name) or (f"{name}=={PINS[name][1]}" if name in PINS else name)
                  for name in requirements]
        source = ENVIRONMENTS / f"{environment}.in"
        source.write_text("".join(f"{line}\n" for line in pinned))
        overridden = []
        if environment in OVERRIDES:
            overrides = ENVIRONMENTS / f"{environment}.overrides"
            overrides.write_text("".join(f"{line}\n" for line in OVERRIDES[environment]))
            overridden = ["--override", str(overrides.relative_to(ROOT))]
        locked = (ENVIRONMENTS / f"{environment}.txt").relative_to(ROOT)
        # Resolved afresh: uv takes an existing output's pins and hashes as preferences, and its cache keeps
        # what another index once offered of a package.
        (ROOT / locked).unlink(missing_ok=True)
        subprocess.run(["uv", "pip", "compile", "--quiet", "--refresh", "--generate-hashes", "--universal",
                        "--python-version", "3.12", *overridden, str(source.relative_to(ROOT)),
                        "-o", str(locked)], check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("lock", "checksums"))
    if parser.parse_args().command == "lock":
        lock()
    else:
        write_checksums()


if __name__ == "__main__":
    main()
