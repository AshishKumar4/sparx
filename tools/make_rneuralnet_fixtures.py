"""Write RNeuralNet-Research's own outputs, local rewards and weights for sparx's parity test.

Compiles the original `header.h`, `neuron.h`, `neuron.cpp`, `Processor.h`
and `Processor.cpp` of github.com/AshishKumar4/RNeuralNet-Research,
unchanged, with `tools/rneuralnet_driver.cpp`, which runs them
single-threaded in the order `sparx.learn.RNeuralNet` describes, and a
`gnuplot-iostream.h` that plots nothing (the original plots from inside
its loops). The network is drawn as `NeuralNet_init` draws its own, at a
size a test can inspect: 40 neurons of 4 connections each (none receiving
more than 4) with normal weights of deviation 1/2 and `Myelin` 0 to 19,
thresholds normal around 2 with deviation 0.5, 3 input neurons of 2
connections, 2 output neurons, each output neuron's connection to the
reward feeder made with its others. It runs 80 ticks of sparse random
inputs, large enough that outputs cross their thresholds, with rewards of
1, -0.5 and 2 at ticks 25, 50 and 70, and records each tick every
neuron's last output (`OldOutput`), and at each reward every neuron's local
reward and every weight.

Saves to `tests/fixtures/rneuralnet.npz`. The committed fixture came from
RNeuralNet-Research d4b7803a5bbe87747d27a7137cc05a756bef42f7 with g++ 13.3.0:

    git clone https://github.com/AshishKumar4/RNeuralNet-Research ../ref-RNeuralNet-Research
    python tools/make_rneuralnet_fixtures.py [path to the cloned repository]
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from references import require_checkout

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "rneuralnet.npz"
SOURCES = ("header.h", "neuron.h", "neuron.cpp", "Processor.h", "Processor.cpp")
NEURONS, FAN, INPUTS, INPUT_FAN, OUTPUTS, TICKS = 40, 4, 3, 2, 2, 80
REWARDS = {25: 1.0, 50: -0.5, 70: 2.0}
NO_PLOTS = """#pragma once
// Plotting is disabled: the simulation loop sends nothing to gnuplot.
struct Gnuplot {
  template <class T> Gnuplot& operator<<(const T&) { return *this; }
  template <class T> void send1d(const T&) {}
};
"""


def network(rng):
    """Connections in the order `NeuralNet_init` makes them, with each output neuron's connection to the
    feeder among its others."""
    threshold = rng.normal(2.0, 0.5, NEURONS).astype(np.float32)
    received = np.zeros(NEURONS, int)
    feeder = NEURONS + INPUTS
    taken = rng.permutation(NEURONS)[:INPUTS * INPUT_FAN + OUTPUTS]
    fed, outputs = taken[:INPUTS * INPUT_FAN].reshape(INPUTS, INPUT_FAN), np.sort(taken[INPUTS * INPUT_FAN:])
    connections = []
    for source in range(NEURONS):
        chosen = []
        for _ in range(FAN):
            free = [j for j in range(NEURONS) if j != source and received[j] < FAN and j not in chosen]
            if not free:
                break
            chosen.append(int(rng.choice(free)))
            received[chosen[-1]] += 1
        connections += [(source, j, rng.normal(0, 1 / np.sqrt(FAN)), int(rng.integers(0, 20)))
                        for j in chosen]
        if source in outputs:
            connections.append((source, feeder, 1.0, 4))
    connections += [(NEURONS + i, int(j), 1.0, 1) for i in range(INPUTS) for j in fed[i]]
    pre, post, weight, myelin = (np.asarray(c) for c in zip(*connections, strict=True))
    return threshold, pre, post, weight.astype(np.float32), myelin, outputs


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ref-RNeuralNet-Research"
    require_checkout(repo, "RNeuralNet-Research")
    commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    version = subprocess.run(["g++", "--version"], capture_output=True, text=True, check=True)
    compiler = version.stdout.splitlines()[0]
    rng = np.random.default_rng(0)
    threshold, pre, post, weight, myelin, outputs = network(rng)
    x = ((rng.random((TICKS, INPUTS)) < 0.3) * rng.normal(0.0, 3.0, (TICKS, INPUTS))).astype(np.float32)
    reward = np.zeros(TICKS, np.float32)
    reward[list(REWARDS)] = list(REWARDS.values())

    feeder = NEURONS + INPUTS
    kind = np.where(post == feeder, 1, np.where(pre >= NEURONS, 2, 0))
    ends = np.where(kind == 2, pre - NEURONS, pre), np.where(kind == 1, -1, post)
    spec = [f"{NEURONS} {INPUTS} {OUTPUTS} {len(pre)} {TICKS}", " ".join(f"{v:.9g}" for v in threshold),
            " ".join(map(str, outputs))]
    rows = zip(kind, *ends, weight, myelin, strict=True)
    spec += [f"{k} {a} {b} {w:.9g} {m}" for k, a, b, w, m in rows]
    spec += [" ".join(f"{v:.9g}" for v in row) for row in x] + [" ".join(f"{v:.9g}" for v in reward)]
    with tempfile.TemporaryDirectory() as build:
        build = Path(build)
        for name in SOURCES:
            shutil.copy(repo / name, build / name)
        (build / "gnuplot-iostream.h").write_text(NO_PLOTS)
        shutil.copy(ROOT / "tools" / "rneuralnet_driver.cpp", build / "driver.cpp")
        subprocess.run(["g++", "-w", "-O2", "-std=gnu++11", "driver.cpp", "-o", "driver"], cwd=build,
                       check=True)
        subprocess.run([build / "driver", build / "trace.txt"], input="\n".join(spec), text=True, check=True,
                       stdout=subprocess.DEVNULL)
        trace = (build / "trace.txt").read_text().splitlines()
        lines = [np.array(line.split(), np.float32) for line in trace]
    activity, credit, weights = [], [], []
    for t in range(TICKS):
        activity.append(lines.pop(0)[:feeder])  # the feeder never sends, so it has no last output
        if reward[t]:
            credit.append(lines.pop(0))
            weights.append(lines.pop(0))
    np.savez_compressed(OUT, threshold=threshold, pre=pre, post=post, weight=weight, myelin=myelin,
                        inputs=INPUTS, outputs=outputs, x=x, reward=reward, activity=np.stack(activity),
                        credit=np.stack(credit), weights=np.stack(weights), commit=commit, compiler=compiler)
    print(f"wrote {OUT} from {commit} with {compiler}")


if __name__ == "__main__":
    main()
