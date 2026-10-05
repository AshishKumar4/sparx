"""Write the per-neuron firing rates of Shiu et al.'s published whole-brain simulations.

Their repository (github.com/philshiu/Drosophila_brain_model, cloned next
to sparx as `ref-shiu`) ships the spike times of their example experiments
on FlyWire v630: 21 sugar-sensing neurons activated at 200 Hz (`sugarR`;
their stimulated neurons fire at 200 Hz, as their notebook says, though
their `model.py` now defaults to 150 Hz)
and at 100 Hz (`sugarR_100Hz`), 30 trials of 1 s each, from their Brian2
model. This reduces them to each spiking neuron's mean rate and its
standard deviation over trials, in `tests/fixtures/shiu.npz`, which
`tests/test_connectome.py` compares sparx's `shiu2024` against.

    pip install pandas pyarrow
    python tools/make_shiu_fixtures.py [path to the cloned repository]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "shiu.npz"
TRIALS, DURATION = 30, 1.0


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ref-shiu"
    cases = {}
    for name in ("sugarR", "sugarR_100Hz"):
        spikes = pd.read_parquet(repo / "results" / "example" / f"{name}.parquet")
        counts = spikes.groupby(["flywire_id", "trial"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=range(TRIALS), fill_value=0)
        rates = counts.to_numpy() / DURATION
        cases[f"{name}/ids"] = counts.index.to_numpy(np.int64)
        cases[f"{name}/rate"] = rates.mean(1)
        cases[f"{name}/std"] = rates.std(1)
        print(name, len(counts), "neurons spiked; top rates", np.sort(rates.mean(1))[::-1][:5].round(1))
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
