"""Write Elephant's spike-train distances for sparx's parity tests.

Random spike trains on a 0.5 ms grid, their van Rossum distances at three
time constants and Victor-Purpura distances at three costs, from Elephant
(NeuralEnsemble), to `tests/fixtures/elephant.npz`.

    uv pip sync tools/environments/elephant.txt
    python tools/make_elephant_fixtures.py
"""

from pathlib import Path

import elephant
import neo
import numpy as np
import quantities as pq
from elephant.spike_train_dissimilarity import van_rossum_distance, victor_purpura_distance
from references import require

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "elephant.npz"
DT, STEPS, TRAINS = 0.5, 400, 6


def main():
    require("elephant")
    rng = np.random.default_rng(0)
    spikes = (rng.random((STEPS, TRAINS)) < 0.03).astype(np.float64)
    trains = [neo.SpikeTrain(np.flatnonzero(spikes[:, i]) * DT * pq.ms, t_stop=STEPS * DT * pq.ms)
              for i in range(TRAINS)]
    cases = {"spikes": spikes, "dt": np.array(DT), "meta/elephant": np.array(elephant.__version__)}
    for tau in (2.0, 10.0, 50.0):
        cases[f"van_rossum/{tau}"] = van_rossum_distance(trains, time_constant=tau * pq.ms)
    for cost in (0.0, 0.1, 2.0):
        cases[f"victor_purpura/{cost}"] = victor_purpura_distance(trains, cost_factor=cost / pq.ms)
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (elephant {elephant.__version__})")


if __name__ == "__main__":
    main()
