"""Write brian2modelfitting's coincidence factors for sparx's parity tests.

Pairs of spike trains on a 0.1 ms grid over 2 s: a reference train
with a 3 ms dead time, and a model train that is the same train, the same
jittered, jittered with spikes dropped and added, an independent train of
another rate, one spike, or none. For each pair and three precisions,
`get_gamma_factor(..., rate_correction=False)`, which is 1 - Γ, to
`tests/fixtures/gamma.npz`.

    pip install brian2modelfitting==0.4 "numpy<2" "scipy<1.14" "bayesian-optimization<2"
    python tools/make_gamma_fixtures.py
"""

from pathlib import Path

import brian2
import brian2modelfitting
import numpy as np
from brian2 import ms
from brian2modelfitting.metric import get_gamma_factor

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gamma.npz"
DT, STEPS = 0.1, 20000
WINDOWS = (0.5, 2.0, 4.0)


def train(rng, rate, dead=30):
    """Spike steps at about `rate` Hz with at least `dead` steps between spikes."""
    gaps = dead + rng.exponential(1e4 / rate - dead, STEPS)
    steps = np.cumsum(gaps).astype(int)
    return steps[steps < STEPS]


def jitter(rng, steps, width):
    return np.unique(np.clip(steps + rng.integers(-width, width + 1, len(steps)), 0, STEPS - 1))


def main():
    rng = np.random.default_rng(0)
    data, model = [], []
    for _ in range(3):
        reference = train(rng, 25.0)
        kept = jitter(rng, reference, 25)[rng.random(len(reference)) > 0.3] if len(reference) else reference
        variants = (reference, jitter(rng, reference, 10), jitter(rng, reference, 25),
                    np.union1d(kept, train(rng, 8.0)), train(rng, 60.0), reference[:1], reference[:0])
        for steps in variants:
            data.append(reference)
            model.append(steps)
    raster = np.zeros((2, STEPS, len(data)), np.uint8)
    for i, (d, m) in enumerate(zip(data, model, strict=True)):
        raster[0, d, i] = raster[1, m, i] = 1
    cases = {"data": raster[0], "model": raster[1], "dt": np.array(DT),
             "meta/brian2modelfitting": np.array(brian2modelfitting.__version__),
             "meta/brian2": np.array(brian2.__version__)}
    for window in WINDOWS:
        cases[f"one_minus_gamma/{window}"] = np.array([
            float(get_gamma_factor(m * DT * ms, d * DT * ms, window * ms, STEPS * DT * ms, DT * ms,
                                   rate_correction=False))
            for d, m in zip(data, model, strict=True)])
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (brian2modelfitting {brian2modelfitting.__version__}, brian2 {brian2.__version__})")


if __name__ == "__main__":
    main()
