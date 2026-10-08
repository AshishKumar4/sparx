"""Statistics of and distances between spike trains, for comparing networks that cannot match spike
for spike.

Spikes are `[steps, neurons]` arrays of 0 and 1 (or booleans) at step `dt`
ms. These read a finished run (a `sparx.graph.simulate` record, a reference
simulator's output) and have no gradient. The module imports NumPy alone,
so the fixture tools (`tools/make_brunel_fixtures.py`,
`tools/make_brian2_benchmarks.py`) load this file in a NEST or Brian2
environment without JAX or sparx, and the statistics they record for the
reference are the code the tests apply to sparx.

Van Rossum's distance (`sparx.losses.van_rossum`) also compares spike
trains. It lives in `sparx.losses` because it is written in JAX to train a
network by its gradient, as `sparx.objectives.ActivityFitObjective` does; here it would make
the tools above load JAX. The split is by use: a differentiable training
target is a loss, a measurement of a finished run is here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["coincidence_factor", "cv_isi", "population_fano", "rates_hz", "spike_steps", "victor_purpura"]


def rates_hz(spikes: np.ndarray, dt: float) -> np.ndarray:
    """Each neuron's mean rate in Hz, over a run of `dt` ms steps.

    `sparx.rates.firing_rates` reads a different quantity during training,
    each layer's mean rate in spikes per step from the sown collection.
    """
    spikes = np.asarray(spikes)
    return spikes.sum(0) / (spikes.shape[0] * dt / 1000.0)


def spike_steps(spikes: np.ndarray) -> list[np.ndarray]:
    """Each neuron's spike steps."""
    steps, neurons = np.nonzero(np.asarray(spikes))
    order = np.argsort(neurons, kind="stable")
    return np.split(steps[order], np.cumsum(np.bincount(neurons, minlength=np.shape(spikes)[1]))[:-1])


def cv_isi(spikes: np.ndarray, minimum: int = 3) -> np.ndarray:
    """The coefficient of variation of each neuron's interspike intervals, for neurons with at least
    `minimum` intervals; 1 for a Poisson process, 0 for a clock."""
    cvs = []
    for steps in spike_steps(spikes):
        isi = np.diff(steps)
        if len(isi) >= minimum:
            cvs.append(isi.std() / isi.mean())
    return np.asarray(cvs)


def population_fano(spikes: np.ndarray, dt: float, window: float = 1.0) -> float:
    """The Fano factor of the population's spike count in windows of `window` ms.

    Near 1 when neurons fire independently (Brunel's asynchronous states),
    far above 1 when they fire together (his synchronous states).
    """
    spikes = np.asarray(spikes)
    per = max(1, round(window / dt))
    usable = spikes.shape[0] // per * per
    counts = spikes[:usable].sum(1).reshape(-1, per).sum(1)
    return float(counts.var() / counts.mean()) if counts.mean() > 0 else 0.0


def victor_purpura(times_a: np.ndarray, times_b: np.ndarray, cost: float) -> float:
    """Victor and Purpura's (J. Neurophysiol. 1996) distance between two spike trains (times in ms).

    The cheapest way to turn one train into the other, at 1 to add or remove
    a spike and `cost` per ms to move one: `cost = 0` counts the difference
    in spike counts, a large `cost` counts unmatched spikes. Computed by
    their dynamic program, `O(len(a) len(b))`; not differentiable.
    """
    a, b = np.sort(np.asarray(times_a, float)), np.sort(np.asarray(times_b, float))
    table = np.zeros((len(a) + 1, len(b) + 1))
    table[:, 0] = np.arange(len(a) + 1)
    table[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        moves = table[i - 1, :-1] + cost * np.abs(a[i - 1] - b)
        row = table[i]
        for j in range(1, len(b) + 1):
            row[j] = min(table[i - 1, j] + 1, row[j - 1] + 1, moves[j - 1])
    return float(table[-1, -1])


def coincidence_factor(spikes: np.ndarray, reference: np.ndarray, dt: float, window: float) -> np.ndarray:
    """Each neuron's coincidence factor Γ, how well its spikes in `spikes` predict those in `reference`,
    both `[steps, neurons]` at step `dt` ms, with a precision of `window` ms.

        Γ = (N_coinc - 2 nu Δ N_ref) / (½ (N + N_ref) (1 - 2 nu Δ))

    `N_coinc` counts the reference's spikes with one of the neuron's within
    `Δ`, `window` rounded to steps, and `nu = N / duration` is the rate of
    the neuron in `spikes`, so `2 nu Δ N_ref` is what a Poisson train of that
    rate meets by chance. Γ is 1 for the same spikes, 0 on average for a
    Poisson train, and below 0 for fewer coincidences than chance. This is
    Jolivet and Gerstner's (J. Physiol. Paris 2004, eq. 13) form of Kistler
    et al.'s (Neural Comput. 1997) measure, the score of spike-time
    prediction in Jolivet et al.'s (J. Neurosci. Methods 2008) benchmark.
    brian2modelfitting's `get_gamma_factor` counts the same coincidences
    and takes `nu` from the reference; the two agree where the counts do.
    A neuron silent in both trains has none (NaN).
    """
    spikes, reference = np.asarray(spikes), np.asarray(reference)
    within = round(window / dt)
    delta, duration = within * dt, spikes.shape[0] * dt
    gamma = []
    for steps, target in zip(spike_steps(spikes), spike_steps(reference), strict=True):
        n, n_ref = len(steps), len(target)
        if n + n_ref == 0:
            gamma.append(np.nan)
            continue
        coincident = 0
        if n:
            after = np.clip(np.searchsorted(steps, target), 0, n - 1)
            before = np.clip(after - 1, 0, n - 1)
            nearest = np.minimum(np.abs(steps[after] - target), np.abs(steps[before] - target))
            coincident = int(np.sum(nearest <= within))
        chance = 2 * (n / duration) * delta
        gamma.append((coincident - chance * n_ref) / (0.5 * (n + n_ref) * (1 - chance)))
    return np.asarray(gamma)
