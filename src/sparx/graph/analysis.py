"""Statistics of spike trains, for comparing networks that cannot match spike for spike.

NumPy only, so reference tools load this file outside sparx. Spikes are
`[steps, neurons]` arrays of 0 and 1 (or booleans) at step `dt` ms.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cv_isi", "firing_rates", "population_fano", "spike_steps", "victor_purpura"]


def firing_rates(spikes: np.ndarray, dt: float) -> np.ndarray:
    """Each neuron's mean rate, Hz."""
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
