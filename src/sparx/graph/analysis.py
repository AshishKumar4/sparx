"""Statistics of spike trains, for comparing networks that cannot match spike for spike.

NumPy only, so reference tools load this file outside sparx. Spikes are
`[steps, neurons]` arrays of 0 and 1 (or booleans) at step `dt` ms.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cv_isi", "firing_rates", "population_fano", "spike_steps"]


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
