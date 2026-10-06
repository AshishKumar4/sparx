"""Metrics over a spiking objective's evaluation, registered in dew's `metrics` table.

`Accuracy` is registered as `spike_accuracy`, a name of sparx's own, so a
dew release that adds a plain `accuracy` cannot collide with it. It reports
under `accuracy`, so a run logs `val/accuracy` (or `<split>/accuracy`).
"""

from __future__ import annotations

import numpy as np
from dew.artifacts import TokenScores
from dew.objectives.base import Batch, Shown
from dew.registry import metrics

__all__ = ["Accuracy"]


@metrics("spike_accuracy")
class Accuracy:
    """The share of examples whose predicted class is their label, over a whole pass.

    It reads the `TokenScores` that `SpikingClassifierObjective.evaluate`
    returns, one row per example. Each example counts by its weight, so the
    copies `sparx.datasets.whole_batches` adds to fill the last batch count
    for nothing.
    """

    name = "accuracy"
    reads = TokenScores
    shown = Shown(better="higher", percent=True)

    def __call__(self, scores: TokenScores, batch: Batch) -> tuple[float, float]:
        weights = np.asarray(scores.weights[:, 0], np.float64)
        return float(np.sum(np.asarray(scores.correct[:, 0]) * weights)), float(np.sum(weights))

    def merge(self, accumulated: tuple[float, float],
              contribution: tuple[float, float]) -> tuple[float, float]:
        return accumulated[0] + contribution[0], accumulated[1] + contribution[1]

    def finalize(self, accumulated: tuple[float, float]) -> float:
        total, count = accumulated
        if count <= 0:
            raise ValueError("accuracy: no counted example in the validation pass")
        return total / count
