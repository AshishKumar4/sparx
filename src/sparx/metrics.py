"""Metrics over a spiking objective's evaluation, which dew's `Trainer.fit(metrics=...)` takes.

`Accuracy` reports under `accuracy`, so a run logs `val/accuracy` (or
`<split>/accuracy`), and a run's record names it by import path,
`sparx.metrics:Accuracy`.
"""

from __future__ import annotations

import numpy as np
from dew.artifacts import Artifact, TokenScores
from dew.objectives.base import Batch, Shown, mean_of_totals, merge_totals

__all__ = ["Accuracy"]


class Accuracy:
    """The share of examples whose predicted class is their label, over a whole pass.

    It reads the `TokenScores` that `SpikingClassifierObjective.evaluate`
    returns, one row per example, each counted by its weight. Dew's
    validation pass hands a metric the real rows alone, so the repeats that
    fill a split's last batch count for nothing.
    """

    name = "accuracy"
    reads = TokenScores
    shown = Shown(better="higher", percent=True)

    def __call__(self, scores: Artifact, batch: Batch, /) -> tuple[float, float]:
        assert isinstance(scores, TokenScores), "the trainer hands a metric the artifact it reads"
        weights = np.asarray(scores.weights[:, 0], np.float64)
        return float(np.sum(np.asarray(scores.correct[:, 0]) * weights)), float(np.sum(weights))

    merge = staticmethod(merge_totals)

    def finalize(self, accumulated: tuple[float, float]) -> float:
        if accumulated[1] <= 0:
            raise ValueError("accuracy: no counted example in the validation pass")
        return mean_of_totals(accumulated)
