"""Neuromorphic datasets as dense, binned spike counts.

`shd` reads the Spiking Heidelberg Digits (Cramer et al., "The Heidelberg
Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks",
IEEE TNNLS 2020): spoken digits 0-9 in English and German, 20 classes,
rendered as spikes on 700 cochlear channels. Each record becomes a
`[steps, channels]` array of spike counts, batch-major as dew's loaders
expect; `sparx.encode.Events()` moves the time axis to the front. `SHD` is the
same data as a registered dew dataset spec (`datasets["shd"]`), which a
recipe names on its command line (`data:shd`). `write_synthetic_shd` writes
small files in SHD's layout, which smoke runs and tests read in its place.

`holdout`, `whole_batches` and `evaluation_pass` split records and score
every record of a split under dew's `Trainer`.

Reading the files needs h5py (`pip install "sparxml[datasets]"`).
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from dew.data import Dataset, DatasetSpec

# Remove when AshishKumar4/dew#39 merges: `Reader` and `Tokenize` in dew.data.dataset's `__all__`.
from dew.data.dataset import Reader, Tokenize
from dew.registry import datasets
from numpy.typing import ArrayLike

__all__ = ["SHD", "SHD_URL", "WEIGHT", "Binning", "bin_events", "evaluation_pass", "holdout", "shd",
           "whole_batches", "write_synthetic_shd"]

SHD_URL = "https://zenkelab.org/datasets/shd/{split}.h5.gz"
_CHANNELS = 700


type Binning = Literal["grid", "events"]
"""How `bin_events` cuts time into steps.

- `grid`: equal bins from time 0, the usual binning.
- `events`: SpikingJelly's SHD frames by duration, in the releases SNN-delays
  (Hammouamri et al., ICLR 2024) trained on: each step opens at the first
  event not yet counted and holds every event within one step's duration of
  it. Silences longer than a step are dropped, so a recording is shorter
  than on the grid and its timing is compressed; times are scaled to
  milliseconds in the file's own float16, as theirs are.
"""


def bin_events(times: ArrayLike, units: ArrayLike, steps: int, max_time: float, channels: int,
               source_channels: int = _CHANNELS, binning: Binning = "grid") -> np.ndarray:
    """Count one record's spikes into `[steps, channels]` uint8 bins.

    A step lasts `max_time / steps`. On the `grid`, time `[0, max_time)` is
    cut into `steps` equal bins and events at or after `max_time` are
    dropped; with `binning="events"` steps open at events (`Binning`) and
    steps past the `steps`-th are dropped. `channels` must divide
    `source_channels`; adjacent source channels are pooled into each output
    channel. Counts saturate at 255.
    """
    if source_channels % channels:
        raise ValueError(f"channels must divide {source_channels}, not {channels}")
    if binning == "grid":
        step = np.floor(np.asarray(times, np.float64) / max_time * steps).astype(np.int64)
    elif binning == "events":
        step = _event_frames(np.asarray(times), round(1000 * max_time / steps, 9))
    else:
        raise ValueError(f"binning must be grid or events, not {binning!r}")
    kept = step < steps
    counts = np.zeros((steps, channels), np.int64)
    np.add.at(counts, (step[kept], np.asarray(units, np.int64)[kept] // (source_channels // channels)), 1)
    return np.minimum(counts, 255).astype(np.uint8)


def _event_frames(times: np.ndarray, duration: float) -> np.ndarray:
    """Each event's frame under `Binning`'s `events`, `duration` in ms.

    SpikingJelly's `integrate_events_by_fixed_duration_shd` scans the events
    once: a frame starts at event `l` and takes every following event `r`
    while `t[r] - t[l] <= duration`. Its arithmetic is kept, `1000 * t` in
    the times' dtype, so the frames are theirs exactly.
    """
    t = 1000 * times
    if t.size == 0:
        return np.zeros(0, np.int64)
    starts = [0]
    while True:
        start = starts[-1]
        beyond = t[start:] - t[start] > duration
        if not beyond.any():
            break
        starts.append(start + int(np.argmax(beyond)))
    return np.searchsorted(np.asarray(starts), np.arange(t.size), side="right") - 1


def _download(split: str, cache: Path) -> Path:
    """The decompressed `{split}.h5` in `cache`, fetched and decompressed once."""
    path = cache / f"{split}.h5"
    if path.exists():
        return path
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{split}.h5.gz"
    if not archive.exists():
        partial = archive.with_suffix(".gz.partial")
        urllib.request.urlretrieve(SHD_URL.format(split=split), partial)
        partial.rename(archive)
    with gzip.open(archive) as source, open(path.with_suffix(".h5.partial"), "wb") as target:
        shutil.copyfileobj(source, target)
    path.with_suffix(".h5.partial").rename(path)
    return path


def shd(split: Literal["train", "test"], steps: int = 100, max_time: float = 1.4, channels: int = 700,
        cache: str | Path | None = None, path: str | Path | None = None,
        binning: Binning = "grid") -> dict[str, np.ndarray]:
    """The SHD `split` as `{"spikes": uint8 [N, steps, channels], "label": int32 [N]}`.

    `steps` bins over the first `max_time` seconds. Every spike of the train
    split falls before 1.37 s, and 100 steps of 14 ms is the binning of
    Zenke's SpyTorch SHD tutorial. `binning="events"` is SNN-delays' binning
    (`Binning`); at 10 ms its recordings last at most 124 steps (train) and
    105 (test), so `steps=124, max_time=1.24` keeps every event.
    The file is read from `path` when given, otherwise downloaded once into
    `cache` (default `~/.cache/sparx`; 131 MB for train, 38 MB for test) and
    decompressed beside it.
    """
    import h5py

    if path is None:
        path = _download(f"shd_{split}", Path.home() / ".cache" / "sparx" if cache is None else Path(cache))
    with h5py.File(path, "r") as file:
        nodes = [file.get(name) for name in ("spikes/times", "spikes/units", "labels")]
        times, units, labels = nodes
        if not (isinstance(times, h5py.Dataset) and isinstance(units, h5py.Dataset)
                and isinstance(labels, h5py.Dataset)):
            raise ValueError(f"{path} lacks SHD's spikes/times, spikes/units and labels datasets")
        spikes = np.stack([bin_events(times[i], units[i], steps, max_time, channels, binning=binning)
                           for i in range(len(times))])
        return {"spikes": spikes, "label": np.asarray(labels, np.int32)}


@datasets("shd")
@dataclass(frozen=True)
class SHD(DatasetSpec):
    """The Spiking Heidelberg Digits as a dew dataset: the train split to train on, the test
    split to validate on (SHD has no separate validation split), binned by `shd`.

    Records are `{"spikes": uint8 [steps, channels], "label": int32}`; `sparx.encode.Events()`
    turns a batch into the network's time-major input. Both splits are held in memory
    (about 700 MB at 100 steps over 700 channels), shuffled from `seed` every epoch, and
    each process reads its share of every batch.
    """

    steps: int = 100
    max_time: float = 1.4
    channels: int = 700
    cache: str | None = None
    binning: Binning = "grid"

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        self.uncaptioned(tokenize)
        train = shd("train", self.steps, self.max_time, self.channels, self.cache, binning=self.binning)
        test = shd("test", self.steps, self.max_time, self.channels, self.cache, binning=self.binning)
        # Until AshishKumar4/dew#40 merges, dew scores the test split in whole batches and leaves out
        # the last partial one; `evaluation_pass` scores every record meanwhile.
        return Dataset.from_records(train, batch=batch, seed=self.seed, validation=test, loading=self.loading)


def write_synthetic_shd(directory: str | Path, records: int = 64, seed: int = 0) -> Path:
    """Write `shd_train.h5` and `shd_test.h5` in SHD's layout into `directory`, and return it.

    Each split holds `records` recordings of 60 spikes over 1.4 s, labelled 0
    or 1 by which half of the 700 channels fires, so a network learns them in
    a few steps. `shd(split, cache=directory)` and `SHD(cache=directory)` read
    them as they read SHD's own files, which lets a smoke run or a test
    exercise the whole path without the 169 MB download.
    """
    import h5py

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    half = _CHANNELS // 2
    for split in ("train", "test"):
        labels = rng.integers(0, 2, records).astype(np.uint16)
        with h5py.File(directory / f"shd_{split}.h5", "w") as file:
            times = file.create_dataset("spikes/times", (records,), dtype=h5py.vlen_dtype(np.float32))
            units = file.create_dataset("spikes/units", (records,), dtype=h5py.vlen_dtype(np.uint16))
            for i, label in enumerate(labels):
                times[i] = np.sort(rng.uniform(0, 1.4, 60)).astype(np.float32)
                units[i] = (rng.integers(0, half, 60) + half * int(label)).astype(np.uint16)
            file.create_dataset("labels", data=labels)
    return directory


def holdout(records: Mapping[str, np.ndarray], fraction: float,
            seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split `records` (columns of equal length) into a random `1 - fraction` and `fraction`.

    SHD has no validation split, and SNN-delays selects its epochs on the
    test set; holding out part of the training set gives a validation set
    to select on, so the test accuracy stays an estimate.
    """
    sizes = {len(column) for column in records.values()}
    if len(sizes) != 1:
        raise ValueError(f"columns of one length split together, not lengths {sorted(sizes)}")
    total = sizes.pop()
    held = round(fraction * total)
    if not 0 < held < total:
        raise ValueError(f"holding out {fraction} of {total} records leaves one side empty")
    order = np.random.default_rng(seed).permutation(total)
    kept, out = np.sort(order[held:]), np.sort(order[:held])
    return ({name: column[kept] for name, column in records.items()},
            {name: column[out] for name, column in records.items()})


# Remove when AshishKumar4/dew#40 merges: `validation_pass` fills the last batch itself and marks
# each row in `dew.data.dataset.COUNTED`, so `WEIGHT`, `whole_batches` and `evaluation_pass` go.
WEIGHT = "weight"
"""The batch field evaluation weighs each example by, when a batch holds it.

`whole_batches` writes it: 1 for a record, 0 for the copies that fill the
last batch, so a split of any size is scored over exactly its records."""


def whole_batches(records: Mapping[str, np.ndarray], batch: int) -> dict[str, np.ndarray]:
    """`records` filled to whole batches with copies of its first record, weighted under `WEIGHT`.

    dew scores a split in whole batches only, so the records past the last
    whole one would go unscored. Each record has weight 1 and each copy 0,
    which `SpikingClassifierObjective`'s evaluation and `sparx.metrics.Accuracy`
    honor.
    """
    total = len(next(iter(records.values())))
    fill = -total % batch
    padded = {name: np.concatenate([column, np.repeat(column[:1], fill, axis=0)])
              for name, column in records.items()}
    padded[WEIGHT] = np.concatenate([np.ones(total, np.float32), np.zeros(fill, np.float32)])
    return padded


def evaluation_pass(records: Mapping[str, np.ndarray], batch: int) -> Reader:
    """One pass over every record, in order, for `Trainer.fit(validation={...})`.

    The records are filled to whole batches first (`whole_batches`), so a
    split such as SHD's 2264 test recordings is scored over all of them.
    """
    padded = whole_batches(records, batch)
    reader = Dataset.from_records(padded, batch=batch, validation=padded).val
    assert reader is not None  # from_records reads a validation split it is given
    return reader
