"""Neuromorphic datasets as dense, binned spike counts.

`shd` reads the Spiking Heidelberg Digits (Cramer et al., "The Heidelberg
Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks",
IEEE TNNLS 2020): spoken digits 0-9 in English and German, 20 classes,
rendered as spikes on 700 cochlear channels. Each record becomes a
`[steps, channels]` array of spike counts, batch-major as dew's loaders
expect; `sparx.dew.Events()` moves the time axis to the front. `SHD` is the
same data as a registered dew dataset spec (`datasets["shd"]`), which a
recipe names on its command line.

Reading the files needs h5py (`pip install "sparxml[datasets]"`).
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from dew.data import Dataset, DatasetSpec
from dew.data.dataset import Tokenize
from dew.registry import datasets
from numpy.typing import ArrayLike

__all__ = ["SHD", "SHD_URL", "bin_events", "shd"]

SHD_URL = "https://zenkelab.org/datasets/shd/{split}.h5.gz"
_CHANNELS = 700


def bin_events(times: ArrayLike, units: ArrayLike, steps: int, max_time: float, channels: int,
               source_channels: int = _CHANNELS) -> np.ndarray:
    """Count one record's spikes into `[steps, channels]` uint8 bins.

    Time `[0, max_time)` is cut into `steps` equal bins and events at or
    after `max_time` are dropped. `channels` must divide `source_channels`;
    adjacent source channels are pooled into each output channel. Counts
    saturate at 255.
    """
    if source_channels % channels:
        raise ValueError(f"channels must divide {source_channels}, not {channels}")
    step = np.floor(np.asarray(times, np.float64) / max_time * steps).astype(np.int64)
    kept = step < steps
    counts = np.zeros((steps, channels), np.int64)
    np.add.at(counts, (step[kept], np.asarray(units, np.int64)[kept] // (source_channels // channels)), 1)
    return np.minimum(counts, 255).astype(np.uint8)


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
        cache: str | Path | None = None, path: str | Path | None = None) -> dict[str, np.ndarray]:
    """The SHD `split` as `{"spikes": uint8 [N, steps, channels], "label": int32 [N]}`.

    `steps` bins over the first `max_time` seconds. Every spike of the train
    split falls before 1.37 s, and 100 steps of 14 ms is the binning of
    Zenke's SpyTorch SHD tutorial.
    The file is read from `path` when given, otherwise downloaded once into
    `cache` (default `~/.cache/sparx`; 131 MB for train, 38 MB for test) and
    decompressed beside it.
    """
    import h5py

    if path is None:
        path = _download(f"shd_{split}", Path.home() / ".cache" / "sparx" if cache is None else Path(cache))
    with h5py.File(path, "r") as data:
        nodes = [data.get(name) for name in ("spikes/times", "spikes/units", "labels")]
        times, units, labels = nodes
        if not (isinstance(times, h5py.Dataset) and isinstance(units, h5py.Dataset)
                and isinstance(labels, h5py.Dataset)):
            raise ValueError(f"{path} lacks SHD's spikes/times, spikes/units and labels datasets")
        spikes = np.stack([bin_events(times[i], units[i], steps, max_time, channels)
                           for i in range(len(times))])
        return {"spikes": spikes, "label": np.asarray(labels, np.int32)}


@datasets("shd")
@dataclass(frozen=True)
class SHD(DatasetSpec):
    """The Spiking Heidelberg Digits as a dew dataset: the train split to train on, the test
    split to validate on (SHD has no separate validation split), binned by `shd`.

    Records are `{"spikes": uint8 [steps, channels], "label": int32}`; `sparx.dew.Events()`
    turns a batch into the network's time-major input. Both splits are held in memory
    (about 700 MB at 100 steps over 700 channels), shuffled from `seed` every epoch, and
    each process reads its share of every batch.
    """

    steps: int = 100
    max_time: float = 1.4
    channels: int = 700
    cache: str | None = None

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        self.uncaptioned(tokenize)
        train = shd("train", self.steps, self.max_time, self.channels, self.cache)
        test = shd("test", self.steps, self.max_time, self.channels, self.cache)
        return Dataset.from_records(train, batch=batch, seed=self.seed, validation=test, loading=self.loading)
