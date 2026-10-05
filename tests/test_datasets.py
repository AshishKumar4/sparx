import numpy as np
import pytest

from sparx.datasets import bin_events, shd


def test_events_are_counted_into_their_time_and_channel_bins():
    times = np.asarray([0.0, 0.0139, 0.014, 0.5, 0.5, 1.3999, 1.4, 2.0])
    units = np.asarray([0, 699, 3, 10, 10, 5, 5, 5])
    out = bin_events(times, units, steps=100, max_time=1.4, channels=700)
    assert out.shape == (100, 700) and out.dtype == np.uint8
    expected = np.zeros((100, 700), np.uint8)
    expected[0, 0] = 1
    expected[0, 699] = 1
    expected[1, 3] = 1  # 14 ms opens the second bin
    expected[35, 10] = 2  # two spikes in one bin are counted
    expected[99, 5] = 1  # spikes at or after max_time are dropped
    np.testing.assert_array_equal(out, expected)


def test_adjacent_channels_pool_when_downsampled():
    out = bin_events(np.zeros(4), np.asarray([0, 4, 5, 699]), steps=2, max_time=1.0, channels=140)
    np.testing.assert_array_equal(out[0, :2], [2, 1])
    assert out[0, 139] == 1 and out.sum() == 4


def test_event_binning_opens_a_step_at_each_event_past_the_last_steps_reach():
    # 10 ms steps: the first holds events within 10 ms of 0 (10 ms itself included), the next
    # opens at 10.5 ms, and the silence after 20.5 ms is dropped, so 40 ms lands in step 2.
    times = np.asarray([0.0, 0.004, 0.010, 0.0105, 0.020, 0.040, 0.0405], np.float16)
    units = np.asarray([1, 2, 3, 4, 5, 6, 7])
    out = bin_events(times, units, steps=4, max_time=0.04, channels=700, binning="events")
    assert [sorted(np.flatnonzero(row).tolist()) for row in out] == [[1, 2, 3], [4, 5], [6, 7], []]
    # The grid puts the same events in 10 ms bins from 0.
    grid = bin_events(times.astype(np.float64), units, steps=5, max_time=0.05, channels=700)
    assert [sorted(np.flatnonzero(row).tolist()) for row in grid] == [[1, 2], [3, 4], [5], [], [6, 7]]
    empty = bin_events(np.zeros(0), np.zeros(0), steps=3, max_time=0.03, channels=700, binning="events")
    assert not empty.any()
    with pytest.raises(ValueError, match="binning"):
        bin_events(times, units, steps=4, max_time=0.04, channels=700, binning="edges")  # type: ignore[arg-type]


def test_channels_must_divide_the_source():
    with pytest.raises(ValueError, match="divide"):
        bin_events(np.zeros(1), np.zeros(1), steps=2, max_time=1.0, channels=300)


def test_shd_reads_the_published_file_layout(tmp_path):
    h5py = pytest.importorskip("h5py")  # the optional dependency `sparxml[datasets]`
    path = tmp_path / "shd_test.h5"
    with h5py.File(path, "w") as file:
        ragged = h5py.vlen_dtype(np.float32)
        times = file.create_dataset("spikes/times", (2,), dtype=ragged)
        units = file.create_dataset("spikes/units", (2,), dtype=h5py.vlen_dtype(np.uint16))
        times[0], units[0] = np.asarray([0.0, 0.75], np.float32), np.asarray([1, 2], np.uint16)
        times[1], units[1] = np.asarray([1.0], np.float32), np.asarray([699], np.uint16)
        file.create_dataset("labels", data=np.asarray([3, 19], np.uint16))
    data = shd("test", steps=10, max_time=1.4, path=path)
    assert data["spikes"].shape == (2, 10, 700)
    np.testing.assert_array_equal(data["label"], [3, 19])
    assert data["spikes"][0, 0, 1] == 1 and data["spikes"][0, 5, 2] == 1
    assert data["spikes"][1, 7, 699] == 1 and data["spikes"].sum() == 3


def test_shd_refuses_a_file_without_its_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("labels", data=np.zeros(2, np.uint16))
    with pytest.raises(ValueError, match="lacks SHD"):
        shd("test", path=path)
