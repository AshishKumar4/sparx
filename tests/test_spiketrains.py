from pathlib import Path

import numpy as np
import pytest

from sparx.spiketrains import rates_hz, victor_purpura

ELEPHANT = np.load(Path(__file__).parent / "fixtures" / "elephant.npz")


def test_rates_hz_counts_spikes_per_second_of_a_run_in_ms_steps():
    spikes = np.zeros((1000, 3), bool)
    spikes[::100, 0] = True  # 10 spikes in 100 ms
    spikes[::10, 1] = True  # 100 spikes in 100 ms
    np.testing.assert_allclose(rates_hz(spikes, 0.1), [100.0, 1000.0, 0.0])


@pytest.mark.parametrize("cost", [0.0, 0.1, 2.0])
def test_victor_purpura_matches_elephant(cost):
    spikes, dt = ELEPHANT["spikes"], float(ELEPHANT["dt"])
    times = [np.flatnonzero(spikes[:, i]) * dt for i in range(spikes.shape[1])]
    got = np.array([[victor_purpura(a, b, cost) for b in times] for a in times])
    np.testing.assert_allclose(got, ELEPHANT[f"victor_purpura/{cost}"], rtol=1e-12, atol=1e-12)
