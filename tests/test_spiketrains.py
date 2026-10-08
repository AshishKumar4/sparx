from pathlib import Path

import numpy as np
import pytest

from sparx.spiketrains import coincidence_factor, rates_hz, victor_purpura

ELEPHANT = np.load(Path(__file__).parent / "fixtures" / "elephant.npz")
GAMMA = np.load(Path(__file__).parent / "fixtures" / "gamma.npz")


def test_rates_hz_counts_spikes_per_second_of_a_run_in_ms_steps():
    spikes = np.zeros((1000, 3), bool)
    spikes[::100, 0] = True  # 10 spikes in 100 ms
    spikes[::10, 1] = True  # 100 spikes in 100 ms
    np.testing.assert_allclose(rates_hz(spikes, 0.1), [100.0, 1000.0, 0.0])  # observed 0 relative


@pytest.mark.parametrize("cost", [0.0, 0.1, 2.0])
def test_victor_purpura_matches_elephant(cost):
    spikes, dt = ELEPHANT["spikes"], float(ELEPHANT["dt"])
    times = [np.flatnonzero(spikes[:, i]) * dt for i in range(spikes.shape[1])]
    got = np.array([[victor_purpura(a, b, cost) for b in times] for a in times])
    # Observed 7.1e-15.
    np.testing.assert_allclose(got, ELEPHANT[f"victor_purpura/{cost}"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("window", [0.5, 2.0, 4.0])
def test_coincidence_factor_counts_brian2modelfittings_coincidences(window):
    data, model, dt = GAMMA["data"], GAMMA["model"], float(GAMMA["dt"])
    gamma = coincidence_factor(model, data, dt, window)
    reference = 1.0 - GAMMA[f"one_minus_gamma/{window}"]
    n, n_ref, duration = model.sum(0), data.sum(0), data.shape[0] * dt

    def coincidences(gamma, rate):
        chance = 2 * rate * window
        return gamma * 0.5 * (n + n_ref) * (1 - chance) + chance * n_ref

    # The same coincidences, each Γ taking nu from its own train: the model's here, the reference's there.
    np.testing.assert_allclose(coincidences(gamma, n / duration), coincidences(reference, n_ref / duration),
                               rtol=0, atol=1e-9)  # observed 4.0e-15
    same = n == n_ref
    assert same.sum() >= 6
    np.testing.assert_allclose(gamma[same], reference[same], rtol=0, atol=1e-12)  # observed 5.6e-17
