"""Write the statistics of NEST's Brunel (2000) network in each of his regimes, for sparx's comparison.

Runs NEST's `brunel_delta_nest.py` model at `ORDER` (4 ORDER excitatory
and ORDER inhibitory neurons) in the four regimes of Brunel's Figure 8,
with several seeds, and saves the excitatory population's mean rate, mean
CV of interspike intervals and population Fano factor over
`[SKIP, DURATION]` ms to `tests/fixtures/brunel.npz`. Chaotic networks do
not match spike for spike, so `tests/test_graph.py` compares these
statistics, with the spread over seeds as the scale of agreement. The
statistics are `sparx.spiketrains`, loaded from its file.

    conda env create -f tools/environments/nest.yml
    python tools/make_brunel_fixtures.py
"""

import importlib.util
from pathlib import Path

import nest
import numpy as np
from references import require

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "brunel.npz"
ORDER, DURATION, SKIP, DT, SEEDS = 500, 400.0, 100.0, 0.1, tuple(range(1, 9))
REGIMES = {"sr": (3.0, 2.0), "ai": (5.0, 2.0), "si_fast": (6.0, 4.0), "si_slow": (4.5, 0.9)}

spec = importlib.util.spec_from_file_location("spiketrains", ROOT / "src" / "sparx" / "spiketrains.py")
spiketrains = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spiketrains)


def run(g, eta, seed, j=0.1, delay=1.5):
    nest.ResetKernel()
    nest.resolution = DT
    nest.rng_seed = seed
    excitatory, inhibitory = 4 * ORDER, ORDER
    c_e, c_i = excitatory // 10, inhibitory // 10
    params = {"C_m": 250.0, "tau_m": 20.0, "t_ref": 2.0, "E_L": 0.0, "V_reset": 10.0, "V_m": 0.0,
              "V_th": 20.0}
    e = nest.Create("iaf_psc_delta", excitatory, params=params)
    i = nest.Create("iaf_psc_delta", inhibitory, params=params)
    rate = eta * 20.0 / (j * c_e * 20.0) * 1000.0 * c_e
    noise = nest.Create("poisson_generator", params={"rate": rate})
    nest.Connect(noise, e + i, syn_spec={"weight": j, "delay": delay})
    nest.Connect(e, e + i, {"rule": "fixed_indegree", "indegree": c_e}, {"weight": j, "delay": delay})
    nest.Connect(i, e + i, {"rule": "fixed_indegree", "indegree": c_i}, {"weight": -g * j, "delay": delay})
    recorder = nest.Create("spike_recorder")
    nest.Connect(e, recorder)
    nest.Simulate(DURATION)
    events = recorder.events
    spikes = np.zeros((round(DURATION / DT), excitatory), bool)
    spikes[np.rint(events["times"] / DT).astype(int) - 1, events["senders"] - e[0].global_id] = True
    window = spikes[round(SKIP / DT):]
    return (spiketrains.rates_hz(window, DT).mean(), spiketrains.cv_isi(window).mean(),
            spiketrains.population_fano(window, DT))


def main():
    require("nest-simulator")
    nest.set_verbosity("M_ERROR")
    cases = {"meta/nest": np.array(nest.__version__), "meta/order": np.array(ORDER),
             "meta/duration": np.array(DURATION), "meta/skip": np.array(SKIP)}
    for name, (g, eta) in REGIMES.items():
        stats = np.array([run(g, eta, seed) for seed in SEEDS])
        cases[f"{name}/stats"] = stats
        print(name, "rate, cv, fano per seed\n", stats.round(3))
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (nest {nest.__version__})")


if __name__ == "__main__":
    main()
