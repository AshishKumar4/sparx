"""Write the statistics of Brian2's CUBA and COBA networks (Brette et al. 2007's benchmarks) for sparx.

Both are Vogels and Abbott's (2005) network: 3,200 excitatory and 800
inhibitory LIF neurons, connected with probability 0.02, no external
input, activity sustained from random initial voltages. CUBA is Brian2's
`examples/CUBA.py` (current-based synapses in voltage units, `exact`
integration, `El = -49 mV` above threshold); COBA has conductance-based
synapses (reversal 0 and -80 mV, 6 and 67 nS, rest -60 mV, random initial
conductances), integrated by `exponential_euler`. Synapses act without
delay. Saves, per seed, the excitatory and inhibitory mean rates, the mean
CV of interspike intervals and the population Fano factor over
`[SKIP, DURATION]` ms to `tests/fixtures/benchmarks.npz`.

    uv pip sync tools/environments/brian2.txt
    python tools/make_brian2_benchmarks.py
"""

import importlib.util
from pathlib import Path

import brian2 as b2
import numpy as np
from references import require

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "benchmarks.npz"
DURATION, SKIP, DT, SEEDS = 500.0, 100.0, 0.1, (1, 2, 3, 4)

spec = importlib.util.spec_from_file_location("spiketrains", ROOT / "src" / "sparx" / "spiketrains.py")
spiketrains = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spiketrains)

CUBA = """
dv/dt = (ge + gi - (v - El)) / taum : volt (unless refractory)
dge/dt = -ge / taue : volt
dgi/dt = -gi / taui : volt
"""
COBA = """
dv/dt = (gl * (El - v) + ge * (Ee - v) + gi * (Ei - v)) / Cm : volt (unless refractory)
dge/dt = -ge / taue : siemens
dgi/dt = -gi / taui : siemens
"""


def run(model, seed):
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.seed(seed)
    b2.defaultclock.dt = DT * b2.ms
    namespace = {"taum": 20 * b2.ms, "taue": 5 * b2.ms, "taui": 10 * b2.ms, "Vt": -50 * b2.mV,
                 "Vr": -60 * b2.mV, "Ee": 0 * b2.mV, "Ei": -80 * b2.mV, "gl": 10 * b2.nS, "Cm": 200 * b2.pF}
    if model == "cuba":
        namespace.update(El=-49 * b2.mV, we=60 * 0.27 / 10 * b2.mV, wi=-20 * 4.5 / 10 * b2.mV)
        group = b2.NeuronGroup(4000, CUBA, threshold="v > Vt", reset="v = Vr", refractory=5 * b2.ms,
                               method="exact", namespace=namespace)
        group.ge, group.gi = 0 * b2.mV, 0 * b2.mV
    else:
        namespace.update(El=-60 * b2.mV, we=6 * b2.nS, wi=67 * b2.nS)
        group = b2.NeuronGroup(4000, COBA, threshold="v > Vt", reset="v = Vr", refractory=5 * b2.ms,
                               method="exponential_euler", namespace=namespace)
        group.ge = "(randn() * 1.5 + 4) * 10 * nS"
        group.gi = "(randn() * 12 + 20) * 10 * nS"
    group.v = "Vr + rand() * (Vt - Vr)"
    excitatory = b2.Synapses(group, group, on_pre="ge += we", namespace=namespace)
    inhibitory = b2.Synapses(group, group, on_pre="gi += wi", namespace=namespace)
    excitatory.connect("i < 3200", p=0.02)
    inhibitory.connect("i >= 3200", p=0.02)
    monitor = b2.SpikeMonitor(group)
    b2.run(DURATION * b2.ms)
    spikes = np.zeros((round(DURATION / DT), 4000), bool)
    spikes[np.rint(np.asarray(monitor.t / b2.ms) / DT).astype(int), np.asarray(monitor.i)] = True
    window = spikes[round(SKIP / DT):]
    e, i = window[:, :3200], window[:, 3200:]
    return (spiketrains.rates_hz(e, DT).mean(), spiketrains.rates_hz(i, DT).mean(),
            spiketrains.cv_isi(e).mean(), spiketrains.population_fano(e, DT))


def main():
    require("brian2")
    cases = {"meta/brian2": np.array(b2.__version__), "meta/duration": np.array(DURATION),
             "meta/skip": np.array(SKIP)}
    for model in ("cuba", "coba"):
        stats = np.array([run(model, seed) for seed in SEEDS])
        cases[f"{model}/stats"] = stats
        print(model, "rate e, rate i, cv e, fano e per seed\n", stats.round(3))
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (brian2 {b2.__version__})")


if __name__ == "__main__":
    main()
