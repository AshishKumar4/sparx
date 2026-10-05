"""Write Brian2's single-neuron responses for sparx's parity tests.

Runs a conductance-based LIF (Brian2's `exponential_euler`) and a
current-based one (`exact`), driven by the weighted arrivals of
`tests/fixtures/nest.npz` (run `tools/make_nest_fixtures.py` first) plus a
constant current. Arrivals are added to the synaptic variables at the end
of each step (`run_regularly(when="end")`), after the threshold and reset,
which is when Brian2's `on_pre` lands a spike sent with no delay. Saves the
voltage after every step and the spikes to `tests/fixtures/brian2.npz`.

    pip install brian2==<version below>
    python tools/make_brian2_fixtures.py
"""

from pathlib import Path

import brian2 as b2
import numpy as np

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUT = FIXTURES / "brian2.npz"

COBA = """
dv/dt = (g_l * (e_l - v) + ge * (e_ex - v) + gi * (e_in - v) + i_e) / c_m : volt (unless refractory)
dge/dt = -ge / tau_ex : siemens
dgi/dt = -gi / tau_in : siemens
"""
CUBA = """
dv/dt = (g_l * (e_l - v) + ie + ii + i_e) / c_m : volt (unless refractory)
die/dt = -ie / tau_ex : amp
dii/dt = -ii / tau_in : amp
"""


HH = """
dv/dt = (g_na * m**3 * h * (e_na - v) + g_k * n**4 * (e_k - v) + g_l * (e_l - v) + i_e) / c_m : volt
dm/dt = alpha_m * (1 - m) - beta_m * m : 1
dh/dt = alpha_h * (1 - h) - beta_h * h : 1
dn/dt = alpha_n * (1 - n) - beta_n * n : 1
alpha_m = 0.1 * (v / mV + 40) / (1 - exp(-(v / mV + 40) / 10)) / ms : Hz
beta_m = 4 * exp(-(v / mV + 65) / 18) / ms : Hz
alpha_h = 0.07 * exp(-(v / mV + 65) / 20) / ms : Hz
beta_h = 1 / (1 + exp(-(v / mV + 35) / 10)) / ms : Hz
alpha_n = 0.01 * (v / mV + 55) / (1 - exp(-(v / mV + 55) / 10)) / ms : Hz
beta_n = 0.125 * exp(-(v / mV + 65) / 80) / ms : Hz
i_e : amp
"""


def hodgkin_huxley(currents, dt, steps):
    """NEST's `hh_psc_alpha` membrane in Brian2 by `exponential_euler`, at rest under each current."""
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = dt * b2.ms
    namespace = {"g_na": 12000 * b2.nS, "g_k": 3600 * b2.nS, "g_l": 30 * b2.nS, "e_na": 50 * b2.mV,
                 "e_k": -77 * b2.mV, "e_l": -54.402 * b2.mV, "c_m": 100 * b2.pF}
    group = b2.NeuronGroup(len(currents), HH, method="exponential_euler", namespace=namespace)
    rest = -65.0
    rates = {"m": (0.1 * (rest + 40) / (1 - np.exp(-(rest + 40) / 10)), 4 * np.exp(-(rest + 65) / 18)),
             "h": (0.07 * np.exp(-(rest + 65) / 20), 1 / (1 + np.exp(-(rest + 35) / 10))),
             "n": (0.01 * (rest + 55) / (1 - np.exp(-(rest + 55) / 10)), 0.125 * np.exp(-(rest + 65) / 80))}
    group.v = rest * b2.mV
    for gate, (alpha, beta) in rates.items():
        setattr(group, gate, alpha / (alpha + beta))
    group.i_e = np.asarray(currents) * b2.pA
    monitor = b2.StateMonitor(group, "v", record=True, when="end")
    b2.run(steps * dt * b2.ms)
    return {"v": np.asarray(monitor.v / b2.mV).T, "currents": np.asarray(currents)}


SHIU = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
rfc : second
"""


def shiu(seed=0, size=120, steps=3000):
    """Shiu et al.'s (2024) whole-brain neuron model, as their `model.py` states it, on a small random graph.

    Their equations, reset (`v = v_rst; g = 0`), refractoriness (`g`
    frozen while refractory, none for stimulated neurons), synapses
    (`g += w`, 1.8 ms delay, weight = signed synapse count x 0.275 mV), and
    stimulation onto `v` (their `PoissonInput` weight, 68.75 mV), here
    from fixed random trains so the run is deterministic.
    """
    rng = np.random.default_rng(seed)
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = 0.1 * b2.ms
    namespace = {"v_0": -52 * b2.mV, "v_rst": -52 * b2.mV, "v_th": -45 * b2.mV, "t_mbr": 20 * b2.ms,
                 "tau": 5 * b2.ms}
    neurons = b2.NeuronGroup(size, SHIU, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
                             refractory="rfc", namespace=namespace)
    neurons.v, neurons.g, neurons.rfc = -52 * b2.mV, 0 * b2.mV, 2.2 * b2.ms
    stimulated = np.arange(10)
    neurons.rfc[stimulated] = 0 * b2.ms
    pre, post = np.nonzero(rng.random((size, size)) < 0.08)
    counts = rng.integers(1, 30, len(pre)) * np.where(rng.random(len(pre)) < 0.75, 1, -1)
    synapses = b2.Synapses(neurons, neurons, "w : volt", on_pre="g += w", delay=1.8 * b2.ms)
    synapses.connect(i=pre, j=post)
    synapses.w = counts * 0.275 * b2.mV
    times = [(i, step) for i in stimulated for step in np.flatnonzero(rng.random(steps - 10) < 0.015)]
    indices, stamp = np.array([t[0] for t in times]), np.array([t[1] for t in times])
    generator = b2.SpikeGeneratorGroup(size, indices, stamp * 0.1 * b2.ms)
    drive = b2.Synapses(generator, neurons, on_pre="v += 68.75 * mV")
    drive.connect(j="i")
    monitor = b2.SpikeMonitor(neurons)
    b2.run(steps * 0.1 * b2.ms)
    fired = np.zeros((steps, size))
    fired[np.rint(np.asarray(monitor.t / b2.ms) / 0.1).astype(int), np.asarray(monitor.i)] = 1
    return {"spikes": fired, "pre": pre, "post": post, "counts": counts, "stim_neuron": indices,
            "stim_step": stamp}


def run(equations, method, unit, arrivals, nest, model):
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    dt = float(nest["meta/dt"])
    b2.defaultclock.dt = dt * b2.ms
    steps, n = arrivals["ex"].shape

    def p(name):
        return float(nest[f"{model}/param/{name}"])

    g_l = p("g_L") if f"{model}/param/g_L" in nest else p("C_m") / p("tau_m")
    namespace = {"g_l": g_l * b2.nS, "e_l": p("E_L") * b2.mV, "c_m": p("C_m") * b2.pF,
                 "i_e": p("I_e") * b2.pA,
                 "e_ex": 0 * b2.mV, "e_in": -85 * b2.mV, "tau_ex": 2.0 * b2.ms, "tau_in": 5.0 * b2.ms,
                 "v_th": p("V_th") * b2.mV, "v_reset": p("V_reset") * b2.mV,
                 "a_ex": b2.TimedArray(arrivals["ex"] * unit, dt=dt * b2.ms),
                 "a_in": b2.TimedArray(arrivals["in"] * unit, dt=dt * b2.ms)}
    group = b2.NeuronGroup(n, equations, threshold="v >= v_th", reset="v = v_reset",
                           refractory=p("t_ref") * b2.ms, method=method, namespace=namespace)
    group.v = p("E_L") * b2.mV
    names = ("ge", "gi") if "ge" in equations else ("ie", "ii")
    group.run_regularly(f"{names[0]} += a_ex(t, i)\n{names[1]} += a_in(t, i)", when="end")
    monitor = b2.StateMonitor(group, "v", record=True, when="end")
    spikes = b2.SpikeMonitor(group)
    b2.run(steps * dt * b2.ms)
    fired = np.zeros((steps, n))
    fired[np.rint(np.asarray(spikes.t / b2.ms) / dt).astype(int), np.asarray(spikes.i)] = 1
    return {"v": np.asarray(monitor.v / b2.mV).T, "spikes": fired}


def main():
    nest = np.load(FIXTURES / "nest.npz")
    cases = {"meta/brian2": np.array(b2.__version__)}
    coba = {"ex": nest["iaf_cond_exp/arrivals_ex"], "in": -nest["iaf_cond_exp/arrivals_in"]}
    cuba = {"ex": nest["iaf_psc_exp/arrivals_ex"], "in": nest["iaf_psc_exp/arrivals_in"]}
    for name, out in (("coba", run(COBA, "exponential_euler", b2.nS, coba, nest, "iaf_cond_exp")),
                      ("cuba", run(CUBA, "exact", b2.pA, cuba, nest, "iaf_psc_exp"))):
        cases.update({f"{name}/{k}": v for k, v in out.items()})
        print(name, "spikes per neuron", out["spikes"].sum(0))
    cases.update({f"shiu/{k}": v for k, v in shiu().items()})
    print("shiu spikes", cases["shiu/spikes"].sum())
    hh = hodgkin_huxley(list(nest["hh_currents/currents"]), 0.1, 1000)
    cases.update({f"hh/{k}": v for k, v in hh.items()})
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (brian2 {b2.__version__})")


if __name__ == "__main__":
    main()
