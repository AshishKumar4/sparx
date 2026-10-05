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
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (brian2 {b2.__version__})")


if __name__ == "__main__":
    main()
