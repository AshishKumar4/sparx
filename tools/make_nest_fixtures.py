"""Write NEST's single-neuron responses for sparx's parity tests.

Drives NEST's integrate-and-fire models with random weighted spike trains on
an excitatory and an inhibitory input plus a constant current, and records
the membrane voltage every step and the spikes. Saves them with the inputs
to `tests/fixtures/nest.npz`, which `tests/test_reference_nest.py` compares
`sparx.dynamics` against.

    pip install nest-simulator==<version below>
    python tools/make_nest_fixtures.py

Times are mapped onto sparx's steps (`sparx.dynamics.core`): NEST stamps an
event that arrives at time `T` into the step ending at `T`, so arrivals at
`T` are sparx's input of step `T / dt - 1`, and the voltage NEST records at
`T` is sparx's after that step.
"""

from pathlib import Path

import nest
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "nest.npz"
DT, STEPS, NEURONS, DELAY = 0.1, 3000, 3, 0.1
COMMON = {"C_m": 250.0, "tau_m": 10.0, "t_ref": 2.0, "E_L": -70.0, "V_reset": -70.0, "V_th": -55.0,
          "I_e": 300.0}
COND = {"C_m": 250.0, "g_L": 16.6667, "t_ref": 2.0, "E_L": -70.0, "V_reset": -70.0, "V_th": -55.0,
        "I_e": 200.0, "E_ex": 0.0, "E_in": -85.0}
CASES = {
    "iaf_psc_exp": ({**COMMON, "tau_syn_ex": 2.0, "tau_syn_in": 5.0}, (50.0, 400.0)),
    "iaf_psc_alpha": ({**COMMON, "tau_syn_ex": 2.0, "tau_syn_in": 5.0}, (50.0, 400.0)),
    "iaf_psc_delta": ({**COMMON}, (0.2, 2.0)),
    "iaf_cond_exp": ({**COND, "tau_syn_ex": 2.0, "tau_syn_in": 5.0}, (1.0, 8.0)),
    "iaf_cond_alpha": ({**COND, "tau_syn_ex": 2.0, "tau_syn_in": 5.0}, (1.0, 8.0)),
    "iaf_cond_beta": ({**COND, "tau_rise_ex": 0.5, "tau_decay_ex": 2.0, "tau_rise_in": 1.0,
                       "tau_decay_in": 5.0}, (1.0, 8.0)),
    "aeif_psc_exp": ({"I_e": 800.0, "tau_syn_ex": 2.0, "tau_syn_in": 5.0, "t_ref": 0.0}, (50.0, 400.0)),
    "aeif_cond_exp": ({"I_e": 800.0, "tau_syn_ex": 2.0, "tau_syn_in": 5.0, "t_ref": 1.0}, (1.0, 8.0)),
}
# Naud, Marcille, Clopath and Gerstner (Biol. Cybern. 2008), Table 1: the
# firing patterns of AdEx under a current step. C pF, g_L nS, E_L mV,
# V_T mV, Delta_T mV, tau_w ms, a nS, b pA, V_reset mV, I pA.
NAUD = {
    "tonic": (200, 10, -70, -50, 2, 30, 2, 0, -58, 500),
    "adapting": (200, 12, -70, -50, 2, 300, 2, 60, -58, 500),
    "initial_burst": (130, 18, -58, -50, 2, 150, 4, 120, -50, 400),
    "regular_bursting": (200, 10, -58, -50, 2, 120, 2, 100, -46, 210),
    "delayed_accelerating": (200, 12, -70, -50, 2, 300, -10, 0, -58, 300),
    "delayed_regular_bursting": (100, 10, -65, -50, 2, 90, -10, 30, -47, 110),
    "transient": (100, 10, -65, -50, 2, 90, 10, 100, -47, 180),
    "irregular": (100, 12, -60, -50, 2, 130, -11, 30, -48, 160),
}
NAUD_STEPS = 5000
# Izhikevich (2003), Figure 2: (a, b, c, d) per class.
IZHIKEVICH = {
    "regular_spiking": (0.02, 0.2, -65.0, 8.0),
    "intrinsically_bursting": (0.02, 0.2, -55.0, 4.0),
    "chattering": (0.02, 0.2, -50.0, 2.0),
    "fast_spiking": (0.1, 0.2, -65.0, 2.0),
    "low_threshold_spiking": (0.02, 0.25, -65.0, 2.0),
    "thalamo_cortical": (0.02, 0.25, -65.0, 0.05),
    "resonator": (0.1, 0.26, -65.0, 2.0),
}
IZHIKEVICH_TIME = 300.0


def trains(rng, low, high):
    """Random input times on the grid and weights, per neuron: excitatory and inhibitory."""
    out = []
    for _ in range(NEURONS):
        neuron = {}
        for sign, rate in (("ex", 0.15), ("in", 0.06)):
            steps = np.flatnonzero(rng.random(STEPS - 20) < rate) + 1
            weights = rng.uniform(low, high, len(steps)) * (1 if sign == "ex" else -1)
            neuron[sign] = (steps, weights)
        out.append(neuron)
    return out


def run(model, params, inputs):
    nest.ResetKernel()
    nest.resolution = DT
    neurons = nest.Create(model, NEURONS, params=params)
    meter = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": DT})
    recorder = nest.Create("spike_recorder")
    arrivals = {sign: np.zeros((STEPS, NEURONS)) for sign in ("ex", "in")}
    for i, neuron in enumerate(inputs):
        for sign, (steps, weights) in neuron.items():
            generator = nest.Create("spike_generator",
                                    params={"spike_times": steps * DT, "spike_weights": weights})
            nest.Connect(generator, neurons[i], syn_spec={"weight": 1.0, "delay": DELAY})
            arrival = np.rint(steps + DELAY / DT).astype(int) - 1
            np.add.at(arrivals[sign][:, i], arrival, weights)
    nest.Connect(meter, neurons)
    nest.Connect(neurons, recorder)
    nest.Simulate((STEPS + 1) * DT)  # the multimeter records the step ending at T one step later
    events = meter.events
    v = np.zeros((STEPS, NEURONS))
    first = neurons[0].global_id
    keep = events["times"] <= STEPS * DT + DT / 2
    steps = np.rint(events["times"][keep] / DT).astype(int) - 1
    v[steps, events["senders"][keep] - first] = events["V_m"][keep]
    spikes = np.zeros((STEPS, NEURONS))
    sent = recorder.events
    keep = sent["times"] <= STEPS * DT + DT / 2
    spikes[np.rint(sent["times"][keep] / DT).astype(int) - 1, sent["senders"][keep] - first] = 1
    return {"v": v, "spikes": spikes, **{f"arrivals_{k}": a for k, a in arrivals.items()}}


def naud(name, values):
    """One `aeif_cond_exp` neuron at rest under Naud et al.'s current from t = 0."""
    keys = ("C_m", "g_L", "E_L", "V_th", "Delta_T", "tau_w", "a", "b", "V_reset", "I_e")
    params = {**dict(zip(keys, map(float, values), strict=True)), "V_peak": 0.0, "t_ref": 0.0}
    nest.ResetKernel()
    nest.resolution = DT
    neuron = nest.Create("aeif_cond_exp", params={**params, "V_m": params["E_L"]})
    recorder = nest.Create("spike_recorder")
    nest.Connect(neuron, recorder)
    nest.Simulate(NAUD_STEPS * DT)
    spikes = np.zeros(NAUD_STEPS)
    spikes[np.rint(recorder.events["times"] / DT).astype(int) - 1] = 1
    return {"spikes": spikes, **{f"param/{k}": np.array(v) for k, v in params.items()}}


def izhikevich(values, dt, consistent, current=10.0, trains=None):
    """One `izhikevich` neuron per entry of `trains` (or one), under constant `current`, at `dt`."""
    a, b, c, d = values
    count = 1 if trains is None else len(trains)
    steps = round(IZHIKEVICH_TIME / dt)
    nest.ResetKernel()
    nest.resolution = dt
    # NEST starts u at -13 whatever b is; Izhikevich's code starts it at b * v.
    neurons = nest.Create("izhikevich", count, params={"a": a, "b": b, "c": c, "d": d, "I_e": current,
                                                      "V_m": -65.0, "U_m": b * -65.0,
                                                      "consistent_integration": consistent})
    meter = nest.Create("multimeter", params={"record_from": ["V_m", "U_m"], "interval": dt})
    recorder = nest.Create("spike_recorder")
    arrivals = np.zeros((steps, count))
    for i, train in enumerate(trains or []):
        times, weights = train
        generator = nest.Create("spike_generator",
                                params={"spike_times": times * dt, "spike_weights": weights})
        nest.Connect(generator, neurons[i], syn_spec={"weight": 1.0, "delay": dt})
        np.add.at(arrivals[:, i], times, weights)
    nest.Connect(meter, neurons)
    nest.Connect(neurons, recorder)
    nest.Simulate((steps + 1) * dt)
    first = neurons[0].global_id
    events = meter.events
    keep = events["times"] <= steps * dt + dt / 2
    rows = np.rint(events["times"][keep] / dt).astype(int) - 1
    v, u = np.zeros((steps, count)), np.zeros((steps, count))
    v[rows, events["senders"][keep] - first] = events["V_m"][keep]
    u[rows, events["senders"][keep] - first] = events["U_m"][keep]
    spikes = np.zeros((steps, count))
    sent = recorder.events
    keep = sent["times"] <= steps * dt + dt / 2
    spikes[np.rint(sent["times"][keep] / dt).astype(int) - 1, sent["senders"][keep] - first] = 1
    return {"v": v, "u": u, "spikes": spikes, "arrivals": arrivals}


def main():
    nest.set_verbosity("M_ERROR")
    rng = np.random.default_rng(0)
    cases = {"meta/nest": np.array(nest.__version__), "meta/dt": np.array(DT)}
    for model, (params, (low, high)) in CASES.items():
        out = run(model, params, trains(rng, low, high))
        cases.update({f"{model}/{k}": v for k, v in out.items()})
        cases.update({f"{model}/param/{k}": np.array(v) for k, v in params.items()})
        print(model, "spikes per neuron", out["spikes"].sum(0))
    for name, values in NAUD.items():
        out = naud(name, values)
        cases.update({f"naud/{name}/{k}": v for k, v in out.items()})
        print("naud", name, "spikes", out["spikes"].sum())
    for name, values in IZHIKEVICH.items():
        for label, dt, consistent in (("published_1", 1.0, False), ("published_01", 0.1, False),
                                      ("euler_01", 0.1, True)):
            out = izhikevich(values, dt, consistent)
            kept = ("spikes", "v", "u") if label == "published_1" else ("spikes",)
            cases.update({f"izhikevich/{name}/{label}/{k}": out[k] for k in kept})
        print("izhikevich", name, "spikes", out["spikes"].sum())
    # Random delta input on top of a weaker current, three neurons, both schemes.
    delta = []
    for _ in range(3):
        times = np.flatnonzero(rng.random(round(IZHIKEVICH_TIME / 0.1) - 20) < 0.03) + 1
        delta.append((times, rng.uniform(-3.0, 6.0, len(times))))
    for label, consistent in (("euler", True), ("published", False)):
        out = izhikevich(IZHIKEVICH["regular_spiking"], 0.1, consistent, current=4.0, trains=delta)
        cases.update({f"izhikevich/delta_{label}/{k}": v for k, v in out.items()})
        print("izhikevich delta", label, "spikes", out["spikes"].sum(0))
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (nest {nest.__version__})")


if __name__ == "__main__":
    main()
