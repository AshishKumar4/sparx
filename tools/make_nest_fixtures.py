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
}


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


def main():
    nest.set_verbosity("M_ERROR")
    rng = np.random.default_rng(0)
    cases = {"meta/nest": np.array(nest.__version__), "meta/dt": np.array(DT)}
    for model, (params, (low, high)) in CASES.items():
        out = run(model, params, trains(rng, low, high))
        cases.update({f"{model}/{k}": v for k, v in out.items()})
        cases.update({f"{model}/param/{k}": np.array(v) for k, v in params.items()})
        print(model, "spikes per neuron", out["spikes"].sum(0))
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (nest {nest.__version__})")


if __name__ == "__main__":
    main()
