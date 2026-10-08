"""Write NEST's response of two passive cells coupled by a gap junction, for sparx's gap junction tests.

NEST's gap-junction neurons are Hodgkin-Huxley models (`hh_psc_alpha_gap`);
with their sodium and potassium conductances set to zero, each is a passive
membrane, `C dV/dt = -g_L (V - E_L) + I_e + I_gap`, which sparx's
`GradedPotential` integrates. Two cells start at different voltages under
different currents and are joined by one `gap_junction` of conductance `g`
(nS), with NEST's waveform relaxation (`use_wfr`, its default) and without
it. Saves the voltages after every step to `tests/fixtures/nest_gap.npz`.

    conda env create -f tools/environments/nest.yml
    python tools/make_nest_gap_fixtures.py

The voltage NEST records at `T` is sparx's after step `T / dt - 1`.
"""

from pathlib import Path

import nest
import numpy as np
from references import require

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "nest_gap.npz"
DT, STEPS = 0.1, 200
CELL = {"C_m": 40.0, "g_L": 10.0, "E_L": -70.0, "g_Na": 0.0, "g_Kv1": 0.0, "g_Kv3": 0.0}
CURRENTS = (200.0, -100.0)
START = (-70.0, -50.0)
COUPLINGS = (5.0, 50.0)


def run(g: float, wfr: bool) -> np.ndarray:
    nest.ResetKernel()
    nest.resolution = DT
    nest.use_wfr = wfr
    cells = nest.Create("hh_psc_alpha_gap", 2, params=CELL)
    for cell, current, v in zip(cells, CURRENTS, START, strict=True):
        cell.set({"I_e": current, "V_m": v})
    nest.Connect(cells[0], cells[1], {"rule": "one_to_one", "make_symmetric": True},
                 {"synapse_model": "gap_junction", "weight": g})
    meter = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": DT})
    nest.Connect(meter, cells)
    nest.Simulate((STEPS + 1) * DT)  # the multimeter records the step ending at T one step later
    events = meter.events
    v = np.zeros((STEPS, 2))
    keep = events["times"] <= STEPS * DT + DT / 2
    steps = np.rint(events["times"][keep] / DT).astype(int) - 1
    v[steps, events["senders"][keep] - cells[0].global_id] = events["V_m"][keep]
    return v


def main():
    require("nest-simulator")
    nest.set_verbosity("M_ERROR")
    cases = {"meta/nest": np.array(nest.__version__), "meta/dt": np.array(DT),
             "currents": np.array(CURRENTS), "start": np.array(START),
             **{f"param/{k}": np.array(v) for k, v in CELL.items()}}
    for g in COUPLINGS:
        for wfr in (True, False):
            cases[f"g{g:g}/{'wfr' if wfr else 'single'}"] = run(g, wfr)
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (nest {nest.__version__})")


if __name__ == "__main__":
    main()
