"""The simulators page's traces: sparx beside NEST, Brian2 and Izhikevich's own code, on the inputs of the
committed reference fixtures, run by the same helpers the parity tests use (tests/test_simulators.py):

    python site/lab/parity.py --out site/public/data/parity.json

Each case keeps both voltage traces at the fixture's resolution, both spike trains, and the largest
difference between the voltages up to the first spike that differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests")]

import test_simulators as ts  # noqa: E402

from sparx.dynamics import IZHIKEVICH_2004, IzhikevichState, SynapticInput, izhikevich_2004, run  # noqa: E402


def case(
    name: str, title: str, reference: str, kind: str, mine, theirs, dt: float, note: str, test: str
) -> dict:
    (fired, v), (their_fired, their_v) = mine, theirs
    fired, their_fired = np.asarray(fired) > 0, np.asarray(their_fired) > 0
    differ = np.flatnonzero(fired != their_fired)
    until = differ[0] if len(differ) else len(fired)
    difference = np.abs(np.asarray(v) - np.asarray(their_v))
    error = float(difference[:until].max())
    return {
        "name": name,
        "title": title,
        "reference": reference,
        "kind": kind,
        "dt": dt,
        "note": note,
        "test": test,
        "sparx": {"v": np.round(v, 4).tolist(), "spikes": np.flatnonzero(fired).tolist()},
        "difference": [float(f"{d:.2e}") for d in difference],
        "theirs": {"v": np.round(their_v, 4).tolist(), "spikes": np.flatnonzero(their_fired).tolist()},
        "error": error,
        "same_spikes": not len(differ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="site/public/data/parity.json")
    args = parser.parse_args()
    nest, brian2, dt = ts.NEST, ts.BRIAN2, ts.DT
    cases = []

    model = "iaf_psc_exp"
    cell = ts.PointNeuron(
        ts.nest_lif(model),
        {
            "ex": ts.Receptor(ts.Exponential(ts.param(model, "tau_syn_ex"))),
            "in": ts.Receptor(ts.Exponential(ts.param(model, "tau_syn_in"))),
        },
    )
    fired, v = ts.nest_run(cell, model)
    cases.append(
        case(
            "lif",
            "LIF, exponential current synapses",
            "NEST iaf_psc_exp",
            "exact",
            (fired[:, 0], v[:, 0]),
            (nest[f"{model}/spikes"][:, 0], nest[f"{model}/v"][:, 0]),
            dt,
            "Both integrate the membrane and the synaptic currents exactly over each step.",
            "test_current_synapses_match_nest_to_rounding",
        )
    )

    model = "iaf_cond_exp"
    fired, v = ts.nest_run(ts.conductance_cell(model), model, sign=-1.0)
    cases.append(
        case(
            "cond",
            "LIF, conductance synapses",
            "NEST iaf_cond_exp (adaptive RK45)",
            "tolerance",
            (fired[:, 0], v[:, 0]),
            (nest[f"{model}/spikes"][:, 0], nest[f"{model}/v"][:, 0]),
            dt,
            "sparx holds each conductance at its exact mean over the step, a second-order scheme; NEST "
            "integrates adaptively to within 1e-9 mV of the truth.",
            "test_conductance_synapses_fire_with_nest_spike_for_spike",
        )
    )

    cell = ts.conductance_cell(model, hold="start", t_ref=ts.param(model, "t_ref") - dt)
    fired, v = ts.nest_run(cell, model, sign=-1.0)
    cases.append(
        case(
            "brian2",
            "LIF, conductance synapses",
            "Brian2 exponential_euler",
            "exact",
            (fired[:, 0], v[:, 0]),
            (brian2["coba/spikes"][:, 0], brian2["coba/v"][:, 0]),
            dt,
            'With hold="start" sparx holds the conductance at its value at the start of the step, as '
            "Brian2 does, and counts refractoriness one step shorter, as Brian2 does.",
            "test_brian2_exponential_euler_is_the_start_of_step_hold",
        )
    )

    fired, v = ts.izhikevich_run("regular_spiking", "published_1", exact=True)
    cases.append(
        case(
            "izhikevich",
            "Izhikevich, regular spiking",
            "NEST izhikevich",
            "exact",
            (fired, v),
            (
                nest["izhikevich/regular_spiking/published_1/spikes"][:, 0],
                nest["izhikevich/regular_spiking/published_1/v"][:, 0],
            ),
            1.0,
            'Run op by op in float64 with order="nest", it is NEST\'s run to the last bit.',
            "test_izhikevich_published_scheme_is_nests_to_the_last_bit",
        )
    )

    patterns = []
    figure = ts.IZHIKEVICH_FIGURE_1
    for panel, pattern in zip("ABCDEFGHIJKLMNOPQRST", IZHIKEVICH_2004, strict=True):
        data = {key.split("/")[1]: figure[key] for key in figure if key[0] == panel}
        neuron = izhikevich_2004(pattern)
        with jax.enable_x64(new_val=True):
            state = IzhikevichState(jnp.asarray([data["V0"]]), jnp.asarray([data["u0"]]))
            (spikes, v), _ = run(
                neuron,
                SynapticInput(jnp.asarray(data["II"])[:, None]),
                dt=float(data["tau"]),
                state=state,
                record=lambda s: s.v,
            )
        v = np.asarray(v[:, 0])
        theirs = np.asarray(data["VV"])
        fired = np.asarray(spikes.value[:, 0]) > 0
        quiet = theirs != 30
        patterns.append(
            {
                "panel": panel,
                "pattern": pattern,
                "dt": float(data["tau"]),
                "current": np.round(np.asarray(data["II"]), 6).tolist(),
                "sparx": np.round(np.where(fired, 30.0, v), 4).tolist(),
                "theirs": np.round(theirs, 4).tolist(),
                "same_spikes": bool(np.array_equal(np.flatnonzero(fired), np.flatnonzero(~quiet))),
                "spikes": int(fired.sum()),
                "error": float(np.abs(v[quiet & ~fired] - theirs[quiet & ~fired]).max()),
            }
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"cases": cases, "patterns": patterns}))


if __name__ == "__main__":
    main()
