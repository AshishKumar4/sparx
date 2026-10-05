"""sparx.dynamics against NEST and Brian2, the reference neural simulators, and a fine RK4 ground truth.

Fixtures come from `tools/make_nest_fixtures.py` and
`tools/make_brian2_fixtures.py`: three neurons per model, 300 ms at
0.1 ms, driven by random weighted spike trains on an excitatory and an
inhibitory receptor and a constant current.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reference

from sparx.dynamics import (
    LIF,
    Alpha,
    Arrivals,
    BiExponential,
    Delta,
    Exponential,
    PointNeuron,
    Receptor,
    integrate,
)

FIXTURES = Path(__file__).parent / "fixtures"
NEST = np.load(FIXTURES / "nest.npz")
BRIAN2 = np.load(FIXTURES / "brian2.npz")
DT = float(NEST["meta/dt"])
REVERSAL = {"ex": 0.0, "in": -85.0}


def param(model, name):
    return float(NEST[f"{model}/param/{name}"])


def nest_lif(model, **overrides):
    if f"{model}/param/tau_m" in NEST:
        tau_m = param(model, "tau_m")
    else:
        tau_m = param(model, "C_m") / param(model, "g_L")
    fields = {"tau_m": tau_m, "c_m": param(model, "C_m"), "e_l": param(model, "E_L"),
              "v_th": param(model, "V_th"), "v_reset": param(model, "V_reset"),
              "t_ref": param(model, "t_ref"), "reversal": REVERSAL}
    return LIF(**{**fields, **overrides})


def run(cell, model, sign=1.0):
    """`cell` driven by `model`'s fixture inputs in float64: spikes and the voltage after every step.
    Conductance weights are stored with the inhibitory sign; `sign=-1` makes them positive."""
    arrivals = {"ex": NEST[f"{model}/arrivals_ex"], "in": sign * NEST[f"{model}/arrivals_in"]}
    with jax.enable_x64(new_val=True):
        inputs = Arrivals(jnp.full(arrivals["ex"].shape, param(model, "I_e")),
                          {k: jnp.asarray(v) for k, v in arrivals.items()})
        (spikes, v), _ = integrate(cell, inputs, DT, record=lambda state: state.neuron.v)
    return np.asarray(spikes.fired), np.asarray(v)


@pytest.mark.parametrize(("model", "synapse"), [("iaf_psc_exp", Exponential), ("iaf_psc_alpha", Alpha)])
def test_current_synapses_match_nest_to_rounding(model, synapse):
    # NEST integrates these exactly (Rotter and Diesmann 1999), as sparx does.
    cell = PointNeuron(nest_lif(model), {"ex": Receptor(synapse(param(model, "tau_syn_ex"))),
                                         "in": Receptor(synapse(param(model, "tau_syn_in")))})
    fired, v = run(cell, model)
    np.testing.assert_array_equal(fired, NEST[f"{model}/spikes"])
    np.testing.assert_allclose(v, NEST[f"{model}/v"], atol=1e-11)
    assert fired.sum() >= 20


def test_delta_synapses_match_nest_to_rounding():
    cell = PointNeuron(nest_lif("iaf_psc_delta"), {"ex": Receptor(Delta()), "in": Receptor(Delta())})
    fired, v = run(cell, "iaf_psc_delta")
    np.testing.assert_array_equal(fired, NEST["iaf_psc_delta/spikes"])
    np.testing.assert_allclose(v, NEST["iaf_psc_delta/v"], atol=1e-11)


CONDUCTANCES = {
    "iaf_cond_exp": lambda m: (Exponential(param(m, "tau_syn_ex")), Exponential(param(m, "tau_syn_in"))),
    "iaf_cond_alpha": lambda m: (Alpha(param(m, "tau_syn_ex")), Alpha(param(m, "tau_syn_in"))),
    "iaf_cond_beta": lambda m: (BiExponential(param(m, "tau_rise_ex"), param(m, "tau_decay_ex")),
                                BiExponential(param(m, "tau_rise_in"), param(m, "tau_decay_in"))),
}


def conductance_cell(model, hold="mean", **overrides):
    ex, inh = CONDUCTANCES[model](model)
    return PointNeuron(nest_lif(model, **overrides), {"ex": Receptor(ex, "conductance"),
                                                      "in": Receptor(inh, "conductance")}, hold=hold)


@pytest.mark.parametrize("model", list(CONDUCTANCES))
def test_conductance_synapses_fire_with_nest_spike_for_spike(model):
    # NEST integrates conductance models by adaptive RK45 to within 1e-9 mV
    # of the truth here; sparx holds each conductance at its average over
    # the step, a second-order scheme, and stays within 2e-3 mV. Trajectories
    # that close can only part where one of them grazes the threshold.
    tolerance = 2e-3
    fired, v = run(conductance_cell(model), model, sign=-1.0)
    expected, expected_v = NEST[f"{model}/spikes"], NEST[f"{model}/v"]
    for neuron in range(fired.shape[1]):
        differ = np.flatnonzero(fired[:, neuron] != expected[:, neuron])
        until = differ[0] if len(differ) else len(fired)
        if len(differ):
            silent = v if fired[until, neuron] == 0 else expected_v
            assert abs(silent[until, neuron] - param(model, "V_th")) < tolerance
        assert until > 0.9 * len(fired)
        np.testing.assert_allclose(v[:until, neuron], expected_v[:until, neuron], atol=tolerance)
        assert expected[:until, neuron].sum() >= 30


def test_brian2_exponential_euler_is_the_start_of_step_hold():
    # Brian2 counts refractoriness from the start of the step in which the
    # neuron crossed, one step less than NEST and sparx (docs/fidelity.md).
    cell = conductance_cell("iaf_cond_exp", hold="start", t_ref=param("iaf_cond_exp", "t_ref") - DT)
    fired, v = run(cell, "iaf_cond_exp", sign=-1.0)
    np.testing.assert_array_equal(fired, BRIAN2["coba/spikes"])
    np.testing.assert_allclose(v, BRIAN2["coba/v"], atol=1e-9)


def test_brian2_exact_current_synapses_are_sparxs():
    model = "iaf_psc_exp"
    cell = PointNeuron(nest_lif(model, t_ref=param(model, "t_ref") - DT),
                       {"ex": Receptor(Exponential(2.0)), "in": Receptor(Exponential(5.0))})
    fired, v = run(cell, model)
    np.testing.assert_array_equal(fired, BRIAN2["cuba/spikes"])
    np.testing.assert_allclose(v, BRIAN2["cuba/v"], atol=1e-9)


def test_conductance_hold_converges_at_second_order_to_the_rk4_truth():
    model, steps = "iaf_cond_exp", 300
    errors = []
    for dt, factor in ((DT, 1), (DT / 2, 2)):
        arrivals = {"ex": np.zeros((steps * factor, 3)), "in": np.zeros((steps * factor, 3))}
        arrivals["ex"][factor - 1::factor] = NEST[f"{model}/arrivals_ex"][:steps]
        arrivals["in"][factor - 1::factor] = -NEST[f"{model}/arrivals_in"][:steps]
        current = np.full((steps * factor, 3), param(model, "I_e"))
        truth, truth_spikes = reference.conductance_lif(
            arrivals, current, c_m=param(model, "C_m"), g_l=param(model, "g_L"), e_l=param(model, "E_L"),
            v_th=0.0, v_reset=param(model, "V_reset"), t_ref=param(model, "t_ref"),
            reversal=REVERSAL, kernels={"ex": lambda s: np.exp(-s / 2.0), "in": lambda s: np.exp(-s / 5.0)},
            dt=dt, substeps=10)
        assert truth_spikes.sum() == 0  # below threshold, so the error is the integrator's alone
        cell = conductance_cell(model, v_th=0.0)
        with jax.enable_x64(new_val=True):
            inputs = Arrivals(jnp.asarray(current), {k: jnp.asarray(a) for k, a in arrivals.items()})
            (_, v), _ = integrate(cell, inputs, dt, record=lambda state: state.neuron.v)
        errors.append(np.abs(np.asarray(v) - truth).max())
    assert errors[0] < 1e-3
    assert 3.5 < errors[0] / errors[1] < 4.5
