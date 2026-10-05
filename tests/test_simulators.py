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
    IZHIKEVICH_2003,
    LIF,
    AdEx,
    Alpha,
    Arrivals,
    BiExponential,
    Delta,
    Exponential,
    HodgkinHuxley,
    PointNeuron,
    Receptor,
    SynapticInput,
    integrate,
    izhikevich_2003,
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


NAUD = sorted({key.split("/")[1] for key in NEST.files if key.startswith("naud/")})


def naud_adex(name, **overrides):
    def p(key):
        return float(NEST[f"naud/{name}/param/{key}"])

    fields = {"c_m": p("C_m"), "g_l": p("g_L"), "e_l": p("E_L"), "v_t": p("V_th"), "delta_t": p("Delta_T"),
              "tau_w": p("tau_w"), "a": p("a"), "b": p("b"), "v_reset": p("V_reset"), "v_peak": p("V_peak")}
    return AdEx(**{**fields, **overrides}), p("I_e")


def naud_spikes(name, **overrides):
    neuron, current = naud_adex(name, **overrides)
    steps = len(NEST[f"naud/{name}/spikes"])
    with jax.enable_x64(new_val=True):
        spikes, _ = integrate(neuron, SynapticInput(jnp.full((steps, 1), current)), DT)
    return np.flatnonzero(np.asarray(spikes.fired[:, 0]))


@pytest.mark.parametrize("name", [name for name in NAUD if name != "irregular"])
def test_adex_fires_naud_patterns_with_nest(name):
    # NEST integrates adaptively; ten RK4 substeps keep every spike of these
    # 500 ms runs within 0.8 ms of NEST's.
    expected = np.flatnonzero(NEST[f"naud/{name}/spikes"])
    got = naud_spikes(name)
    assert len(got) == len(expected)
    assert np.abs(got - expected).max() <= 8


@pytest.mark.parametrize("name", ["tonic", "regular_bursting"])
def test_adex_converges_to_nest_with_substeps(name):
    expected = np.flatnonzero(NEST[f"naud/{name}/spikes"])
    coarse = np.abs(naud_spikes(name, substep=0.1) - expected).max()
    fine = np.abs(naud_spikes(name, substep=0.001) - expected).max()
    assert fine <= 1 < coarse


def test_adex_irregular_pattern_is_irregular_in_both():
    # Naud's irregular parameters are chaotic: spike times part from NEST's
    # after a few spikes, and the statistics are what is reproduced.
    got, expected = naud_spikes("irregular"), np.flatnonzero(NEST["naud/irregular/spikes"])
    assert abs(len(got) - len(expected)) <= 2
    for train in (got, expected):
        isi = np.diff(train)
        assert isi.std() / isi.mean() > 0.3


def test_adex_patterns_are_the_published_ones():
    def isi(name):
        return np.diff(naud_spikes(name)) * DT

    tonic = isi("tonic")
    assert tonic[-10:].std() / tonic[-10:].mean() < 0.02
    adapting = isi("adapting")
    assert np.all(np.diff(adapting) > -0.5) and adapting[-1] > 5 * adapting[0]
    burst = isi("initial_burst")
    assert np.all(burst[:2] < 10) and np.all(burst[2:] > 50)
    bursting = isi("regular_bursting")
    assert np.all(bursting[2::2] > 100) and np.all(bursting[3::2] < 10)
    accelerating = isi("delayed_accelerating")
    assert naud_spikes("delayed_accelerating")[0] * DT > 30 and np.all(np.diff(accelerating) < 0.5)
    delayed = isi("delayed_regular_bursting")
    assert naud_spikes("delayed_regular_bursting")[0] * DT > 50 and np.sum(delayed > 50) >= 3
    transient = naud_spikes("transient") * DT
    assert 1 <= len(transient) <= 3 and transient.max() < 100


@pytest.mark.parametrize(("model", "kind"), [("aeif_psc_exp", "current"), ("aeif_cond_exp", "conductance")])
def test_adex_with_synapses_fires_with_nest(model, kind):
    neuron = AdEx(t_ref=param(model, "t_ref"), reversal=REVERSAL)
    cell = PointNeuron(neuron, {"ex": Receptor(Exponential(param(model, "tau_syn_ex")), kind),
                                "in": Receptor(Exponential(param(model, "tau_syn_in")), kind)})
    fired, _ = run(cell, model, sign=-1.0 if kind == "conductance" else 1.0)
    expected = NEST[f"{model}/spikes"]
    for neuron in range(fired.shape[1]):
        got, want = np.flatnonzero(fired[:, neuron]), np.flatnonzero(expected[:, neuron])
        assert len(got) == len(want) >= 5
        # Held conductances (second order) meet the stiff upswing: a spike
        # can land up to 0.5 ms from NEST's adaptive solution.
        assert np.abs(got - want).max() <= (2 if kind == "current" else 5)


IZHIKEVICH_SCHEMES = {"published_1": (1.0, "published"), "published_01": (0.1, "published"),
                      "euler_01": (0.1, "euler")}


def izhikevich_run(kind, label, exact=False):
    dt, scheme = IZHIKEVICH_SCHEMES[label]
    steps = len(NEST[f"izhikevich/{kind}/{label}/spikes"])
    with jax.enable_x64(new_val=True), jax.disable_jit(exact):
        inputs = SynapticInput(jnp.full((steps, 1), 10.0))
        neuron = izhikevich_2003(kind, scheme=scheme)
        (spikes, v), _ = integrate(neuron, inputs, dt, record=lambda state: state.v)
    return np.asarray(spikes.fired[:, 0]), np.asarray(v[:, 0])


@pytest.mark.parametrize("kind", ["regular_spiking", "fast_spiking", "resonator"])
def test_izhikevich_published_scheme_is_nests_to_the_last_bit(kind):
    # Compiled XLA contracts and reassociates the quadratic's arithmetic
    # (19% of one step's elements differ in the last bit from NumPy's), and
    # the quadratic membrane amplifies that; op by op, it is NEST's run.
    fired, v = izhikevich_run(kind, "published_1", exact=True)
    np.testing.assert_array_equal(fired, NEST[f"izhikevich/{kind}/published_1/spikes"][:, 0])
    np.testing.assert_array_equal(v, NEST[f"izhikevich/{kind}/published_1/v"][:, 0])


@pytest.mark.parametrize("label", ["published_01", "euler_01"])
@pytest.mark.parametrize("kind", list(IZHIKEVICH_2003))
def test_izhikevich_classes_fire_with_nest(kind, label):
    fired, _ = izhikevich_run(kind, label)
    got, expected = np.flatnonzero(fired), np.flatnonzero(NEST[f"izhikevich/{kind}/{label}/spikes"][:, 0])
    # Compiled, last-bit rounding moves single spikes by a step or two, and
    # the reset carries that on: every spike stays within 1 ms of NEST's.
    assert len(got) == len(expected) >= 7
    assert np.abs(got - expected).max() <= 10


@pytest.mark.parametrize("scheme", ["euler", "published"])
def test_izhikevich_delta_input_matches_nest(scheme):
    # NEST adds a delta input to the voltage under its Euler scheme, and as
    # a current of the weight for one step under the published one, as
    # Izhikevich's network code adds its synaptic input to I.
    case = f"izhikevich/delta_{scheme}"
    arrivals = NEST[f"{case}/arrivals"]
    with jax.enable_x64(new_val=True):
        arrivals = jnp.asarray(arrivals)
        inputs = (SynapticInput(4.0, jump=arrivals) if scheme == "euler"
                  else SynapticInput(4.0 + arrivals))
        (spikes, v), _ = integrate(izhikevich_2003("regular_spiking", scheme=scheme), inputs, 0.1,
                                   record=lambda state: state.v)
    np.testing.assert_array_equal(np.asarray(spikes.fired), NEST[f"{case}/spikes"])
    np.testing.assert_allclose(np.asarray(v), NEST[f"{case}/v"], atol=1e-9)


def hh_under_currents(steps, **fields):
    currents = jnp.asarray(NEST["hh_currents/currents"])
    with jax.enable_x64(new_val=True):
        inputs = SynapticInput(jnp.broadcast_to(jnp.asarray(currents, jnp.float64), (steps, len(currents))))
        (spikes, v), _ = integrate(HodgkinHuxley(**fields), inputs, DT, record=lambda state: state.v)
    return np.asarray(spikes.fired), np.asarray(v)


def spike_shift(fired, expected):
    """The largest shift, in steps, between matching spikes of each neuron; None if a count differs."""
    shifts = []
    for neuron in range(fired.shape[1]):
        got, want = np.flatnonzero(fired[:, neuron]), np.flatnonzero(expected[:, neuron])
        if len(got) != len(want):
            return None
        shifts.append(np.abs(got - want).max(initial=0))
    return max(shifts)


def test_hodgkin_huxley_fires_with_nest():
    # Four currents from just above rheobase to 4 nA, 300 ms each.
    fired, _ = hh_under_currents(3000)
    assert spike_shift(fired, NEST["hh_currents/spikes"]) <= 1
    assert fired.sum() >= 90


def test_hodgkin_huxley_rk4_is_closest_to_nest_without_strong_inhibition():
    fired, v = hh_under_currents(3000, scheme="rk4", substep=0.025)
    np.testing.assert_array_equal(fired, NEST["hh_currents/spikes"])
    np.testing.assert_allclose(v, NEST["hh_currents/v"], atol=2e-2)


def test_hodgkin_huxley_strang_is_second_order():
    _, truth = hh_under_currents(200, scheme="rk4", substep=0.001)
    errors = [np.abs(hh_under_currents(200, substep=step)[1] - truth).max() for step in (0.02, 0.01)]
    assert 3.5 < errors[0] / errors[1] < 4.5


def test_hodgkin_huxley_exponential_euler_is_brian2s():
    _, v = hh_under_currents(1000, scheme="exponential_euler", substep=None)
    np.testing.assert_allclose(v[:100], BRIAN2["hh/v"][:100], atol=1e-10)
    # Later, rounding differences carried through each spike's upswing grow.
    np.testing.assert_allclose(v, BRIAN2["hh/v"], atol=1e-4)


@pytest.mark.parametrize("scheme", ["strang", "rk4"])
def test_hodgkin_huxley_with_alpha_synapses(scheme):
    # Inhibition drives one neuron below -128 mV, where RK4 at 0.025 ms
    # diverges and the splitting does not.
    model = "hh_psc_alpha"
    cell = PointNeuron(HodgkinHuxley(scheme=scheme, substep=0.01 if scheme == "strang" else 0.025),
                       {"ex": Receptor(Alpha(param(model, "tau_syn_ex"))),
                        "in": Receptor(Alpha(param(model, "tau_syn_in")))})
    fired, v = run(cell, model)
    if scheme == "rk4":
        assert np.isnan(v).any()
        return
    assert v.min() < -125
    np.testing.assert_array_equal(fired, NEST[f"{model}/spikes"])
