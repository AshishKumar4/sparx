"""Plasticity rules against NEST: the weight each synapse transmits at every presynaptic spike.

Fixtures from `tools/make_nest_fixtures.py`: four synapses between parrot
neurons replaying random trains for 2 s at 0.1 ms, with a 1 ms dendritic
delay.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.dynamics import PairSTDP, TripletSTDP, TsodyksMarkram

NEST = np.load(Path(__file__).parent / "fixtures" / "nest.npz")
# Traces decay by one factor per step here, 20,000 multiplications over a
# run, where NEST exponentiates each interval: they agree to about 1e-12.
RTOL = 1e-10
DT = float(NEST["meta/dt"])
DELAY = 1.0


def fixture(name):
    prefix = f"plasticity/{name}/"
    return {key[len(prefix):]: NEST[key] for key in NEST.files if key.startswith(prefix)}


def nest_weights(case):
    """NEST's transmitted weights per synapse, in time order."""
    order = np.lexsort((case["weight_times"], case["weight_senders"]))
    senders, weights = case["weight_senders"][order], case["weights"][order]
    return [weights[senders == i] for i in range(case["pre"].shape[1])]


def transmitted(weights, pre):
    """The weight after each step at which the synapse's presynaptic neuron spiked."""
    return [weights[pre[:, i] > 0, i] for i in range(pre.shape[1])]


def run_stdp(rule, case):
    pre, post = case["pre"], case["post"]
    shift = round(DELAY / DT)
    arrivals = np.zeros_like(post)
    arrivals[shift:] = post[:-shift]  # postsynaptic spikes reach the synapse a dendritic delay later
    n = pre.shape[1]
    index = jnp.arange(n)
    with jax.enable_x64(new_val=True):
        traces = rule.init_state(n, n, jnp.float64)

        def step(carry, spikes):
            traces, weights = carry
            traces, weights = rule.step(traces, weights, *spikes, index, index, DT, modulators={})
            return (traces, weights), weights

        weights0 = jnp.full(n, float(case["initial"]), jnp.float64)
        _, weights = jax.lax.scan(step, (traces, weights0), (jnp.asarray(pre), jnp.asarray(arrivals)))
    return transmitted(np.asarray(weights), pre)


@pytest.mark.parametrize("name", ["stdp_additive", "stdp_multiplicative"])
def test_pair_stdp_is_nests(name):
    case = fixture(name)
    p = {key[len("param/"):]: float(value) for key, value in case.items() if key.startswith("param/")}
    rule = PairSTDP(tau_plus=p.get("tau_plus", 20.0), tau_minus=p.get("tau_minus", 20.0), lambda_=p["lambda"],
                    alpha=p["alpha"], mu_plus=p["mu_plus"], mu_minus=p["mu_minus"], w_max=p["Wmax"])
    got = run_stdp(rule, case)
    for mine, theirs in zip(got, nest_weights(case), strict=True):
        assert len(mine) == len(theirs) >= 40
        np.testing.assert_allclose(mine, theirs, rtol=RTOL)  # observed 6.4e-14 relative
    # The rule did something: weights moved well away from where they started.
    assert np.ptp(np.concatenate(got)) > 20


def test_triplet_stdp_is_nests():
    case = fixture("stdp_triplet")
    p = {key[len("param/"):]: float(value) for key, value in case.items() if key.startswith("param/")}
    rule = TripletSTDP(tau_minus=p["tau_minus"], tau_y=p["tau_minus_triplet"], a2_plus=p["Aplus"],
                       a3_plus=p["Aplus_triplet"], a2_minus=p["Aminus"], a3_minus=p["Aminus_triplet"],
                       w_max=p["Wmax"])
    got = run_stdp(rule, case)
    for mine, theirs in zip(got, nest_weights(case), strict=True):
        assert len(mine) == len(theirs) >= 40
        np.testing.assert_allclose(mine, theirs, rtol=RTOL)  # observed 1.9e-13 relative
    assert np.ptp(np.concatenate(got)) > 5


@pytest.mark.parametrize("name", ["tsodyks_depressing", "tsodyks_facilitating"])
def test_tsodyks_markram_is_nests(name):
    case = fixture(name)
    rule = TsodyksMarkram(U=float(case["param/U"]), tau_rec=float(case["param/tau_rec"]),
                          tau_fac=float(case["param/tau_fac"]))
    pre = case["pre"]
    with jax.enable_x64(new_val=True):

        def step(state, spikes):
            return rule.step(state, spikes, DT)

        _, efficacy = jax.lax.scan(step, rule.init_state((pre.shape[1],), jnp.float64), jnp.asarray(pre))
    got = transmitted(np.asarray(efficacy) * float(case["initial"]), pre)
    for mine, theirs in zip(got, nest_weights(case), strict=True):
        assert len(mine) == len(theirs) >= 40
        np.testing.assert_allclose(mine, theirs, rtol=RTOL)  # observed 3.6e-13 relative


def test_tsodyks_markram_depresses_and_facilitates():
    spikes = jnp.zeros((400, 1)).at[::20].set(1.0)  # every 2 ms
    for rule, trend in ((TsodyksMarkram(U=0.5, tau_rec=800.0), -1), (TsodyksMarkram(U=0.03, tau_rec=100.0,
                                                                                    tau_fac=1000.0), 1)):
        _, efficacy = jax.lax.scan(lambda s, x, rule=rule: rule.step(s, x, DT), rule.init_state((1,)), spikes)
        released = np.asarray(efficacy[::20, 0])
        assert released[0] == pytest.approx(float(rule.U))
        assert np.sign(released[3] - released[0]) == trend
