"""Neuron models in physical units (ms, mV, pA, nS, pF; see `sparx.dynamics.core`).

Every model here has a refractory period after a spike, measured on the step
grid as NEST's `iaf_*` models count it: a neuron that fires holds its
reset voltage for `t_ref` ms, `round(t_ref / dt)` steps. Spikes pass
gradients through a surrogate, as in `sparx.cells`, and the reset sets the
voltage exactly while its gradient follows the spike.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from sparx.dynamics.core import Spikes, SynapticInput, crossing, exact_linear, membrane_dtype, response, rk4
from sparx.surrogate import ATan, Surrogate, spike

__all__ = ["LIF", "RECEPTORS", "LIFState"]

RECEPTORS: Mapping[str, float] = {"ampa": 0.0, "nmda": 0.0, "gaba_a": -80.0, "gaba_b": -95.0}
"""Reversal potentials (mV) of the common receptors, the defaults models read
conductances against: AMPA and NMDA 0 mV, GABA-A -80 mV (Cl-), GABA-B
-95 mV (K+), as in Brette et al.'s simulator benchmarks (2007) and the
cortical models built on them."""


@struct.dataclass
class MgBlock:
    """The fraction of NMDA receptors not blocked by magnesium at voltage `v` (mV).

        B(v) = 1 / (1 + [Mg] / 3.57 exp(-0.062 v))

    Jahr and Stevens (J. Neurosci. 1990), `mg` the extracellular
    concentration in mM. A conductance-based model scales its `nmda`
    conductance by `B` at the voltage at the start of each step.
    """

    mg: float = 1.0

    def __call__(self, v: jax.Array) -> jax.Array:
        return 1 / (1 + self.mg / 3.57 * jnp.exp(-0.062 * v))


def _reset(v: jax.Array, fired: jax.Array, to: jax.Array | float) -> jax.Array:
    """`to` where fired, exactly, with the gradient of `v + fired * (to - v)`."""
    return jnp.where(fired > 0, to, v) + (fired - jax.lax.stop_gradient(fired)) * (to - v)


class LIFState(NamedTuple):
    v: jax.Array
    refractory: jax.Array
    """Milliseconds of refractoriness left."""


@struct.dataclass
class LIF:
    """Leaky integrate-and-fire with current and conductance input.

        C dv/dt = -g_L (v - E_L) + sum_k g_k (E_k - v) + I,    g_L = C / tau_m

    At `v >= v_th` the neuron fires, `v` is set to `v_reset` and held there
    for `t_ref`. With the inputs constant over a step the equation is linear
    in `v`, and the update is its exact solution: `v` relaxes toward
    `(g_L E_L + sum g_k E_k + I) / (g_L + sum g_k)` with time constant
    `C / (g_L + sum g_k)`, and synaptic current waveforms are integrated
    exactly against that time constant, as NEST's `iaf_psc_exp` and
    `iaf_psc_alpha` do. Conductances are read against `reversal`, by
    receptor name, held over the step as Brian2's `exponential_euler`
    holds them; a receptor in `gates` is scaled by its gate at the voltage
    at the start of the step (NMDA's magnesium block by default). A voltage
    `jump` from delta synapses lands after the step's integration, before
    the threshold test, and is lost during refractoriness, as in NEST's
    `iaf_psc_delta`. The defaults are the cortical cell of Brette et al.'s
    benchmarks (2007): 20 ms, 200 pF, rest -60 mV, threshold -50 mV, reset
    -60 mV, 5 ms refractory.
    """

    tau_m: jax.Array | float = 20.0
    c_m: jax.Array | float = 200.0
    e_l: jax.Array | float = -60.0
    v_th: jax.Array | float = -50.0
    v_reset: jax.Array | float = -60.0
    t_ref: float = struct.field(pytree_node=False, default=5.0)
    reversal: Mapping[str, float] = struct.field(pytree_node=False, default_factory=lambda: dict(RECEPTORS))
    gates: Mapping[str, MgBlock] = struct.field(pytree_node=False,
                                                default_factory=lambda: {"nmda": MgBlock()})
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> LIFState:
        dtype = membrane_dtype(dtype)
        return LIFState(jnp.full(shape, self.e_l, dtype), jnp.zeros(shape, dtype))

    def step(self, state: LIFState, inputs: SynapticInput, dt: float) -> tuple[LIFState, Spikes]:
        g_l = self.c_m / self.tau_m
        conductance = {name: g * self.gates[name](state.v) if name in self.gates else g
                       for name, g in inputs.conductance.items()}
        g_total = g_l + sum(conductance.values(), jnp.zeros(()))
        drive = g_l * self.e_l + inputs.current + sum(
            (g * self.reversal[name] for name, g in conductance.items()), jnp.zeros(()))
        tau = self.c_m / g_total
        integrated = exact_linear(state.v, drive / g_total, tau, dt) + sum(
            (response(term, tau, dt) for term in inputs.currents), jnp.zeros(())) / self.c_m + inputs.jump
        held = state.refractory > dt / 2
        v = jnp.where(held, self.v_reset, integrated)
        fired = spike(v - self.v_th, self.surrogate) * (1 - held)
        offset = jnp.where(fired > 0, crossing(state.v, v, self.v_th), 1.0)
        refractory = jnp.where(fired > 0, self.t_ref, jnp.maximum(state.refractory - dt, 0))
        dtype = state.v.dtype
        return (LIFState(_reset(v, fired, self.v_reset).astype(dtype), refractory.astype(dtype)),
                Spikes(fired, offset))


def _synaptic(model, inputs: SynapticInput, v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The total held conductance (nS) and its drive `sum g E` (pA) at voltage `v`, gates applied."""
    conductance = {name: g * model.gates[name](v) if name in model.gates else g
                   for name, g in inputs.conductance.items()}
    total = sum(conductance.values(), jnp.zeros(()))
    drive = sum((g * model.reversal[name] for name, g in conductance.items()), jnp.zeros(()))
    return total, drive


class AdExState(NamedTuple):
    v: jax.Array
    w: jax.Array
    """Adaptation current, pA."""
    refractory: jax.Array


@struct.dataclass
class AdEx:
    """The adaptive exponential integrate-and-fire neuron (Brette and Gerstner, J. Neurophysiol. 2005).

        C dv/dt = -g_L (v - E_L) + g_L D_T exp((v - v_T) / D_T) - w + I + sum_k g_k (E_k - v)
        tau_w dw/dt = a (v - E_L) - w

    At `v >= v_peak` it fires, `v` is set to `v_reset` and `w` grows by `b`;
    `v` then holds for `t_ref` (0 by default). The right-hand side reads the
    voltage capped at `v_peak`, and reads `v_reset` while refractory, as
    NEST's `aeif_*` models do, which keeps the exponential finite through
    the upswing. Integrated by `substeps` RK4 steps per step, with synaptic
    currents evaluated at each stage and conductances held; a spike is
    detected and reset at the substep that crosses, and the rest of the step
    continues from the reset, as NEST's adaptive integration does.

    The upswing makes the equation stiff. Against NEST's adaptive RK45 at
    `dt = 0.1` over 500 ms of Naud et al.'s patterns, spike times drift by
    up to 5.3 ms with one substep, 0.4 ms with ten (the default) and
    0.1 ms with a hundred (`tests/test_simulators.py`); an exponential
    Rosenbrock step was less accurate than RK4 at every substep count
    tried. Lower `substeps` to trade that accuracy for speed in large
    networks. The defaults are
    NEST's, the parameters of Brette and Gerstner's Figure 2.
    """

    c_m: jax.Array | float = 281.0
    g_l: jax.Array | float = 30.0
    e_l: jax.Array | float = -70.6
    v_t: jax.Array | float = -50.4
    delta_t: jax.Array | float = 2.0
    v_peak: jax.Array | float = 0.0
    v_reset: jax.Array | float = -60.0
    a: jax.Array | float = 4.0
    b: jax.Array | float = 80.5
    tau_w: jax.Array | float = 144.0
    t_ref: float = struct.field(pytree_node=False, default=0.0)
    substeps: int = struct.field(pytree_node=False, default=10)
    reversal: Mapping[str, float] = struct.field(pytree_node=False, default_factory=lambda: dict(RECEPTORS))
    gates: Mapping[str, MgBlock] = struct.field(pytree_node=False,
                                                default_factory=lambda: {"nmda": MgBlock()})
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> AdExState:
        dtype = membrane_dtype(dtype)
        zeros = jnp.zeros(shape, dtype)
        return AdExState(jnp.full(shape, self.e_l, dtype), zeros, zeros)

    def step(self, state: AdExState, inputs: SynapticInput, dt: float) -> tuple[AdExState, Spikes]:
        g_syn, syn_drive = _synaptic(self, inputs, state.v)
        h = dt / self.substeps

        def f(held):
            def f(s, y):
                v, w = y
                v = jnp.where(held, self.v_reset, jnp.minimum(v, self.v_peak))
                spike_current = self.g_l * self.delta_t * jnp.exp((v - self.v_t) / self.delta_t)
                dv = (-self.g_l * (v - self.e_l) + spike_current - w + inputs.current_at(s)
                      + syn_drive - g_syn * v) / self.c_m
                return jnp.where(held, 0.0, dv), (self.a * (v - self.e_l) - w) / self.tau_w
            return f

        # A spike resets the membrane at the substep it crosses on, and the
        # rest of the step integrates from the reset (refractory if t_ref > 0),
        # as NEST's aeif models reset within their adaptive steps.
        def detect(i, before, v, w, held, fired, offset):
            crossed = spike(v - self.v_peak, self.surrogate) * (1 - held)
            first = (crossed > 0) & (fired == 0)
            offset = jnp.where(first, (i + crossing(before, v, self.v_peak)) / self.substeps, offset)
            held = held | (crossed > 0) if self.t_ref > 0 else held
            fired = fired + crossed * (1 - fired)
            return _reset(v, crossed, self.v_reset), w + crossed * self.b, held, fired, offset

        def substep(i, carry):
            v, w, held, fired, offset = carry
            after, w = rk4(f(held), (v, w), h, start=i * h)
            return detect(i, v, after, w, held, fired, offset)

        held = state.refractory > dt / 2
        carry = (state.v, state.w, held, jnp.zeros_like(state.v), jnp.ones_like(state.v))
        v, w, held, fired, offset = jax.lax.fori_loop(0, self.substeps, substep, carry)
        # A delta synapse's jump lands at the end of the step, before the threshold test.
        v, w, held, fired, offset = detect(self.substeps - 1, v, v + inputs.jump * (1 - held), w, held, fired,
                                           offset)
        v = jnp.where(held, self.v_reset, v)
        refractory = jnp.where(fired > 0, self.t_ref, jnp.maximum(state.refractory - dt, 0))
        dtype = state.v.dtype
        return AdExState(v.astype(dtype), w.astype(dtype), refractory.astype(dtype)), Spikes(fired, offset)
