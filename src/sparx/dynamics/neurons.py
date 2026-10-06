"""Neuron models in physical units (ms, mV, pA, nS, pF; see `sparx.dynamics.core`).

Every spiking model here has a refractory period after a spike, measured on
the step grid as NEST's `iaf_*` models count it: a neuron that fires holds
its reset voltage for `t_ref` ms, `round(t_ref / dt)` steps. One model,
`GradedPotential`, never spikes, and its output is a release rate graded
with its voltage. Spikes pass gradients through a surrogate, as in
`sparx.dynamics.ml`, and the reset (`fire`) sets the voltage exactly while
its gradient follows the spike.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from sparx.dynamics.core import (
    Output,
    SynapticInput,
    crossing,
    exact_linear,
    fire,
    jump_after_threshold,
    membrane_dtype,
    response,
    rk4,
    substeps,
)
from sparx.surrogate import ATan, Surrogate, spike

__all__ = [
    "IZHIKEVICH_2003",
    "IZHIKEVICH_2004",
    "LIF",
    "RECEPTORS",
    "AdEx",
    "AdExState",
    "GradedPotential",
    "GradedPotentialState",
    "HodgkinHuxley",
    "HodgkinHuxleyState",
    "Izhikevich",
    "IzhikevichState",
    "LIFState",
    "MgBlock",
    "izhikevich_2003",
    "izhikevich_2004",
]

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


def _synaptic(model: LIF | GradedPotential | AdEx | Izhikevich | HodgkinHuxley, inputs: SynapticInput,
              v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The total held conductance (nS) and its drive `sum g E` (pA) at voltage `v`, gates applied."""
    conductance = {name: g * model.gates[name](v) if name in model.gates else g
                   for name, g in inputs.conductance.items()}
    total = sum(conductance.values(), jnp.zeros(()))
    drive = sum((g * model.reversal[name] for name, g in conductance.items()), jnp.zeros(()))
    return total, drive


def _leaky(model: LIF | GradedPotential, v: jax.Array, inputs: SynapticInput, dt: float) -> jax.Array:
    """`v` after a step of the leaky membrane on `inputs`, its jump included: exact.

    With the conductances and the current held, the membrane relaxes toward
    `(g_L E_L + sum g_k E_k + I) / (g_L + sum g_k)` with time constant
    `C / (g_L + sum g_k)`, and each synaptic current waveform adds its exact
    response against that time constant.
    """
    g_l = model.c_m / model.tau_m
    g_syn, syn_drive = _synaptic(model, inputs, v)
    g_total = g_l + g_syn
    drive = g_l * model.e_l + inputs.current + syn_drive
    tau = model.c_m / g_total
    return exact_linear(v, drive / g_total, tau, dt) + sum(
        (response(term, tau, dt) for term in inputs.currents), jnp.zeros(())) / model.c_m + inputs.jump


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
    t_ref: jax.Array | float = 5.0
    """Refractory period, ms; per neuron when an array."""
    reversal: Mapping[str, float] = struct.field(pytree_node=False, default_factory=lambda: dict(RECEPTORS))
    gates: Mapping[str, MgBlock] = struct.field(pytree_node=False,
                                                default_factory=lambda: {"nmda": MgBlock()})
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())
    graded = False

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> LIFState:
        dtype = membrane_dtype(dtype)
        return LIFState(jnp.full(shape, self.e_l, dtype), jnp.zeros(shape, dtype))

    def step(self, state: LIFState, inputs: SynapticInput, dt: float) -> tuple[LIFState, Output]:
        integrated = _leaky(self, state.v, inputs, dt)
        held = state.refractory > dt / 2
        v = jnp.where(held, self.v_reset, integrated)
        reset, fired = fire(v, jnp.where(held, jnp.inf, self.v_th), self.surrogate, self.v_reset)
        offset = jnp.where(fired > 0, crossing(state.v, v, self.v_th), 1.0)
        refractory = jnp.where(fired > 0, self.t_ref, jnp.maximum(state.refractory - dt, 0))
        dtype = state.v.dtype
        return LIFState(reset.astype(dtype), refractory.astype(dtype)), Output(fired, offset)

    def is_refractory(self, state: LIFState, dt: float) -> jax.Array:
        return state.refractory > dt / 2

    def after_threshold(self, state: LIFState, jump: jax.Array, fired: jax.Array) -> LIFState:
        return state._replace(v=jump_after_threshold(state.v, jump, fired))


class GradedPotentialState(NamedTuple):
    v: jax.Array


@struct.dataclass
class GradedPotential:
    """A neuron that never spikes and releases transmitter as a sigmoidal function of its voltage.

        C dv/dt = -g_L (v - E_L) + sum_k g_k (E_k - v) + I,    g_L = C / tau_m
        release(v) = 1 / (1 + exp((v_half - v) / slope))

    The membrane is `LIF`'s without a threshold, solved exactly over each
    step for held inputs, with conductances, gates, synaptic current
    waveforms as `LIF` reads them. The output is the
    release at the end of the step, as a fraction of the maximal rate, in
    [0, 1]. A `Graded` synapse holds it over the next step and filters it
    with its own kinetics, so the weight of a projection from a graded
    population is the current (pA) or conductance (nS) of its synapses at
    full release.

    The sigmoid is the graded transmission of the stomatogastric network
    models of Prinz, Bucher and Marder (Nature Neuroscience 2004),
    `s(V_pre) = 1 / (1 + exp((V_th - V_pre) / delta))`, whose `V_th` of
    -35 mV and `delta` of 5 mV are the defaults here. Much of the fly's
    visual system is non-spiking in this sense, and Lappalainen et al.'s
    connectome-constrained model of it (Nature 2024) also uses passive
    point neurons that transmit a function of their voltage, there a
    rectified linear one. The membrane's defaults are `LIF`'s.
    """

    tau_m: jax.Array | float = 20.0
    c_m: jax.Array | float = 200.0
    e_l: jax.Array | float = -60.0
    v_half: jax.Array | float = -35.0
    """The voltage of half the maximal release, mV."""
    slope: jax.Array | float = 5.0
    """The voltage over which release grows by a factor of e near its foot, mV."""
    reversal: Mapping[str, float] = struct.field(pytree_node=False, default_factory=lambda: dict(RECEPTORS))
    gates: Mapping[str, MgBlock] = struct.field(pytree_node=False,
                                                default_factory=lambda: {"nmda": MgBlock()})
    graded = True

    def release(self, v: jax.Array) -> jax.Array:
        """The release at voltage `v`, as a fraction of the maximum."""
        return jax.nn.sigmoid((v - self.v_half) / self.slope)

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> GradedPotentialState:
        return GradedPotentialState(jnp.full(shape, self.e_l, membrane_dtype(dtype)))

    def step(self, state: GradedPotentialState, inputs: SynapticInput,
             dt: float) -> tuple[GradedPotentialState, Output]:
        v = _leaky(self, state.v, inputs, dt).astype(state.v.dtype)
        rate = self.release(v)
        return GradedPotentialState(v), Output(rate, jnp.ones_like(rate))

    def is_refractory(self, state: GradedPotentialState, dt: float) -> jax.Array:
        return jnp.zeros(jnp.shape(state.v), bool)

    def after_threshold(self, state: GradedPotentialState, jump: jax.Array,
                        fired: jax.Array) -> GradedPotentialState:
        # It has no threshold and no reset, so the jump always lands.
        return GradedPotentialState(jump_after_threshold(state.v, jump, None))


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
    the upswing. Integrated by RK4 in substeps of at most `substep` ms, with synaptic
    currents evaluated at each stage and conductances held; a spike is
    detected and reset at the substep that crosses, and the rest of the step
    continues from the reset, as NEST's adaptive integration does.

    The upswing makes the equation stiff. Against NEST's adaptive RK45
    over 500 ms of Naud et al.'s patterns, spike times drift by up to
    5.3 ms with substeps of 0.1 ms, 0.4 ms with 0.01 ms (the default) and
    0.1 ms with 0.001 ms (`tests/test_simulators.py`); an exponential
    Rosenbrock step was less accurate than RK4 at every length tried.
    Lengthen `substep` (None: one per step) to trade that accuracy for
    speed in large networks. The defaults are
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
    substep: float | None = struct.field(pytree_node=False, default=0.01)
    reversal: Mapping[str, float] = struct.field(pytree_node=False, default_factory=lambda: dict(RECEPTORS))
    gates: Mapping[str, MgBlock] = struct.field(pytree_node=False,
                                                default_factory=lambda: {"nmda": MgBlock()})
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())
    graded = False

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> AdExState:
        dtype = membrane_dtype(dtype)
        zeros = jnp.zeros(shape, dtype)
        return AdExState(jnp.full(shape, self.e_l, dtype), zeros, zeros)

    def step(self, state: AdExState, inputs: SynapticInput, dt: float) -> tuple[AdExState, Output]:
        g_syn, syn_drive = _synaptic(self, inputs, state.v)
        count = substeps(dt, self.substep)
        h = dt / count

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
            reset, crossed = fire(v, jnp.where(held, jnp.inf, self.v_peak), self.surrogate, self.v_reset)
            first = (crossed > 0) & (fired == 0)
            offset = jnp.where(first, (i + crossing(before, v, self.v_peak)) / count, offset)
            held = held | (crossed > 0) if self.t_ref > 0 else held
            fired = fired + crossed * (1 - fired)
            return reset, w + crossed * self.b, held, fired, offset

        def substep(i, carry):
            v, w, held, fired, offset = carry
            after, w = rk4(f(held), (v, w), h, start=i * h)
            return detect(i, v, after, w, held, fired, offset)

        held = state.refractory > dt / 2
        carry = (state.v, state.w, held, jnp.zeros_like(state.v), jnp.ones_like(state.v))
        v, w, held, fired, offset = jax.lax.fori_loop(0, count, substep, carry)
        # A delta synapse's jump lands at the end of the step, before the threshold test.
        v, w, held, fired, offset = detect(count - 1, v, v + inputs.jump * (1 - held), w, held, fired,
                                           offset)
        v = jnp.where(held, self.v_reset, v)
        refractory = jnp.where(fired > 0, self.t_ref, jnp.maximum(state.refractory - dt, 0))
        dtype = state.v.dtype
        return AdExState(v.astype(dtype), w.astype(dtype), refractory.astype(dtype)), Output(fired, offset)

    def is_refractory(self, state: AdExState, dt: float) -> jax.Array:
        return state.refractory > dt / 2

    def after_threshold(self, state: AdExState, jump: jax.Array, fired: jax.Array) -> AdExState:
        return state._replace(v=jump_after_threshold(state.v, jump, fired))


class IzhikevichState(NamedTuple):
    v: jax.Array
    u: jax.Array


@struct.dataclass
class Izhikevich:
    """Izhikevich's simple model (IEEE Trans. Neural Netw. 2003), in ms and mV.

        dv/dt = 0.04 v^2 + 5 v + 140 - u + I,    du/dt = a (b v - u)

    At `v >= v_th` (30 mV) it fires, `v` is set to `c` and `u` grows by
    `d`. `I` is in the model's own units, as in the paper. `scheme` names
    the integration: `"published"` takes two half-steps of `v` and then
    one step of `u` from the new `v`, the 2003 paper's code (at `dt = 1`
    with `order="izhikevich"` it is that code to the last bit, and its
    firing patterns are this scheme's); `"euler"` is the forward Euler step
    of both from the old values, NEST's `consistent_integration`;
    `"semi_implicit"` is one Euler step of `v` and then one of `u` from the
    new `v`, the code of the 2004 paper's twenty firing patterns
    (`izhikevich_2004`). Synaptic current waveforms are read at the start
    of the step and conductances at the voltage of each update. The
    defaults are the regular spiking cell; the 2003 paper's classes are
    `izhikevich_2003`.

    `order` names the arithmetic of the quadratic term, which the membrane
    amplifies from the last bit to whole spikes over a long run:
    `"izhikevich"` squares first, `0.04 * v**2`, as his MATLAB code does;
    `"nest"` multiplies `(0.04 * v) * v`, as NEST's `izhikevich` does.

    `quadratic` holds the coefficients of `v^2`, `v` and 1, and
    `du/dt = a (b (v - v_u) - u_decay u)`: the defaults are the equations
    above, and two of the 2004 patterns change them (class 1 excitability
    and the integrator take `4.1 v + 108`, accommodation `du/dt = a b (v + 65)`).
    """

    a: jax.Array | float = 0.02
    b: jax.Array | float = 0.2
    c: jax.Array | float = -65.0
    d: jax.Array | float = 8.0
    v_th: jax.Array | float = 30.0
    v_init: jax.Array | float = -65.0
    v_u: jax.Array | float = 0.0
    u_decay: jax.Array | float = 1.0
    quadratic: tuple[float, float, float] = struct.field(pytree_node=False, default=(0.04, 5.0, 140.0))
    scheme: Literal["published", "euler", "semi_implicit"] = struct.field(pytree_node=False,
                                                                         default="published")
    order: Literal["izhikevich", "nest"] = struct.field(pytree_node=False, default="izhikevich")
    reversal: Mapping[str, float] = struct.field(pytree_node=False, default_factory=lambda: dict(RECEPTORS))
    gates: Mapping[str, MgBlock] = struct.field(pytree_node=False,
                                                default_factory=lambda: {"nmda": MgBlock()})
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())
    graded = False

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> IzhikevichState:
        dtype = membrane_dtype(dtype)
        v = jnp.full(shape, self.v_init, dtype)
        return IzhikevichState(v, self.b * v)

    def step(self, state: IzhikevichState, inputs: SynapticInput,
             dt: float) -> tuple[IzhikevichState, Output]:
        current = inputs.current_at(0.0)

        k2, k1, k0 = self.quadratic

        def dv(v, u):
            g, drive = _synaptic(self, inputs, v)
            square = k2 * v ** 2 if self.order == "izhikevich" else k2 * v * v
            return square + k1 * v + k0 - u + current + drive - g * v

        def du(v, u):
            # `dt a (...)` multiplies `dt a` first, as NEST and Izhikevich's code both do.
            return dt * self.a * (self.b * (v - self.v_u) - self.u_decay * u)

        v, u = state
        if self.scheme == "euler":
            v, u = v + dt * dv(v, u), u + du(v, u)
        elif self.scheme == "semi_implicit":
            v = v + dt * dv(v, u)
            u = u + du(v, u)
        else:
            v = v + dt / 2 * dv(v, u)
            v = v + dt / 2 * dv(v, u)
            u = u + du(v, u)
        v = v + inputs.jump
        reset, fired = fire(v, self.v_th, self.surrogate, self.c)
        offset = jnp.where(fired > 0, crossing(state.v, v, self.v_th), 1.0)
        dtype = state.v.dtype
        return (IzhikevichState(reset.astype(dtype), (u + fired * self.d).astype(dtype)),
                Output(fired, offset))

    def is_refractory(self, state: IzhikevichState, dt: float) -> jax.Array:
        return jnp.zeros(jnp.shape(state.v), bool)

    def after_threshold(self, state: IzhikevichState, jump: jax.Array, fired: jax.Array) -> IzhikevichState:
        return state._replace(v=jump_after_threshold(state.v, jump, fired))


def izhikevich_2003(kind: str, **fields) -> Izhikevich:
    """The cortical and thalamic classes of Izhikevich (2003), Figure 2, by their names there."""
    a, b, c, d = IZHIKEVICH_2003[kind]
    return Izhikevich(a=a, b=b, c=c, d=d, **fields)


IZHIKEVICH_2003: Mapping[str, tuple[float, float, float, float]] = {
    "regular_spiking": (0.02, 0.2, -65.0, 8.0),
    "intrinsically_bursting": (0.02, 0.2, -55.0, 4.0),
    "chattering": (0.02, 0.2, -50.0, 2.0),
    "fast_spiking": (0.1, 0.2, -65.0, 2.0),
    "low_threshold_spiking": (0.02, 0.25, -65.0, 2.0),
    "thalamo_cortical": (0.02, 0.25, -65.0, 0.05),
    "resonator": (0.1, 0.26, -65.0, 2.0),
}
"""`(a, b, c, d)` of each class in Izhikevich (2003), Figure 2."""


def izhikevich_2004(pattern: str, **fields) -> Izhikevich:
    """The neuron of one of the twenty firing patterns of Izhikevich (2004), Figure 1, by its name
    (`IZHIKEVICH_2004`), integrated as the paper's code does (`scheme="semi_implicit"`)."""
    return Izhikevich(**{"scheme": "semi_implicit", **IZHIKEVICH_2004[pattern], **fields})


_CLASS_1 = {"quadratic": (0.04, 4.1, 108.0)}
IZHIKEVICH_2004: Mapping[str, Mapping[str, float | tuple[float, float, float]]] = {
    name: {"a": a, "b": b, "c": c, "d": d, **extra} for name, (a, b, c, d), extra in [
        ("tonic_spiking", (0.02, 0.2, -65.0, 6.0), {}),
        ("phasic_spiking", (0.02, 0.25, -65.0, 6.0), {}),
        ("tonic_bursting", (0.02, 0.2, -50.0, 2.0), {}),
        ("phasic_bursting", (0.02, 0.25, -55.0, 0.05), {}),
        ("mixed_mode", (0.02, 0.2, -55.0, 4.0), {}),
        ("spike_frequency_adaptation", (0.01, 0.2, -65.0, 8.0), {}),
        ("class_1_excitable", (0.02, -0.1, -55.0, 6.0), _CLASS_1),
        ("class_2_excitable", (0.2, 0.26, -65.0, 0.0), {}),
        ("spike_latency", (0.02, 0.2, -65.0, 6.0), {}),
        ("subthreshold_oscillations", (0.05, 0.26, -60.0, 0.0), {}),
        ("resonator", (0.1, 0.26, -60.0, -1.0), {}),
        ("integrator", (0.02, -0.1, -55.0, 6.0), _CLASS_1),
        ("rebound_spike", (0.03, 0.25, -60.0, 4.0), {}),
        ("rebound_burst", (0.03, 0.25, -52.0, 0.0), {}),
        ("threshold_variability", (0.03, 0.25, -60.0, 4.0), {}),
        ("bistability", (0.1, 0.26, -60.0, 0.0), {}),
        ("depolarizing_after_potential", (1.0, 0.2, -60.0, -21.0), {}),
        ("accommodation", (0.02, 1.0, -55.0, 4.0), {"v_u": -65.0, "u_decay": 0.0}),
        ("inhibition_induced_spiking", (-0.02, -1.0, -60.0, 8.0), {}),
        ("inhibition_induced_bursting", (-0.026, -1.0, -45.0, -2.0), {}),
    ]
}
"""The fields of each of Izhikevich's (2004) twenty firing patterns, Figure 1 (A) to (T), as his code
sets them. Each pattern also has its own input protocol and step, which `tests/test_simulators.py`
reads from his code's run."""


def _vtrap(x: jax.Array, y: float) -> jax.Array:
    """`x / (1 - exp(-x / y))`, continuous through `x = 0` where it is `y`."""
    near = jnp.abs(x / y) < 1e-6
    safe = jnp.where(near, 1.0, x)
    return jnp.where(near, y + x / 2, safe / -jnp.expm1(-safe / y))


class HodgkinHuxleyState(NamedTuple):
    v: jax.Array
    m: jax.Array
    h: jax.Array
    n: jax.Array
    refractory: jax.Array


@struct.dataclass
class HodgkinHuxley:
    """The squid giant axon (Hodgkin and Huxley, J. Physiol. 1952), as NEST's `hh_psc_alpha` states it.

        C dv/dt = -g_Na m^3 h (v - E_Na) - g_K n^4 (v - E_K) - g_L (v - E_L) + I
        dx/dt = alpha_x(v) (1 - x) - beta_x(v) x,    x in m, h, n

    with the rates (1/ms) in the modern convention, rest near -65 mV:
    `alpha_n = 0.01 (v + 55) / (1 - exp(-(v + 55) / 10))`,
    `beta_n = 0.125 exp(-(v + 65) / 80)`,
    `alpha_m = 0.1 (v + 40) / (1 - exp(-(v + 40) / 10))`,
    `beta_m = 4 exp(-(v + 65) / 18)`, `alpha_h = 0.07 exp(-(v + 65) / 20)`,
    `beta_h = 1 / (1 + exp(-(v + 35) / 10))`. Conductances are NEST's,
    for a 100 pF membrane (1 uF/cm^2 over 1e-4 cm^2).

    The membrane has no reset; a spike is its peak. As in NEST, a spike is
    reported at the step where the voltage is at or above `v_spike` (0 mV)
    and has begun to fall, and none is reported within `t_ref` of one.

    `scheme` names the integration, in substeps of at most `substep` ms:

    - `"strang"` (the default): half a substep of each gate with the
      voltage held (exact, Rush and Larsen 1978), a substep of the voltage
      with the gates held (exact), and half a substep of the gates again.
      Second order, and stable at any substep, since each part is solved
      exactly. At 0.01 ms (the default) it fires with NEST's adaptive
      solution within a step.
    - `"rk4"`: classical RK4 of the whole system. At 0.025 ms it is within
      0.01 mV of NEST, but the gates grow stiff under hyperpolarization
      (`beta_m` is 130/ms at -128 mV) and RK4 diverges there; shorter
      substeps only move the limit.
    - `"exponential_euler"`: gates and voltage each advanced exactly from
      the start of the substep, Brian2's `exponential_euler` (with
      `substep=None`, Brian2's step exactly); stable and first order.
    """

    c_m: jax.Array | float = 100.0
    g_na: jax.Array | float = 12000.0
    g_k: jax.Array | float = 3600.0
    g_l: jax.Array | float = 30.0
    e_na: jax.Array | float = 50.0
    e_k: jax.Array | float = -77.0
    e_l: jax.Array | float = -54.402
    v_init: jax.Array | float = -65.0
    v_spike: jax.Array | float = 0.0
    t_ref: float = struct.field(pytree_node=False, default=2.0)
    scheme: Literal["strang", "rk4", "exponential_euler"] = struct.field(pytree_node=False, default="strang")
    substep: float | None = struct.field(pytree_node=False, default=0.01)
    reversal: Mapping[str, float] = struct.field(pytree_node=False, default_factory=lambda: dict(RECEPTORS))
    gates: Mapping[str, MgBlock] = struct.field(pytree_node=False,
                                                default_factory=lambda: {"nmda": MgBlock()})
    surrogate: Surrogate = struct.field(pytree_node=False, default=ATan())
    graded = False

    @staticmethod
    def rates(v: jax.Array) -> tuple[tuple[jax.Array, jax.Array], ...]:
        """`(alpha, beta)` of m, h and n at `v`, in 1/ms."""
        m = (0.1 * _vtrap(v + 40, 10.0), 4 * jnp.exp(-(v + 65) / 18))
        h = (0.07 * jnp.exp(-(v + 65) / 20), 1 / (1 + jnp.exp(-(v + 35) / 10)))
        n = (0.01 * _vtrap(v + 55, 10.0), 0.125 * jnp.exp(-(v + 65) / 80))
        return m, h, n

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype) -> HodgkinHuxleyState:
        dtype = membrane_dtype(dtype)
        v = jnp.full(shape, self.v_init, dtype)
        m, h, n = (alpha / (alpha + beta) for alpha, beta in self.rates(v))
        return HodgkinHuxleyState(v, m, h, n, jnp.zeros(shape, dtype))

    def step(self, state: HodgkinHuxleyState, inputs: SynapticInput,
             dt: float) -> tuple[HodgkinHuxleyState, Output]:
        g_syn, syn_drive = _synaptic(self, inputs, state.v)
        count = substeps(dt, self.substep)
        step = dt / count

        def f(s, y):
            v, m, h, n = y
            (am, bm), (ah, bh), (an, bn) = self.rates(v)
            current = (-self.g_na * m ** 3 * h * (v - self.e_na) - self.g_k * n ** 4 * (v - self.e_k)
                       - self.g_l * (v - self.e_l) + syn_drive - g_syn * v + inputs.current_at(s))
            return current / self.c_m, am * (1 - m) - bm * m, ah * (1 - h) - bh * h, an * (1 - n) - bn * n

        def gates_over(m, h, n, v, length):
            return [exact_linear(x, alpha / (alpha + beta), 1 / (alpha + beta), length)
                    for x, (alpha, beta) in zip((m, h, n), self.rates(v), strict=True)]

        def exponential_euler(i, y):
            v, m, h, n = y
            g_na, g_k = self.g_na * m ** 3 * h, self.g_k * n ** 4
            g_total = g_na + g_k + self.g_l + g_syn
            drive = (g_na * self.e_na + g_k * self.e_k + self.g_l * self.e_l + syn_drive
                     + inputs.current_at(i * step))
            return exact_linear(v, drive / g_total, self.c_m / g_total, step), *gates_over(m, h, n, v, step)

        def strang(i, y):
            v, m, h, n = y
            m, h, n = gates_over(m, h, n, v, step / 2)
            g_na, g_k = self.g_na * m ** 3 * h, self.g_k * n ** 4
            g_total = g_na + g_k + self.g_l + g_syn
            drive = (g_na * self.e_na + g_k * self.e_k + self.g_l * self.e_l + syn_drive
                     + inputs.current_at((i + 0.5) * step))
            v = exact_linear(v, drive / g_total, self.c_m / g_total, step)
            return v, *gates_over(m, h, n, v, step / 2)

        y = (state.v, state.m, state.h, state.n)
        if self.scheme == "strang":
            v, m, h, n = strang(0, y) if count == 1 else jax.lax.fori_loop(0, count, strang, y)
        elif self.scheme == "rk4":
            v, m, h, n = rk4(f, y, step, count)
        else:
            v, m, h, n = (exponential_euler(0, y) if count == 1
                          else jax.lax.fori_loop(0, count, exponential_euler, y))
        v = v + inputs.jump
        free = state.refractory <= dt / 2
        fired = spike(v - self.v_spike, self.surrogate) * (v < state.v) * free
        refractory = jnp.where(fired > 0, self.t_ref, jnp.maximum(state.refractory - dt, 0))
        dtype = state.v.dtype
        new = HodgkinHuxleyState(*(x.astype(dtype) for x in (v, m, h, n, refractory)))
        # The peak is found a step late, so the spike is stamped at the step's end.
        return new, Output(fired, jnp.ones_like(fired))

    def is_refractory(self, state: HodgkinHuxleyState, dt: float) -> jax.Array:
        return state.refractory > dt / 2

    def after_threshold(self, state: HodgkinHuxleyState, jump: jax.Array,
                        fired: jax.Array) -> HodgkinHuxleyState:
        # No reset follows a spike here, so the jump lands whether or not it fired.
        return state._replace(v=jump_after_threshold(state.v, jump, None))
