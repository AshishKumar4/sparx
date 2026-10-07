"""Synaptic plasticity: short-term depression and facilitation, and spike-timing-dependent plasticity.

Rules are clock-driven: each step they see which presynaptic neurons spiked
and which postsynaptic spikes reached the synapse, and update per-neuron
traces and per-edge weights. An edge list `pre[E]`, `post[E]` names each
synapse's neurons; traces live on neurons, so `N` traces serve `E` edges.

Spike times are on the step grid. Where two events share a step, rules
apply them in the order NEST does: a postsynaptic spike pairs only with
presynaptic spikes strictly before it, a presynaptic spike only with
postsynaptic spikes strictly before it, and a facilitation from a
postsynaptic spike lands before the depression of a presynaptic spike in
the same step. NEST evaluates these rules lazily, at each presynaptic
spike, from the spike history; the clock-driven updates apply the same
operations in the same order, so the weights it transmits are the same.

Delays are dendritic, as in NEST's STDP synapses: a presynaptic spike is at
the synapse when it is sent, and a postsynaptic spike reaches it `delay`
later. The caller passes postsynaptic spikes as they arrive.

A rule also reads the network's neuromodulators each step, by name
(`sparx.graph.Modulator`): volume-transmitted concentrations such as
dopamine's, the third factor of a three-factor rule. `PairSTDP` and
`TripletSTDP` are two-factor and ignore them; `DopamineSTDP` is
Izhikevich's (2007) reward-modulated STDP, which reads dopamine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from flax import struct

__all__ = ["DopamineSTDP", "DopamineTraces", "PairSTDP", "Plasticity", "STDPTraces", "TripletSTDP",
           "TripletTraces", "TsodyksMarkram", "TsodyksMarkramState"]


class Plasticity[Traces](Protocol):
    """A rule that changes the weights of a projection's edges with the spikes on either side.

    Its traces live on neurons and edges, `init_state(pre, post, edges,
    dtype)` for `pre` presynaptic and `post` postsynaptic neurons joined by
    `edges` synapses, and `step` advances them and the weights `[E]` of the
    edges `pre[E] -> post[E]` by one step.
    """

    def init_state(self, pre: int, post: int, edges: int, dtype: jnp.dtype = jnp.float32) -> Traces:
        """The traces with no spike yet."""
        ...

    def modulated_by(self) -> Mapping[str, float | None]:
        """The modulators the rule reads, by name, each with the time constant (ms) the rule assumes
        its concentration decays with, or None where the rule reads the concentration alone.

        A network refuses a rule that reads a modulator it lacks, or one whose time constant differs.
        """
        ...

    def step(self, traces: Traces, weights: jax.Array, pre_spikes: jax.Array, post_arrivals: jax.Array,
             pre: jax.Array, post: jax.Array, dt: float, *,
             modulators: Mapping[str, jax.Array]) -> tuple[Traces, jax.Array]:
        """One step: traces decay over `dt`, then this step's spikes update weights and traces.

        `pre_spikes[N_pre]` and `post_arrivals[N_post]` are 0 or 1;
        `weights[E]` belong to the edges `pre[E] -> post[E]`. `modulators`
        holds each neuromodulator's concentration after this step's release,
        by name, a scalar each (with a leading trial axis under `vmap`).
        """
        ...


class TsodyksMarkramState(NamedTuple):
    recovered: jax.Array
    """The fraction of resources available, between spikes recovering toward 1."""
    facilitation: jax.Array
    """What the next spike adds to `U` for its release probability, decaying toward 0."""


@struct.dataclass
class TsodyksMarkram:
    """Short-term depression and facilitation (Tsodyks and Markram 1997; Markram et al. 1998).

    A spike releases a fraction `u` of the available resources `x` and
    transmits `w u x`. Between spikes `x` recovers toward 1 with `tau_rec`
    and `u` relaxes toward `U` with `tau_fac` (no facilitation when
    `tau_fac` is 0). At a spike, with `h` since the last one:

        u = U + u_last (1 - U) exp(-h / tau_fac)
        x = 1 + (x_last (1 - u_last) - 1) exp(-h / tau_rec)

    the update of NEST's `tsodyks2_synapse`, kept per presynaptic neuron and
    integrated exactly between spikes. A synapse starts at rest, fully
    recovered and unfacilitated, so its first spike transmits `w U`. NEST
    transmits its initial `w u x` at the first spike, and its initial `u`
    is 0.5 whatever `U` is; it agrees with sparx when `u` is set to `U`.
    """

    U: jax.Array | float = 0.5
    tau_rec: jax.Array | float = 800.0
    tau_fac: jax.Array | float = 0.0

    def init_state(self, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> TsodyksMarkramState:
        """Fully recovered and unfacilitated, per presynaptic neuron."""
        return TsodyksMarkramState(jnp.ones(shape, dtype), jnp.zeros(shape, dtype))

    def step(self, state: TsodyksMarkramState, spikes: jax.Array,
             dt: float) -> tuple[TsodyksMarkramState, jax.Array]:
        """Advance `dt` and release for this step's `spikes`.

        Returns the efficacy `u x` per presynaptic neuron, 0 where silent.
        """
        recovered = 1 + (state.recovered - 1) * jnp.exp(-dt / self.tau_rec)
        facilitates = self.tau_fac > 0
        decay = jnp.where(facilitates, jnp.exp(-dt / jnp.where(facilitates, self.tau_fac, 1.0)), 0.0)
        facilitation = state.facilitation * decay
        u = self.U + facilitation
        efficacy = spikes * u * recovered
        facilitation = jnp.where(spikes > 0, u * (1 - self.U), facilitation)
        return TsodyksMarkramState(recovered - efficacy, facilitation), efficacy


class STDPTraces(NamedTuple):
    pre: jax.Array
    post: jax.Array


@struct.dataclass
class PairSTDP:
    """All-to-all pair STDP with soft or hard bounds: NEST's `stdp_synapse` (Guetig et al. 2003).

    With weights normalized by `w_max`, a postsynaptic spike potentiates
    and a presynaptic spike depresses:

        w <- min(w + lambda (1 - w)^mu_plus K_pre, 1)
        w <- max(w - alpha lambda w^mu_minus K_post, 0)

    where `K_pre` sums `exp(-dt / tau_plus)` over earlier presynaptic
    spikes and `K_post` sums `exp(-dt / tau_minus)` over earlier
    postsynaptic arrivals. `mu = 0` is additive STDP (Song et al. 2000),
    `mu = 1` multiplicative (van Rossum et al. 2000).
    """

    tau_plus: jax.Array | float = 20.0
    tau_minus: jax.Array | float = 20.0
    lambda_: jax.Array | float = 0.01
    alpha: jax.Array | float = 1.0
    mu_plus: jax.Array | float = 1.0
    mu_minus: jax.Array | float = 1.0
    w_max: jax.Array | float = 100.0

    def init_state(self, pre: int, post: int, edges: int, dtype: jnp.dtype = jnp.float32) -> STDPTraces:
        return STDPTraces(jnp.zeros(pre, dtype), jnp.zeros(post, dtype))

    def modulated_by(self) -> Mapping[str, float | None]:
        return {}

    def step(self, traces: STDPTraces, weights: jax.Array, pre_spikes: jax.Array, post_arrivals: jax.Array,
             pre: jax.Array, post: jax.Array, dt: float, *,
             modulators: Mapping[str, jax.Array]) -> tuple[STDPTraces, jax.Array]:
        """As `Plasticity.step`."""
        k_pre = traces.pre * jnp.exp(-dt / self.tau_plus)
        k_post = traces.post * jnp.exp(-dt / self.tau_minus)
        w = weights / self.w_max
        potentiated = jnp.minimum(w + self.lambda_ * (1 - w) ** self.mu_plus * k_pre[pre], 1.0)
        w = jnp.where(post_arrivals[post] > 0, potentiated, w)
        depressed = jnp.maximum(w - self.alpha * self.lambda_ * w ** self.mu_minus * k_post[post], 0.0)
        w = jnp.where(pre_spikes[pre] > 0, depressed, w)
        return STDPTraces(k_pre + pre_spikes, k_post + post_arrivals), w * self.w_max


class TripletTraces(NamedTuple):
    pre: jax.Array
    pre_triplet: jax.Array
    post: jax.Array
    post_triplet: jax.Array


@struct.dataclass
class TripletSTDP:
    """All-to-all triplet STDP (Pfister and Gerstner 2006): NEST's `stdp_triplet_synapse`.

    A postsynaptic spike potentiates by `K_pre (A2+ + A3+ y)` and a
    presynaptic spike depresses by `K_post (A2- + A3- r)`, weights clipped
    to `[0, w_max]`, where `K_pre`, `r` are presynaptic traces with
    `tau_plus`, `tau_x` and `K_post`, `y` postsynaptic traces with
    `tau_minus`, `tau_y`, each read just before the spike that uses it.
    The defaults are Pfister and Gerstner's all-to-all fit to visual
    cortex, whose `A2+` (5e-10) is effectively zero. NEST's synapse has the
    same defaults, but keeps `tau_minus` and `tau_y` on the postsynaptic
    neuron, where they default to 20 and 110 ms.
    """

    tau_plus: jax.Array | float = 16.8
    tau_x: jax.Array | float = 101.0
    tau_minus: jax.Array | float = 33.7
    tau_y: jax.Array | float = 125.0
    a2_plus: jax.Array | float = 5e-10
    a3_plus: jax.Array | float = 6.2e-3
    a2_minus: jax.Array | float = 7e-3
    a3_minus: jax.Array | float = 2.3e-4
    w_max: jax.Array | float = 100.0

    def init_state(self, pre: int, post: int, edges: int, dtype: jnp.dtype = jnp.float32) -> TripletTraces:
        return TripletTraces(jnp.zeros(pre, dtype), jnp.zeros(pre, dtype), jnp.zeros(post, dtype),
                             jnp.zeros(post, dtype))

    def modulated_by(self) -> Mapping[str, float | None]:
        return {}

    def step(self, traces: TripletTraces, weights: jax.Array, pre_spikes: jax.Array, post_arrivals: jax.Array,
             pre: jax.Array, post: jax.Array, dt: float, *,
             modulators: Mapping[str, jax.Array]) -> tuple[TripletTraces, jax.Array]:
        """As `PairSTDP.step`."""
        k_pre = traces.pre * jnp.exp(-dt / self.tau_plus)
        r = traces.pre_triplet * jnp.exp(-dt / self.tau_x)
        k_post = traces.post * jnp.exp(-dt / self.tau_minus)
        y = traces.post_triplet * jnp.exp(-dt / self.tau_y)
        potentiated = jnp.minimum(weights + k_pre[pre] * (self.a2_plus + self.a3_plus * y[post]), self.w_max)
        w = jnp.where(post_arrivals[post] > 0, potentiated, weights)
        depressed = jnp.maximum(w - k_post[post] * (self.a2_minus + self.a3_minus * r[pre]), 0.0)
        w = jnp.where(pre_spikes[pre] > 0, depressed, w)
        return TripletTraces(k_pre + pre_spikes, r + pre_spikes, k_post + post_arrivals, y + post_arrivals), w


class DopamineTraces(NamedTuple):
    pre: jax.Array
    post: jax.Array
    eligibility: jax.Array
    """Each synapse's eligibility `c`, `[E]`."""
    dopamine: jax.Array
    """The dopamine concentration at the end of the last step, which this step's integral starts from."""


@struct.dataclass
class DopamineSTDP:
    """Reward-modulated STDP (Izhikevich 2007): NEST's `stdp_dopamine_synapse` (Potjans et al. 2010).

    Pair STDP writes each synapse's eligibility `c` instead of its weight,
    and the weight follows the eligibility gated by the concentration `n`
    of the network's modulator `modulator`, above a baseline `b`:

        dc/dt = -c / tau_c + a_plus K_pre delta(t - t_post) - a_minus K_post delta(t - t_pre)
        dw/dt = c (n - b)

    `K_pre` sums `exp(-dt / tau_plus)` over earlier presynaptic spikes and
    `K_post` sums `exp(-dt / tau_minus)` over earlier postsynaptic arrivals,
    each read just before the spike that uses it, as in `PairSTDP`. Between
    steps `c` decays with `tau_c` and `n` with `tau_n`, so the weight
    integrates their product exactly, as NEST does between events; `tau_n`
    must be the modulator's `tau`, which the network checks, so it is a
    Python number and not traced. A modulator
    whose `release` is `1 / tau_n` adds NEST's increment per dopamine spike.

    The weight is clipped to `[w_min, w_max]` after every step, where NEST
    clips at events (spikes, dopamine arrivals and the volume transmitter's
    deliveries). The two agree while `n - b` keeps its sign between
    dopamine releases, which it always does for `b = 0`, or while the
    weight stays inside its bounds. The defaults are NEST's, Izhikevich's
    (2007) values.
    """

    modulator: str = struct.field(pytree_node=False, default="dopamine")
    tau_plus: jax.Array | float = 20.0
    tau_minus: jax.Array | float = 20.0
    tau_c: jax.Array | float = 1000.0
    tau_n: float = struct.field(pytree_node=False, default=200.0)
    a_plus: jax.Array | float = 1.0
    a_minus: jax.Array | float = 1.5
    b: jax.Array | float = 0.0
    w_min: jax.Array | float = 0.0
    w_max: jax.Array | float = 200.0

    def init_state(self, pre: int, post: int, edges: int, dtype: jnp.dtype = jnp.float32) -> DopamineTraces:
        return DopamineTraces(jnp.zeros(pre, dtype), jnp.zeros(post, dtype), jnp.zeros(edges, dtype),
                              jnp.zeros((), dtype))

    def modulated_by(self) -> Mapping[str, float | None]:
        return {self.modulator: self.tau_n}

    def step(self, traces: DopamineTraces, weights: jax.Array, pre_spikes: jax.Array,
             post_arrivals: jax.Array, pre: jax.Array, post: jax.Array, dt: float, *,
             modulators: Mapping[str, jax.Array]) -> tuple[DopamineTraces, jax.Array]:
        """As `Plasticity.step`: the weight over the step, then this step's spikes on the eligibility."""
        c, n = traces.eligibility, traces.dopamine
        both = 1 / self.tau_c + 1 / self.tau_n
        # The integral of c(s) (n(s) - b) over the step, both decaying from where the last step left them.
        gained = c * (-n * jnp.expm1(-both * dt) / both + self.b * self.tau_c * jnp.expm1(-dt / self.tau_c))
        w = jnp.clip(weights + gained, self.w_min, self.w_max)
        k_pre = traces.pre * jnp.exp(-dt / self.tau_plus)
        k_post = traces.post * jnp.exp(-dt / self.tau_minus)
        c = (c * jnp.exp(-dt / self.tau_c) + self.a_plus * k_pre[pre] * post_arrivals[post]
             - self.a_minus * k_post[post] * pre_spikes[pre])
        return DopamineTraces(k_pre + pre_spikes, k_post + post_arrivals, c, modulators[self.modulator]), w
