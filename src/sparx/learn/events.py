"""Exact spike times and their gradients: the event-based learning of EventProp (Wunderlich and Pehle 2021).

    times = spike_times(inputs, weights, neuron, horizon=50.0, capacity=4)    # [B, N, capacity]

Neurons are leaky integrate-and-fire with exponential current synapses, in
continuous time:

    tau_m dV/dt = -V + I,    tau_syn dI/dt = -I,    an input spike adds its weight to I,

and a neuron fires when `V` reaches `v_th`, after which `V` is 0. Times
are in ms, as in `sparx.dynamics`, and so are spike times. `V` is the
membrane voltage in mV above rest, and `I` the synaptic current in pA
through a membrane resistance of 1 GOhm, so a current of 1 pA held would
settle the voltage at 1 mV. A weight is the jump of the current, in pA.
`sparx.dynamics.LeakyIntegrateAndFire` with `c_m = tau_m` (a leak conductance of 1 nS),
`e_l = v_reset = 0` and no refractory period, driven by an
`Exponential(tau_syn)` synapse, is this neuron (`tests/test_learn.py`
checks it on a 1 us grid).

Between two input spikes the membrane is a sum of two exponentials, which
has at most one maximum, so its first threshold crossing lies on a rising,
monotone branch: bisection finds it to the precision of the dtype. One
Newton step from that root, taken with the root itself held fixed, gives
the crossing time exactly the derivative the implicit function theorem
gives, `dt*/dp = -(dV/dp) / (dV/dt)` at the crossing, so the gradient of
any loss of spike times through this simulation is EventProp's: the exact
gradient of the event-based dynamics, with each neuron's spike count
fixed. `tests/test_learn.py` checks it against finite differences.

Spike times are padded with `inf` to a fixed `capacity` per neuron; a
neuron that would fire more often is counted in the second return value.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = ["EventLIF", "first_spike_cross_entropy", "spike_times"]


@dataclass(frozen=True)
class EventLIF:
    """The membrane and synaptic time constants `tau_m` and `tau_syn` (ms), and the threshold `v_th` (mV
    above rest).

    `crossings` bounds the spikes one neuron fires between two consecutive
    input spikes, `iterations` the bisection steps that find each. The
    closed form of the membrane divides by `tau_syn - tau_m`, so the two
    must differ.
    """

    tau_m: float = 20.0
    tau_syn: float = 5.0
    v_th: float = 1.0
    crossings: int = 2
    iterations: int = 60

    def __post_init__(self):
        if self.tau_m == self.tau_syn:
            raise ValueError("EventLIF needs tau_m != tau_syn")


def _trajectory(neuron: EventLIF, v0, i0, delta):
    """`V` and `dV/dt` a time `delta` after a state `(v0, i0)`, with no input in between."""
    c = i0 * neuron.tau_syn / (neuron.tau_syn - neuron.tau_m)
    a = v0 - c
    em, es = jnp.exp(-delta / neuron.tau_m), jnp.exp(-delta / neuron.tau_syn)
    return a * em + c * es, -a * em / neuron.tau_m - c * es / neuron.tau_syn


def _crossing(neuron: EventLIF, v0, i0, length):
    """The first time in `[0, length]` the membrane reaches threshold, or `inf`.

    The membrane `a e^{-t/tau_m} + c e^{-t/tau_s}` has its one stationary
    point where its derivative vanishes; before it the membrane is monotone,
    so a crossing, if any, lies in `[0, min(peak, length)]` and the
    threshold is reached there at the bound or not at all.
    """
    c = i0 * neuron.tau_syn / (neuron.tau_syn - neuron.tau_m)
    a = v0 - c
    ratio = -(c * neuron.tau_m) / jnp.where(a == 0, 1e-30, a * neuron.tau_syn)
    rate = 1 / neuron.tau_syn - 1 / neuron.tau_m
    peak = jnp.where(ratio > 0, jnp.log(jnp.where(ratio > 0, ratio, 1.0)) / rate, -1.0)
    upper = jnp.where((peak > 0) & (peak < length), peak, length)
    upper = jnp.where(jnp.isfinite(upper), upper, 10 * neuron.tau_m + 10 * neuron.tau_syn)
    reaches = _trajectory(neuron, v0, i0, upper)[0] >= neuron.v_th

    stop = jax.lax.stop_gradient

    def bisect(_, bounds):
        low, high = bounds
        middle = (low + high) / 2
        above = _trajectory(neuron, stop(v0), stop(i0), middle)[0] >= neuron.v_th
        return jnp.where(above, low, middle), jnp.where(above, middle, high)

    _, root = jax.lax.fori_loop(0, neuron.iterations, bisect, (jnp.zeros_like(upper), stop(upper)))
    root = stop(root)
    # One Newton step with the root held fixed: its value is the root, its
    # derivative the implicit function theorem's.
    value, slope = _trajectory(neuron, v0, i0, root)
    time = root - (value - neuron.v_th) / stop(jnp.where(slope > 0, slope, 1.0))
    return jnp.where(reaches, time, jnp.inf)


def spike_times(inputs: jax.Array, weights: jax.Array, neuron: EventLIF, horizon: float,
                capacity: int) -> tuple[jax.Array, jax.Array]:
    """The output spike times of a layer of `neuron`s driven by input spike times.

    `inputs` `[B, M, K]` holds each input's spike times (ms, `inf` where
    absent), `weights` `[M, N]` (pA). Returns the spike times `[B, N, capacity]`
    up to `horizon` ms, `inf` where absent, and the count of spikes beyond
    capacity `[B]`. Differentiable in the weights and the input times.
    """

    def one(times):
        flat = times.reshape(-1)
        source = jnp.repeat(jnp.arange(times.shape[0]), times.shape[1])
        order = jnp.argsort(flat)
        event_time, event_source = flat[order], source[order]
        size = weights.shape[1]
        zeros = jnp.zeros(size, weights.dtype)

        def advance(state, end):
            """Run each neuron from `state` (at time `t`) to `end`, recording crossings."""
            t, v, i, out, count, lost = state
            for _ in range(neuron.crossings):
                at = _crossing(neuron, v, i, end - t)
                fires = jnp.isfinite(at)
                slot = jax.nn.one_hot(jnp.minimum(count, capacity - 1), capacity, dtype=bool)
                record = fires[:, None] & slot & (count < capacity)[:, None]
                out = jnp.where(record, (t + jnp.where(fires, at, 0.0))[:, None], out)
                lost = lost + jnp.sum(fires & (count >= capacity)).astype(lost.dtype)
                count = count + fires.astype(count.dtype)
                # A neuron that fired restarts from the reset at its spike.
                i = jnp.where(fires, i * jnp.exp(-jnp.where(fires, at, 0.0) / neuron.tau_syn), i)
                v = jnp.where(fires, 0.0, v)
                t = jnp.where(fires, t + jnp.where(fires, at, 0.0), t)
            v_end, _ = _trajectory(neuron, v, i, end - t)
            i_end = i * jnp.exp(-(end - t) / neuron.tau_syn)
            return jnp.full_like(v, end), v_end, i_end, out, count, lost

        def step(state, event):
            when, source = event
            end = jnp.minimum(when, horizon)
            state = advance(state, end)
            t, v, i, out, count, lost = state
            arrives = when < horizon
            i = i + jnp.where(arrives, weights[source], 0.0)
            return (t, v, i, out, count, lost), None

        state = (zeros, zeros, zeros, jnp.full((size, capacity), jnp.inf, weights.dtype),
                 jnp.zeros(size, jnp.int32), jnp.zeros((), jnp.int32))
        state, _ = jax.lax.scan(step, state, (event_time, event_source))
        state = advance(state, jnp.asarray(horizon, weights.dtype))
        return state[3], state[5]

    return jax.vmap(one)(inputs)


def first_spike_cross_entropy(times: jax.Array, labels: jax.Array, tau: float = 5.0,
                              silent: float = 1e4) -> jax.Array:
    """Cross entropy of `softmax(-t_first / tau)` over output neurons, the time-to-first-spike loss of
    Göltz et al. (2021) and Wunderlich and Pehle's latency tasks.

    `times` `[B, N, K]`, `labels` `[B]`; returns the batch mean. A silent
    neuron counts as firing at `silent` ms. A silent neuron carries no
    gradient whatever this is, so a large value makes silence a cliff in
    the loss; the horizon of the simulation keeps the loss bounded.
    """
    first = jnp.min(times, axis=-1)
    first = jnp.where(jnp.isfinite(first), first, silent)
    return jnp.mean(-jax.nn.log_softmax(-first / tau)[jnp.arange(labels.shape[0]), labels])
