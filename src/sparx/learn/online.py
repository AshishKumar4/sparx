"""Online learning: gradients computed as a sequence runs, in memory that does not grow with its length.

Backpropagation through time stores every step to run the sequence
backwards. The rules here carry what the gradient needs forward instead:

- `eprop`: e-prop (Bellec et al., Nature Communications 2020) for a
  recurrent spiking layer and a leaky readout. Each synapse keeps an
  eligibility trace, the derivative of its neuron's spike through the
  neuron's own state, and a learning signal from the readout weights it.
  It equals backpropagation with the gradient through the recurrent spikes
  cut, and with the true learning signal it is exactly backpropagation
  (their equation 1); `tests/test_learn.py` checks both identities, as
  their own numerical verifications do.
- `ottt`: online training through time (Xiao et al., NeurIPS 2022) for a
  feedforward spiking stack. Membranes and resets are detached from the
  gradient, each step's loss backpropagates through the layers at that
  step only, and a weight's gradient pairs the error with its input's
  presynaptic trace, as their `WrapedSNNOp` does.

Both work with any elementwise model of `sparx.dynamics.ml`, stepped at
`dt`, and take the loss as a function of each step's output, summed over
steps. Time constants (`tau`) are in the unit of `dt`, so with the default
`dt = 1` they count steps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from dew.nn.precision import at_least_fp32
from flax.typing import PrecisionLike
from jax.extend.core import Jaxpr, Literal, Var

from sparx.dynamics import Dense, LICell, NeuronModel, RecurrentCell, SynapticInput, decay

__all__ = ["EPropParams", "OTTTLayer", "accumulate", "bptt_loss", "eligibility_traces", "eprop",
           "eprop_forward", "ottt", "ottt_dense"]

Loss = Callable[[jax.Array, jax.Array], jax.Array]
"""`loss(output_t, target_t)`: one step's loss, a scalar (averaged over the batch as the caller likes)."""


def _step[State](cell: NeuronModel[State], state: State, x: jax.Array, dt: float) -> tuple[State, jax.Array]:
    """One step of `cell` on the jump `x`; the new state and the spikes."""
    state, spikes = cell.step(state, SynapticInput(jump=x), dt)
    return state, spikes.value


def accumulate[P, C, X](step: Callable[[P, C, X], tuple[jax.Array, C]], params: P, carry: C,
                        xs: X) -> tuple[jax.Array, P, C]:
    """Sum each step's loss and its gradient over a sequence, holding the carry fixed in each step's gradient.

    `step(params, carry, x_t) -> (loss_t, carry)`. Exact for a model whose
    carry the gradient does not cross (a detached membrane, as OTTT has),
    and then memory stays that of one step. Returns the total loss, the
    summed gradient, and the final carry.
    """

    def one(state, x_t):
        carry, total, grads = state
        (loss, new), g = jax.value_and_grad(step, has_aux=True)(params, jax.lax.stop_gradient(carry), x_t)
        return (new, total + loss, jax.tree.map(jnp.add, grads, g)), None

    zeros = jax.tree.map(jnp.zeros_like, params)
    (carry, total, grads), _ = jax.lax.scan(one, (carry, jnp.zeros(()), zeros), xs)
    return total, grads, carry


class EPropParams(NamedTuple):
    """A recurrent layer, `x_t = u_t @ w_in + z_{t-1} @ w_rec`, and its leaky readout,
    `y_t = decay(tau, dt) y_{t-1} + z_t @ w_out + b_out`."""

    w_in: jax.Array
    w_rec: jax.Array
    w_out: jax.Array
    b_out: jax.Array


def _readout(tau: float) -> LICell:
    """The leaky readout of time constant `tau`, a decay per unit of time that the step raises to `dt`."""
    return LICell(decay(tau))


def _promoted(params: EPropParams, inputs: jax.Array) -> tuple[EPropParams, jax.Array]:
    """The parameters and inputs in one dtype, the widest of theirs and float32, so every carry of a scan
    keeps the dtype it starts in: float64 weights promote a float32 input's products."""
    dtype = at_least_fp32(jnp.result_type(inputs, *params))
    return EPropParams(*(p.astype(dtype) for p in params)), inputs.astype(dtype)


def eprop_forward(cell: NeuronModel, params: EPropParams, inputs: jax.Array, *, tau: float, dt: float = 1.0,
                  cut_recurrence: bool = False,
                  precision: PrecisionLike = None) -> tuple[jax.Array, jax.Array]:
    """The network `eprop` trains, over time-major `inputs` `[T, B, in]`: a `RecurrentCell` of `cell`
    and an `LICell` readout of time constant `tau`, both stepped at `dt`.

    Returns the readout `[T, B, out]` and the recurrent layer's spikes
    `[T, B, N]`, in the widest dtype of the inputs, the parameters and
    float32. With `cut_recurrence` the gradient stops at the fed-back
    spikes, which makes BPTT's gradient e-prop's. `precision` is the
    matrix products'.
    """
    params, inputs = _promoted(params, inputs)
    layer = RecurrentCell(cell, Dense(params.w_rec, precision), cut_gradient=cut_recurrence)
    readout = _readout(tau)

    # One scan over both models, with each step's products inside it. Two
    # `run`s, the input and readout products taken over all steps at once,
    # made BPTT's gradient 30 ms where this takes 23 ms, at
    # `benchmarks/bench_eprop.py`'s shapes on a 4-core CPU.
    def step(carry, u):
        state, y = carry
        state, spikes = layer.step(state, SynapticInput(jump=jnp.matmul(u, params.w_in, precision=precision)),
                                   dt)
        jump = jnp.matmul(spikes.value, params.w_out, precision=precision) + params.b_out
        y, out = readout.step(y, SynapticInput(jump=jump), dt)
        return (state, y), (out.value, spikes.value)

    batch, size = inputs.shape[1], params.w_rec.shape[0]
    carry = (layer.init_state((batch, size), inputs.dtype),
             readout.init_state((batch, params.w_out.shape[1]), inputs.dtype))
    return jax.lax.scan(step, carry, inputs)[1]


def bptt_loss(cell: NeuronModel, params: EPropParams, inputs: jax.Array, targets: jax.Array, loss: Loss, *,
              tau: float, dt: float = 1.0, cut_recurrence: bool = False,
              precision: PrecisionLike = None) -> jax.Array:
    """The summed loss of `eprop_forward`'s readout; its gradient is BPTT's, or e-prop's with
    `cut_recurrence`."""
    outputs, _ = eprop_forward(cell, params, inputs, tau=tau, dt=dt, cut_recurrence=cut_recurrence,
                               precision=precision)
    return jnp.sum(jax.vmap(loss)(outputs, targets))


def _sources(jaxpr: Jaxpr) -> list[frozenset[int]]:
    """The inputs each output of `jaxpr` depends on, by index.

    The walk is conservative: an equation's outputs depend on all of its
    inputs, even when it holds a nested jaxpr that would say otherwise, so
    a dependence may be claimed that is not there, never missed.
    """
    sources: dict[Var, frozenset[int]] = {v: frozenset([i]) for i, v in enumerate(jaxpr.invars)}

    def of(v: Var | Literal) -> frozenset[int]:
        return sources.get(v, frozenset()) if isinstance(v, Var) else frozenset()

    for eqn in jaxpr.eqns:
        merged = frozenset().union(*(of(v) for v in eqn.invars))
        for out in eqn.outvars:
            sources[out] = merged
    return [of(v) for v in jaxpr.outvars]


class _Structure(NamedTuple):
    """Which state variables of a model need an eligibility vector per synapse, and which need less.

    The model has `d` state variables; index `d` stands for the input `x`
    among the columns and for the spike among the rows. `general` variables
    keep an eligibility vector per synapse, `[B, N, P]`. A `filtered`
    variable `k` follows `h_k <- decay_k h_k + gain_k x` with constants the
    same for every neuron, so its eligibility vector is `gain_k` times the
    presynaptic activity filtered by `decay_k`, Bellec et al.'s `z_bar`,
    `[B, P]`. The rest carry no eligibility: no input reaches them through
    a gradient (the refractory count). `uses[k]` lists the columns row `k`
    of the step's Jacobian can be nonzero in.
    """

    general: tuple[int, ...]
    filtered: tuple[int, ...]
    decays: tuple[jax.Array, ...]
    gains: tuple[jax.Array, ...]
    uses: tuple[frozenset[int], ...]


def _structure(cell: NeuronModel, dtype: jnp.dtype, dt: float) -> _Structure:
    """`cell`'s `_Structure`, read from the jaxpr of its step's forward derivative on one neuron.

    The derivative of the new state by the old state and the input is
    linear in their tangents, so an output tangent that does not depend on
    an input tangent has a zero coefficient there, for every state; one
    that depends on no primal value has constant coefficients. A model
    whose constants are per neuron shapes the one-neuron outputs like
    them, which keeps it off the `filtered` path, whose filter is shared
    by every neuron.
    """
    leaves, tree = jax.tree.flatten(cell.init_state((), dtype))
    d = len(leaves)
    x = jnp.zeros((), dtype)

    def step(leaves: list[jax.Array], x: jax.Array) -> tuple[list[jax.Array], jax.Array]:
        new, z = _step(cell, jax.tree.unflatten(tree, leaves), x, dt)
        return jax.tree.leaves(new), z

    def tangents(*args: jax.Array) -> list[jax.Array]:
        primal, tangent = list(args[:d + 1]), list(args[d + 1:])
        new, z = jax.jvp(step, (primal[:d], primal[d]), (tangent[:d], tangent[d]))[1]
        return [*new, z]

    rows = _sources(jax.make_jaxpr(tangents)(*leaves, x, *leaves, x).jaxpr)
    uses = tuple(frozenset(j - d - 1 for j in row if j > d) for row in rows)
    primal_free = [not any(j <= d for j in row) for row in rows]

    silent = {k for k in range(d) if d not in uses[k]}
    while any(not uses[k] <= silent for k in silent):
        silent = {k for k in silent if uses[k] <= silent}
    shapes = [jnp.shape(leaf) for leaf in jax.eval_shape(tangents, *leaves, x, *leaves, x)]
    filtered = tuple(k for k in range(d) if k not in silent and primal_free[k] and shapes[k] == ()
                     and uses[k] - silent <= {k, d})
    general = tuple(k for k in range(d) if k not in silent and k not in filtered)

    def coefficient(k: int, j: int) -> jax.Array:
        unit = [jnp.ones_like(t) if i == j else jnp.zeros_like(t) for i, t in enumerate([*leaves, x])]
        return tangents(*leaves, x, *unit)[k]

    return _Structure(general, filtered, tuple(coefficient(k, k) for k in filtered),
                      tuple(coefficient(k, d) for k in filtered), uses)


class _Eligibility(NamedTuple):
    """The eligibility vectors of the general state variables `[B, N, P]` and the filtered presynaptic
    activity of the filtered ones `[B, P]`."""

    vectors: tuple[jax.Array, ...]
    presynaptic: tuple[jax.Array, ...]


def _trace_step[State](cell: NeuronModel[State], structure: _Structure, dt: float, params: EPropParams,
                       state: State, u: jax.Array, z: jax.Array, eligibility: _Eligibility,
                       precision: PrecisionLike) -> tuple[State, jax.Array, jax.Array, _Eligibility]:
    """One step of the recurrent layer and of its synapses' eligibility, on the input `u` `[B, in]` and the
    previous step's spikes `z` `[B, N]`, the presynaptic activity `pre = [u, z]`.

    The eligibility vector of synapse `p -> n` is `eps_t = J_t eps_{t-1} +
    dh_t/dx_t pre_t`, and its trace `e_t = dz_t/dh_{t-1} eps_{t-1} +
    dz_t/dx_t pre_t`, with `J_t` the neuron's state Jacobian. A model is
    elementwise, so one forward derivative with a tangent of ones on one
    state variable gives that column of every neuron's Jacobian; the
    columns are evaluated together. Returns the new state, the spikes, the
    traces `[B, N, P]` and the new eligibility.
    """
    pre = jnp.concatenate([u, z], -1)
    x = jnp.matmul(u, params.w_in, precision=precision) + jnp.matmul(z, params.w_rec, precision=precision)
    leaves, tree = jax.tree.flatten(state)
    d = len(leaves)

    def step(leaves: list[jax.Array], x: jax.Array) -> tuple[list[jax.Array], jax.Array]:
        new, z = _step(cell, jax.tree.unflatten(tree, leaves), x, dt)
        return jax.tree.leaves(new), z

    (new, z), linear = jax.linearize(step, leaves, x)
    columns = (*structure.general, *structure.filtered, d)
    basis = [jnp.stack([jnp.full_like(leaf, i == j) for j in columns]) for i, leaf in enumerate([*leaves, x])]

    def column(*tangent: jax.Array) -> list[jax.Array]:
        new, z = linear(list(tangent[:d]), tangent[d])
        return [*new, z]

    derivative = jax.vmap(column)(*basis)
    coefficient = {(k, j): derivative[k][c] for k in range(d + 1) for c, j in enumerate(columns)}

    def row(k: int) -> jax.Array:
        """The eligibility of state variable `k` after this step, or the trace for `k = d`."""
        out = coefficient[k, d][..., None] * pre[:, None, :]
        for j, vector in zip(structure.general, eligibility.vectors, strict=True):
            if j in structure.uses[k]:
                out = out + coefficient[k, j][..., None] * vector
        filtered = zip(structure.filtered, structure.gains, eligibility.presynaptic, strict=True)
        for j, gain, activity in filtered:
            if j in structure.uses[k]:
                out = out + (coefficient[k, j] * gain)[..., None] * activity[:, None, :]
        return out

    trace = row(d)
    presynaptic = zip(structure.decays, eligibility.presynaptic, strict=True)
    updated = _Eligibility(tuple(row(k) for k in structure.general),
                           tuple(decay * activity + pre for decay, activity in presynaptic))
    return jax.tree.unflatten(tree, new), z, trace, updated


def _start(structure: _Structure, batch: int, size: int, fan_in: int, dtype: jnp.dtype) -> _Eligibility:
    return _Eligibility(tuple(jnp.zeros((batch, size, fan_in), dtype) for _ in structure.general),
                        tuple(jnp.zeros((batch, fan_in), dtype) for _ in structure.filtered))


def eprop(cell: NeuronModel, params: EPropParams, inputs: jax.Array, targets: jax.Array, loss: Loss, *,
          tau: float, dt: float = 1.0, feedback: jax.Array | None = None,
          precision: PrecisionLike = None) -> tuple[jax.Array, EPropParams]:
    """e-prop's gradients for `eprop_forward`'s network, computed online.

    Each synapse `i -> j` keeps an eligibility vector, the derivative of
    neuron `j`'s state by the weight through `j`'s own dynamics, advanced
    every step by the neuron's state Jacobian; its eligibility trace is the
    spike's derivative through it. The readout's leak filters the traces,
    so the learning signal `dloss_t/dy_t @ feedback.T` of each step weights
    them directly. `feedback` is `w_out` (symmetric e-prop) unless given
    (random e-prop). The readout's own gradients are exact.

    The vectors are derived for any elementwise model from the forward
    derivative of its step. A state variable whose input enters with
    constant coefficients shared by all neurons (the membrane of a model
    with a detached reset) needs only the filtered presynaptic activity,
    and one the gradient never reaches (a refractory count) needs nothing.
    Memory is `B x N x P` for the filtered traces, plus as much for each
    remaining state variable, `P = in + N`, whatever the sequence length.
    It runs in the widest dtype of the inputs, the parameters and float32,
    and `precision` is its matrix products'. Returns the summed loss and the
    gradients, each in its parameter's dtype.
    """
    given = params
    params, inputs = _promoted(params, inputs)
    dtype = inputs.dtype
    batch, size = inputs.shape[1], params.w_rec.shape[0]
    fan_in = inputs.shape[2] + size
    feedback = params.w_out if feedback is None else feedback.astype(dtype)
    structure = _structure(cell, dtype, dt)
    readout = _readout(tau)
    kappa = readout.decay ** dt

    def step(carry, xs):
        state, z, y, eligibility, filtered, z_bar, leak, total, grad_w, grad_out, grad_b = carry
        u, target = xs
        state, z, trace, eligibility = _trace_step(cell, structure, dt, params, state, u, z, eligibility,
                                                   precision)
        jump = jnp.matmul(z, params.w_out, precision=precision) + params.b_out
        y, out = readout.step(y, SynapticInput(jump=jump), dt)
        value, dy = jax.value_and_grad(loss)(out.value, target)
        filtered = kappa * filtered + trace
        z_bar = kappa * z_bar + z
        leak = kappa * leak + 1  # the bias accumulates through the readout's leak too
        signal = jnp.matmul(dy, feedback.T, precision=precision)  # [B, N]
        # A product and a sum over the batch: XLA on CPU runs the einsum
        # "bn,bnp->pn" as a batched matrix product, three times slower here.
        grad_w = grad_w + (signal[..., None] * filtered).sum(0)
        return (state, z, y, eligibility, filtered, z_bar, leak, total + value, grad_w,
                grad_out + jnp.matmul(z_bar.T, dy, precision=precision), grad_b + leak * dy.sum(0)), None

    carry = (cell.init_state((batch, size), dtype), jnp.zeros((batch, size), dtype),
             readout.init_state((batch, params.w_out.shape[1]), dtype),
             _start(structure, batch, size, fan_in, dtype), jnp.zeros((batch, size, fan_in), dtype),
             jnp.zeros((batch, size), dtype), jnp.zeros((), dtype), jnp.zeros((), dtype),
             jnp.zeros((size, fan_in), dtype), jnp.zeros_like(params.w_out), jnp.zeros_like(params.b_out))
    final, _ = jax.lax.scan(step, carry, (inputs, targets))
    total, grad_w, grad_out, grad_b = final[7:]
    n_in = inputs.shape[2]
    grad_w = grad_w.T
    grads = EPropParams(grad_w[:n_in], grad_w[n_in:], grad_out, grad_b)
    return total, EPropParams(*(g.astype(p.dtype) for g, p in zip(grads, given, strict=True)))


def eligibility_traces(cell: NeuronModel, params: EPropParams, inputs: jax.Array, *,
                       dt: float = 1.0, precision: PrecisionLike = None) -> jax.Array:
    """Every step's eligibility traces `dz_t/dW` through each neuron's own state, `[T, B, N, in + N]`,
    for analysis; `eprop` uses them as they are made instead of storing them."""
    params, inputs = _promoted(params, inputs)
    dtype = inputs.dtype
    batch, size = inputs.shape[1], params.w_rec.shape[0]
    structure = _structure(cell, dtype, dt)

    def step(carry, u):
        state, z, eligibility = carry
        state, z, trace, eligibility = _trace_step(cell, structure, dt, params, state, u, z, eligibility,
                                                   precision)
        return (state, z, eligibility), trace

    carry = (cell.init_state((batch, size), dtype), jnp.zeros((batch, size), dtype),
             _start(structure, batch, size, inputs.shape[2] + size, dtype))
    return jax.lax.scan(step, carry, inputs)[1]


@jax.custom_vjp
def ottt_dense(spikes: jax.Array, trace: jax.Array, weight: jax.Array) -> jax.Array:
    """`spikes @ weight` whose weight gradient pairs the error with the presynaptic `trace`, OTTT's op."""
    return spikes @ weight


def _ottt_forward(spikes, trace, weight):
    return spikes @ weight, (trace, weight)


def _ottt_backward(saved, g):
    trace, weight = saved
    grad_weight = trace.reshape(-1, trace.shape[-1]).T @ g.reshape(-1, g.shape[-1])
    return g @ weight.T, jnp.zeros_like(trace), grad_weight


ottt_dense.defvjp(_ottt_forward, _ottt_backward)


class OTTTLayer(NamedTuple):
    weight: jax.Array
    bias: jax.Array


def ottt(cells: Sequence[NeuronModel], layers: Sequence[OTTTLayer], inputs: jax.Array, targets: jax.Array,
         loss: Loss, *, tau: float, dt: float = 1.0) -> tuple[jax.Array, list[OTTTLayer]]:
    """OTTT's gradients for a feedforward stack, computed online.

    `layers[k]` feeds `cells[k]`, and one more layer reads the last cell's
    spikes out, so `len(layers) == len(cells) + 1`. The first layer sees the
    input as it is; every later one sees spikes, and learns from their
    trace `a_t = decay(tau, dt) a_{t-1} + s_t` (their `rate_tracking`).
    Their trace decays by `1 - dt / tau`, the forward Euler step of the
    same leak; `tau = -dt / log(1 - dt / tau_theirs)` gives their decay.
    Membranes carry no gradient between steps, so each step's loss
    differentiates through that step alone; give the cells
    `detach_reset=True`, as their `OnlineLIFNode` has.
    Returns the summed loss and the gradients.
    """
    if len(layers) != len(cells) + 1:
        raise ValueError("ottt needs one more layer than cells: the readout")
    batch = inputs.shape[1]
    kept = decay(tau, dt)

    def step(params, carry, xs):
        # A spike's trace includes the spike itself, as their `rate_tracking`
        # does, so each op reads the trace updated with this step's spikes.
        u, target = xs
        states, traces = carry
        h, new_states, new_traces = u, [], []
        for k, cell in enumerate(cells):
            layer = params[k]
            if k == 0:
                x = h @ layer.weight + layer.bias
            else:
                x = ottt_dense(h, new_traces[k - 1], layer.weight) + layer.bias
            state, h = _step(cell, states[k], x, dt)
            new_states.append(state)
            new_traces.append(kept * traces[k] + jax.lax.stop_gradient(h))
        x = ottt_dense(h, new_traces[-1], params[-1].weight) + params[-1].bias
        return loss(x, target), (new_states, new_traces)

    sizes = [layer.weight.shape[1] for layer in layers[:-1]]
    states = [cell.init_state((batch, n), inputs.dtype) for cell, n in zip(cells, sizes, strict=True)]
    traces = [jnp.zeros((batch, n), inputs.dtype) for n in sizes]
    total, grads, _ = accumulate(step, list(layers), (states, traces), (inputs, targets))
    return total, [OTTTLayer(*g) for g in grads]
