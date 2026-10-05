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

Both work with any elementwise cell of `sparx.cells` and take the loss as a
function of each step's output, summed over steps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp

from sparx.cells import Cell

__all__ = ["EPropParams", "OTTTLayer", "accumulate", "bptt_loss", "eligibility_traces", "eprop", "ottt",
           "ottt_dense"]

Loss = Callable[[jax.Array, jax.Array], jax.Array]
"""`loss(output_t, target_t)`: one step's loss, a scalar (averaged over the batch as the caller likes)."""


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
    `y_t = kappa y_{t-1} + z_t @ w_out + b_out`."""

    w_in: jax.Array
    w_rec: jax.Array
    w_out: jax.Array
    b_out: jax.Array


def _run(cell: Cell, params: EPropParams, kappa: float, inputs: jax.Array, cut_recurrence: bool):
    """The recurrent layer and readout over time-major `inputs` `[T, B, in]`; outputs `[T, B, out]`."""
    batch, size = inputs.shape[1], params.w_rec.shape[0]

    def step(carry, u):
        state, z, y = carry
        recurrent = jax.lax.stop_gradient(z) if cut_recurrence else z
        state, z = cell.step(state, u @ params.w_in + recurrent @ params.w_rec)
        y = kappa * y + z @ params.w_out + params.b_out
        return (state, z, y), (y, z)

    carry = (cell.init_state((batch, size), inputs.dtype), jnp.zeros((batch, size), inputs.dtype),
             jnp.zeros((batch, params.w_out.shape[1]), inputs.dtype))
    _, (ys, zs) = jax.lax.scan(step, carry, inputs)
    return ys, zs


def bptt_loss(cell: Cell, params: EPropParams, kappa: float, inputs: jax.Array, targets: jax.Array,
              loss: Loss, cut_recurrence: bool = False) -> jax.Array:
    """The summed loss of the layer `eprop` trains; its gradient is BPTT's, or e-prop's with
    `cut_recurrence` (the gradient stopped at the recurrent spikes)."""
    ys, _ = _run(cell, params, kappa, inputs, cut_recurrence)
    return jnp.sum(jax.vmap(loss)(ys, targets))


def _jacobians(cell: Cell, state, x):
    """Per-neuron derivatives of one step: the state Jacobian `[B, N, d, d]`, the state's and the
    spike's derivatives by the input `[B, N, d]`, `[B, N]`, and the spike's by the state `[B, N, d]`."""
    leaves, tree = jax.tree.flatten(state)

    def f(leaves, x):
        new, z = cell.step(jax.tree.unflatten(tree, leaves), x)
        return jax.tree.leaves(new), z

    d = len(leaves)
    zeros = [jnp.zeros_like(leaf) for leaf in leaves]
    columns, dz_dh = [], []
    for k in range(d):
        tangent = [jnp.ones_like(leaf) if i == k else zero for i, (leaf, zero) in enumerate(zip(leaves, zeros,
                                                                                                strict=True))]
        _, (dnew, dz) = jax.jvp(f, (leaves, x), (tangent, jnp.zeros_like(x)))
        columns.append(jnp.stack(dnew, -1))
        dz_dh.append(dz)
    _, (dnew, dz_dx) = jax.jvp(f, (leaves, x), (zeros, jnp.ones_like(x)))
    return jnp.stack(columns, -1), jnp.stack(dnew, -1), dz_dx, jnp.stack(dz_dh, -1)


def eprop(cell: Cell, params: EPropParams, kappa: float, inputs: jax.Array, targets: jax.Array, loss: Loss,
          feedback: jax.Array | None = None) -> tuple[jax.Array, EPropParams]:
    """e-prop's gradients for a recurrent layer of `cell` and its leaky readout, computed online.

    Each synapse `i -> j` keeps an eligibility vector, the derivative of
    neuron `j`'s state by the weight through `j`'s own dynamics, advanced
    every step by the neuron's state Jacobian; its eligibility trace is the
    spike's derivative through it. The readout's leak `kappa` filters the
    traces, so the learning signal `dloss_t/dy_t @ feedback.T` of each step
    weights them directly. `feedback` is `w_out` (symmetric e-prop) unless
    given (random e-prop). The readout's own gradients are exact.

    Memory is `B x N x (in + N) x d` for `d` state variables per neuron,
    whatever the sequence length. Returns the summed loss and the gradients.
    """
    batch, size = inputs.shape[1], params.w_rec.shape[0]
    fan_in = inputs.shape[2] + size
    feedback = params.w_out if feedback is None else feedback
    state = cell.init_state((batch, size), inputs.dtype)
    d = len(jax.tree.leaves(state))

    def step(carry, xs):
        state, z, y, epsilon, filtered, z_bar, leak, total, grad_w, grad_out, grad_b = carry
        u, target = xs
        pre = jnp.concatenate([u, z], -1)  # [B, in + N]
        x = u @ params.w_in + z @ params.w_rec
        jacobian, dh_dx, dz_dx, dz_dh = _jacobians(cell, state, x)
        # e_t = dz_t/dh_{t-1} eps_{t-1} + dz_t/dx_t pre_t, per synapse [B, N, P]
        trace = jnp.einsum("bnd,bnpd->bnp", dz_dh, epsilon) + dz_dx[:, :, None] * pre[:, None, :]
        epsilon = (jnp.einsum("bnde,bnpe->bnpd", jacobian, epsilon)
                   + dh_dx[:, :, None, :] * pre[:, None, :, None])
        state, z = cell.step(state, x)
        y = kappa * y + z @ params.w_out + params.b_out
        value, dy = jax.value_and_grad(loss)(y, target)
        filtered = kappa * filtered + trace
        z_bar = kappa * z_bar + z
        leak = kappa * leak + 1  # the bias accumulates through the readout's leak too
        signal = dy @ feedback.T  # [B, N]
        grad_w = grad_w + jnp.einsum("bn,bnp->pn", signal, filtered)
        return (state, z, y, epsilon, filtered, z_bar, leak, total + value, grad_w,
                grad_out + z_bar.T @ dy, grad_b + leak * dy.sum(0)), None

    dtype = inputs.dtype
    carry = (state, jnp.zeros((batch, size), dtype), jnp.zeros((batch, params.w_out.shape[1]), dtype),
             jnp.zeros((batch, size, fan_in, d), dtype), jnp.zeros((batch, size, fan_in), dtype),
             jnp.zeros((batch, size), dtype), jnp.zeros((), dtype), jnp.zeros((), dtype),
             jnp.zeros((fan_in, size), dtype),
             jnp.zeros_like(params.w_out), jnp.zeros_like(params.b_out))
    final, _ = jax.lax.scan(step, carry, (inputs, targets))
    total, grad_w, grad_out, grad_b = final[7:]
    n_in = inputs.shape[2]
    return total, EPropParams(grad_w[:n_in], grad_w[n_in:], grad_out, grad_b)


def eligibility_traces(cell: Cell, params: EPropParams, inputs: jax.Array) -> jax.Array:
    """Every step's eligibility traces `dz_t/dW` through each neuron's own state, `[T, B, N, in + N]`,
    for analysis; `eprop` uses them as they are made instead of storing them."""
    batch, size = inputs.shape[1], params.w_rec.shape[0]
    state = cell.init_state((batch, size), inputs.dtype)
    d = len(jax.tree.leaves(state))
    fan_in = inputs.shape[2] + size

    def step(carry, u):
        state, z, epsilon = carry
        pre = jnp.concatenate([u, z], -1)
        x = u @ params.w_in + z @ params.w_rec
        jacobian, dh_dx, dz_dx, dz_dh = _jacobians(cell, state, x)
        trace = jnp.einsum("bnd,bnpd->bnp", dz_dh, epsilon) + dz_dx[:, :, None] * pre[:, None, :]
        epsilon = (jnp.einsum("bnde,bnpe->bnpd", jacobian, epsilon)
                   + dh_dx[:, :, None, :] * pre[:, None, :, None])
        state, z = cell.step(state, x)
        return (state, z, epsilon), trace

    carry = (state, jnp.zeros((batch, size), inputs.dtype), jnp.zeros((batch, size, fan_in, d), inputs.dtype))
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


def ottt(cells: Sequence[Cell], layers: Sequence[OTTTLayer], decay: float, inputs: jax.Array,
         targets: jax.Array, loss: Loss) -> tuple[jax.Array, list[OTTTLayer]]:
    """OTTT's gradients for a feedforward stack, computed online.

    `layers[k]` feeds `cells[k]`, and one more layer reads the last cell's
    spikes out, so `len(layers) == len(cells) + 1`. The first layer sees the
    input as it is; every later one sees spikes, and learns from their
    trace `a_t = decay a_{t-1} + s_t` (their `rate_tracking`, `decay` being
    their `1 - 1/tau`). Membranes carry no gradient between steps, so each
    step's loss differentiates through that step alone; give the cells
    `detach_reset=True`, as their `OnlineLIFNode` has.
    Returns the summed loss and the gradients.
    """
    if len(layers) != len(cells) + 1:
        raise ValueError("ottt needs one more layer than cells: the readout")
    batch = inputs.shape[1]

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
            state, h = cell.step(states[k], x)
            new_states.append(state)
            new_traces.append(decay * traces[k] + jax.lax.stop_gradient(h))
        x = ottt_dense(h, new_traces[-1], params[-1].weight) + params[-1].bias
        return loss(x, target), (new_states, new_traces)

    sizes = [layer.weight.shape[1] for layer in layers[:-1]]
    states = [cell.init_state((batch, n), inputs.dtype) for cell, n in zip(cells, sizes, strict=True)]
    traces = [jnp.zeros((batch, n), inputs.dtype) for n in sizes]
    total, grads, _ = accumulate(step, list(layers), (states, traces), (inputs, targets))
    return total, [OTTTLayer(*g) for g in grads]
