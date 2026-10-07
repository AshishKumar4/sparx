"""Predictive coding and PC-ALM: weight updates from local energy minimization over a stack of layers.

A stack of `L` layers predicts each layer's activity from the one below,
`f_l(params_l, h_{l-1})`, from the input `h_0 = x` to the output. Predictive
coding (PC) holds the hidden activities `h_1 .. h_{L-1}` free, starts them at
the forward pass, and relaxes them by gradient descent on each example's
energy

    E = loss(f_L(h_{L-1}), y) + rho / 2 sum_l ||r_l + lambda_l / rho||^2,   r_l = h_l - f_l(h_{l-1}),

then steps the weights down the gradient of `E` at the relaxed activity,
where each layer's update reads only its own prediction error and the
activity below it. PC's multipliers `lambda_l` stay zero. PC-ALM, augmented
Lagrangian predictive coding (Seely and Gould 2026, arXiv 2605.31022), adds
`alpha r_l` to each layer's multiplier after every activity step (their
eq. 12), so `E` is the augmented Lagrangian of the constraints `r_l = 0`,
written as their eq. 9, each prediction target moved by `-lambda_l / rho`.
In a linear network its weight update converges to backpropagation's
gradient, and in their deep residual MLPs it tracks backpropagation where
PC falls behind.

`PredictiveCoding` holds the inference schedule (`alpha = 0` is PC) and
computes the settled activity and multipliers, the energy and the weight
gradient for any stack of `blocks`, each `block(params, below) ->
prediction`; `sequential_blocks` reads them off a Flax `nn.Sequential`.
`residual_mlp` builds the residual MLP of their experiments at their
mean-field scales. `tests/test_predictive.py` checks every quantity against
their JAX reference (github.com/SakanaAI/pc-alm).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

__all__ = [
    "Block",
    "PredictiveCoding",
    "ResidualBlock",
    "Settled",
    "residual_mlp",
    "sequential_blocks",
    "squared_error",
]

type Params = Mapping[str, jax.Array] | jax.Array
"""One layer's parameters."""
type Block = Callable[[Params, jax.Array], jax.Array]
"""One layer: its prediction of its activity `[B, out]` from its parameters and the activity below."""
type Loss = Callable[[jax.Array, jax.Array], jax.Array]
"""The output's loss per example, `loss(prediction, target) -> [B]`."""


def squared_error(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Half the squared error per example, the loss of their experiments."""
    return 0.5 * jnp.sum((prediction - target) ** 2, axis=-1)


class Settled(NamedTuple):
    activity: tuple[jax.Array, ...]
    """Each hidden layer's activity after inference, `h_1 .. h_{L-1}`, `[B, width]` each."""
    multipliers: tuple[jax.Array, ...]
    """Each hidden layer's multiplier `lambda_l` the weight update reads; zero for PC."""


def _forward(blocks: Sequence[Block], params: Sequence[Params], x: jax.Array) -> list[jax.Array]:
    """Every layer's activity in the forward pass, `h_1 .. h_L`."""
    activity, below = [], x
    for block, held in zip(blocks, params, strict=True):
        below = block(held, below)
        activity.append(below)
    return activity


def _errors(blocks: Sequence[Block], params: Sequence[Params], x: jax.Array,
            activity: Sequence[jax.Array]) -> list[jax.Array]:
    """Each hidden layer's prediction error `r_l = h_l - f_l(h_{l-1})`; the output's is the loss's."""
    below = [x, *activity[:-1]]
    return [h - block(held, z)
            for block, held, z, h in zip(blocks[:-1], params[:-1], below, activity, strict=True)]


@struct.dataclass
class PredictiveCoding:
    """Predictive coding's inference and weight update; `alpha > 0` makes it PC-ALM.

    The activity relaxes in `budget` cycles of `inner_steps` steps of
    `state_lr` down each example's own energy, every cycle but the last
    followed by a multiplier step `lambda_l += alpha r_l`. `alpha = 0` with
    one inner step is PC with `budget` activity steps. `credit` says which
    multipliers the weight update reads: those its last activity steps ran
    with (`"before"`, their `pre_dual_energy` and the paper's Algorithm 1),
    or those one more multiplier step gives (`"after"`, their
    `post_dual_energy`). `rho` is the penalty on the prediction errors, 1
    in their experiments, where `state_lr` is near `1 / lambda_max` of the
    energy's Hessian in the activity (their `eta_best_by_cell.csv`) and
    `budget` is `2 L`.
    """

    budget: int = struct.field(pytree_node=False)
    state_lr: float | jax.Array
    rho: float | jax.Array = 1.0
    alpha: float | jax.Array = 0.0
    inner_steps: int = struct.field(pytree_node=False, default=1)
    credit: Literal["before", "after"] = struct.field(pytree_node=False, default="before")

    def __post_init__(self) -> None:
        if self.budget < 1 or self.inner_steps < 1:
            raise ValueError(f"inference takes at least one step: budget {self.budget}, inner_steps "
                             f"{self.inner_steps}")
        if self.credit not in ("before", "after"):
            raise ValueError(f"credit is before or after the last multiplier step, not {self.credit!r}")

    def energy(self, blocks: Sequence[Block], params: Sequence[Params], x: jax.Array, y: jax.Array,
               settled: Settled, loss: Loss = squared_error) -> jax.Array:
        """Each example's energy `[B]` at `settled`: the output's loss and the shifted prediction errors."""
        output = blocks[-1](params[-1], settled.activity[-1])
        errors = _errors(blocks, params, x, settled.activity)
        shifted = sum(jnp.sum((r + m / self.rho) ** 2, axis=-1)
                      for r, m in zip(errors, settled.multipliers, strict=True))
        return loss(output, y) + 0.5 * self.rho * shifted

    def settle(self, blocks: Sequence[Block], params: Sequence[Params], x: jax.Array, y: jax.Array,
               loss: Loss = squared_error) -> Settled:
        """The hidden activity after inference from the forward pass, and the multipliers the update reads."""
        if len(blocks) < 2:
            raise ValueError("predictive coding needs a hidden layer: at least two blocks")
        activity = tuple(_forward(blocks, params, x)[:-1])
        multipliers = tuple(jnp.zeros_like(h) for h in activity)

        def descend(state: Settled, _: None) -> tuple[Settled, None]:
            # Each example's activity follows its own energy, so the batch's sum is the one to differentiate.
            def summed(activity: tuple[jax.Array, ...]) -> jax.Array:
                return jnp.sum(self.energy(blocks, params, x, y, Settled(activity, state.multipliers), loss))

            grads = jax.grad(summed)(state.activity)
            stepped = tuple(h - self.state_lr * g for h, g in zip(state.activity, grads, strict=True))
            return Settled(stepped, state.multipliers), None

        def relax(state: Settled) -> Settled:
            return jax.lax.scan(descend, state, None, length=self.inner_steps)[0]

        def ascend(state: Settled) -> Settled:
            errors = _errors(blocks, params, x, state.activity)
            return Settled(state.activity, tuple(m + self.alpha * r
                                                 for m, r in zip(state.multipliers, errors, strict=True)))

        def cycle(state: Settled, _: None) -> tuple[Settled, None]:
            return ascend(relax(state)), None

        state = jax.lax.scan(cycle, Settled(activity, multipliers), None, length=self.budget - 1)[0]
        state = relax(state)
        return ascend(state) if self.credit == "after" else state

    def gradient(self, blocks: Sequence[Block], params: Sequence[Params], x: jax.Array, y: jax.Array,
                 loss: Loss = squared_error, rows: jax.Array | None = None) -> list[Params]:
        """The weight update: the gradient of the energy summed over the batch's rows at the settled state.

        `rows` weighs each example (1 for a real row, 0 for a repeat); their
        reference takes the mean over the batch, the gradient here divided by
        the batch size.
        """
        settled = jax.lax.stop_gradient(self.settle(blocks, params, x, y, loss))
        weights = jnp.ones(x.shape[0], x.dtype) if rows is None else rows

        def total(held: Sequence[Params]) -> jax.Array:
            return jnp.sum(weights * self.energy(blocks, held, x, y, settled, loss))

        return jax.grad(total)(list(params))


def sequential_blocks(model: nn.Sequential, params: Mapping[str, Params]) -> tuple[list[Block], list[Params]]:
    """The blocks and parameters of a Flax `nn.Sequential`, one layer of the stack per element.

    An element without parameters, such as `nn.relu`, is a layer of its own
    with empty parameters, so its output is held free like any other.
    """
    held = [params.get(f"layers_{k}", {}) for k in range(len(model.layers))]
    return [_block(layer) for layer in model.layers], held


def _block(layer: Callable[..., jax.Array] | nn.Module) -> Block:
    """One element of an `nn.Sequential` as a block: a module applied with its parameters, or a function."""

    def block(params: Params, below: jax.Array) -> jax.Array:
        if not isinstance(layer, nn.Module):
            return layer(below)
        assert isinstance(params, Mapping), "a module's parameters are a mapping"
        out = layer.apply({"params": params}, below)
        assert not isinstance(out, tuple)  # no mutable collection, so apply returns the output alone
        return out

    return block


ACTIVATIONS: Mapping[str, Callable[[jax.Array], jax.Array]] = {
    "linear": lambda x: x, "tanh": jnp.tanh, "relu": jax.nn.relu}


class ResidualBlock(nn.Module):
    """A layer of Seely and Gould's residual MLP: `scale * f(below) @ kernel`, plus `below` when `skip`.

    The input layer reads the input as it is (`activation=None`); every
    other layer applies the activation to the activity below first. The
    kernel starts at `N(0, 1)` entrywise and `scale` sets its effective size.
    """

    features: int
    scale: float
    activation: Literal["linear", "tanh", "relu"] | None = None
    skip: bool = False

    @nn.compact
    def __call__(self, below: jax.Array) -> jax.Array:
        kernel = self.param("kernel", nn.initializers.normal(1.0), (below.shape[-1], self.features))
        inputs = below if self.activation is None else ACTIVATIONS[self.activation](below)
        out = self.scale * (inputs @ kernel)
        return out + below if self.skip else out


def residual_mlp(width: int, depth: int, inputs: int, outputs: int,
                 activation: Literal["linear", "tanh", "relu"] = "relu") -> nn.Sequential:
    """Their residual MLP of `depth` layers at their mean-field scales (`pcalm.model.model_scales`).

    The input layer scales by `1 / sqrt(inputs)`, the `depth - 2` interior
    layers, each with an identity skip, by `1 / sqrt(width * depth)`, and
    the readout by `1 / width` (Innocenti et al.'s parameterization, their
    Appendix B).
    """
    if depth < 2:
        raise ValueError(f"the stack needs a hidden layer and a readout, depth >= 2, not {depth}")
    layers = [ResidualBlock(width, inputs ** -0.5)]
    layers += [ResidualBlock(width, (width * depth) ** -0.5, activation, skip=True) for _ in range(depth - 2)]
    layers.append(ResidualBlock(outputs, 1 / width, activation))
    return nn.Sequential(layers)
