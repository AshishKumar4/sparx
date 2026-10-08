"""Reward diffusion and the network it was made for: RNeuralNet-Research, rebuilt deterministically.

Ashish Kumar Singh's RNeuralNet-Research (github.com/AshishKumar4/RNeuralNet-Research,
commit d4b7803, 2018) is a randomly wired recurrent network of graded
neurons (`sparx.dynamics.PulseCell`) whose connections queue weighted
messages for a few steps, trained by a reward that spreads backward from a
reward feeder neuron. Its program runs the forward pass, the neurons'
refresh and the learning in threads that share the neurons without
synchronization, so what it computes depends on how they interleave.
`RNeuralNet` fixes one order, each tick:

    1. each input neuron takes its value,
    2. every neuron computes its output from its sum (`Global_Refresher`),
    3. one pass over the connections in the order they were made, each one
       first sending its source's output if the source is ready, then
       delivering a message that is due (`Global_ForwardProcessor`),
    4. on a tick with a reward, the reward spreads from the feeder and every
       weight changes (`Global_Adjuster`, which the original leaves
       commented out, then `Global_Teacher` and `Global_Renew`),

and computes it clocked. Every neuron is ready every tick (its activation
sets `Fired` whatever the sum), sends its output on all its connections in
one stretch of the pass and empties its sum, so a tick is one step of
`PulseCell`s fed back through a `Sparse` wiring with a delay per connection.
A message sent in one pass is delivered `Myelin + 1` passes later, at the
connection's place in the pass, and reaches its target's output that pass
if the target sends later in the pass, the next one otherwise; that is the
connection's delay. `tools/make_rneuralnet_fixtures.py` runs the original
classes and functions, unchanged, in this order, and the tests match them.

The reward rule, as the notes on the project reconstruct it from
`Global_RewardSpreader` and `Neurite_t::WeightPostProcessor`: a unit `j`
holding a local reward `c_j` passes each presynaptic unit `i` the share

    p_{i|j} = exp|a_i| / sum_{k -> j} exp|a_k|
    c_i += p_{i|j} c_j

of it, a softmax of the absolute activity `a` of its inputs (each unit's
last output), and each connection changes by `dw_{ji} = eta c_j p_{i|j}`,
with `eta` 0.01 (`W_CONST`). The original spreads depth first from the
feeder and lets each unit pass on only the reward it holds when the spread
first reaches it (`Var2`), so what a unit receives afterwards, along
another path, stops there, and the result depends on the order of the
connections. `paths="all"` is the repair the notes propose: every path
counts, each connection passing on `discount` of its share, `c = r +
discount P c`, which converges to `(I - discount P)^-1 r` for a discount
below 1 whatever the cycles.

The notes also show why neither is a gradient. The share is positive
whatever the sign of an input or of its weight, so two inputs of +1 and -1
that should move their weights apart move them together; a reward of `R +
100` changes the weights differently from `R`, although it orders the
outcomes the same; and the activity it reads is each unit's last output,
not the activity that earned the reward. sparx keeps the rule as a
baseline: `sparx.objectives.RNeuralNetObjective` trains the network by it,
or by REINFORCE through the network, under dew's trainer, and
`examples/reward_diffusion.py` runs the comparison the notes propose.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from sparx.dynamics import MembraneState, PulseCell, RecurrentCell, RecurrentState, Sparse, run

__all__ = ["Diffusion", "RNeuralNet", "reward_diffusion", "reward_shares"]

type Paths = Literal["first", "all"]

MYELIN = 20
"""The original draws each connection's `Myelin` from 0 to 19 (`rand() % 20`)."""


@struct.dataclass
class RNeuralNet:
    """RNeuralNet-Research's network, clocked: `neurons` internal neurons, `inputs` input neurons and the
    reward feeder, in that order, as one recurrent layer.

    `cell` is a `PulseCell` per unit fed back through a `Sparse` wiring
    with each connection's delay; the input neurons pass their values on
    unchanged (a threshold of minus infinity) and the feeder sends nothing.
    `outputs` are the internal neurons wired to the feeder, whose last
    outputs share the reward first. Build one with `wire`, from connections
    in the order the original makes them, or `random`, as its
    `NeuralNet_init` draws them; `run` it over inputs `[T, ..., inputs]`
    and `learn` from a reward.
    """

    cell: RecurrentCell[MembraneState, None]
    outputs: jax.Array
    neurons: int = struct.field(pytree_node=False)
    inputs: int = struct.field(pytree_node=False)

    @property
    def wiring(self) -> Sparse:
        assert isinstance(self.cell.wiring, Sparse), "an RNeuralNet is wired sparsely"
        return self.cell.wiring

    @property
    def feeder(self) -> int:
        """The reward feeder's unit, after the internal and the input neurons."""
        return self.neurons + self.inputs

    @classmethod
    def wire(cls, threshold: np.ndarray, pre: np.ndarray, post: np.ndarray, weight: np.ndarray,
             myelin: np.ndarray, *, inputs: int, outputs: np.ndarray) -> RNeuralNet:
        """The network of the connections `pre[e] -> post[e]`, in the order they were made.

        Units count the `len(threshold)` internal neurons, then `inputs`
        input neurons, then the feeder. Each connection has its weight and
        the original's `Myelin`; its delay follows from its place in the
        pass (see the module). A neuron sends once a tick, so its outgoing
        connections must be made one after another: `NeuralNet_init` makes
        an output neuron's connection to the feeder after every other, which
        in the pass splits its sending in two, and here it is made with the
        neuron's others. No connection may enter an input neuron, leave the
        feeder, or return to its source.
        """
        neurons = len(threshold)
        size = neurons + inputs + 1
        pre, post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
        if np.any(pre == post) or np.any((post >= neurons) & (post < size - 1)) or np.any(pre == size - 1):
            raise ValueError("connections run between distinct units, into no input neuron and out of "
                             "no feeder")
        made = np.arange(len(pre))
        first = np.full(size, len(pre))
        np.minimum.at(first, pre, made)
        if np.any(first[pre] + np.bincount(pre, minlength=size)[pre] <= made):
            raise ValueError("each unit's outgoing connections must be made one after another, so it "
                             "sends once a tick")
        # A message reaches its target's output in the pass it arrives if the target sends later.
        delay = np.asarray(myelin, np.int64) + 1 + (made > first[post])
        thresholds = np.concatenate([np.asarray(threshold, np.float32), np.full(inputs, -np.inf, np.float32),
                                     np.zeros(1, np.float32)])
        wiring = Sparse(jnp.asarray(pre, jnp.int32), jnp.asarray(post, jnp.int32),
                        jnp.asarray(weight, jnp.float32), size, jnp.asarray(delay, jnp.int32),
                        int(delay.max(initial=1)))
        cell = RecurrentCell(PulseCell(jnp.asarray(thresholds)), wiring)
        return cls(cell, jnp.asarray(outputs, jnp.int32), neurons, inputs)

    @classmethod
    def random(cls, seed: int, neurons: int, inputs: int, outputs: int, *, fan: int = 16,
               input_fan: int = 5, myelin: int | None = None) -> RNeuralNet:
        """A network drawn as `NeuralNet_init` draws it, from a numpy generator seeded with `seed`.

        Thresholds are normal around 2 with deviation 0.5. Each neuron
        connects to `fan` others (`MAX_AXIONS`), none receiving more than
        `fan` (`MAX_DENDRITES`), with a normal weight of deviation
        `1 / sqrt(fan)` and a `Myelin` from 0 to 19; one that finds no free
        target makes fewer. Each input neuron connects to `input_fan`
        neurons with weight 1 and `Myelin` 1, and `outputs` neurons connect
        to the feeder with weight 1 and `Myelin` 4, no neuron taking two of
        these. The original wires all but its last neuron, which an
        off-by-one leaves unconnected; here every neuron is wired. `myelin`
        gives every connection between neurons that `Myelin` in place of
        the draw, one delay throughout but for the pass order's tick.
        """
        special = inputs * input_fan + outputs
        if special > neurons:
            raise ValueError(f"{inputs} inputs of {input_fan} and {outputs} outputs need {special} distinct "
                             f"neurons, more than {neurons}")
        rng = np.random.default_rng(seed)
        threshold = rng.normal(2.0, 0.5, neurons)
        received = np.zeros(neurons, np.int64)
        taken = rng.permutation(neurons)[:special]
        fed = taken[:inputs * input_fan].reshape(inputs, input_fan)
        chosen = np.sort(taken[inputs * input_fan:])
        feeder = neurons + inputs
        connections: list[tuple[int, int, float, int]] = []
        for source in range(neurons):
            targets: list[int] = []
            for _ in range(fan):
                free = np.flatnonzero(received < fan)
                free = free[(free != source) & ~np.isin(free, targets)]
                if not len(free):
                    break
                targets.append(int(rng.choice(free)))
                received[targets[-1]] += 1
            weights = rng.normal(0.0, 1 / np.sqrt(fan), len(targets))
            myelins = rng.integers(0, MYELIN, len(targets)) if myelin is None else [myelin] * len(targets)
            connections += zip([source] * len(targets), targets, weights, myelins, strict=True)
            if source in chosen:
                connections.append((source, feeder, 1.0, 4))
        connections += [(neurons + i, int(j), 1.0, 1) for i in range(inputs) for j in fed[i]]
        pre, post, weight, drawn = (np.asarray(c) for c in zip(*connections, strict=True))
        return cls.wire(threshold, pre, post, weight, drawn, inputs=inputs, outputs=chosen)

    def drive(self, x: jax.Array) -> jax.Array:
        """The layer's input `[..., size]` for the input neurons' values `x` `[..., inputs]`."""
        zeros = jnp.zeros((*x.shape[:-1], self.neurons), x.dtype)
        return jnp.concatenate([zeros, x, jnp.zeros_like(x[..., :1])], axis=-1)

    def run(self, x: jax.Array, state: RecurrentState[MembraneState, None] | None = None
            ) -> tuple[jax.Array, RecurrentState[MembraneState, None]]:
        """Each unit's output each tick `[T, ..., size]` over input values `x` `[T, ..., inputs]`, and the
        final state; `state` None starts with nothing on its way."""
        out, state = run(self.cell, self.drive(x), state)
        return out.value, state

    def learn(self, activity: jax.Array, reward: jax.Array | float, *, paths: Paths = "first",
              discount: float = 1.0, eta: float = 0.01) -> tuple[RNeuralNet, Diffusion]:
        """The network after `reward` spreads from the feeder over the outputs `activity` `[size]` of the
        tick it came in (`reward_diffusion`), and what it spread to."""
        wiring = self.wiring
        diffusion = reward_diffusion(wiring, activity, reward, root=self.feeder, paths=paths,
                                     discount=discount, eta=eta)
        weight = wiring.weight + diffusion.change.astype(wiring.weight.dtype)
        wiring = dataclasses.replace(wiring, weight=weight)
        return dataclasses.replace(self, cell=dataclasses.replace(self.cell, wiring=wiring)), diffusion


class Diffusion(NamedTuple):
    """What a reward spread to: each connection's weight change, each unit's local reward and each
    connection's share of its target's."""

    change: jax.Array
    credit: jax.Array
    share: jax.Array


def reward_shares(wiring: Sparse, activity: jax.Array) -> jax.Array:
    """Each connection's share of its target's reward, `[E]`: a softmax of `|activity|` at the sources
    of the target's incoming connections.

    `activity` is each unit's last output, `[size]`. The exponentials are
    taken less each target's largest, so they stay finite where the
    original's float exponentials overflow (an activity past 88); elsewhere
    the shares are the original's up to rounding.
    """
    a = jnp.abs(activity)[wiring.pre]
    largest = jax.ops.segment_max(a, wiring.post, wiring.size)
    e = jnp.exp(a - largest[wiring.post])
    return e / jax.ops.segment_sum(e, wiring.post, wiring.size)[wiring.post]


def reward_diffusion(wiring: Sparse, activity: jax.Array, reward: jax.Array | float, *, root: int,
                     paths: Paths = "first", discount: float = 1.0, eta: float = 0.01) -> Diffusion:
    """Spread `reward` from the unit `root` backward along `wiring`, and each connection's weight change.

    For one example: `activity` is each unit's last output, `[size]`, and
    `reward` a scalar; `jax.vmap` it over a batch. `paths="first"` is the
    original's depth-first spread, where each unit passes on the reward it
    holds when the spread first reaches it, through its incoming
    connections in their order in the wiring; a unit that holds no reward
    then, or has no inputs, passes nothing and may be reached again.
    `paths="all"` passes on every path's reward, which needs a `discount`
    below 1 to converge on a cycle. Either way each connection passes
    `discount` times its share (1 in the original), and a connection whose
    target holds no reward keeps its weight.
    """
    if paths == "all" and not 0 <= discount < 1:
        raise ValueError(f"every path's reward converges for a discount in [0, 1), not {discount}")
    if paths not in ("first", "all"):
        raise ValueError(f"paths is first or all, not {paths!r}")
    share = reward_shares(wiring, activity)
    reward = jnp.asarray(reward, share.dtype)
    spread = _first_visits if paths == "first" else _all_paths
    credit = spread(wiring, discount * share, reward, root)
    target = credit[wiring.post]
    change = jnp.where(target != 0, eta * target * share, 0.0)
    return Diffusion(change, credit, share)


def _first_visits(wiring: Sparse, passed: jax.Array, reward: jax.Array, root: int) -> jax.Array:
    """`Global_RewardSpreader`'s recursion, iterated over an explicit stack.

    A frame is a unit, how far it has got through its incoming connections
    (first handing each source its share, then entering each source in
    turn) and the reward it was entered with. One iteration takes one
    connection, or leaves a unit that has taken all of them twice.
    """
    size = wiring.size
    order = jnp.argsort(wiring.post, stable=True)
    source, passed = wiring.pre[order], passed[order]
    targets = wiring.post[order]
    units = jnp.arange(size)
    start = jnp.searchsorted(targets, units, side="left")
    inputs = jnp.searchsorted(targets, units, side="right") - start
    credit = jnp.zeros(size, passed.dtype).at[root].set(reward)
    if not len(order):
        return credit
    last = len(order) - 1
    entered = (reward != 0) & (inputs[root] > 0)
    stack = jnp.zeros(size + 1, jnp.int32).at[0].set(root)
    taken = jnp.zeros(size + 1, jnp.int32)
    held = jnp.zeros(size + 1, passed.dtype).at[0].set(reward)
    visited = jnp.zeros(size, bool).at[root].set(entered)
    depth = entered.astype(jnp.int32)

    def busy(carry: tuple[jax.Array, ...]) -> jax.Array:
        return carry[-1] > 0

    def take(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        credit, stack, taken, held, visited, depth = carry
        top = depth - 1
        unit, k = stack[top], taken[top]
        n = inputs[unit]
        sharing, entering = k < n, (k >= n) & (k < 2 * n)
        e = jnp.clip(start[unit] + jnp.where(sharing, k, k - n), 0, last)
        child = source[e]
        credit = credit.at[child].add(jnp.where(sharing, held[top] * passed[e], 0.0))
        enter = entering & ~visited[child] & (inputs[child] > 0) & (credit[child] != 0)
        taken = taken.at[top].set(k + 1)
        stack = stack.at[depth].set(jnp.where(enter, child, stack[depth]))
        taken = taken.at[depth].set(jnp.where(enter, 0, taken[depth]))
        held = held.at[depth].set(jnp.where(enter, credit[child], held[depth]))
        visited = visited.at[child].set(visited[child] | enter)
        depth = depth + enter.astype(jnp.int32) - (k >= 2 * n).astype(jnp.int32)
        return credit, stack, taken, held, visited, depth

    return jax.lax.while_loop(busy, take, (credit, stack, taken, held, visited, depth))[0]


def _all_paths(wiring: Sparse, passed: jax.Array, reward: jax.Array, root: int) -> jax.Array:
    """`c = r + P c` by its series, one path length a term, until a term's total falls below rounding.

    Each target passes on at most `discount` of what it holds, so a term
    is at most `discount` times the one before.
    """
    r = jnp.zeros(wiring.size, passed.dtype).at[root].set(reward)
    small = jnp.finfo(passed.dtype).eps * jnp.abs(reward)

    def busy(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
        return jnp.sum(jnp.abs(carry[1])) > small

    def lengthen(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        credit, term = carry
        term = jax.ops.segment_sum(passed * term[wiring.post], wiring.pre, wiring.size)
        return credit + term, term

    return jax.lax.while_loop(busy, lengthen, (r, r))[0]
