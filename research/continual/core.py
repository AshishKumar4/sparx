"""The research notes' recurrent core: modules of leaky tanh units joined by delayed connections, a few of
them fast-plastic.

    h_j(t) = a_j h_j(t-1) + (1 - a_j) tanh(z_j(t))
    z_j(t) = b_j + [U x_t]_j + sum_i (w_ji + alpha_ji q_ji(t)) h_i(t - d_ji)

The notes (chapter 6, section 3A, and their table of initial choices) start
with 16 modules of 256 units, dense within each module, small interfaces
between modules, delays from a small fixed set such as 1, 2, 4 and 8 steps,
time constants that differ by module, and fast plasticity on 16 incoming
connections per unit, `alpha_ji q_ji` added to only those weights. Here the
connections within a module take one step and those between modules a delay
drawn from `delays`; each unit's plastic connections are drawn from all its
inputs; `q` is a `sparx.nn` Hebbian trace (`ModulatedTrace` is the notes'
signed, modulated rule, `RetroactiveTrace` its form with an eligibility
trace). The input projection `U` and the readout sit outside, as in `sparx.nn`.

Everything is `sparx.dynamics` and `sparx.nn`: a `RecurrentCell` of a
`RateCell` over a `Sparse` wiring with per-edge delays, and `FastWeights` on
the picked connections, which keep traces for those alone.
"""

import functools
import math
from collections.abc import Sequence
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from sparx.dynamics import FastWeights, RateCell, RecurrentCell, Sparse
from sparx.nn import HebbianTrace, Neuron, adopt


class Structure(NamedTuple):
    """A core's connections: sources, targets and delays per edge, and the indices of the plastic ones."""

    pre: np.ndarray
    post: np.ndarray
    delay: np.ndarray
    plastic: np.ndarray


@functools.lru_cache(maxsize=8)
def structure(modules: int, units: int, interface: int, delays: tuple[int, ...], plastic: int,
              seed: int) -> Structure:
    """Every pair within a module, one step apart; `interface` inputs per unit from units of the other
    modules, each with a delay drawn from `delays`; and `plastic` of each unit's inputs picked to adapt."""
    rng = np.random.default_rng(seed)
    size = modules * units
    local = np.arange(units)
    module = np.repeat(np.arange(modules), units * units)
    local_pre = module * units + np.tile(np.repeat(local, units), modules)
    local_post = module * units + np.tile(np.tile(local, units), modules)
    targets = np.repeat(np.arange(size), interface)
    if modules > 1:
        other = rng.integers(0, modules - 1, len(targets))
        other += other >= targets // units  # any module but the target's own
        sources = other * units + rng.integers(0, units, len(targets))
        far_delay = rng.choice(np.asarray(delays), len(targets))
    else:
        targets = sources = far_delay = np.zeros(0, np.int64)
    pre = np.concatenate([local_pre, sources]).astype(np.int32)
    post = np.concatenate([local_post, targets]).astype(np.int32)
    delay = np.concatenate([np.ones(len(local_pre), np.int64), far_delay]).astype(np.int32)
    # Each unit's inputs, in rows, and `plastic` of each row picked at random.
    rows = np.argsort(post, kind="stable").reshape(size, -1)
    picked = rng.random(rows.shape).argsort(axis=1)[:, :plastic]
    chosen = np.sort(np.take_along_axis(rows, picked, axis=1).ravel()).astype(np.int32)
    return Structure(pre, post, delay, chosen)


class ModularCore(Neuron):
    """The notes' core as a `sparx.nn` layer over input currents `[T, ..., modules * units]`.

    Module `m`'s units have the time constant `taus[m % len(taus)]` in
    steps; recurrent weights start normal with standard deviation `gain`
    over the square root of a unit's inputs; with a `rule`, each unit's
    `plastic` picked inputs carry fast weights, `alpha` per connection
    starting at `alpha_init`. `seed` draws the structure, which is not a
    parameter, so a run rebuilds it from its record.
    """

    modules: int = 16
    units: int = 256
    interface: int = 16
    delays: Sequence[int] = (1, 2, 4, 8)
    taus: Sequence[float] = (1.0, 2.0, 4.0, 8.0)
    plastic: int = 16
    rule: HebbianTrace | None = None
    alpha_init: nn.initializers.Initializer = nn.initializers.normal(0.01)
    gain: float = 1.0
    seed: int = 0

    @property
    def size(self) -> int:
        return self.modules * self.units

    def build(self, x: jax.Array) -> RecurrentCell:
        s = structure(self.modules, self.units, self.interface, tuple(self.delays), self.plastic, self.seed)
        fan_in = self.units + (self.interface if self.modules > 1 else 0)
        weight = self.param("weight", nn.initializers.normal(self.gain / math.sqrt(fan_in)), (len(s.pre),),
                            jnp.float32)
        bias = self.param("bias", nn.initializers.zeros_init(), (self.size,), jnp.float32)
        tau = np.repeat(np.resize(np.asarray(self.taus, np.float64), self.modules), self.units)
        decay = jnp.asarray(np.exp(-1.0 / tau), jnp.float32)
        wiring = Sparse(jnp.asarray(s.pre), jnp.asarray(s.post), weight, self.size, jnp.asarray(s.delay),
                        max(self.delays))
        fast = None
        if self.rule is not None and self.plastic:
            alpha = self.param("alpha", self.alpha_init, (len(s.plastic),), jnp.float32)
            fast = FastWeights(alpha, adopt(self.rule, self, "rule")(self.size), jnp.asarray(s.plastic))
        return RecurrentCell(RateCell(decay, bias), wiring, fast)
