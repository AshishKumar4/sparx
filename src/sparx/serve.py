"""Serving streaming spiking models: many sessions, each with its own neuron state, in one batched program.

    server = StreamServer(model, variables, slots=8, frame=10, sample_shape=(40,))
    session = server.open()
    future = server.submit(session, chunk)     # [10, 40]: one frame of this session's stream
    server.run()                               # one batched step per round of pending frames
    outputs = future.result()                  # [10, ...]

A run that `dew.pipeline` reloads serves as `StreamServer(classifier.model,
classifier.variables, ...)`; its frames are the encoder's output, time-major.

A spiking model that streams (the `state` collection, the guide's "Streaming")
carries its neurons' state from one call to the next. A server keeps that
state for `slots` sessions as the rows of one resident batch: each step
runs one frame of every session that has one waiting, in one jitted call,
and a session without a frame keeps its state exactly (its row is not
advanced, not advanced on zeros). A session's outputs over its frames are
then the outputs of one call over its whole stream. Closing a session
frees its row.

A session's first frame runs from rest, the state every layer starts from
when the `state` collection holds nothing. The server never builds a rest
state itself: a step whose rows include a session's first frame also runs
the model without carried state and takes those rows from that run. Any
model whose layers follow the `state` convention serves this way, whatever
it does between them, and only a step that starts a session pays for a
second pass.

This is dew's slot-scheduling `Server` for text (a resident KV cache
refilled from a queue) applied to neuron state; dew's own server is
specific to token generation (AshishKumar4/dew#30).
"""

from __future__ import annotations

import collections
import functools
from concurrent.futures import Future

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from dew.objectives.base import Variables

from sparx.nn import STATE

__all__ = ["StreamServer"]


class StreamServer:
    """Sessions of a streaming model, `slots` at a time, advanced a `frame` of steps per round.

    `variables` holds every collection the model reads (`params`, and
    `batch_stats` for a model with batch norm); a `state` collection in it
    is ignored, since each session starts at rest. Construction runs the
    model once on abstract inputs, so a model that cannot stream fails here
    with the reason.
    """

    def __init__(self, model: nn.Module, variables: Variables, *, slots: int, frame: int,
                 sample_shape: tuple[int, ...], dtype: jnp.dtype | type = jnp.float32):
        self.model, self.slots, self.frame = model, slots, frame
        self.variables = {name: tree for name, tree in variables.items() if name != STATE}
        self.sample_shape, self.dtype = tuple(sample_shape), dtype
        self.axes = self._check()  # each state leaf's batch axis
        self.state: Variables | None = None  # until the first step, no row has run
        self.free = list(range(slots))
        self.sessions: dict[int, int] = {}
        self.fresh: set[int] = set()  # slots whose session has not run a frame yet
        self.pending: dict[int, collections.deque[tuple[np.ndarray, Future]]] = {}
        self._next = 0
        self._step = jax.jit(self._advance, static_argnames=("carry", "start"))

    def _check(self) -> Variables:
        """Each state leaf's batch axis, the one that grows with the batch; refuses a model that cannot serve.

        A neuron's state is `[B, ...]` and a history window's `[steps, B,
        ...]`, so the axis is found by running the model abstractly at two
        batch sizes.
        """
        name = type(self.model).__name__
        shapes = []
        for batch in (self.slots, self.slots + 1):
            frames = jax.ShapeDtypeStruct((self.frame, batch, *self.sample_shape), self.dtype)
            try:
                shapes.append(jax.eval_shape(functools.partial(self.model.apply, mutable=[STATE]),
                                             self.variables, frames))
            except Exception as error:
                raise ValueError(f"{name} cannot be served: one frame {frames.shape} with the 'state' "
                                 f"collection mutable raised {type(error).__name__}: {error}") from error
        (outputs, state), (_, wider) = shapes
        for leaf in jax.tree.leaves(outputs):
            if leaf.shape[:2] != (self.frame, self.slots):
                raise ValueError(f"{name} cannot be served: a served model returns time-major outputs "
                                 f"[frame, slots, ...] = [{self.frame}, {self.slots}, ...], and this one "
                                 f"returned {leaf.shape}")

        def batch_axis(leaf: jax.ShapeDtypeStruct, grown: jax.ShapeDtypeStruct) -> int:
            axes = [axis for axis, (a, b) in enumerate(zip(leaf.shape, grown.shape, strict=True)) if a != b]
            if len(axes) != 1:
                raise ValueError(f"{name} cannot be served: a state leaf of shape {leaf.shape} has no single "
                                 "batch axis, so sessions cannot hold rows of it")
            return axes[0]

        return jax.tree.map(batch_axis, dict(state), dict(wider))

    def _advance(self, state: Variables | None, frames: jax.Array, active: jax.Array, fresh: jax.Array,
                 *, carry: bool, start: bool) -> tuple[jax.Array, Variables]:
        """One frame of every row: rows that continue run from `state`, `fresh` rows from rest.

        `carry` says some running row continues and `start` that some
        begins, so a step runs the model once unless it has both. Rows not
        `active` keep their state.
        """
        def apply(variables: Variables) -> tuple[jax.Array, Variables]:
            return self.model.apply(variables, frames, mutable=[STATE])

        if state is None:  # no row has run, so every row that runs begins here
            return apply(self.variables)
        if not carry:
            outputs, updated = apply(self.variables)
        else:
            outputs, updated = apply({**self.variables, **state})
            if start:
                begun_outputs, begun = apply(self.variables)
                outputs = _rows(fresh, begun_outputs, outputs, 1)
                updated = _rows(fresh, begun, updated, self.axes)
        return outputs, _rows(active, updated, state, self.axes)

    def open(self) -> int:
        """Start a session at rest in a free slot; returns its id."""
        if not self.free:
            raise RuntimeError(f"all {self.slots} slots hold sessions; close one first")
        slot = self.free.pop(0)
        session, self._next = self._next, self._next + 1
        self.sessions[session] = slot
        self.fresh.add(slot)
        self.pending[session] = collections.deque()
        return session

    def close(self, session: int) -> None:
        """End a session; its slot takes the next one. Frames still waiting are cancelled."""
        for _, future in self.pending.pop(session):
            future.cancel()
        slot = self.sessions.pop(session)
        self.fresh.discard(slot)
        self.free.append(slot)

    def submit(self, session: int, frame: np.ndarray | jax.Array) -> Future:
        """Queue one frame `[frame, *sample_shape]` of `session`'s stream; the future holds its outputs."""
        frame = np.asarray(frame, self.dtype)
        if frame.shape != (self.frame, *self.sample_shape):
            raise ValueError(f"a frame is {(self.frame, *self.sample_shape)}, got {frame.shape}")
        future: Future = Future()
        self.pending[session].append((frame, future))
        return future

    def step(self) -> int:
        """Run the oldest waiting frame of every session that has one; returns how many ran."""
        ready = {session: queue.popleft() for session, queue in self.pending.items() if queue}
        if not ready:
            return 0
        frames = np.zeros((self.frame, self.slots, *self.sample_shape), self.dtype)
        active = np.zeros(self.slots, bool)
        for session, (frame, _) in ready.items():
            frames[:, self.sessions[session]] = frame
            active[self.sessions[session]] = True
        starting = self.fresh & {self.sessions[session] for session in ready}
        fresh = np.isin(np.arange(self.slots), list(starting))
        outputs, self.state = self._step(self.state, jnp.asarray(frames), jnp.asarray(active),
                                         jnp.asarray(fresh), carry=len(starting) < len(ready),
                                         start=bool(starting))
        self.fresh -= starting
        outputs = np.asarray(outputs)
        for session, (_, future) in ready.items():
            future.set_result(outputs[:, self.sessions[session]])
        return len(ready)

    def run(self) -> None:
        """Step until no frame waits."""
        while self.step():
            pass


def _rows[Tree](mask: jax.Array, chosen: Tree, other: Tree, axes: Variables | int) -> Tree:
    """`chosen`'s rows where `mask` holds and `other`'s elsewhere, each leaf's rows on its axis in `axes`."""
    def pick(axis: int, a: jax.Array, b: jax.Array) -> jax.Array:
        shape = [1] * a.ndim
        shape[axis] = mask.shape[0]
        return jnp.where(mask.reshape(shape), a, b)

    if isinstance(axes, int):
        return jax.tree.map(functools.partial(pick, axes), chosen, other)
    return jax.tree.map(pick, axes, chosen, other)
