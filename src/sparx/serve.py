"""Serving streaming spiking models: many sessions, each with its own neuron state, in one batched program.

    server = StreamServer(model, params, slots=8, frame=10, sample_shape=(40,))
    session = server.open()
    future = server.submit(session, chunk)     # [10, 40]: one frame of this session's stream
    server.run()                               # one batched step per round of pending frames
    outputs = future.result()                  # [10, ...]

A spiking model that streams (the `state` collection, README "Streaming")
carries its neurons' state from one call to the next. A server keeps that
state for `slots` sessions as the rows of one resident batch: each step
runs one frame of every session that has one waiting, in one jitted call,
and a session without a frame keeps its state exactly (its row is not
advanced, not advanced on zeros). A session's outputs over its frames are
then the outputs of one call over its whole stream. Opening a session
starts its row at rest; closing it frees the row.

This is dew's slot-scheduling `Server` for text (a resident KV cache
refilled from a queue) applied to neuron state; dew's own server is
specific to token generation (AshishKumar4/dew#30).
"""

from __future__ import annotations

import collections
from concurrent.futures import Future
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["StreamServer"]


class StreamServer:
    """Sessions of a streaming model, `slots` at a time, advanced a `frame` of steps per round."""

    def __init__(self, model: nn.Module, params: Any, *, slots: int, frame: int,
                 sample_shape: tuple[int, ...], dtype: Any = jnp.float32):
        self.model, self.params, self.slots, self.frame = model, dict(params), slots, frame
        self.sample_shape, self.dtype = tuple(sample_shape), dtype
        empty = jnp.zeros((0, slots, *self.sample_shape), dtype)
        _, rest = model.apply(self.params, empty, mutable=["state"])  # zero steps: every row at rest
        self.rest = rest
        self.state = rest
        self.free = list(range(slots))
        self.sessions: dict[int, int] = {}
        self.pending: dict[int, collections.deque] = {}
        self._next = 0
        self._step = jax.jit(self._advance)

    def _advance(self, state, frames, active):
        outputs, updated = self.model.apply({**self.params, **state}, frames, mutable=["state"])

        def keep(new, old):
            mask = active.reshape((self.slots,) + (1,) * (new.ndim - 1))
            return jnp.where(mask, new, old)

        return outputs, jax.tree.map(keep, updated, state)

    def open(self) -> int:
        """Start a session at rest in a free slot; returns its id."""
        if not self.free:
            raise RuntimeError(f"all {self.slots} slots hold sessions; close one first")
        slot = self.free.pop(0)
        session, self._next = self._next, self._next + 1
        self.sessions[session] = slot
        self.pending[session] = collections.deque()
        self.state = jax.tree.map(lambda leaf, rest: leaf.at[slot].set(rest[slot]), self.state, self.rest)
        return session

    def close(self, session: int) -> None:
        """End a session; its slot takes the next one. Frames still waiting are cancelled."""
        for _, future in self.pending.pop(session):
            future.cancel()
        self.free.append(self.sessions.pop(session))

    def submit(self, session: int, frame: Any) -> Future:
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
        outputs, self.state = self._step(self.state, jnp.asarray(frames), jnp.asarray(active))
        outputs = np.asarray(outputs)
        for session, (_, future) in ready.items():
            future.set_result(outputs[:, self.sessions[session]])
        return len(ready)

    def run(self) -> None:
        """Step until no frame waits."""
        while self.step():
            pass
