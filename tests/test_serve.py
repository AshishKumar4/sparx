"""A served session's outputs are a direct streaming call's."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparx.nn import LI, LIF
from sparx.serve import StreamServer


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        return LI(tau=4.0)(nn.Dense(3)(LIF(tau=3.0)(nn.Dense(16)(x))))


def test_interleaved_sessions_equal_their_own_streams():
    rng = np.random.default_rng(0)
    frame, features = 5, 6
    model = Net()
    with jax.enable_x64(new_val=True):
        params = model.init(jax.random.key(0), jnp.zeros((1, 1, features)))
        params = jax.tree.map(lambda w: 2.0 * w, params)
        lengths = {"a": 4, "b": 2, "c": 3}
        streams = {name: rng.random((frames * frame, features)) for name, frames in lengths.items()}
        server = StreamServer(model, params, slots=2, frame=frame, sample_shape=(features,),
                              dtype=jnp.float64)
        a, b = server.open(), server.open()
        futures = {"a": [], "b": [], "c": []}
        # Sessions arrive and leave at different times; c takes b's slot.
        for k in range(2):
            futures["a"].append(server.submit(a, streams["a"][k * frame:(k + 1) * frame]))
        futures["b"].append(server.submit(b, streams["b"][:frame]))
        server.run()
        futures["b"].append(server.submit(b, streams["b"][frame:]))
        server.step()  # b alone: a's row must not move
        server.close(b)
        c = server.open()
        for k in range(3):
            futures["c"].append(server.submit(c, streams["c"][k * frame:(k + 1) * frame]))
        for k in range(2, 4):
            futures["a"].append(server.submit(a, streams["a"][k * frame:(k + 1) * frame]))
        server.run()
        for name, parts in futures.items():
            served = np.concatenate([f.result() for f in parts])
            direct = np.asarray(model.apply(params, jnp.asarray(streams[name])[:, None]))[:, 0]
            np.testing.assert_allclose(served, direct, rtol=1e-12, atol=1e-12)
            assert np.abs(direct).max() > 0.1


def test_the_server_refuses_more_sessions_than_slots_and_bad_frames():
    model = Net()
    params = model.init(jax.random.key(0), jnp.zeros((1, 1, 4)))
    server = StreamServer(model, params, slots=1, frame=2, sample_shape=(4,))
    session = server.open()
    with pytest.raises(RuntimeError, match="slots"):
        server.open()
    with pytest.raises(ValueError, match="frame"):
        server.submit(session, np.zeros((3, 4)))
