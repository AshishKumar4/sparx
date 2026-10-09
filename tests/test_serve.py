"""A served session's outputs are a direct streaming call's."""

import re
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sparx
from sparx.models import SEWResNet, SpikingMLP
from sparx.nn import ALIF, LI, LIF, PSN
from sparx.serve import StreamServer
from sparx.tasks import bound_call

GUIDE = Path(__file__).parents[1] / "docs" / "guide.md"


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        return LI(tau=4.0)(nn.Dense(3)(LIF(tau=3.0)(nn.Dense(16)(x))))


def test_interleaved_sessions_equal_their_own_streams():
    rng = np.random.default_rng(0)
    frame, features = 5, 6
    model = Net()
    with jax.enable_x64(new_val=True):
        variables = model.init(jax.random.key(0), jnp.zeros((1, 1, features)))
        variables = jax.tree.map(lambda w: 2.0 * w, variables)
        lengths = {"a": 4, "b": 2, "c": 3}
        streams = {name: rng.random((frames * frame, features)) for name, frames in lengths.items()}
        server = StreamServer(model, variables, slots=2, frame=frame, sample_shape=(features,),
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
            direct = np.asarray(model.apply(variables, jnp.asarray(streams[name])[:, None]))[:, 0]
            np.testing.assert_allclose(served, direct, rtol=1e-12, atol=1e-12)  # observed 0
            assert np.abs(direct).max() > 0.1


def test_the_server_refuses_more_sessions_than_slots_and_bad_frames():
    model = Net()
    variables = model.init(jax.random.key(0), jnp.zeros((1, 1, 4)))
    server = StreamServer(model, variables, slots=1, frame=2, sample_shape=(4,))
    session = server.open()
    with pytest.raises(RuntimeError, match="slots"):
        server.open()
    with pytest.raises(ValueError, match="frame"):
        server.submit(session, np.zeros((3, 4)))


def _guide_convnet():
    blocks = re.findall(r"```python\n(.*?)\n```", GUIDE.read_text(), re.S)
    (block,) = [b for b in blocks if b.startswith("class ConvNet")]
    scope = {"nn": nn, "sparx": sparx}
    exec(block, scope)
    return scope["ConvNet"]()


# Every model of sparx.models, plus the guide's ConvNet. Each reshapes or
# pools between its layers, and the ConvNet flattens with `reshape(..., -1)`.
SERVED = {
    "mlp": (lambda: SpikingMLP((8,), 3), (2, 3)),
    "recurrent_alif_mlp": (lambda: SpikingMLP((8,), 3, neuron=ALIF(tau=3.0), recurrent=True), (6,)),
    "delayed_mlp": (lambda: SpikingMLP((8,), 3, delays=3), (6,)),
    "sew_resnet": (lambda: SEWResNet((1, 1, 1, 1), 3, width=4, stem="small"), (6, 6, 1)),
    "guide_convnet": (_guide_convnet, (6, 6, 1)),
}


@pytest.mark.parametrize("name", list(SERVED))
def test_every_model_serves_its_direct_call(name):
    # SEW ResNet takes `train`, which the server binds to False as a trained classifier does.
    build, sample = SERVED[name]
    model, frame = build(), 2
    method = bound_call(model, train=False, kwargs={})
    rng = np.random.default_rng(1)
    with jax.enable_x64(new_val=True):
        variables = model.init(jax.random.key(0), jnp.zeros((1, 1, *sample)), method=method)
        variables = jax.tree.map(lambda w: 3.0 * w, variables)
        streams = {"a": rng.random((3 * frame, *sample)), "b": rng.random((2 * frame, *sample))}
        server = StreamServer(model, variables, slots=3, frame=frame, sample_shape=sample, dtype=jnp.float64)
        # The first step starts a alone, the second starts b while a continues
        # and the third continues both, so each path through a step runs.
        a = server.open()
        futures = {"a": [server.submit(a, streams["a"][:frame])], "b": []}
        server.run()
        b = server.open()
        for k in (1, 2):
            futures["a"].append(server.submit(a, streams["a"][k * frame:(k + 1) * frame]))
            futures["b"].append(server.submit(b, streams["b"][(k - 1) * frame:k * frame]))
            server.run()
        for session, parts in futures.items():
            served = np.concatenate([f.result() for f in parts])
            direct = model.apply(variables, jnp.asarray(streams[session])[:, None], method=method)
            np.testing.assert_allclose(served, np.asarray(direct)[:, 0], rtol=1e-12, atol=1e-12)  # observed 0
            assert np.abs(np.diff(direct, axis=0)).max() > 1e-3  # the outputs move over time


class Scaled(nn.Module):
    """Net with its readout scaled by a keyword a classifier records, which every call needs."""

    @nn.compact
    def __call__(self, x, scale):
        return Net()(x) * scale


def test_the_server_calls_the_model_with_the_classifiers_recorded_arguments():
    model, rng = Scaled(), np.random.default_rng(2)
    # In float64, as the tests above: an A100 runs float32 matrix products in TF32, rounded
    # differently for the server's batch than for the direct call's.
    with jax.enable_x64(new_val=True):
        variables = model.init(jax.random.key(0), jnp.zeros((1, 1, 4)), 1.0)
        server = StreamServer(model, variables, slots=1, frame=3, sample_shape=(4,), call={"scale": 2.5},
                              dtype=jnp.float64)
        session, stream = server.open(), rng.random((6, 4))
        futures = [server.submit(session, stream[:3]), server.submit(session, stream[3:])]
        server.run()
        direct = model.apply(variables, jnp.asarray(stream)[:, None], 2.5)[:, 0]
        served = np.concatenate([f.result() for f in futures])
    np.testing.assert_allclose(served, direct, rtol=1e-12, atol=1e-12)
    assert np.abs(direct).max() > 0.1


def test_a_cancelled_frame_is_dropped_and_every_other_frame_runs():
    # a's first frame is cancelled: a runs its second frame from rest, and b, stepped beside it, gets its
    # outputs.
    model, rng = Net(), np.random.default_rng(3)
    variables = jax.tree.map(lambda w: 2.0 * w, model.init(jax.random.key(0), jnp.zeros((1, 1, 4))))
    server = StreamServer(model, variables, slots=2, frame=3, sample_shape=(4,))
    a, b = server.open(), server.open()
    streams = {name: rng.random((6, 4)).astype(np.float32) for name in "ab"}
    dropped = server.submit(a, streams["a"][:3])
    kept = server.submit(a, streams["a"][3:])
    others = [server.submit(b, streams["b"][:3]), server.submit(b, streams["b"][3:])]
    assert dropped.cancel()
    assert server.step() == 2 and kept.done() and others[0].done() and not others[1].done()
    server.run()
    assert dropped.cancelled()
    direct = {name: np.asarray(model.apply(variables, jnp.asarray(stream)[:, None]))[:, 0]
              for name, stream in (("a", streams["a"][3:]), ("b", streams["b"]))}
    np.testing.assert_allclose(kept.result(), direct["a"], rtol=1e-6)  # observed 0
    np.testing.assert_allclose(np.concatenate([f.result() for f in others]), direct["b"], rtol=1e-6)


def test_a_step_that_raises_fails_its_frames_and_keeps_every_session(monkeypatch):
    model = Net()
    variables = model.init(jax.random.key(0), jnp.zeros((1, 1, 4)))
    server = StreamServer(model, variables, slots=1, frame=2, sample_shape=(4,))
    session = server.open()
    stream = np.random.default_rng(4).random((4, 4)).astype(np.float32)
    first = server.submit(session, stream[:2])
    server.run()
    held = server.state

    def fails(*args, **kwargs):
        raise RuntimeError("the device is gone")

    monkeypatch.setattr(server, "_step", fails)
    failed = server.submit(session, stream[2:])
    with pytest.raises(RuntimeError, match="device is gone"):
        server.step()
    assert isinstance(failed.exception(timeout=0), RuntimeError)
    assert server.state is held and first.done()


class TimeMean(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.mean(LIF()(nn.Dense(4)(x)), axis=0)


@pytest.mark.parametrize(("model", "reason"), [(nn.Sequential([nn.Dense(4), PSN()]), "PSN mixes every step"),
                                               (TimeMean(), "time-major outputs")])
def test_a_model_that_cannot_stream_is_refused_with_its_reason(model, reason):
    variables = model.init(jax.random.key(0), jnp.zeros((3, 1, 4)))
    with pytest.raises(ValueError, match=f"cannot be served.*{reason}"):
        StreamServer(model, variables, slots=2, frame=3, sample_shape=(4,))
