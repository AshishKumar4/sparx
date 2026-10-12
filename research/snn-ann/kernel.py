"""Four Pallas/Triton launches for a batch-one racer step, with optional zero-block skipping.

The launch boundaries are the two convolutional populations: different CTAs need each other's
outputs, and Triton has no grid-wide barrier. Each convolution includes its membrane update. The
64 input splits, each in four CTAs over features, compute the dense drive's partial sums, then one CTA
finalizes its units and readout. Convolutional windows reduce over groups of taps and channels together.
The camera's ON/OFF inputs and LIF outputs are binary, so their products select float32 weights before
the float64 sum. Graded products and every sum use float64, not TF32. Weights and stored state stay
float32 to meet the float64-reference correctness bar.
This is an accuracy-first kernel, not a tensor-core implementation.
"""

from __future__ import annotations

import base64
import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

NAMES = ("conv0", "conv1", "hidden", "readout")
SHAPES = ((16, 32, 16), (8, 16, 32), (128,), (2,))
DENSE_BLOCK = 64
DENSE_SPLITS = 64


class State(NamedTuple):
    conv0: jax.Array
    conv1: jax.Array
    hidden: jax.Array
    readout: jax.Array


class Parameters(NamedTuple):
    conv0_kernel: jax.Array
    conv0_bias: jax.Array
    conv1_kernel: jax.Array
    conv1_bias: jax.Array
    dense_kernel: jax.Array
    dense_bias: jax.Array
    readout_kernel: jax.Array
    readout_bias: jax.Array


@dataclasses.dataclass(frozen=True)
class Config:
    decay: float
    readout_decay: float
    graded: bool = False


def from_model(model: dict) -> tuple[Parameters, Config]:
    """Decode the trained racer's float32 arrays without changing their layout."""
    vision = model["vision"]
    c = vision["config"]
    expected = ([32, 64, 2], [16, 32], [5, 3], 128, 2)
    actual = (c["shape"], c["features"], c["kernels"], c["hidden"], c["outputs"])
    if actual != expected or c["neuron"] not in ("lif", "relu") or c.get("bits", 1) != 1:
        raise ValueError(f"This kernel requires the two one-bit racer arms at their real shapes: {actual}")

    def array(path):
        leaf = vision["params"][path]
        return np.frombuffer(base64.b64decode(leaf["data"]), np.float32).reshape(leaf["shape"]).copy()

    paths = ("Conv_0", "Conv_1", "Dense_0", "Dense_1")
    params = Parameters(*(array(f"{path}/{kind}") for path in paths for kind in ("kernel", "bias")))
    return params, Config(c["decay"], c["readout_decay"], c["neuron"] == "relu")


def at_rest(dtype=jnp.float32) -> State:
    return State(*(jnp.zeros(shape, dtype) for shape in SHAPES))


def _fire(drive, before, config: Config):
    # Carry arithmetic in double too, then round the membrane once for storage.
    gain = 1 - config.decay if config.graded else 1.0
    v = config.decay * before.astype(jnp.float64) + gain * drive
    if config.graded:
        after = v.astype(before.dtype)
        return after, jnp.maximum(after, 0)
    fired = v >= 1.0
    return jnp.where(fired, 0, v).astype(before.dtype), fired.astype(before.dtype)


def _conv_kernel(x_ref, w_ref, b_ref, v_ref, after_ref, out_ref, *, layer, config, skip):
    rows, columns, channels, features, size, pixels, feature_block, taps = (
        (32, 64, 2, 16, 5, 2, 16, 25) if layer == 0 else (16, 32, 16, 32, 3, 1, 32, 4)
    )
    out_columns = columns // 2
    top = (size - 2) // 2  # SAME uses the smaller half of odd padding before.
    p = pl.program_id(0) * pixels + jnp.arange(pixels, dtype=jnp.int32)
    q = jnp.arange(pl.next_power_of_2(taps * channels), dtype=jnp.int32)
    c = lax.rem(q, np.int32(channels))
    f = pl.program_id(1) * feature_block + jnp.arange(feature_block, dtype=jnp.int32)
    r0 = lax.div(p, np.int32(out_columns)) * 2 - top
    c0 = lax.rem(p, np.int32(out_columns)) * 2 - top

    def group(k, acc):
        tap = k.astype(jnp.int32) * taps + lax.div(q, np.int32(channels))
        r = r0[:, None] + lax.div(tap, np.int32(size))[None, :]
        col = c0[:, None] + lax.rem(tap, np.int32(size))[None, :]
        valid_tap = (q < taps * channels) & (tap < size * size)
        valid = valid_tap[None, :] & (r >= 0) & (r < rows) & (col >= 0) & (col < columns)
        if layer == 0:
            address = c[None, :] * rows * columns + r * columns + col
        else:
            address = (r * columns + col) * channels + c[None, :]
        x = plt.load(x_ref.at[address], mask=valid, other=0)

        def compute(a):
            w = plt.load(w_ref.at[tap[:, None], c[:, None], f[None, :]],
                         mask=valid_tap[:, None], other=0)
            if layer == 0 or not config.graded:
                # The camera's events and LIF outputs are binary. Selecting the float32 weight is
                # exactly their product, before the same float64 reduction.
                product = jnp.where(x[:, :, None] != 0, w[None, :, :], 0).astype(jnp.float64)
            else:
                product = x.astype(jnp.float64)[:, :, None] * w.astype(jnp.float64)[None, :, :]
            return a + jnp.sum(product, axis=1)

        return lax.cond(jnp.max(jnp.abs(x)) != 0, compute, lambda a: a, acc) if skip else compute(acc)

    # Four-tap groups avoid padding the second convolution's 144 terms to 256 and keep its live tile
    # smaller. The first convolution takes its complete window in one reduction.
    drive = lax.fori_loop(np.int32(0), np.int32((size * size + taps - 1) // taps), group,
                         jnp.zeros((pixels, feature_block), jnp.float64))
    drive = drive + plt.load(b_ref.at[f]).astype(jnp.float64)[None, :]
    ix = p[:, None] * features + f[None, :]
    before = plt.load(v_ref.at[ix])
    after, out = _fire(drive, before, config)
    plt.store(after_ref.at[ix], after)
    plt.store(out_ref.at[ix], out)


def _dense_partial_kernel(x_ref, w_ref, partial_ref, *, skip, binary):
    f = pl.program_id(1) * 32 + jnp.arange(32, dtype=jnp.int32)
    c = jnp.arange(DENSE_BLOCK, dtype=jnp.int32)
    pid = pl.program_id(0)
    row = pid * DENSE_BLOCK + c
    x = plt.load(x_ref.at[row])

    def compute():
        w = plt.load(w_ref.at[row[:, None], f[None, :]])
        product = (jnp.where(x[:, None] != 0, w, 0).astype(jnp.float64) if binary
                   else x.astype(jnp.float64)[:, None] * w.astype(jnp.float64))
        return jnp.sum(product, axis=0)

    drive = (lax.cond(jnp.max(jnp.abs(x)) != 0, compute, lambda: jnp.zeros((32,), jnp.float64))
             if skip else compute())
    plt.store(partial_ref.at[pid, f], drive)


def _dense_finalize_kernel(partial_ref, input_ref, w_ref, b_ref, wr_ref, br_ref, v_ref, r_ref,
                           after_ref, readout_ref, out_ref, *, config, skip):
    f = jnp.arange(128, dtype=jnp.int32)
    split = jnp.arange(DENSE_SPLITS, dtype=jnp.int32)
    partial = plt.load(partial_ref.at[split[:, None], f[None, :]])
    drive = jnp.sum(partial, axis=0)
    # Strongly typed scalar indices also work in Pallas's CPU state-discharge interpreter.
    speed_row = jnp.asarray(4096, jnp.int32)
    speed = plt.load(input_ref.at[speed_row]).astype(jnp.float64)

    def add_speed(a):
        speed_weights = plt.load(w_ref.at[speed_row, f]).astype(jnp.float64)
        return a + speed * speed_weights

    # The final input block contains only speed. Its weights also remain behind the skip branch.
    drive = lax.cond(speed != 0, add_speed, lambda a: a, drive) if skip else add_speed(drive)
    drive = drive + plt.load(b_ref.at[f]).astype(jnp.float64)
    after, out = _fire(drive, plt.load(v_ref.at[f]), config)
    plt.store(after_ref.at[f], after)
    plt.store(out_ref.at[f], out)
    o = jnp.arange(2, dtype=jnp.int32)
    wr = plt.load(wr_ref.at[f[:, None], o[None, :]])
    product = (jnp.where(out[:, None] != 0, wr, 0).astype(jnp.float64) if not config.graded
               else out.astype(jnp.float64)[:, None] * wr.astype(jnp.float64))
    readout_drive = jnp.sum(product, axis=0)
    readout_drive = readout_drive + plt.load(br_ref.at[o]).astype(jnp.float64)
    readout = config.readout_decay * plt.load(r_ref.at[o]).astype(jnp.float64) + readout_drive
    plt.store(readout_ref.at[o], readout.astype(r_ref.dtype))


def step_with_activity(params: Parameters, state: State, x: jax.Array, *, config: Config,
                       skip: bool = True, interpret: bool = False):
    """The step on binary ON/OFF pixels plus speed, and population outputs for correctness measurements."""
    if x.shape != (4097,):
        raise ValueError(f"Expected one flattened ON/OFF image and speed, got {x.shape}")
    conv = []
    h = x
    for layer in range(2):
        w, b = params[2 * layer:2 * layer + 2]
        shape = state[layer].shape
        features = shape[-1]
        pixels = 2 if layer == 0 else 1
        feature_block = 16 if layer == 0 else 32
        out_shape = (jax.ShapeDtypeStruct((state[layer].size,), state[layer].dtype),) * 2
        after, h = pl.pallas_call(
            partial(_conv_kernel, layer=layer, config=config, skip=skip),
            out_shape=out_shape, grid=(state[layer].size // features // pixels, features // feature_block),
            compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
            interpret=interpret, name=f"racer_conv{layer}_unit",
        )(h, w.reshape((-1, w.shape[-2], w.shape[-1])), b, state[layer].reshape(-1))
        conv.append((after.reshape(shape), h.reshape(shape)))
    dense_partial = pl.pallas_call(
        partial(_dense_partial_kernel, skip=skip, binary=not config.graded),
        out_shape=jax.ShapeDtypeStruct((DENSE_SPLITS, 128), jnp.float64),
        grid=(DENSE_SPLITS, 4), compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret, name="racer_dense_partial",
    )(h, params.dense_kernel)
    hidden, readout, out = pl.pallas_call(
        partial(_dense_finalize_kernel, config=config, skip=skip),
        out_shape=(jax.ShapeDtypeStruct((128,), state.hidden.dtype),
                   jax.ShapeDtypeStruct((2,), state.readout.dtype),
                   jax.ShapeDtypeStruct((128,), state.hidden.dtype)),
        grid=(1,), compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret, name="racer_dense_unit_readout",
    )(dense_partial, x, params.dense_kernel, params.dense_bias, params.readout_kernel, params.readout_bias,
      state.hidden, state.readout)
    after = State(conv[0][0], conv[1][0], hidden, readout)
    return after, readout, (conv[0][1], conv[1][1], out)


def step(params: Parameters, state: State, x: jax.Array, *, config: Config,
         skip: bool = True, interpret: bool = False):
    """(state, input) -> (state', readout'), with weights/configuration held fixed."""
    after, readout, _ = step_with_activity(params, state, x, config=config, skip=skip, interpret=interpret)
    return after, readout


def _reference_conv(x: np.ndarray, w: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Float64 SAME convolution in vision.ts's tap-row, tap-column, channel order."""
    rows, columns, channels = x.shape
    size, _, _, features = w.shape
    top = max((rows // 2 - 1) * 2 + size - rows, 0) // 2
    left = max((columns // 2 - 1) * 2 + size - columns, 0) // 2
    padded = np.pad(x, ((top, size - 1 - top), (left, size - 1 - left), (0, 0)))
    into = np.zeros((rows // 2, columns // 2, features), np.float64)
    for di in range(size):
        for dj in range(size):
            window = padded[di:di + rows:2, dj:dj + columns:2]
            for ch in range(channels):
                into += window[..., ch, None] * w[di, dj, ch]
    return into + bias


def _reference_dense(x: np.ndarray, w: np.ndarray, bias: np.ndarray) -> np.ndarray:
    into = np.zeros(w.shape[-1], np.float64)
    # Skip exact zeros, as the TypeScript implementation does. Other terms stay in input order.
    for i in np.flatnonzero(x):
        into += x[i] * w[i]
    return into + bias


def reference_step(params: Parameters, state: State, x: np.ndarray, config: Config):
    """Independent NumPy float64 reference, including all four carried populations."""
    p = Parameters(*(np.asarray(w, np.float64) for w in params))
    x = np.asarray(x, np.float64)
    h = x[:4096].reshape(2, 32, 64).transpose(1, 2, 0)
    after, activity = [], []
    for k in range(3):
        if k < 2:
            drive = _reference_conv(h, p[2 * k], p[2 * k + 1])
        else:
            joined = np.concatenate([h.reshape(-1), x[4096:]])
            drive = _reference_dense(joined, p.dense_kernel, p.dense_bias)
        v = config.decay * state[k] + ((1 - config.decay) * drive if config.graded else drive)
        h = np.maximum(v, 0) if config.graded else (v >= 1).astype(np.float64)
        if not config.graded:
            v = np.where(h != 0, 0, v)
        after.append(v)
        activity.append(h)
    drive = _reference_dense(h, p.readout_kernel, p.readout_bias)
    readout = config.readout_decay * state.readout + drive
    return State(*after, readout), readout, tuple(activity)
