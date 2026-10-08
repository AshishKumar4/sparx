"""Render sparx's clips into `docs/assets`: networks running and learning, from sparx's own runs.

    python tools/make_clips.py [--mp4] [names...]

Each clip is a sequence of SVG frames drawn with `tools/drawing.py`, in a
light and a dark version, rasterized by `tools/render_frames.cjs` in
headless Chromium and encoded by ffmpeg as an animated WebP, which a
README shows inline. `--mp4` also writes an MP4. The data is real:

    network   Brunel's network at the paper's size (12,500 LIF neurons),
              asynchronous irregular, `sparx.graph.models.brunel`
    training  a spiking classifier learning MNIST, 784-200-10 LIF trained
              by surrogate gradients for 400 steps
    messages  RNeuralNet's delayed messages crossing its connections
              (`sparx.learn.RNeuralNet`), each edge with its own delay

Needs ffmpeg, Node with Playwright's Chromium (`npm root -g` holds
playwright), and the MNIST files `sparx.datasets.mnist` downloads.
"""

from __future__ import annotations

import os
import shutil
import string
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from drawing import THEMES, Canvas, Theme
from make_figures import brunel_raster

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"
FPS = 30


def card(theme: Theme, width: float, height: float, title: str, subtitle: str) -> Canvas:
    """A frame, on the canvas's rounded surface: its title and its subtitle."""
    c = Canvas(width, height, theme)
    c.text(36, 56, title, size=28, weight=600)
    c.inline(36, 90, subtitle, size=19, fill=theme.muted)
    return c


# The network: a raster revealed as time passes, and the population's rate under it.

def network_frames(theme: Theme) -> Iterator[Canvas]:
    raster = brunel_raster()
    duration, cells = 300.0, 200
    counts = np.histogram(raster[:, 0], bins=np.arange(0, duration + 1, 1.0))[0]
    rate = np.convolve(counts / cells / 1e-3, np.ones(5) / 5, mode="same")  # Hz, 5 ms window
    width, height = 1280, 600
    x0, x1, y0, y1 = 70.0, 1220.0, 126.0, 440.0
    r0, r1 = 470.0, 560.0
    frames = 150
    for k in range(frames + 30):
        now = duration * min(k, frames) / frames
        c = card(theme, width, height, "12,500 LIF neurons in Brunel's balanced network",
                 "`sparx.graph.models.brunel(2500, g=5.0, eta=2.0)`: 200 of the excitatory cells, "
                 "asynchronous and irregular")
        c.text(x1, 52, f"{now:5.1f} ms", size=24, weight=500, family="mono", anchor="end", fill=theme.ink2)
        shown = raster[raster[:, 0] <= now]
        age = now - shown[:, 0]
        parts = []
        for (t, n), a in zip(shown, age, strict=True):
            x, y = x0 + (x1 - x0) * t / duration, y0 + (y1 - y0) * n / cells
            r = 1.8 + 2.2 * np.exp(-a / 3.0)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}"/>')
        c.add(f'<g fill="{theme.orange}">{"".join(parts)}</g>')
        cursor = x0 + (x1 - x0) * now / duration
        c.line(cursor, y0 - 6, cursor, r1, stroke=theme.ink2, width=1, opacity=0.5)
        steps = int(now)
        if steps > 1:
            top = 80.0
            points = [(x0 + (x1 - x0) * (i + 0.5) / duration, r1 - (r1 - r0) * min(rate[i], top) / top)
                      for i in range(steps)]
            c.polyline(points, stroke=theme.blue, width=1.8)
        c.line(x0, r1, x1, r1, stroke=theme.hairline, width=1)
        c.line(x0, r1 - (r1 - r0) / 2, x1, r1 - (r1 - r0) / 2, stroke=theme.hairline, width=1, dash="2 6")
        c.text(x1 + 4, r1 - (r1 - r0) / 2 + 4, "40", size=18, fill=theme.muted)
        c.text(x0, r0 - 10, "population rate, Hz", size=19, fill=theme.blue)
        yield c


# Training: a spiking classifier on MNIST, its spikes and its accuracy as it learns.

STEPS, HIDDEN, CHECKPOINTS, EVERY = 20, 200, 41, 10
FEATURED = 7  # the test digit the clip follows (a 9)


def training_run() -> dict[str, np.ndarray]:
    """Train 784-200-10 LIF on rate-coded MNIST for 400 steps of 128 images, Adam at 1e-3, the loss the
    cross entropy of the output spike counts; every 10 steps record the test accuracy on 2,000 images
    and the featured digit's input, hidden and output spikes."""
    import flax.linen as nn
    import optax

    import sparx
    from sparx.datasets import mnist

    class Net(nn.Module):
        @nn.compact
        def __call__(self, spikes: jax.Array) -> tuple[jax.Array, jax.Array]:
            hidden = sparx.nn.LIF(tau=2.0)(nn.Dense(HIDDEN)(spikes))
            return hidden, sparx.nn.LIF(tau=2.0)(nn.Dense(10)(hidden))

    train, test = mnist("train"), mnist("test")
    encoder = sparx.encode.RateEncoder(steps=STEPS)
    net = Net()
    test_x, test_y = test["image"][:2000].reshape(-1, 784), test["label"][:2000]
    params = net.init(jax.random.key(0), jnp.zeros((STEPS, 1, 784)))
    optimizer = optax.adam(1e-3)
    state = optimizer.init(params)

    def loss(params: dict, spikes: jax.Array, labels: jax.Array) -> jax.Array:
        counts = net.apply(params, spikes)[1].sum(0)
        return optax.softmax_cross_entropy_with_integer_labels(counts, labels).mean()

    @jax.jit
    def step(params: dict, state: optax.OptState, key: jax.Array, x: jax.Array, y: jax.Array):
        grads = jax.grad(loss)(params, encoder(key, x), y)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(params, updates), state

    @jax.jit
    def look(params: dict, key: jax.Array) -> tuple[jax.Array, ...]:
        spikes = encoder(key, test_x)
        hidden, out = net.apply(params, spikes)
        accuracy = jnp.mean(jnp.argmax(out.sum(0), -1) == test_y)
        return accuracy, spikes[:, FEATURED], hidden[:, FEATURED], out[:, FEATURED]

    rng = np.random.default_rng(0)
    record: dict[str, list] = {"accuracy": [], "input": [], "hidden": [], "output": []}
    for k in range((CHECKPOINTS - 1) * EVERY + 1):
        if k % EVERY == 0:
            for name, value in zip(record, look(params, jax.random.key(1)), strict=True):
                record[name].append(np.asarray(value))
        batch = rng.choice(len(train["label"]), 128, replace=False)
        params, state = step(params, state, jax.random.key(10 + k), train["image"][batch].reshape(-1, 784),
                             train["label"][batch])
    out = {name: np.stack(values) for name, values in record.items()}
    out["image"], out["label"] = test["image"][FEATURED], np.asarray(test["label"][FEATURED])
    return out


def training_frames(theme: Theme) -> Iterator[Canvas]:
    path = ROOT / ".cache" / "figures" / "training.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **training_run())
    run = np.load(path)
    width, height = 1280, 490
    per = 6  # frames per checkpoint
    total = CHECKPOINTS * per
    for k in range(total + 45):
        i = min(k // per, CHECKPOINTS - 1)
        c = card(theme, width, height, "A spiking classifier learning MNIST",
                 "784 inputs, 200 `LIF`, 10 `LIF` outputs; surrogate gradients, 20 time steps a digit")
        c.text(width - 36, 52, f"step {i * EVERY:3d}", size=24, weight=500, family="mono", anchor="end",
               fill=theme.ink2)
        # The digit, and the spikes that encode it.
        image = run["image"]
        px, gx, gy = 6, 40, 128
        for yy in range(28):
            for xx in range(28):
                if image[yy, xx] > 30:
                    c.rect(gx + xx * px, gy + yy * px, px - 0.6, px - 0.6, fill=theme.ink,
                           opacity=float(image[yy, xx]) / 255)
        c.text(gx, gy + 28 * px + 28, "a test digit", size=19, fill=theme.muted)
        # The hidden layer's spikes for it, time across.
        hx, hy, hw, hh = 260, 128, 420, 300
        c.rect(hx, hy, hw, hh, r=10, fill=theme.soft(theme.blue, 0.05), stroke=theme.panel_line)
        hidden = run["hidden"][i]
        times, cells = np.nonzero(hidden)
        dots = [f'<circle cx="{hx + 12 + (hw - 24) * t / (STEPS - 1):.1f}" '
                f'cy="{hy + 8 + (hh - 16) * n / HIDDEN:.1f}" r="1.9"/>'
                for t, n in zip(times, cells, strict=True)]
        c.add(f'<g fill="{theme.orange}">{"".join(dots)}</g>')
        c.text(hx, hy + hh + 28, f"hidden spikes: {len(times)}", size=19, fill=theme.muted)
        # The output neurons' spike counts; the most active is the answer.
        counts = run["output"][i].sum(0)
        ox, oy, ow = 730, 128, 230
        bar = 26
        for d in range(10):
            y = oy + d * (bar + 4)
            n = counts[d] / STEPS
            chosen = d == int(np.argmax(counts)) and counts.max() > 0
            c.rect(ox + 28, y, max(ow * n, 1.5), bar, r=4,
                   fill=theme.orange if chosen else theme.soft(theme.ink2, 0.3))
            c.text(ox + 8, y + 18, str(d), size=19, family="mono", anchor="middle",
                   fill=theme.ink if d == run["label"] else theme.muted)
        c.text(ox, oy + 10 * (bar + 4) + 20, "output spike counts", size=19, fill=theme.muted)
        # Test accuracy so far.
        ax, ay, aw, ah = 1010, 128, 234, 300
        acc = run["accuracy"][: i + 1]
        c.rect(ax, ay, aw, ah, r=10, fill="none", stroke=theme.panel_line)
        for level in (0.25, 0.5, 0.75, 1.0):
            yy = ay + ah - ah * level
            c.line(ax, yy, ax + aw, yy, stroke=theme.hairline, width=1, dash="2 6")
            c.text(ax - 6, yy + 4, f"{int(level * 100)}", size=17, fill=theme.muted, anchor="end")
        if len(acc) > 1:
            c.polyline([(ax + aw * j / (CHECKPOINTS - 1), ay + ah - ah * a) for j, a in enumerate(acc)],
                       stroke=theme.blue, width=2.2)
        c.circle(ax + aw * i / (CHECKPOINTS - 1), ay + ah - ah * acc[-1], 4.5, fill=theme.blue)
        c.text(ax, ay + ah + 28, f"test accuracy {100 * acc[-1]:.1f}%", size=21, weight=500, fill=theme.blue)
        yield c


# Messages: RNeuralNet's neurons send every tick, and each connection delivers after its own delay.

def messages_frames(theme: Theme) -> Iterator[Canvas]:
    from sparx.learn import RNeuralNet

    net = RNeuralNet.random(2, 14, 2, 2, fan=3, input_fan=2)
    ticks, per = 36, 5
    x = np.zeros((ticks, 2), np.float32)
    x[2, 0], x[9, 1] = 3.0, 3.0
    out = np.asarray(net.run(jnp.asarray(x))[0])
    w = net.wiring
    pre, post, weight, delay = (np.asarray(a) for a in (w.pre, w.post, w.weight, w.delay))
    keep = post != net.feeder
    pre, post, weight, delay = pre[keep], post[keep], weight[keep], delay[keep]
    width, height = 1280, 600
    cx, cy, radius = 760.0, 330.0, 200.0
    angles = 2 * np.pi * np.arange(net.neurons) / net.neurons - np.pi / 2
    where = {i: (cx + radius * np.cos(a), cy + radius * np.sin(a)) for i, a in enumerate(angles)}
    for k in range(net.inputs):
        where[net.neurons + k] = (260.0, cy - 90 + 180 * k)
    outputs = set(np.asarray(net.outputs).tolist())
    for f in range(ticks * per + 40):
        now = min(f, ticks * per) / per
        c = card(theme, width, height, "RNeuralNet: messages in flight",
                 "`sparx.learn.RNeuralNet`: every neuron sends each tick; each connection takes its own "
                 "1 to 21 ticks")
        c.text(width - 36, 52, f"tick {now:4.1f}", size=24, weight=500, family="mono", anchor="end",
               fill=theme.ink2)
        for a, b in zip(pre, post, strict=True):
            (x1, y1), (x2, y2) = where[a], where[b]
            c.line(x1, y1, x2, y2, stroke=theme.ink2, width=0.8, opacity=0.18)
        dots = {True: [], False: []}
        for a, b, wt, d in zip(pre, post, weight, delay, strict=True):
            (x1, y1), (x2, y2) = where[a], where[b]
            for t in range(max(0, int(np.floor(now - d))), int(np.ceil(now))):
                if not t < now <= t + d:
                    continue
                message = wt * out[t, a]
                if abs(message) < 1e-3:
                    continue
                frac = (now - t) / d
                size = 2.4 + 3.6 * min(abs(message), 1.5) / 1.5
                mx, my = x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac
                dots[bool(message > 0)].append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="{size:.2f}"/>')
        c.add(f'<g fill="{theme.blue}" opacity="0.75">{"".join(dots[False])}</g>')
        c.add(f'<g fill="{theme.orange}">{"".join(dots[True])}</g>')
        tick = min(int(now), ticks - 1)
        for unit, (ux, uy) in where.items():
            value = out[tick, unit]
            is_input = unit >= net.neurons
            fill = theme.soft(theme.orange, min(1.0, 0.15 + 0.5 * max(value, 0))) if value > 0 else theme.page
            output = unit in outputs
            c.circle(ux, uy, 15 if is_input else 13, fill=fill, stroke=theme.aqua if output else theme.ink2,
                     width=2.4 if output else 1.4)
        c.text(260, cy - 122, "cue A", size=19, fill=theme.ink2, anchor="middle")
        c.text(260, cy + 128, "cue B", size=19, fill=theme.ink2, anchor="middle")
        c.text(60, 474, "positive message", size=19, fill=theme.orange)
        c.text(60, 504, "negative message", size=19, fill=theme.blue)
        c.text(60, 534, "output neuron", size=19, fill=theme.aqua)
        yield c


CLIPS: dict[str, Callable[[Theme], Iterator[Canvas]]] = {
    "network": network_frames,
    "training": training_frames,
    "messages": messages_frames,
}


def render(name: str, theme: Theme, mp4: bool) -> None:
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        files = []
        for k, frame in enumerate(CLIPS[name](theme)):
            path = work / f"frame{k:04d}.svg"
            path.write_text(frame.svg(title=name, description=name, glyphs=string.printable))
            files.append(str(path))
        modules = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, check=True).stdout
        env = {**os.environ, "NODE_PATH": modules.strip()}
        subprocess.run(["node", str(ROOT / "tools" / "render_frames.cjs"), str(work), "1", *files], env=env,
                       check=True)
        frames = str(work / "frame%04d.png")
        stem = OUT / f"{name}-{theme.name}"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", frames,
                        "-vf", "scale=1024:-1:flags=lanczos", "-c:v", "libwebp_anim", "-lossless", "0",
                        "-quality", "70", "-loop", "0",
                        f"{stem}.webp"], check=True)
        if mp4:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", frames,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                            f"{stem}.mp4"], check=True)
    for suffix in ("webp", "mp4") if mp4 else ("webp",):
        path = Path(f"{stem}.{suffix}")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("make_clips needs ffmpeg")
    mp4 = "--mp4" in sys.argv[1:]
    for name in [arg for arg in sys.argv[1:] if arg != "--mp4"] or list(CLIPS):
        for theme in THEMES.values():
            render(name, theme, mp4)


if __name__ == "__main__":
    main()
