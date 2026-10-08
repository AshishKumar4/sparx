"""Draw sparx's banner and diagrams into `docs/assets`, each in a light and a dark version.

    python tools/make_figures.py [names...]

Every figure is SVG built by hand with `tools/drawing.py`, and the data a
figure plots comes from sparx itself: the banner's spikes are a Brunel
network that `sparx.graph.models.brunel` simulates, and its membrane is
an `LIFCell` run on noisy input. A README shows the version that matches
the reader's theme through `<picture>`. With names, only those figures
are drawn. `node tools/render_frames.cjs` rasterizes them to check by eye.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from drawing import THEMES, Canvas, Theme

from sparx.dynamics import LIFCell, run
from sparx.graph import SpikeRaster, simulate
from sparx.graph.models import brunel
from sparx.surrogate import ATan, spike

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"
SANS = "sans"


def cached(name: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
    """`compute()`, saved under `.cache/figures` and read back from there on later runs."""
    path = ROOT / ".cache" / "figures" / f"{name}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, compute())
    return np.load(path)


def brunel_raster(neurons: int = 200, duration: float = 300.0) -> np.ndarray:
    """Spike times (ms) and neuron indices of `neurons` excitatory cells of Brunel's network at the
    paper's size (12,500 neurons), asynchronous irregular (g = 5, eta = 2), after 100 ms of settling."""
    def compute() -> np.ndarray:
        network = brunel(2500, g=5.0, eta=2.0)
        result = simulate(network, network.init(jax.random.key(0)), duration=duration + 100.0,
                          key=jax.random.key(1), monitors={"e": SpikeRaster("e")}, chunk=100.0)
        spikes = np.asarray(result.records["e"])[round(100.0 / 0.1):, :neurons]
        steps, cells = np.nonzero(spikes)
        return np.stack([steps * 0.1, cells], axis=1)

    return cached(f"brunel-{neurons}-{duration:g}", compute)


def lif_trace(steps: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """The membrane before each reset and the spikes of one `LIFCell` (decay 0.9) on noisy drive."""
    rng = np.random.default_rng(3)
    drive = jnp.asarray(0.115 + 0.07 * rng.standard_normal((steps, 1)), jnp.float32)
    cell = LIFCell(0.9, threshold=1.0, reset="zero")
    (out, membrane), _ = run(cell, drive, record=lambda state: state.v)
    spikes = np.asarray(out.value)[:, 0]
    # The recorded membrane is after the reset; put the crossing back where it fired.
    before = np.asarray(membrane)[:, 0].copy()
    before[spikes > 0] = 1.0
    return before, spikes


def banner(theme: Theme, data: dict) -> Canvas:
    width, height = 1600, 420
    c = Canvas(width, height, theme)
    dark = theme.name == "dark"
    top, bottom = ("#ffffff", "#f3f6fb") if not dark else ("#0d1117", "#111a2c")
    c.defs.append(f'<linearGradient id="sky" x1="0" y1="0" x2="1" y2="1">'
                  f'<stop offset="0" stop-color="{top}"/><stop offset="1" stop-color="{bottom}"/>'
                  '</linearGradient><linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">'
                  '<stop offset="0" stop-color="#fff" stop-opacity="0"/>'
                  '<stop offset="0.28" stop-color="#fff" stop-opacity="1"/></linearGradient>'
                  '<mask id="reveal"><rect x="600" y="0" width="1000" height="420" fill="url(#fade)"/>'
                  '</mask>')
    c.rect(0.5, 0.5, width - 1, height - 1, r=28, fill="url(#sky)", stroke=theme.panel_line)

    # The raster: each dot one spike of one neuron, time to the right.
    raster = data["raster"]
    x0, x1, y0, y1 = 640.0, 1545.0, 58.0, 262.0
    t_max, cells = raster[:, 0].max(), raster[:, 1].max() + 1
    dots = [f'<circle cx="{x0 + (x1 - x0) * t / t_max:.1f}" cy="{y0 + (y1 - y0) * n / cells:.1f}" r="1.9"/>'
            for t, n in raster]
    opacity = 1.0 if dark else 0.85
    c.add(f'<g mask="url(#reveal)" fill="{theme.orange}" opacity="{opacity}">{"".join(dots)}</g>')

    # One membrane under it, rising to the threshold and resetting at each spike.
    v, spikes = data["membrane"]
    m0, m1 = 304.0, 366.0
    xs = x0 + (x1 - x0) * np.arange(len(v)) / (len(v) - 1)
    ys = m1 - (m1 - m0) * np.clip(v, 0, 1)
    points = []
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        points.append((x, y))
        if spikes[i] > 0:
            points.append((x, m1))
    c.add('<g mask="url(#reveal)">')
    c.line(x0, m0, x1, m0, stroke=theme.muted, width=1, dash="3 5")
    c.polyline(points, stroke=theme.blue, width=1.6)
    for x in xs[spikes > 0]:
        c.line(x, m0 - 20, x, m0 - 7, stroke=theme.orange, width=2)
    c.add("</g>")

    # The name and what it is.
    word = c.text(92, 214, "sparx", size=148, weight=700, family="display", spacing=-4)
    c.circle(92 + word + 20, 214 - 13, 13, fill=theme.orange)
    c.text(96, 270, "Spiking neural networks in JAX", size=34, weight=500, fill=theme.ink2)
    c.text(97, 320, "Train them with gradients or local rules.", size=27, fill=theme.muted)
    c.text(97, 358, "Simulate circuits in millivolts and milliseconds.", size=27, fill=theme.muted)
    return c


# Figures are 1200 px wide, which a README shows at about 880 px, so text is set at 17 px or more and
# reads at 13 px or more on the page.
WIDTH = 1200
TITLE, BODY, SMALL = 27, 19, 17


def card(c: Canvas, x: float, y: float, w: float, title: str, items: list[str], color: str) -> float:
    """A card headed `title` listing `items`, tinted with `color`; returns its height."""
    lines = [line for item in items for line in c.wrap(item, w - 50, size=BODY)]
    height = 66 + 29 * len(lines)
    c.box(x, y, w, height, color=color, alpha=0.06, r=16)
    c.circle(x + 27, y + 32, 6, fill=color)
    c.inline(x + 44, y + 40, title, size=22, weight=600)
    for i, line in enumerate(lines):
        c.inline(x + 26, y + 76 + 29 * i, line, size=BODY, fill=c.theme.ink2)
    return height


def panel_title(c: Canvas, x: float, y: float, title: str, caption: str = "") -> None:
    c.text(x, y, title, size=TITLE, weight=600)
    if caption:
        c.inline(x, y + 34, caption, size=BODY, fill=c.theme.muted)


def architecture(theme: Theme, data: dict) -> Canvas:
    margin, gap = 30, 22
    column = (WIDTH - 2 * margin - 30) / 2
    sides = [
        (margin, "Train", theme.blue, [
            ("`sparx.nn` layers", ["`LIF` `ALIF` `Synaptic` `Rate` `PSN`",
                                   "`Recurrent`: dense, sparse or plastic",
                                   "`DelayedDense` with learned delays"]),
            ("Learning rules", ["surrogate gradients through time", "e-prop, OTTT and EventProp",
                                "PC-ALM, REINFORCE, fast weights"]),
            ("Training on `dew`", ["objectives on dew's `Trainer`", "checkpoints, evaluation, run records",
                                   "a streaming server and NIR export"])]),
        (WIDTH - margin - column, "Simulate", theme.aqua, [
            ("`sparx.dynamics`", ["neurons in mV, ms, pA and nS", "current, conductance and graded synapses",
                                  "STDP and short-term plasticity"]),
            ("`sparx.graph` networks", ["populations, projections, delays",
                                        "Poisson and current inputs, monitors",
                                        "`simulate` in chunks, over devices"]),
            ("Connectomes", ["the FlyWire whole brain (Shiu et al.)", "the male CNS connectome",
                             "FLYNN, a connectome trained by BPTT"])])]
    c = Canvas(WIDTH, 1200, theme)
    bottoms = []
    for x, label, color, cards in sides:
        c.text(x + 4, 46, label.upper(), size=18, weight=600, fill=color, spacing=2.5)
        y = 68.0
        for title, items in cards:
            y += card(c, x, y, column, title, items, color) + gap
        bottoms.append(y - gap)
    # Both sides run their models through one protocol, below them.
    pw, ph = 720, 238
    px, py = (WIDTH - pw) / 2, max(bottoms) + 80
    for (x, _, color, _), bottom in zip(sides, bottoms, strict=True):
        mid = x + column / 2
        towards = px + 120 if x < WIDTH / 2 else px + pw - 120
        c.path(f"M{mid:.1f},{bottom + 4} C{mid:.1f},{bottom + 50} {towards:.1f},{py - 50} "
               f"{towards:.1f},{py - 6}", stroke=color, width=1.8, arrow=color)
    c.box(px, py, pw, ph, color=theme.ink2, alpha=0.04, r=18)
    c.text(WIDTH / 2, py + 48, "One neuron protocol", size=24, weight=600, anchor="middle")
    for i, line in enumerate(["model.init_state(shape, dtype)", "model.step(state, SynapticInput, dt)",
                              "    -> (state, Output)", "run(model, inputs)  # a scan over time"]):
        c.text(px + 110, py + 98 + 34 * i, line, size=19, family="mono")
    # The two families of models that meet it.
    fy, fh, fw = py + ph + 74, 168, (pw - gap) / 2
    families = [
        ("Dimensionless cells", theme.blue,
         ["`LIFCell` `ALIFCell`", "`RateCell` `PulseCell`", "`BernoulliCell`"]),
        ("Physical models", theme.aqua,
         ["`LeakyIntegrateAndFire`", "`AdEx` `Izhikevich`", "`HodgkinHuxley`"])]
    for k, (title, color, items) in enumerate(families):
        fx = px + k * (fw + gap)
        c.box(fx, fy, fw, fh, color=color, alpha=0.07, r=16)
        c.text(fx + 24, fy + 38, title, size=20, weight=600)
        for i, item in enumerate(items):
            c.inline(fx + 24, fy + 76 + 29 * i, item, size=BODY, fill=theme.ink2)
        c.path(f"M{fx + fw / 2:.1f},{fy - 4} L{fx + fw / 2:.1f},{py + ph + 6}", stroke=color, width=1.8,
               arrow=color)
    c.height = fy + fh + 30
    return c


def over_time(theme: Theme, data: dict) -> Canvas:
    c = Canvas(WIDTH, 820, theme)
    # A: the scan. Each step takes the state and the step's input and gives the next state and spikes.
    panel_title(c, 30, 46, "A layer runs over time",
                "`run(model, x)` with `x` of shape `[T, B, F]`: one `jax.lax.scan`")
    steps, sw, sh = 5, 140, 72
    pitch = (WIDTH - 60 - sw) / (steps - 1)
    sy = 190
    for t in range(steps):
        x = 30 + t * pitch
        c.box(x, sy, sw, sh, color=theme.blue, alpha=0.08, r=12)
        c.inline(x + sw / 2, sy + 44, "`step`", size=20, anchor="middle")
        c.path(f"M{x + sw / 2},{sy + sh + 70} L{x + sw / 2},{sy + sh + 6}", stroke=theme.ink2, width=1.6,
               arrow=theme.ink2)
        c.text(x + sw / 2, sy + sh + 98, f"x[{t + 1 if t < steps - 1 else 'T'}]", size=SMALL, family="mono",
               fill=theme.ink2, anchor="middle")
        c.path(f"M{x + sw / 2},{sy - 4} L{x + sw / 2},{sy - 52}", stroke=theme.orange, width=1.8,
               arrow=theme.orange)
        c.text(x + sw / 2 + 12, sy - 30, "spikes", size=SMALL, fill=theme.orange)
        if t:
            c.path(f"M{x - pitch + sw + 4},{sy + sh / 2} L{x - 6},{sy + sh / 2}", stroke=theme.blue, width=2,
                   arrow=theme.blue)
    c.text(30 + sw + (pitch - sw) / 2, sy + sh / 2 - 12, "state", size=SMALL, fill=theme.blue,
           anchor="middle")
    # B: one leaky integrate-and-fire neuron, the membrane rising to threshold and resetting at each spike.
    top, bw = 470, 540
    bx = 30
    panel_title(c, bx, top, "It integrates and fires", "`v[t] = decay * v[t-1] + x[t]`, a spike at `v >= 1`")
    v, spikes = data["membrane"]
    v, spikes = v[:120], spikes[:120]
    y_hi, y_lo = top + 120, top + 300
    xs = bx + bw * np.arange(len(v)) / (len(v) - 1)
    ys = y_lo - (y_lo - y_hi) * np.clip(v, 0, 1.05) / 1.05
    points = []
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        points.append((x, y))
        if spikes[i] > 0:
            points.append((x, y_lo))
    threshold = y_lo - (y_lo - y_hi) / 1.05
    c.line(bx, threshold, bx + bw, threshold, stroke=theme.muted, width=1, dash="3 5")
    c.line(bx, y_lo, bx + bw, y_lo, stroke=theme.hairline, width=1)
    c.polyline(points, stroke=theme.blue, width=2.2)
    for x in xs[spikes > 0]:
        c.line(x, y_hi - 42, x, y_hi - 16, stroke=theme.orange, width=2.4)
    c.text(bx + bw, y_lo + 30, "membrane, reset to 0 at each spike", size=SMALL, fill=theme.blue,
           anchor="end")
    # C: the spike is a step, so its derivative is zero almost everywhere; training uses a smooth stand-in.
    cx, cw = 630, 540
    panel_title(c, cx, top, "Gradients pass a surrogate", "forward a step, backward `ATan()`'s slope")
    u = np.linspace(-2.0, 2.0, 241)
    forward = np.asarray(spike(jnp.asarray(u, jnp.float32), ATan()))
    slope = np.asarray(jax.vmap(jax.grad(lambda x: spike(x, ATan())))(jnp.asarray(u, jnp.float32)))
    g_hi, g_lo = top + 120, top + 300
    gx = cx + cw * (u - u[0]) / (u[-1] - u[0])
    c.line(cx, g_lo, cx + cw, g_lo, stroke=theme.hairline, width=1)
    c.line(cx + cw / 2, g_hi - 10, cx + cw / 2, g_lo, stroke=theme.hairline, width=1)
    c.polyline(list(zip(gx, g_lo - (g_lo - g_hi) * forward, strict=True)), stroke=theme.orange, width=2.4)
    c.polyline(list(zip(gx, g_lo - (g_lo - g_hi) * slope / slope.max(), strict=True)), stroke=theme.violet,
               width=2.4)
    c.text(cx + cw, g_hi - 14, "spike", size=SMALL, fill=theme.orange, anchor="end")
    c.text(cx + cw / 2 + 70, g_hi + 40, "surrogate slope", size=SMALL, fill=theme.violet)
    c.text(cx + cw / 2, g_lo + 30, "v - threshold", size=SMALL, fill=theme.muted, anchor="middle")
    return c


def recurrence(theme: Theme, data: dict) -> Canvas:
    c = Canvas(WIDTH, 640, theme)
    panel_title(c, 30, 46, "One recurrent cell, any wiring",
                "`RecurrentCell(inner, wiring, fast_weights=None)`")
    # The loop: input plus what arrives, through the model, out and back through the wiring.
    y = 230
    c.inline(30, y - 16, "`x[t]`", size=BODY, fill=theme.ink2)
    c.path(f"M30,{y} L120,{y}", stroke=theme.ink2, width=1.8, arrow=theme.ink2)
    c.circle(146, y, 24, fill=theme.page, stroke=theme.ink2, width=1.8)
    c.text(146, y + 8, "+", size=26, weight=500, anchor="middle", fill=theme.ink2)
    c.path(f"M172,{y} L232,{y}", stroke=theme.ink2, width=1.8, arrow=theme.ink2)
    c.box(238, y - 64, 250, 128, color=theme.blue, alpha=0.08, r=16)
    c.inline(363, y - 18, "`inner` model", size=21, weight=600, anchor="middle")
    c.inline(363, y + 16, "`LIFCell` `ALIFCell`", size=SMALL, fill=theme.ink2, anchor="middle")
    c.inline(363, y + 42, "`RateCell` `PulseCell`", size=SMALL, fill=theme.ink2, anchor="middle")
    c.path(f"M492,{y} L604,{y}", stroke=theme.orange, width=2, arrow=theme.orange)
    c.inline(612, y + 7, "`output`", size=BODY, fill=theme.orange)
    # Back along the bottom: the wiring sends each output, and what it sends waits until it arrives.
    loop, wx, ww = 410, 330, 210
    c.path(f"M560,{y} L560,{loop} L{wx + ww + 6},{loop}", stroke=theme.blue, width=1.8, arrow=theme.blue)
    c.box(wx, loop - 36, ww, 72, color=theme.blue, alpha=0.08, r=14)
    c.inline(wx + ww / 2, loop + 7, "`wiring.send`", size=BODY, anchor="middle")
    slots, sw, bx = 4, 36, 172
    c.path(f"M{wx - 4},{loop} L{bx + slots * sw + 6},{loop}", stroke=theme.blue, width=1.8, arrow=theme.blue)
    for k in range(slots):
        c.rect(bx + k * sw, loop - 18, sw - 5, 36, r=6, fill=theme.soft(theme.blue, 0.2 - 0.04 * k),
               stroke=theme.blue, width=1.2)
    c.text(bx + slots * sw / 2, loop + 48, "on its way", size=SMALL, fill=theme.blue, anchor="middle")
    c.path(f"M{bx - 4},{loop} L146,{loop} L146,{y + 28}", stroke=theme.blue, width=1.8, arrow=theme.blue)
    # Fast weights: a Hebbian trace written by the sequence itself adds to the wiring's weights.
    c.box(wx, loop + 74, ww, 62, color=theme.violet, alpha=0.08, r=14)
    c.text(wx + ww / 2, loop + 112, "Hebbian trace", size=BODY, weight=600, anchor="middle")
    c.path(f"M{wx + ww / 2},{loop + 72} L{wx + ww / 2},{loop + 40}", stroke=theme.violet, width=1.8,
           arrow=theme.violet)
    c.inline(wx + ww / 2 + 16, loop + 62, "`+ alpha * hebb`", size=SMALL, fill=theme.violet)
    # The wirings.
    ox = 760
    c.line(ox - 36, 110, ox - 36, 600, stroke=theme.hairline, width=1)
    rows = [("`Dense(weight)`", "every unit to every unit", "dense"),
            ("`Sparse(pre, post, w, n)`", "an edge list, such as a connectome", "sparse"),
            ("`Sparse(..., delay=d)`", "each edge with its own delay", "delay"),
            ("`FastWeights(alpha, rule)`", "plasticity within a sequence", "fast")]
    rng = np.random.default_rng(4)
    for i, (code, caption, kind) in enumerate(rows):
        ry = 120 + 120 * i
        if kind in ("dense", "fast"):
            color = theme.blue if kind == "dense" else theme.violet
            for a in range(5):
                for b in range(5):
                    c.rect(ox + a * 16, ry + b * 16, 13, 13, r=2,
                           fill=theme.soft(color, 0.12 + 0.6 * rng.random()))
        else:
            nodes = [(ox + 10, ry + 12), (ox + 66, ry + 6), (ox + 38, ry + 42), (ox + 74, ry + 66),
                     (ox + 6, ry + 70)]
            for e, (a, b) in enumerate([(0, 1), (1, 3), (2, 0), (2, 3), (4, 2), (3, 4)]):
                (x1, y1), (x2, y2) = nodes[a], nodes[b]
                c.line(x1, y1, x2, y2, stroke=theme.blue, width=1.4, opacity=0.8)
                if kind == "delay":
                    for k in range(1 + e % 3):
                        f = (k + 1) / (2 + e % 3)
                        c.circle(x1 + (x2 - x1) * f, y1 + (y2 - y1) * f, 3, fill=theme.orange)
            for x, yy in nodes:
                c.circle(x, yy, 6.5, fill=theme.page, stroke=theme.blue, width=1.7)
        c.inline(ox + 112, ry + 32, code, size=BODY)
        c.text(ox + 112, ry + 62, caption, size=SMALL, fill=theme.ink2)
    return c


def _steps(c: Canvas, x: float, y: float, n: int = 4, w: float = 50, gap: float = 22) -> list[float]:
    """`n` small step boxes in a row joined by forward arrows; their centers."""
    t = c.theme
    centers = []
    for i in range(n):
        bx = x + i * (w + gap)
        c.rect(bx, y, w, 36, r=8, fill=t.soft(t.blue, 0.1), stroke=t.blue, width=1.3)
        centers.append(bx + w / 2)
        if i:
            c.path(f"M{bx - gap + 2},{y + 18} L{bx - 4},{y + 18}", stroke=t.blue, width=1.4, arrow=t.blue)
    return centers


def _bptt(c: Canvas, x: float, y: float) -> None:
    t = c.theme
    centers = _steps(c, x + 10, y + 16)
    c.path(f"M{centers[-1] + 10},{y + 84} L{centers[0] - 10},{y + 84}", stroke=t.violet, width=2.2,
           arrow=t.violet)
    c.text(centers[-1] + 30, y + 90, "dL", size=SMALL, family="mono", fill=t.violet)


def _trace(c: Canvas, x: float, y: float, w: float, h: float, spikes: list[float], color: str,
           tau: float = 0.12) -> None:
    """A presynaptic trace: a jump at each spike (fractions of `w`) decaying with time constant `tau`."""
    xs = np.linspace(0, 1, 120)
    trace = sum(np.where(xs >= s, np.exp(-(xs - s) / tau), 0.0) for s in spikes)
    trace = trace / trace.max()
    c.polyline([(x + w * a, y + h - h * b) for a, b in zip(xs, trace, strict=True)], stroke=color, width=2.2)
    for s in spikes:
        c.line(x + w * s, y + h + 8, x + w * s, y + h + 24, stroke=c.theme.orange, width=2.2)


def _eprop(c: Canvas, x: float, y: float) -> None:
    t = c.theme
    _trace(c, x + 10, y + 34, 150, 50, [0.1, 0.3, 0.42, 0.75], t.violet, tau=0.18)
    c.text(x + 10, y + 20, "eligibility", size=SMALL, fill=t.violet)
    c.text(x + 190, y + 66, "x", size=20, fill=t.ink2)
    c.path(f"M{x + 236},{y + 4} L{x + 236},{y + 80}", stroke=t.violet, width=2.2, arrow=t.violet)
    c.text(x + 250, y + 30, "learning", size=SMALL, fill=t.violet)
    c.text(x + 250, y + 52, "signal", size=SMALL, fill=t.violet)


def _ottt(c: Canvas, x: float, y: float) -> None:
    t = c.theme
    _trace(c, x + 10, y + 34, 150, 50, [0.12, 0.36, 0.5, 0.8], t.blue)
    c.text(x + 10, y + 20, "presynaptic trace", size=SMALL, fill=t.blue)
    c.text(x + 190, y + 66, "x", size=20, fill=t.ink2)
    for k in range(5):
        c.line(x + 226 + 15 * k, y + 84, x + 226 + 15 * k, y + 84 - (18 + 9 * ((k * 7) % 4)), stroke=t.violet,
               width=3.2)
    c.text(x + 222, y + 20, "error", size=SMALL, fill=t.violet)


def _eventprop(c: Canvas, x: float, y: float) -> None:
    t = c.theme
    xs = np.linspace(0, 1, 100)
    v = np.where(xs < 0.55, 1.15 * (1 - np.exp(-xs / 0.25)), 0.05 + 0.9 * (1 - np.exp(-(xs - 0.55) / 0.3)))
    w, h, top = 290, 62, y + 14
    c.line(x + 10, top + h * 0.12, x + 10 + w, top + h * 0.12, stroke=t.muted, width=1, dash="3 4")
    c.polyline([(x + 10 + w * a, top + h - h * min(b, 0.9)) for a, b in zip(xs, v, strict=True)],
               stroke=t.blue, width=2.2)
    spike_x = x + 10 + w * 0.55
    c.line(spike_x, top - 8, spike_x, top + h + 6, stroke=t.orange, width=2.4)
    c.path(f"M{x + 10 + w},{top + h + 24} L{spike_x + 8},{top + h + 24}", stroke=t.violet, width=2.2,
           arrow=t.violet)
    c.text(spike_x - 8, top + h + 30, "dL/dt", size=SMALL, family="mono", fill=t.violet, anchor="end")


def _pcalm(c: Canvas, x: float, y: float) -> None:
    t = c.theme
    for i, layer in enumerate([3, 4, 3]):
        lx = x + 40 + 120 * i
        for k in range(layer):
            cy = y + 44 + (k - (layer - 1) / 2) * 23
            c.circle(lx, cy, 7.5, fill=t.soft(t.blue, 0.25), stroke=t.blue, width=1.5)
        if i:
            c.path(f"M{lx - 100},{y + 44} L{lx - 20},{y + 44}", stroke=t.blue, width=1.5, arrow=t.blue)
            c.circle(lx - 60, y + 82, 9, fill=t.soft(t.violet, 0.3), stroke=t.violet, width=1.5)
            c.text(lx - 60, y + 110, "error", size=SMALL, fill=t.violet, anchor="middle")


def _reinforce(c: Canvas, x: float, y: float) -> None:
    t = c.theme
    rng = np.random.default_rng(6)
    for row in range(3):
        for k in range(13):
            if rng.random() < 0.3:
                tick = x + 10 + 15 * k
                c.line(tick, y + 10 + 24 * row, tick, y + 26 + 24 * row, stroke=t.orange, width=2.2)
    c.text(x + 10, y + 100, "spikes drawn at random", size=SMALL, fill=t.ink2)
    c.path(f"M{x + 214},{y + 40} L{x + 246},{y + 40}", stroke=t.ink2, width=1.5, arrow=t.ink2)
    c.box(x + 252, y + 16, 70, 48, color=t.violet, alpha=0.12, r=10)
    c.text(x + 287, y + 47, "x R", size=19, family="mono", anchor="middle", fill=t.violet)


def _fast(c: Canvas, x: float, y: float) -> None:
    t = c.theme
    rng = np.random.default_rng(8)
    slow, fast = rng.random((5, 5)), rng.random((5, 5))
    grids = [(slow, x + 10, t.blue), (fast, x + 125, t.violet), (slow + fast, x + 240, t.blue)]
    for grid, gx, color in grids:
        for a in range(5):
            for b in range(5):
                c.rect(gx + a * 15, y + 12 + b * 15, 12, 12, r=2,
                       fill=t.soft(color, 0.12 + 0.35 * grid[b, a]))
    c.text(x + 102, y + 54, "+", size=22, fill=t.ink2, anchor="middle")
    c.text(x + 217, y + 54, "=", size=22, fill=t.ink2, anchor="middle")
    c.text(x + 10, y + 110, "slow", size=SMALL, fill=t.blue)
    c.text(x + 125, y + 110, "this sequence", size=SMALL, fill=t.violet)


def _diffusion(c: Canvas, x: float, y: float) -> None:
    t = c.theme
    nodes = {"r": (x + 290, y + 46), "a": (x + 190, y + 18), "b": (x + 190, y + 78), "c": (x + 80, y + 8),
             "d": (x + 80, y + 50), "e": (x + 80, y + 94)}
    for a, b, w in [("r", "a", 3.4), ("r", "b", 1.8), ("a", "c", 2.4), ("a", "d", 1.2), ("b", "d", 1.0),
                    ("b", "e", 1.4)]:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        c.path(f"M{x1 - 10},{y1} L{x2 + 11},{y2}", stroke=t.violet, width=w, arrow=t.violet)
    for name, (nx, ny) in nodes.items():
        c.circle(nx, ny, 8.5 if name != "r" else 12, fill=t.soft(t.violet if name == "r" else t.blue, 0.25),
                 stroke=t.violet if name == "r" else t.blue, width=1.5)
    c.text(x + 310, y + 52, "R", size=19, weight=600, fill=t.violet)


def _conversion(c: Canvas, x: float, y: float) -> None:
    t = c.theme
    xs = np.linspace(-1, 1, 50)
    c.line(x + 10, y + 90, x + 130, y + 90, stroke=t.hairline, width=1)
    c.polyline([(x + 70 + 60 * a, y + 90 - 70 * max(a, 0)) for a in xs], stroke=t.blue, width=2.4)
    c.text(x + 10, y + 116, "ReLU", size=SMALL, fill=t.blue)
    c.path(f"M{x + 146},{y + 52} L{x + 182},{y + 52}", stroke=t.ink2, width=1.5, arrow=t.ink2)
    for row, rate in enumerate([0.2, 0.5, 0.9]):
        for k in range(10):
            if (k * rate) % 1 < rate:
                tick = x + 196 + 13 * k
                c.line(tick, y + 14 + 30 * row, tick, y + 32 + 30 * row, stroke=t.orange, width=2.2)
    c.text(x + 196, y + 116, "IF spike rates", size=SMALL, fill=t.orange)


def learning(theme: Theme, data: dict) -> Canvas:
    rules = [
        ("Backpropagation through time", "surrogate gradients", _bptt,
         "Errors run back through every step; memory grows with time.", "`jax.grad` through `run`"),
        ("e-prop", "Bellec et al. 2020", _eprop,
         "An eligibility trace per synapse times a learning signal, online.", "`sparx.learn.eprop`"),
        ("OTTT", "Xiao et al. 2022", _ottt,
         "A presynaptic trace times each step's error, without unrolling.", "`sparx.learn.ottt`"),
        ("Exact spike times", "Wunderlich and Pehle 2021", _eventprop,
         "Gradients through spike times in continuous time, as in EventProp.", "`sparx.learn.spike_times`"),
        ("Predictive coding, PC-ALM", "Seely and Gould 2026", _pcalm,
         "Activity relaxes on prediction errors; each layer updates from its own.", "`PredictiveCoding`"),
        ("REINFORCE", "Williams 1992", _reinforce,
         "Neurons fire at random; each synapse's score is scaled by the reward.", "`sparx.learn.reinforce`"),
        ("Fast weights", "Miconi et al. 2018, 2019", _fast,
         "A Hebbian trace each sequence writes; BPTT learns how plastic.", "`FastWeights(alpha, rule)`"),
        ("Reward diffusion", "RNeuralNet-Research, 2018", _diffusion,
         "Reward spreads back by a softmax of activity; it follows no gradient.", "`reward_diffusion`"),
        ("Conversion", "Rueckauer et al. 2017", _conversion,
         "A trained ReLU network becomes IF neurons with balanced thresholds.", "`sparx.learn.convert`"),
    ]
    cols, gap, margin = 3, 20, 30
    cw = (WIDTH - 2 * margin - (cols - 1) * gap) / cols
    ch = 352
    rows = (len(rules) + cols - 1) // cols
    c = Canvas(WIDTH, margin * 2 + rows * ch + (rows - 1) * gap, theme)
    for i, (title, source, draw, caption, api) in enumerate(rules):
        x = margin + (i % cols) * (cw + gap)
        y = margin + (i // cols) * (ch + gap)
        c.box(x, y, cw, ch, r=16)
        c.text(x + 22, y + 40, title, size=21, weight=600)
        c.text(x + 22, y + 66, source, size=SMALL, fill=theme.muted)
        draw(c, x + 12, y + 90)
        for k, line in enumerate(c.wrap(caption, cw - 44, size=SMALL)):
            c.inline(x + 22, y + 248 + 25 * k, line, size=SMALL, fill=theme.ink2)
        c.inline(x + 22, y + ch - 22, api, size=SMALL, fill=theme.violet)
    return c


def _population(c: Canvas, x: float, y: float, r: float, count: int, color: str, seed: int) -> None:
    """A population: a disc of `count` neurons scattered inside, a few of them firing."""
    rng = np.random.default_rng(seed)
    c.circle(x, y, r, fill=c.theme.soft(color, 0.1), stroke=color, width=1.8)
    for _ in range(count):
        a, d = rng.uniform(0, 2 * np.pi), r * 0.82 * np.sqrt(rng.uniform())
        firing = rng.random() < 0.12
        c.circle(x + d * np.cos(a), y + d * np.sin(a), 3.4 if firing else 2.6,
                 fill=c.theme.orange if firing else color, opacity=1.0 if firing else 0.6)


def _inhibit(c: Canvas, start: tuple[float, float], c1: tuple[float, float], c2: tuple[float, float],
             end: tuple[float, float], color: str) -> None:
    """A curved projection from `start` to `end` ending in the bar of an inhibitory synapse."""
    c.path(f"M{start[0]},{start[1]} C{c1[0]},{c1[1]} {c2[0]},{c2[1]} {end[0]},{end[1]}", stroke=color,
           width=2)
    direction = np.subtract(end, c2) / np.linalg.norm(np.subtract(end, c2))
    dx, dy = -direction[1] * 11, direction[0] * 11
    c.line(end[0] - dx, end[1] - dy, end[0] + dx, end[1] + dy, stroke=color, width=3.2)


def circuits(theme: Theme, data: dict) -> Canvas:
    c = Canvas(WIDTH, 900, theme)
    panel_title(c, 30, 46, "A circuit, declared", "`Network(populations, projections, inputs, dt=0.1)`")
    ex, ey, er = 470, 330, 118
    ix, iy, ir = 830, 330, 76
    _population(c, ex, ey, er, 90, theme.aqua, 1)
    _population(c, ix, iy, ir, 36, theme.aqua, 2)
    c.inline(ex, ey + er + 38, '`Population("e", 10000, ...)`', size=SMALL, fill=theme.ink2, anchor="middle")
    c.inline(ix, iy + ir + 38, '`Population("i", 2500, ...)`', size=SMALL, fill=theme.ink2, anchor="middle")
    c.path(f"M{ex + er + 4},{ey - 30} C{ex + er + 80},{ey - 76} {ix - ir - 80},{iy - 66} "
           f"{ix - ir - 6},{iy - 28}", stroke=theme.aqua, width=2, arrow=theme.aqua)
    _inhibit(c, (ix - ir - 4, iy + 28), (ix - ir - 80, iy + 76), (ex + er + 80, ey + 76),
             (ex + er + 8, ey + 34), theme.ink2)
    c.path(f"M{ex - 74},{ey - er + 28} C{ex - 150},{ey - er - 76} {ex - 10},{ey - er - 128} "
           f"{ex - 6},{ey - er - 6}", stroke=theme.aqua, width=2, arrow=theme.aqua)
    _inhibit(c, (ix + 32, iy - ir + 10), (ix + 76, iy - ir - 86), (ix - 44, iy - ir - 96),
             (ix - 24, iy - ir - 6), theme.ink2)
    c.inline(640, ey - 134, "`Projection(..., delay=1.5)`", size=SMALL, fill=theme.ink2, anchor="middle")
    c.text((ex + ix) / 2, ey + 112, "inhibitory", size=SMALL, fill=theme.muted, anchor="middle")
    for k in range(5):
        y = ey - 64 + 32 * k
        for j in range(4):
            if (j + k) % 2 == 0:
                c.circle(130 + 20 * j, y, 3.4, fill=theme.orange)
        c.path(f"M220,{y} L{ex - er - 12},{ey + (y - ey) * 0.5}", stroke=theme.muted, width=1.2,
               arrow=theme.muted)
    c.inline(130, ey + 116, "`PoissonInput`", size=SMALL, fill=theme.ink2)
    # What a run gives back.
    top = 560
    c.line(30, top - 20, WIDTH - 30, top - 20, stroke=theme.hairline, width=1)
    panel_title(c, 30, top + 26, "Simulated in chunks",
                "`simulate(network, variables, duration=600.0, chunk=100.0)`")
    chunks = 6
    cw = (WIDTH - 60 - (chunks - 1) * 6) / chunks
    for k in range(chunks):
        x = 30 + k * (cw + 6)
        c.rect(x, top + 82, cw, 44, r=8, fill=theme.soft(theme.aqua, 0.12), stroke=theme.aqua, width=1.3)
        c.text(x + cw / 2, top + 110, f"{100 * k}-{100 * (k + 1)} ms", size=SMALL, family="mono",
               anchor="middle", fill=theme.ink2)
        c.rect(x + cw - 6, top + 140, 12, 12, r=2, fill=theme.muted)
    c.text(WIDTH - 30, top + 176, "a checkpoint after each chunk", size=SMALL, fill=theme.muted, anchor="end")
    for k, (code, caption) in enumerate([("`SpikeRaster(\"e\")`", "which neurons fired"),
                                         ("`PopulationRate(\"e\")`", "the rate in Hz"),
                                         ("`StateMonitor(\"e\", ...)`", "chosen membranes")]):
        x = 30 + k * 390
        c.circle(x + 8, top + 228, 6, fill=theme.aqua)
        c.inline(x + 24, top + 235, code, size=SMALL)
        c.text(x + 24, top + 263, caption, size=SMALL, fill=theme.ink2)
    c.height = top + 300
    return c


def training(theme: Theme, data: dict) -> Canvas:
    c = Canvas(WIDTH, 560, theme)
    panel_title(c, 30, 46, "Training on dew", "sparx supplies the models and objectives; dew runs them")
    stages = [
        ("Model", "a Flax module", ["`sparx.nn` layers", "`sparx.models`"], theme.blue),
        ("Objective", "loss and evaluation", ["`SpikingClassifierObjective`", "`EPropObjective`, ..."],
         theme.blue),
        ("Trainer", "dew", ["mesh, compiled step", "EMA, checkpoints"], theme.ink2),
        ("Run record", "JSON by import path", ["`run.json`", "reloads in a new process"], theme.ink2),
        ("Use", "after training", ["`StreamServer`", "NIR export"], theme.aqua),
    ]
    w, h, gap = 340, 176, 50
    places = [(30, 120), (30 + w + gap, 120), (30 + 2 * (w + gap), 120), (30 + w + gap, 350), (30, 350)]
    for (title, subtitle, items, color), (x, y) in zip(stages, places, strict=True):
        c.box(x, y, w, h, color=color, alpha=0.06, r=16)
        c.text(x + 24, y + 42, title, size=22, weight=600)
        c.text(x + 24, y + 70, subtitle, size=SMALL, fill=theme.muted)
        for k, item in enumerate(items):
            c.inline(x + 24, y + 112 + 30 * k, item, size=SMALL, fill=theme.ink2)
    for (x1, y1), (x2, y2) in pairwise(places):
        if y1 == y2:
            a, b = (x1 + w + 6, x2 - 8) if x2 > x1 else (x1 - 6, x2 + w + 8)
            c.path(f"M{a},{y1 + h / 2} L{b},{y2 + h / 2}", stroke=theme.ink2, width=1.8, arrow=theme.ink2)
        else:
            c.path(f"M{x1 + w / 2},{y1 + h + 6} L{x1 + w / 2},{y2 + h / 2} L{x2 + w + 8},{y2 + h / 2}",
                   stroke=theme.ink2, width=1.8, arrow=theme.ink2)
    return c


FIGURES: dict[str, tuple[Callable[[Theme, dict], Canvas], str, str]] = {
    "banner": (banner, "sparx",
               "The sparx wordmark beside a raster of spikes from a simulated Brunel network and the "
               "membrane of one leaky integrate-and-fire neuron that resets at each spike."),
    "over_time": (over_time, "A spiking layer over time",
                  "A layer is a scan of one step function over time; a leaky integrate-and-fire neuron's "
                  "membrane rises to threshold and resets at each spike; the spike is a step whose gradient "
                  "is replaced by a smooth surrogate slope."),
    "recurrence": (recurrence, "One recurrent cell, any wiring",
                   "A recurrent cell adds what arrives through its wiring to the input, runs its inner "
                   "model, and sends the output back through a dense, sparse or delayed wiring, optionally "
                   "plus fast weights from a Hebbian trace."),
    "learning": (learning, "Ways a spiking network learns",
                 "Nine ways sparx trains a network, each with a schematic of where its learning signal comes "
                 "from: backpropagation through time, e-prop, OTTT, exact spike-time gradients, predictive "
                 "coding with PC-ALM, REINFORCE, fast weights, reward diffusion and ANN conversion."),
    "circuits": (circuits, "A circuit, declared and simulated",
                 "Two populations of neurons, excitatory and inhibitory, connected by projections with "
                 "delays and driven by Poisson input, simulated in compiled chunks with checkpoints and "
                 "monitors."),
    "training": (training, "Training on dew",
                 "A Flax model and a sparx objective go to dew's trainer, which writes a run record that "
                 "reloads, serves streams and exports to NIR."),
    "architecture": (architecture, "How sparx fits together",
                     "Training tools on the left and simulation tools on the right both run neuron models "
                     "through one protocol, init_state and step, which dimensionless cells and physical "
                     "models alike implement."),
}


def main() -> None:
    names = sys.argv[1:] or list(FIGURES)
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"raster": brunel_raster() if "banner" in names else None, "membrane": lif_trace()}
    for name in names:
        draw, title, description = FIGURES[name]
        for theme in THEMES.values():
            path = OUT / f"{name}-{theme.name}.svg"
            path.write_text(draw(theme, data).svg(title=title, description=description))
            print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
