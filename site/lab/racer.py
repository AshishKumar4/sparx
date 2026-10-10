"""The front page's racer: a spiking network that drives a car around any track, seeing only events.

The car carries a simulated event camera. Each of its 24 x 12 pixels looks at a point on the ground ahead
and fires an ON or OFF event when the logarithm of the brightness there has risen or fallen by `threshold`
since that pixel's last event, as a dynamic vision sensor does; a still view sends nothing. The network
reads the events and the car's speed, and its two readout membranes steer and set the speed.

It learns end to end by gradient descent, with no driver to imitate: through the car, the camera, the
events (each pixel's threshold is sparx's spike, so its surrogate's slope stands in for the step's), and
the network's own spikes. The loss rewards progress along the track and penalizes distance from its
middle.

    python site/lab/racer.py train --out site/public/racer
    python site/lab/racer.py evaluate
    python site/lab/racer.py difficulty
    python site/lab/racer.py record --out site/test/fixtures/racer.json
    python site/lab/racer.py fixture --neuron lif --sees events

The browser runs the same car, camera and network from racer.json (site/src/engines/racer.ts).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import training
from stack import (
    Vision,
    act,
    at_rest,
    from_layers,
    stack,
    stack_fanouts,
    to_layers,
    vision_from_json,
    vision_json,
)

import sparx
from sparx.surrogate import ATan


@dataclasses.dataclass(frozen=True)
class World:
    """The car, the road and the camera, in metres, seconds and radians."""

    dt: float = 0.02
    wheelbase: float = 0.4
    max_steer: float = 0.6
    max_speed: float = 6.0
    speed_tau: float = 0.5
    half_width: float = 0.8
    edge: float = 0.12
    dash_period: float = 1.5
    dash_width: float = 0.08
    threshold: float = 0.15
    columns: int = 24
    rows: int = 12
    near: float = 0.6
    far: float = 7.0
    spread: float = 0.9
    points: int = 128
    behind: int = 4
    ahead: int = 24
    supersample: int = 1
    sees: str = "events"

    @property
    def pixels(self) -> int:
        return self.columns * self.rows

    @property
    def features(self) -> int:
        """What the network reads each step: ON and OFF events, or a frame's log brightness, and the speed."""
        return (2 if self.sees == "events" else 1) * self.pixels + 1

    def ground(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the camera looks, in the car's frame: metres ahead and to the left, `[rows * columns]`, or
        `supersample` squared points per pixel, row-major over a grid that much finer, which `look` averages.

        Rows step geometrically from `near` to `far`, as a camera's rows do over a flat ground, and each
        row spans `spread` times its distance to either side.
        """
        rows, columns = self.rows * self.supersample, self.columns * self.supersample
        forward = self.near * (self.far / self.near) ** (np.arange(rows) / (rows - 1))
        side = self.spread * (2 * np.arange(columns) / (columns - 1) - 1)
        return np.repeat(forward, columns), (forward[:, None] * side[None, :]).reshape(-1)


EVENT = ATan(alpha=8.0)
"""The surrogate of each pixel's threshold, in log brightness: narrower than a neuron's, since the
threshold is 0.15."""


def track_tables(points: jax.Array):
    """A closed track's segments: starts, vectors, squared lengths, and the distance along the track to
    each start. `points` `[..., N, 2]`."""
    vectors = jnp.roll(points, -1, axis=-2) - points
    lengths2 = jnp.sum(vectors**2, -1)
    lengths = jnp.sqrt(lengths2)
    along = jnp.cumsum(lengths, -1) - lengths
    return vectors, lengths2, lengths, along


def locate(world: World, points: jax.Array, tables, x: jax.Array, y: jax.Array, k0: jax.Array):
    """The distance from each point `(x, y)` to the track's middle, the distance along the track there, and
    the direction of the nearest segment, over a window of segments around `k0`. `x`, `y` `[B, ...]`."""
    vectors, lengths2, lengths, along = tables
    n = points.shape[-2]
    window = (k0[:, None] + jnp.arange(-world.behind, world.ahead)) % n
    take = lambda a: jnp.take_along_axis(a, window if a.ndim == 2 else window[..., None], axis=1)  # noqa: E731
    a, d, l2, ln, s0 = take(points), take(vectors), take(lengths2), take(lengths), take(along)
    extra = x.ndim - 1
    shape = (a.shape[0],) + (1,) * extra + (a.shape[1],)
    rx = x[..., None] - a[..., 0].reshape(shape)
    ry = y[..., None] - a[..., 1].reshape(shape)
    dx, dy = d[..., 0].reshape(shape), d[..., 1].reshape(shape)
    t = jnp.clip((rx * dx + ry * dy) / l2.reshape(shape), 0.0, 1.0)
    qx, qy = rx - t * dx, ry - t * dy
    d2 = qx * qx + qy * qy
    best = jnp.argmin(d2, -1)
    pick = lambda v: jnp.take_along_axis(v, best[..., None], -1)[..., 0]  # noqa: E731
    distance = jnp.sqrt(pick(d2) + 1e-6)
    s = pick(s0.reshape(shape) + t * ln.reshape(shape))
    heading = jnp.arctan2(pick(jnp.broadcast_to(dy, d2.shape)), pick(jnp.broadcast_to(dx, d2.shape)))
    return distance, s, heading


def nearest(points: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    """The index of the track point nearest each car, `[B]`."""
    return jnp.argmin((points[..., 0] - x[:, None]) ** 2 + (points[..., 1] - y[:, None]) ** 2, -1)


def brightness(world: World, distance: jax.Array, s: jax.Array, dashes: jax.Array) -> jax.Array:
    """The ground's brightness: dark road, light verge with a soft edge, dashes down the middle."""
    verge = jax.nn.sigmoid((distance - world.half_width) / world.edge)
    period = dashes.reshape(dashes.shape + (1,) * (s.ndim - 1))
    dash = jax.nn.sigmoid((world.dash_width - distance) / 0.03) * jax.nn.sigmoid(
        6.0 * jnp.sin(2 * jnp.pi * s / period)
    )
    return 0.12 + 0.7 * verge + 0.6 * dash * (1 - verge)


def look(world: World, ground, points, tables, dashes, car: jax.Array) -> jax.Array:
    """The log brightness each pixel sees from `car` `[B, 4]` (x, y, heading, speed), the mean over its
    points of the ground: `[B, pixels]`."""
    forward, side = ground
    x, y, psi = car[:, 0], car[:, 1], car[:, 2]
    c, s = jnp.cos(psi)[:, None], jnp.sin(psi)[:, None]
    px = x[:, None] + forward * c - side * s
    py = y[:, None] + forward * s + side * c
    distance, along, _ = locate(world, points, tables, px, py, nearest(points, x, y))
    light, n = brightness(world, distance, along, dashes), world.supersample
    if n > 1:
        light = light.reshape(-1, world.rows, n, world.columns, n).mean((2, 4)).reshape(light.shape[0], -1)
    return jnp.log(light)


def sense(world: World, level: jax.Array, seen: jax.Array):
    """One step of each pixel: an ON event where the log brightness has risen `threshold` above the level of
    its last event, OFF where it has fallen as far; each event moves the level by the threshold."""
    change = seen - level
    on = sparx.spike(change - world.threshold, EVENT)
    off = sparx.spike(-change - world.threshold, EVENT)
    return level + world.threshold * (on - off), on, off


def drive(world: World, car: jax.Array, u: jax.Array) -> jax.Array:
    """One step of the car from the readout's membranes: steering, then speed toward its target."""
    x, y, psi, v = car[:, 0], car[:, 1], car[:, 2], car[:, 3]
    steer = world.max_steer * jnp.tanh(u[:, 0])
    target = world.max_speed * jax.nn.sigmoid(u[:, 1])
    v = v + world.dt * (target - v) / world.speed_tau
    psi = psi + world.dt * v * jnp.tan(steer) / world.wheelbase
    x = x + world.dt * v * jnp.cos(psi)
    y = y + world.dt * v * jnp.sin(psi)
    return jnp.stack([x, y, psi, v], -1)


def network(hidden: tuple[int, ...] = (96, 64), tau: float = 3.0, readout_tau: float = 4.0):
    return stack((*hidden, 2), tau, readout_tau)


def vision(world: World, neuron: str = "lif", tau: float = 3.0, readout_tau: float = 4.0, bits: int = 1,
           delta: float = 0.1, branches: int = 4) -> Vision:
    """The convolutional network over the camera's image: ON and OFF events, or a frame."""
    channels = 2 if world.sees == "events" else 1
    return Vision((world.rows, world.columns, channels), neuron=neuron, tau=tau, readout_tau=readout_tau,
                  bits=bits, delta=delta, branches=branches)


def inputs(world: World, on: jax.Array, off: jax.Array, seen: jax.Array, car: jax.Array) -> jax.Array:
    image = [on, off] if world.sees == "events" else [seen]
    return jnp.concatenate([*image, car[:, 3:4] / world.max_speed], -1)


def race(net, params, world: World, ground, points, car0, steps: int, truncate: int = 0,
         nudge: tuple[int, float] | None = None):
    """Drive each car of `car0` `[B, 4]` around its track `points` `[B, N, 2]` for `steps` steps. With
    `truncate`, gradients flow back through at most that many steps: the state is cut from its past at
    every multiple of it. A `nudge` of `(step, metres)` moves every car sideways, to its left, by `metres`
    at the start of `step`."""
    tables = track_tables(points)
    dashes = jnp.maximum(jnp.round(tables[2].sum(-1) / world.dash_period), 1) / 1.0
    dashes = tables[2].sum(-1) / dashes
    batch = car0.shape[0]

    def step(carry, t):
        if truncate:
            cut = t % truncate == 0
            carry = jax.tree.map(lambda c: jnp.where(cut, jax.lax.stop_gradient(c), c), carry)
        car, level, carried = carry
        if nudge is not None:
            left = jnp.stack([-jnp.sin(car[:, 2]), jnp.cos(car[:, 2])], -1)
            car = jnp.where(t == nudge[0], car.at[:, :2].add(nudge[1] * left), car)
        seen = look(world, ground, points, tables, dashes, car)
        level, on, off = sense(world, level, seen)
        u, carried, spikes = act(net, params, carried, inputs(world, on, off, seen, car))
        car = drive(world, car, u)
        distance, s, heading = locate(
            world, points, tables, car[:, 0], car[:, 1], nearest(points, car[:, 0], car[:, 1])
        )
        along = car[:, 3] * jnp.cos(car[:, 2] - heading)
        return (car, level, carried), (car, u, on, off, spikes, distance, s, along)

    level0 = look(world, ground, points, tables, dashes, car0)
    carried = at_rest(net, params, batch, world.features)
    _, out = jax.lax.scan(jax.checkpoint(step), (car0, level0, carried), jnp.arange(steps))
    return out


@dataclasses.dataclass(frozen=True)
class Tracks:
    """Closed tracks drawn at random: a radius of 6 to 10 m bent by four harmonics, stretched, and run
    either way round. `bend` scales the harmonics; with a `floor`, a track's radius never falls below that
    fraction of its mean, so it stays a simple loop however hard it bends."""

    radius: tuple[float, float] = (6.0, 10.0)
    bend: float = 0.9
    stretch: tuple[float, float] = (0.7, 1.3)
    floor: float = 0.0

    def draw(self, world: World, key: jax.Array, batch: int, bend: jax.Array | float | None = None,
             ) -> jax.Array:
        ks = jax.random.split(key, 6)
        theta = 2 * jnp.pi * jnp.arange(world.points) / world.points
        k = jnp.arange(2, 6)
        amplitude = jax.random.uniform(ks[0], (batch, 4)) * (self.bend if bend is None else bend) / k**1.5
        phase = jax.random.uniform(ks[1], (batch, 4), maxval=2 * jnp.pi)
        r = 1 + jnp.sum(amplitude[:, :, None] * jnp.cos(k[None, :, None] * theta + phase[:, :, None]), 1)
        if self.floor:
            dip = 1 - jnp.min(r, -1, keepdims=True)
            r = 1 + (r - 1) * jnp.minimum(1.0, (1 - self.floor) / jnp.maximum(dip, 1e-6))
        radius = jax.random.uniform(ks[2], (batch, 1), minval=self.radius[0], maxval=self.radius[1])
        stretch = jax.random.uniform(ks[3], (batch, 1), minval=self.stretch[0], maxval=self.stretch[1])
        sign = jnp.where(jax.random.uniform(ks[4], (batch, 1)) < 0.5, 1.0, -1.0)
        points = jnp.stack(
            [radius * stretch * r * jnp.cos(theta), sign * radius / stretch * r * jnp.sin(theta)], -1
        )
        rotate = jax.random.uniform(ks[5], (batch,), maxval=2 * jnp.pi)
        c, s = jnp.cos(rotate)[:, None], jnp.sin(rotate)[:, None]
        return jnp.stack(
            [c * points[..., 0] - s * points[..., 1], s * points[..., 0] + c * points[..., 1]], -1
        )


def starts(world: World, key: jax.Array, points: jax.Array, scatter: float = 1.0) -> jax.Array:
    """A car on each track at a random point, off the middle and askew by up to `scatter` of the usual."""
    ks = jax.random.split(key, 4)
    batch, n = points.shape[:2]
    k = jax.random.randint(ks[0], (batch,), 0, n)
    a = points[jnp.arange(batch), k]
    d = points[jnp.arange(batch), (k + 1) % n] - a
    heading = jnp.arctan2(d[:, 1], d[:, 0])
    offset = scatter * jax.random.uniform(ks[1], (batch,), minval=-0.4, maxval=0.4)
    x = a[:, 0] - offset * jnp.sin(heading)
    y = a[:, 1] + offset * jnp.cos(heading)
    psi = heading + scatter * 0.25 * jax.random.normal(ks[2], (batch,))
    v = scatter * jax.random.uniform(ks[3], (batch,), maxval=4.0)
    return jnp.stack([x, y, psi, v], -1)


def cost(world: World, out, rates: tuple[float, float], reverse: float = 0.0):
    _, _, on, off, spikes, distance, _, along = out
    lateral = (distance / world.half_width) ** 2
    beyond = jax.nn.relu(distance - world.half_width) ** 2
    rate = jnp.mean(spikes, axis=(0, 1))
    band = jnp.mean(jax.nn.relu(rate - rates[1]) ** 2 + jax.nn.relu(rates[0] - rate) ** 2)
    progress = jnp.mean(along) / world.max_speed
    backward = jnp.mean(jax.nn.relu(-along)) / world.max_speed
    loss = jnp.mean(lateral) + 4.0 * jnp.mean(beyond) - 0.5 * progress + 10.0 * band + reverse * backward
    return loss, {
        "distance": jnp.mean(distance),
        "speed": jnp.mean(along),
        "rate": jnp.mean(rate),
        "events": jnp.mean(on + off) * world.pixels,
    }


def model_json(net, params, world: World, meta: dict) -> dict:
    forward, side = world.ground()
    if isinstance(net, Vision):
        weights = {"vision": vision_json(net, params)}
    else:
        weights = {"layers": to_layers(net, params)}
    return {
        "world": dataclasses.asdict(world),
        "ground": {"forward": forward.tolist(), "side": side.tolist()},
        **weights,
        "meta": meta,
    }


def from_json(model: dict):
    net, params = vision_from_json(model["vision"]) if "vision" in model else from_layers(model["layers"])
    world = World(**model["world"])
    ground = (np.asarray(model["ground"]["forward"]), np.asarray(model["ground"]["side"]))
    return net, params, world, ground


def train(args: argparse.Namespace) -> None:
    world = World(columns=args.columns, rows=args.rows, supersample=args.supersample, sees=args.sees)
    tracks = Tracks()
    ground = tuple(jnp.asarray(g, jnp.float32) for g in world.ground())
    if args.net == "conv":
        net = vision(world, args.neuron, args.tau, args.readout_tau, args.bits, args.delta, args.branches)
    else:
        net = network(tuple(args.hidden), args.tau, args.readout_tau)
    params = net.init(jax.random.key(args.seed), jnp.zeros((1, 1, world.features)))["params"]
    if args.curriculum:
        easy, hard = args.curriculum
        tracks = Tracks(bend=hard, floor=args.floor)

    def loss_fn(params, key, progress):
        k1, k2 = jax.random.split(key)
        # With a curriculum, the bends grow from `easy` to `hard` over the first half of the run.
        bend = easy + (hard - easy) * jnp.minimum(1.0, 2 * progress) if args.curriculum else None
        points = tracks.draw(world, k1, args.batch, bend)
        car0 = starts(world, k2, points)
        out = race(net, params, world, ground, points, car0, args.horizon, args.truncate)
        return cost(world, out, (args.rate_low, args.rate_high), args.reverse)

    params, history = training.train(loss_fn, params, lr=args.lr, steps=args.steps, seed=args.seed + 1,
                                     log_every=args.log_every)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "trained": {k: v for k, v in vars(args).items() if k not in ("func", "out")},
        "history": history,
        "backend": jax.default_backend(),
        "jax": jax.__version__,
        "sparx": sparx.__version__,
    }
    model = model_json(net, params, world, meta)
    (out / "racer.json").write_text(json.dumps(model))
    points = Tracks().draw(world, jax.random.key(args.seed + 1000), args.eval_tracks)
    summary = evaluation(net, params, world, ground, points)
    (out / "evaluation.json").write_text(json.dumps(summary))
    print(json.dumps(summary["summary"]))
    if args.difficulty:
        harder = difficulty(net, params, world, ground, args.seed + 2000, args.eval_tracks)
        (out / "difficulty.json").write_text(json.dumps(harder))
        print(json.dumps(harder["by_bend"]))


NUDGE = (100, 0.3)
"""The response probe's nudge: 2 s into the drive, every car moved 0.3 m to its left."""
WINDOW = 50
"""The steps after the nudge over which the response is measured."""


def drives(net, params, world: World, ground, points: jax.Array, car0: jax.Array, steps: int,
           nudge: tuple[int, float] | None = None, chunk: int = 25) -> dict[str, np.ndarray]:
    """`race` on each track of `points`, `chunk` tracks at a time, reduced on the device to what an
    evaluation reads, each `[T, B]` unless noted: the cars every fifth step `[T / 5, B, 4]`, the steering
    after the nudge would come `[WINDOW, B]`, the distance from the middle and the speed along the road,
    the events, the units not silent, and the multiply-adds those inputs and units trigger (`fanouts`)."""
    image, units, extra = (net.fanouts() if isinstance(net, Vision)
                           else stack_fanouts(net, (2 if world.sees == "events" else 1) * world.pixels))
    extras = world.features - image.size
    image, units = jnp.asarray(image, jnp.float32), jnp.asarray(units, jnp.float32)
    ground = tuple(jnp.asarray(g) for g in ground)
    at = NUDGE[0]

    @jax.jit
    def run(p, pts, c):
        cars, u, on, off, spikes, distance, _, along = race(net, p, world, ground, pts, c, steps, nudge=nudge)
        frame = jnp.ones((*on.shape[:2], image.size))  # every pixel's value, every step
        sent = jnp.concatenate([on, off], -1) if world.sees == "events" else frame
        active = (spikes != 0).astype(jnp.float32)
        return {
            "cars": cars[::5],
            "steer": jnp.tanh(u[at:at + WINDOW, :, 0]) if steps >= at + WINDOW else u[:0, :, 0],
            "distance": distance,
            "along": along,
            "events": jnp.sum(on + off, -1),
            "active": jnp.sum(active, -1),
            "triggered": (sent != 0).astype(jnp.float32) @ image + active @ units + extras * extra,
        }

    parts = [jax.device_get(run(params, points[k:k + chunk], car0[k:k + chunk]))
             for k in range(0, points.shape[0], chunk)]
    out = {name: np.concatenate([part[name] for part in parts], 1) for name in parts[0]}
    out["units"], out["dense"] = units.size, int(image.sum() + units.sum()) + extras * extra
    return out


def evaluation(net, params, world: World, ground, points: jax.Array, seconds: float = 30.0) -> dict:
    """Each car starts in the middle of an unseen track of `points`, at rest, pointing along it, and drives
    for `seconds`. A lap counts when the car has gone the track's length forward without leaving the road
    by more than a metre; a car that does leave it that far has failed. A drive long enough to include
    the response probe's window (`response`) runs it too."""
    steps = round(seconds / world.dt)
    tracks = points.shape[0]
    d = points[:, 1] - points[:, 0]
    car0 = jnp.stack([points[:, 0, 0], points[:, 0, 1], jnp.arctan2(d[:, 1], d[:, 0]), jnp.zeros(tracks)], -1)
    out = drives(net, params, world, ground, points, car0, steps)
    length = np.asarray(track_tables(points)[2].sum(-1))
    covered = np.cumsum(out["along"] * world.dt, 0)
    distance = out["distance"]
    crashed = np.maximum.accumulate(distance > world.half_width + 1.0, 0)
    lap = (covered >= length) & ~crashed
    finished = lap.any(0)
    lap_time = np.where(finished, np.argmax(lap, 0) + 1, -1) * world.dt
    off_road = (distance > world.half_width).mean(0)
    bend = tightest(np.asarray(points))
    parameters = sum(int(np.size(leaf)) for leaf in jax.tree.leaves(params))
    latency = {}
    if steps >= NUDGE[0] + WINDOW:
        # The cars still running when the nudge comes, at the end of the step before it.
        running = ~crashed[NUDGE[0] - 1]
        latency = response(net, params, world, ground, points, car0, out["steer"], running)
    per_track = [
        {
            "finished": bool(f),
            "lap_time": float(t) if f else None,
            "off_road": float(o),
            "crashed": bool(crashed[-1, i]),
            "length": float(length[i]),
            "covered": float(covered[-1, i]),
            "tightest": float(bend[i]),
        }
        for i, (f, t, o) in enumerate(zip(finished, lap_time, off_road, strict=True))
    ]
    worst = int(np.argmax(np.where(finished, -1.0, 1.0) * 1000 + off_road))
    return {
        "summary": {
            "tracks": tracks,
            "seconds": seconds,
            "finished": float(finished.mean()),
            "crashed": float(crashed[-1].mean()),
            "median_lap_time": float(np.median(lap_time[finished])) if finished.any() else None,
            "median_speed": float(np.median(out["along"][:, finished].mean(0))) if finished.any() else None,
            "off_road": float(off_road.mean()),
            "events_per_step": float(out["events"].mean()),
            "spikes_per_step": float(out["active"].mean()),
            "units": out["units"],
            "parameters": parameters,
            "dense_macs_per_step": out["dense"],
            "triggered_macs_per_step": float(out["triggered"].mean()),
            **latency,
        },
        "tracks": per_track,
        "worst": {
            "index": worst,
            "points": np.asarray(points[worst]).round(4).tolist(),
            "path": out["cars"][:, worst, :2].round(3).tolist(),
            **per_track[worst],
        },
    }


def response(net, params, world: World, ground, points: jax.Array, car0: jax.Array, steer: np.ndarray,
             running: np.ndarray) -> dict:
    """How fast the network steers back when its car is moved sideways (`NUDGE`): on each track whose car is
    still running at the nudge, the steering's difference from the same drive unnudged, whose steering
    over the `WINDOW` steps from the nudge is `steer` `[WINDOW, B]`. Its onset is the time until that
    difference first reaches a tenth of its largest, and its response the time until it reaches half: the
    medians over the tracks that responded, in ms, and how many did."""
    nudged = drives(net, params, world, ground, points, car0, NUDGE[0] + WINDOW, nudge=NUDGE)["steer"]
    turn = nudged - steer
    peak = np.abs(turn).max(0)
    responded = running & (peak > 1e-3)

    def median_ms(share: float) -> float | None:
        first = np.argmax(np.abs(turn) >= share * peak, 0)
        return float(np.median(first[responded]) * world.dt * 1000) if responded.any() else None

    return {
        "onset_ms": median_ms(0.1),
        "response_ms": median_ms(0.5),
        "responded": int(responded.sum()),
        "nudged": int(running.sum()),
        "nudge_m": NUDGE[1],
    }


def tightest(points: np.ndarray) -> np.ndarray:
    """Each closed track's tightest bend, as the radius in metres of the circle that fits it there: from the
    turn between neighbouring segments over their mean length, averaged over three points. `[B, N, 2]`."""
    vectors = np.roll(points, -1, axis=1) - points
    lengths = np.linalg.norm(vectors, axis=-1)
    heading = np.arctan2(vectors[..., 1], vectors[..., 0])
    turn = np.angle(np.exp(1j * (heading - np.roll(heading, 1, axis=1))))
    curvature = turn / (0.5 * (lengths + np.roll(lengths, 1, axis=1)))
    smooth = (np.roll(curvature, -1, axis=1) + curvature + np.roll(curvature, 1, axis=1)) / 3
    return 1 / np.abs(smooth).max(1)


BENDS = (0.9, 1.2, 1.5, 1.8)
"""The bends of the harder sets: the training tracks' 0.9, then tighter, each with a floor of 0.35."""

RADII = (0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, np.inf)
"""The bins of tightest-bend radius (m) the harder sets are reported in."""


def difficulty(net, params, world: World, ground, seed: int, tracks: int = 200) -> dict:
    """The network on `tracks` unseen tracks at each of `BENDS`, and every track's result by the radius of its
    tightest bend."""
    rows, by_bend = [], {}
    for k, bend in enumerate(BENDS):
        points = Tracks(bend=bend, floor=0.35).draw(world, jax.random.key(seed + k), tracks)
        result = evaluation(net, params, world, ground, points)
        by_bend[str(bend)] = result["summary"]
        rows += [{"bend": bend, **row} for row in result["tracks"]]
    radius = np.array([row["tightest"] for row in rows])
    finished = np.array([row["finished"] for row in rows])
    by_radius = []
    for lo, hi in itertools.pairwise(RADII):
        inside = (radius >= lo) & (radius < hi)
        if inside.any():
            by_radius.append({"from": lo, "to": None if np.isinf(hi) else hi, "tracks": int(inside.sum()),
                              "finished": float(finished[inside].mean())})
    return {"by_bend": by_bend, "by_radius": by_radius, "tracks": rows}


def evaluate(args: argparse.Namespace) -> None:
    net, params, world, ground = from_json(json.loads(Path(args.model).read_text()))
    points = Tracks().draw(world, jax.random.key(args.seed), args.tracks)
    summary = evaluation(net, params, world, ground, points)
    Path(args.out).write_text(json.dumps(summary))
    print(json.dumps(summary["summary"]))


def harder(args: argparse.Namespace) -> None:
    net, params, world, ground = from_json(json.loads(Path(args.model).read_text()))
    result = difficulty(net, params, world, ground, args.seed, args.tracks)
    out = Path(args.out)
    (out / "difficulty.json" if out.is_dir() else out).write_text(json.dumps(result))
    print(json.dumps({"by_bend": result["by_bend"], "by_radius": result["by_radius"]}))


def laps(model: dict, seed: int, steps: int) -> dict:
    """Two laps' worth of driving by `model` in float64 for the browser's engine to match, on two unseen
    tracks, from a scattered start."""
    with jax.enable_x64(new_val=True):
        net, params, world, ground = from_json(model)
        params = jax.tree.map(lambda p: jnp.asarray(p, jnp.float64), params)
        ground = tuple(jnp.asarray(g, jnp.float64) for g in ground)
        points = Tracks().draw(world, jax.random.key(seed), 2).astype(jnp.float64)
        car0 = starts(world, jax.random.key(seed + 1), points)
        cars, us, on, off, spikes, *_ = race(net, params, world, ground, points, car0, steps)
        driven = []
        for b in range(2):
            events = np.concatenate([np.asarray(on[:, b]), np.asarray(off[:, b])], -1)
            driven.append(
                {
                    "points": np.asarray(points[b]).tolist(),
                    "start": np.asarray(car0[b]).tolist(),
                    "cars": np.asarray(cars[:, b]).tolist(),
                    "readout": np.asarray(us[:, b]).tolist(),
                    "events": [np.flatnonzero(row).tolist() for row in events],
                    "spikes": [np.flatnonzero(row).tolist() for row in np.asarray(spikes[:, b])],
                }
            )
    return {"dtype": "float64", "jax": jax.__version__, "laps": driven}


def record(args: argparse.Namespace) -> None:
    """The published network's laps (`laps`)."""
    model = json.loads(Path(args.model).read_text())
    Path(args.out).write_text(json.dumps(laps(model, args.seed, args.steps)))


def fixture(args: argparse.Namespace) -> None:
    """A small convolutional racer at its random start, with its laps (`laps`): what the browser's engine
    must match for the networks the published racer does not use, written to `<out>/racer-conv-<neuron>-
    <sees>.json`."""
    world = World(columns=16, rows=8, supersample=2, sees=args.sees)
    net = vision(world, args.neuron)
    params = net.init(jax.random.key(args.seed), jnp.zeros((1, 1, world.features)))["params"]
    model = json.loads(json.dumps(model_json(net, params, world, {"fixture": True})))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    driven = laps(model, args.seed, args.steps)
    (out / f"racer-conv-{args.neuron}-{args.sees}.json").write_text(json.dumps({"model": model, **driven}))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(required=True)
    t = commands.add_parser("train")
    t.add_argument("--out", default="site/public/racer")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--horizon", type=int, default=150)
    t.add_argument("--hidden", type=int, nargs="+", default=[96, 64])
    t.add_argument("--tau", type=float, default=3.0)
    t.add_argument("--readout-tau", type=float, default=4.0)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--rate-low", type=float, default=0.02)
    t.add_argument("--rate-high", type=float, default=0.3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--eval-tracks", type=int, default=200)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--truncate", type=int, default=0, help="gradients reach back at most this many steps")
    t.add_argument("--curriculum", type=float, nargs=2, metavar=("EASY", "HARD"),
                   help="bend the training tracks from EASY to HARD over the first half of the run")
    t.add_argument("--floor", type=float, default=0.35, help="with a curriculum, the tracks' radius floor")
    t.add_argument("--reverse", type=float, default=0.0,
                   help="the weight of the penalty on driving backwards")
    t.add_argument("--difficulty", action="store_true", help="also score the harder sets (difficulty.json)")
    t.add_argument("--net", choices=("dense", "conv"), default="dense")
    t.add_argument("--neuron", choices=("lif", "relu", "sigma-delta", "dendritic"), default="lif",
                   help="the conv network's units")
    t.add_argument("--bits", type=int, default=1, help="lif: the bits each spike carries")
    t.add_argument("--delta", type=float, default=0.1, help="sigma-delta: the smallest change sent")
    t.add_argument("--branches", type=int, default=4, help="dendritic: the branches of each neuron")
    t.add_argument("--sees", choices=("events", "frames"), default="events")
    t.add_argument("--columns", type=int, default=24)
    t.add_argument("--rows", type=int, default=12)
    t.add_argument("--supersample", type=int, default=1, help="ground points per pixel, squared")
    t.set_defaults(func=train)
    e = commands.add_parser("evaluate")
    e.add_argument("--model", default="site/public/racer/racer.json")
    e.add_argument("--out", default="site/public/racer/evaluation.json")
    e.add_argument("--tracks", type=int, default=200)
    e.add_argument("--seed", type=int, default=1000)
    e.set_defaults(func=evaluate)
    h = commands.add_parser("difficulty")
    h.add_argument("--model", default="site/public/racer/racer.json")
    h.add_argument("--out", default="site/public/racer/difficulty.json")
    h.add_argument("--tracks", type=int, default=200)
    h.add_argument("--seed", type=int, default=2000)
    h.set_defaults(func=harder)
    r = commands.add_parser("record")
    r.add_argument("--model", default="site/public/racer/racer.json")
    r.add_argument("--out", default="site/test/fixtures/racer.json")
    r.add_argument("--steps", type=int, default=500)
    r.add_argument("--seed", type=int, default=31)
    r.set_defaults(func=record)
    f = commands.add_parser("fixture")
    f.add_argument("--out", default="site/test/fixtures")
    f.add_argument("--neuron", choices=("lif", "relu"), default="lif")
    f.add_argument("--sees", choices=("events", "frames"), default="events")
    f.add_argument("--steps", type=int, default=300)
    f.add_argument("--seed", type=int, default=41)
    f.set_defaults(func=fixture)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
