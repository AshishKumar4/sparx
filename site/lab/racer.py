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
    python site/lab/racer.py record --out site/test/fixtures/racer.json

The browser runs the same car, camera and network from racer.json (site/src/engines/racer.ts).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from stack import act, at_rest, from_layers, stack, to_layers

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

    @property
    def pixels(self) -> int:
        return self.columns * self.rows

    def ground(self) -> tuple[np.ndarray, np.ndarray]:
        """Where each pixel looks, in the car's frame: metres ahead and to the left, `[rows * columns]`.

        Rows step geometrically from `near` to `far`, as a camera's rows do over a flat ground, and each
        row spans `spread` times its distance to either side.
        """
        forward = self.near * (self.far / self.near) ** (np.arange(self.rows) / (self.rows - 1))
        side = self.spread * (2 * np.arange(self.columns) / (self.columns - 1) - 1)
        return np.repeat(forward, self.columns), (forward[:, None] * side[None, :]).reshape(-1)


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
    """The log brightness each pixel sees from `car` `[B, 4]` (x, y, heading, speed): `[B, pixels]`."""
    forward, side = ground
    x, y, psi = car[:, 0], car[:, 1], car[:, 2]
    c, s = jnp.cos(psi)[:, None], jnp.sin(psi)[:, None]
    px = x[:, None] + forward * c - side * s
    py = y[:, None] + forward * s + side * c
    distance, along, _ = locate(world, points, tables, px, py, nearest(points, x, y))
    return jnp.log(brightness(world, distance, along, dashes))


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


def inputs(world: World, on: jax.Array, off: jax.Array, car: jax.Array) -> jax.Array:
    return jnp.concatenate([on, off, car[:, 3:4] / world.max_speed], -1)


def race(net, params, world: World, ground, points, car0, steps: int, noise: float = 0.0, key=None):
    """Drive each car of `car0` `[B, 4]` around its track `points` `[B, N, 2]` for `steps` steps."""
    tables = track_tables(points)
    dashes = jnp.maximum(jnp.round(tables[2].sum(-1) / world.dash_period), 1) / 1.0
    dashes = tables[2].sum(-1) / dashes
    batch = car0.shape[0]

    def step(carry, t):
        car, level, carried = carry
        seen = look(world, ground, points, tables, dashes, car)
        level, on, off = sense(world, level, seen)
        u, carried, spikes = act(net, params, carried, inputs(world, on, off, car))
        car = drive(world, car, u)
        distance, s, heading = locate(
            world, points, tables, car[:, 0], car[:, 1], nearest(points, car[:, 0], car[:, 1])
        )
        along = car[:, 3] * jnp.cos(car[:, 2] - heading)
        return (car, level, carried), (car, u, on, off, spikes, distance, s, along)

    level0 = look(world, ground, points, tables, dashes, car0)
    carried = at_rest(net, params, batch, 2 * world.pixels + 1)
    _, out = jax.lax.scan(jax.checkpoint(step), (car0, level0, carried), jnp.arange(steps))
    return out


@dataclasses.dataclass(frozen=True)
class Tracks:
    """Closed tracks drawn at random: a radius of 6 to 10 m bent by four harmonics, stretched, and run
    either way round."""

    radius: tuple[float, float] = (6.0, 10.0)
    bend: float = 0.9
    stretch: tuple[float, float] = (0.7, 1.3)

    def draw(self, world: World, key: jax.Array, batch: int) -> jax.Array:
        ks = jax.random.split(key, 6)
        theta = 2 * jnp.pi * jnp.arange(world.points) / world.points
        k = jnp.arange(2, 6)
        amplitude = jax.random.uniform(ks[0], (batch, 4)) * self.bend / k**1.5
        phase = jax.random.uniform(ks[1], (batch, 4), maxval=2 * jnp.pi)
        r = 1 + jnp.sum(amplitude[:, :, None] * jnp.cos(k[None, :, None] * theta + phase[:, :, None]), 1)
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


def cost(world: World, out, rates: tuple[float, float]):
    _, _, on, off, spikes, distance, _, along = out
    lateral = (distance / world.half_width) ** 2
    beyond = jax.nn.relu(distance - world.half_width) ** 2
    rate = jnp.mean(spikes, axis=(0, 1))
    band = jnp.mean(jax.nn.relu(rate - rates[1]) ** 2 + jax.nn.relu(rates[0] - rate) ** 2)
    progress = jnp.mean(along) / world.max_speed
    loss = jnp.mean(lateral) + 4.0 * jnp.mean(beyond) - 0.5 * progress + 10.0 * band
    return loss, {
        "distance": jnp.mean(distance),
        "speed": jnp.mean(along),
        "rate": jnp.mean(rate),
        "events": jnp.mean(on + off) * world.pixels * 2,
    }


def model_json(net, params, world: World, meta: dict) -> dict:
    forward, side = world.ground()
    return {
        "world": dataclasses.asdict(world),
        "ground": {"forward": forward.tolist(), "side": side.tolist()},
        "layers": to_layers(net, params),
        "meta": meta,
    }


def from_json(model: dict):
    net, params = from_layers(model["layers"])
    world = World(**model["world"])
    ground = (np.asarray(model["ground"]["forward"]), np.asarray(model["ground"]["side"]))
    return net, params, world, ground


def train(args: argparse.Namespace) -> None:
    world, tracks = World(), Tracks()
    ground = tuple(jnp.asarray(g, jnp.float32) for g in world.ground())
    net = network(tuple(args.hidden), args.tau, args.readout_tau)
    params = net.init(jax.random.key(args.seed), jnp.zeros((1, 1, 2 * world.pixels + 1)))["params"]
    warmup = min(100, args.steps // 10)
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, args.steps, args.lr / 20)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)

    def loss_fn(params, key):
        k1, k2 = jax.random.split(key)
        points = tracks.draw(world, k1, args.batch)
        out = race(net, params, world, ground, points, starts(world, k2, points), args.horizon)
        return cost(world, out, (args.rate_low, args.rate_high))

    @jax.jit
    def update(params, opt_state, key):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux, optax.global_norm(grads)

    history, start = [], time.time()
    key = jax.random.key(args.seed + 1)
    for step in range(1, args.steps + 1):
        key, sub = jax.random.split(key)
        params, opt_state, loss, aux, norm = update(params, opt_state, sub)
        if step % args.log_every == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(loss),
                "grad_norm": float(norm),
                "seconds": round(time.time() - start, 1),
            }
            row |= {k: float(v) for k, v in aux.items()}
            history.append(row)
            print(json.dumps(row), flush=True)

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
    summary = evaluation(net, params, world, ground, args.seed + 1000, tracks=args.eval_tracks)
    (out / "evaluation.json").write_text(json.dumps(summary))
    print(json.dumps(summary["summary"]))


def evaluation(
    net, params, world: World, ground, seed: int, tracks: int = 200, seconds: float = 30.0
) -> dict:
    """Each car starts in the middle of an unseen track, at rest, pointing along it, and drives for `seconds`.
    A lap counts when the car has gone the track's length forward without leaving the road by more than a
    metre; a car that does leave it that far has failed."""
    steps = round(seconds / world.dt)
    points = Tracks().draw(world, jax.random.key(seed), tracks)
    d = points[:, 1] - points[:, 0]
    car0 = jnp.stack([points[:, 0, 0], points[:, 0, 1], jnp.arctan2(d[:, 1], d[:, 0]), jnp.zeros(tracks)], -1)
    run = jax.jit(lambda p, pts, c: race(net, p, world, tuple(jnp.asarray(g) for g in ground), pts, c, steps))
    cars, _, on, off, spikes, distance, _, along = run(params, points, car0)
    length = np.asarray(track_tables(points)[2].sum(-1))
    covered = np.cumsum(np.asarray(along) * world.dt, 0)
    distance = np.asarray(distance)
    crashed = np.maximum.accumulate(distance > world.half_width + 1.0, 0)
    lap = (covered >= length) & ~crashed
    finished = lap.any(0)
    lap_time = np.where(finished, np.argmax(lap, 0) + 1, -1) * world.dt
    off_road = (distance > world.half_width).mean(0)
    per_track = [
        {
            "finished": bool(f),
            "lap_time": float(t) if f else None,
            "off_road": float(o),
            "crashed": bool(crashed[-1, i]),
            "length": float(length[i]),
            "covered": float(covered[-1, i]),
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
            "median_speed": float(np.median(np.asarray(along)[:, finished].mean(0)))
            if finished.any()
            else None,
            "off_road": float(off_road.mean()),
            "events_per_step": float(np.mean(np.asarray(on) + np.asarray(off)) * world.pixels * 2),
            "spikes_per_step": float(np.mean(np.asarray(spikes)) * np.asarray(spikes).shape[-1]),
        },
        "tracks": per_track,
        "worst": {
            "index": worst,
            "points": np.asarray(points[worst]).round(4).tolist(),
            "path": np.asarray(cars[::5, worst, :2]).round(3).tolist(),
            **per_track[worst],
        },
    }


def evaluate(args: argparse.Namespace) -> None:
    net, params, world, ground = from_json(json.loads(Path(args.model).read_text()))
    summary = evaluation(net, params, world, ground, args.seed, tracks=args.tracks)
    Path(args.out).write_text(json.dumps(summary))
    print(json.dumps(summary["summary"]))


def record(args: argparse.Namespace) -> None:
    """Two laps' worth of driving in float64 for the browser's engine to match, on two unseen tracks, from a
    scattered start."""
    with jax.enable_x64(new_val=True):
        net, params, world, ground = from_json(json.loads(Path(args.model).read_text()))
        params = jax.tree.map(lambda p: jnp.asarray(p, jnp.float64), params)
        ground = tuple(jnp.asarray(g, jnp.float64) for g in ground)
        points = Tracks().draw(world, jax.random.key(args.seed), 2).astype(jnp.float64)
        car0 = starts(world, jax.random.key(args.seed + 1), points)
        cars, us, on, off, spikes, *_ = race(net, params, world, ground, points, car0, args.steps)
        laps = []
        for b in range(2):
            events = np.concatenate([np.asarray(on[:, b]), np.asarray(off[:, b])], -1)
            laps.append(
                {
                    "points": np.asarray(points[b]).tolist(),
                    "start": np.asarray(car0[b]).tolist(),
                    "cars": np.asarray(cars[:, b]).tolist(),
                    "readout": np.asarray(us[:, b]).tolist(),
                    "events": [np.flatnonzero(row).tolist() for row in events],
                    "spikes": [np.flatnonzero(row).tolist() for row in np.asarray(spikes[:, b])],
                }
            )
    Path(args.out).write_text(json.dumps({"dtype": "float64", "jax": jax.__version__, "laps": laps}))


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
    t.set_defaults(func=train)
    e = commands.add_parser("evaluate")
    e.add_argument("--model", default="site/public/racer/racer.json")
    e.add_argument("--out", default="site/public/racer/evaluation.json")
    e.add_argument("--tracks", type=int, default=200)
    e.add_argument("--seed", type=int, default=1000)
    e.set_defaults(func=evaluate)
    r = commands.add_parser("record")
    r.add_argument("--model", default="site/public/racer/racer.json")
    r.add_argument("--out", default="site/test/fixtures/racer.json")
    r.add_argument("--steps", type=int, default=500)
    r.add_argument("--seed", type=int, default=31)
    r.set_defaults(func=record)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
