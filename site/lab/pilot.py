"""The pilot on sparxml.dev's front page: a spiking network that flies a planar drone to a target.

It learns by gradient descent through its own spikes, with surrogate gradients,
and through a differentiable model of the drone written in JAX. Nothing to
imitate and no reinforcement learning: the loss is how far the drone is from
where it should be, summed over each flight.

    python site/lab/pilot.py train --out site/public/pilot      # writes pilot.json and pilot.nir
    python site/lab/pilot.py evaluate --model site/public/pilot/pilot.json
    python site/lab/pilot.py record --out site/test/fixtures/pilot.json

The browser steps the same network and the same physics from pilot.json
(site/src/engines/pilot.ts); `record` writes the flights the browser's
stepper is checked against (site/test/pilot.test.ts).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import training
from stack import act, at_rest, from_layers, stack, to_layers

import sparx


@dataclasses.dataclass(frozen=True)
class Drone:
    """A planar quadrotor: two rotors on an arm, thrust along the body's up axis.

    SI units. `dt` is the step of both the physics and the network.
    """

    dt: float = 0.01
    mass: float = 1.0
    arm: float = 0.25
    inertia: float = 0.04
    gravity: float = 9.81
    max_thrust: float = 12.0
    drag: float = 0.3
    spin_drag: float = 0.02
    reach: float = 1.5
    bounce: float = 0.3

    @property
    def hover_logit(self) -> float:
        """The readout bias at which both rotors together hold the drone's weight."""
        share = self.mass * self.gravity / 2 / self.max_thrust
        return math.log(share / (1 - share))


def observe(drone: Drone, s: jax.Array, target: jax.Array) -> jax.Array:
    """What the pilot reads: the way to the target, clipped to `reach`, its velocity, attitude and spin."""
    x, y, vx, vy, theta, omega = jnp.moveaxis(s, -1, 0)
    ex = jnp.clip(target[..., 0] - x, -drone.reach, drone.reach) / drone.reach
    ey = jnp.clip(target[..., 1] - y, -drone.reach, drone.reach) / drone.reach
    return jnp.stack([ex, ey, vx / 4, vy / 4, jnp.sin(theta), jnp.cos(theta), omega / 8], axis=-1)


def thrust(drone: Drone, u: jax.Array) -> jax.Array:
    """Each rotor's thrust (N) from the readout's membrane, left then right."""
    return drone.max_thrust * jax.nn.sigmoid(u + drone.hover_logit)


def advance(drone: Drone, s: jax.Array, force: jax.Array, box: jax.Array) -> jax.Array:
    """One semi-implicit Euler step; a wall of the box `[w, h]` (half widths) stops the drone and
    returns `bounce` of its speed."""
    x, y, vx, vy, theta, omega = jnp.moveaxis(s, -1, 0)
    left, right = force[..., 0], force[..., 1]
    total = left + right
    dt = drone.dt
    vx = vx + dt * (-total * jnp.sin(theta) - drone.drag * vx) / drone.mass
    vy = vy + dt * ((total * jnp.cos(theta) - drone.drag * vy) / drone.mass - drone.gravity)
    omega = omega + dt * ((right - left) * drone.arm - drone.spin_drag * omega) / drone.inertia
    x, y, theta = x + dt * vx, y + dt * vy, theta + dt * omega
    w, h = box[..., 0], box[..., 1]
    vx = jnp.where(jnp.abs(x) > w, -drone.bounce * vx, vx)
    vy = jnp.where(jnp.abs(y) > h, -drone.bounce * vy, vy)
    x, y = jnp.clip(x, -w, w), jnp.clip(y, -h, h)
    return jnp.stack([x, y, vx, vy, theta, omega], axis=-1)


def network(hidden: int = 64, tau: float = 3.0, readout_tau: float = 5.0) -> nn.Sequential:
    return stack((hidden, hidden, 2), tau, readout_tau)


def fly(net, params, drone, s0, targets, kicks, box):
    """Fly from `s0` after `targets` `[T, B, 2]`, with velocity kicks `[T, B, 6]` added before each step."""

    def step(carry, inputs):
        s, carried = carry
        target, kick = inputs
        s = s + kick
        u, carried, spikes = act(net, params, carried, observe(drone, s, target))
        s = advance(drone, s, thrust(drone, u), box)
        return (s, carried), (s, u, spikes)

    _, (states, us, spikes) = jax.lax.scan(step, (s0, at_rest(net, params, s0.shape[0], 7)), (targets, kicks))
    return states, us, spikes


@dataclasses.dataclass(frozen=True)
class Flights:
    """How training draws its flights: boxes from phone to ultrawide, any attitude, moving targets, gusts."""

    steps: int = 300
    height: float = 1.5
    widths: tuple[float, float] = (0.9, 4.0)
    inverted: float = 0.4
    speed: float = 1.5
    spin: float = 3.0
    retarget: float = 1.2
    gust: float = 2.0
    gust_speed: float = 2.0
    gust_spin: float = 8.0

    def draw(self, drone: Drone, key: jax.Array, batch: int):
        ks = jax.random.split(key, 10)
        w = jax.random.uniform(ks[0], (batch,), minval=self.widths[0], maxval=self.widths[1])
        box = jnp.stack([w, jnp.full_like(w, self.height)], -1)
        pos = jax.random.uniform(ks[1], (batch, 2), minval=-0.9, maxval=0.9) * box
        vel = jax.random.normal(ks[2], (batch, 2)) * self.speed
        upright = jax.random.normal(ks[3], (batch,)) * 0.4
        anywhere = jax.random.uniform(ks[4], (batch,), minval=-math.pi, maxval=math.pi)
        theta = jnp.where(jax.random.uniform(ks[5], (batch,)) < self.inverted, anywhere, upright)
        omega = jax.random.normal(ks[6], (batch,)) * self.spin
        s0 = jnp.concatenate([pos, vel, theta[:, None], omega[:, None]], -1)
        # Targets hold for exponential times, as a cursor rests and moves on.
        spots = jax.random.uniform(ks[7], (self.steps, batch, 2), minval=-0.85, maxval=0.85) * box
        moves = jax.random.uniform(ks[8], (self.steps, batch)) < drone.dt / self.retarget
        moves = moves.at[0].set(True)
        index = jax.lax.cummax(jnp.where(moves, jnp.arange(self.steps)[:, None], 0), axis=0)
        targets = jnp.take_along_axis(spots, index[..., None], axis=0)
        gusts = jax.random.uniform(ks[9], (self.steps, batch)) < drone.dt / self.gust
        size = jax.random.normal(jax.random.fold_in(ks[9], 1), (self.steps, batch, 6)) * jnp.array(
            [0, 0, self.gust_speed, self.gust_speed, 0, self.gust_spin]
        )
        kicks = jnp.where(gusts[..., None], size, 0.0).at[0].set(0.0)
        return s0, targets, kicks, box


def flight_cost(
    states: jax.Array, targets: jax.Array, spikes: jax.Array, rates: tuple[float, float]
) -> tuple[jax.Array, dict]:
    distance = jnp.sqrt(jnp.sum((states[..., :2] - targets) ** 2, -1) + 1e-4)
    tilt = 1 - jnp.cos(states[..., 4])
    spin = states[..., 5] ** 2
    rate = jnp.mean(spikes, axis=(0, 1))
    band = jnp.mean(jax.nn.relu(rate - rates[1]) ** 2 + jax.nn.relu(rates[0] - rate) ** 2)
    loss = jnp.mean(distance) + 0.05 * jnp.mean(tilt) + 0.002 * jnp.mean(spin) + 10.0 * band
    return loss, {"distance": jnp.mean(distance), "rate": jnp.mean(rate)}


def to_json(net: nn.Sequential, params: dict, drone: Drone, meta: dict) -> dict:
    """The network and the drone for the browser."""
    return {
        "drone": dataclasses.asdict(drone) | {"hover_logit": drone.hover_logit},
        "layers": to_layers(net, params),
        "meta": meta,
    }


def from_json(model: dict) -> tuple[nn.Sequential, dict, Drone]:
    net, params = from_layers(model["layers"])
    fields = {f.name for f in dataclasses.fields(Drone)}
    return net, params, Drone(**{k: v for k, v in model["drone"].items() if k in fields})


def train(args: argparse.Namespace) -> None:
    drone = Drone()
    flights = Flights(
        steps=args.horizon,
        speed=args.speed,
        spin=args.spin,
        gust_speed=args.gust_speed,
        gust_spin=args.gust_spin,
    )
    net = network(args.hidden, args.tau, args.readout_tau)
    params = net.init(jax.random.key(args.seed), jnp.zeros((1, 1, 7)))["params"]
    def loss_fn(params, key):
        s0, targets, kicks, box = flights.draw(drone, key, args.batch)
        states, _, spikes = fly(net, params, drone, s0, targets, kicks, box)
        return flight_cost(states, targets, spikes, (args.rate_low, args.rate_high))

    params, history = training.train(loss_fn, params, lr=args.lr, steps=args.steps, seed=args.seed + 1,
                                     log_every=args.log_every)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "trained": {k: v for k, v in vars(args).items() if k not in ("func", "out")},
        "history": history,
        "backend": jax.default_backend(),
        "devices": str(jax.devices()[0]),
        "jax": jax.__version__,
        "sparx": sparx.__version__,
    }
    model = to_json(net, params, drone, meta)
    model["evaluation"] = evaluation(net, params, drone, args.seed + 2)
    (out / "pilot.json").write_text(json.dumps(model))
    export_nir(net, params, drone, out / "pilot.nir")
    print(json.dumps(model["evaluation"]))


def export_nir(net: nn.Sequential, params: dict, drone: Drone, path: Path) -> None:
    import nir

    from sparx.nir import to_nir

    nir.write(str(path), to_nir(net, {"params": params}, dt=drone.dt))


def evaluation(net, params, drone: Drone, seed: int, flights: int = 1000) -> dict:
    """Flights of 6 s to one fixed target from random starts, half of them from any attitude: how many
    arrive within 15 cm and stay, and how long it takes; and as many thrown, at speeds of about 4 m/s
    and spins of about 15 rad/s."""
    steps = 600
    run = jax.jit(lambda p, *flight: fly(net, p, drone, *flight))

    def arrivals(draw: Flights, key: int):
        s0, targets, kicks, box = draw.draw(drone, jax.random.key(key), flights)
        states, _, spikes = run(params, s0, targets, kicks, box)
        distance = np.asarray(jnp.sqrt(jnp.sum((states[..., :2] - targets) ** 2, -1)))
        settled = np.flip(np.cumprod(np.flip(distance < 0.15, 0), 0), 0).astype(bool)  # close from then on
        first = np.where(settled[-1], np.argmax(settled, 0), steps) * drone.dt
        return s0, settled[-1], first, distance, spikes

    still = Flights(steps=steps, inverted=0.5, gust=1e9, retarget=1e9)
    s0, arrived, first, distance, spikes = arrivals(still, seed)
    inverted = np.cos(np.asarray(s0[:, 4])) < 0
    _, thrown, _, _, _ = arrivals(
        Flights(steps=steps, inverted=0.5, speed=4.0, spin=15.0, gust=1e9, retarget=1e9), seed + 1
    )
    return {
        "flights": flights,
        "seconds": steps * drone.dt,
        "arrived": float(arrived.mean()),
        "arrived_from_inverted": float(arrived[inverted].mean()),
        "inverted_starts": int(inverted.sum()),
        "arrived_thrown": float(thrown.mean()),
        "median_time_s": float(np.median(first[arrived])) if arrived.any() else None,
        "final_distance_median_m": float(np.median(distance[-1])),
        "rate_per_step": float(jnp.mean(spikes)),
    }


def evaluate(args: argparse.Namespace) -> None:
    net, params, drone = from_json(json.loads(Path(args.model).read_text()))
    print(json.dumps(evaluation(net, params, drone, args.seed)))


def record(args: argparse.Namespace) -> None:
    """Flights in float64 for the browser's stepper to match: a desktop's box and a phone's, from any
    attitude, after moving targets, with gusts."""
    with jax.enable_x64(new_val=True):
        net, params, drone = from_json(json.loads(Path(args.model).read_text()))
        params = jax.tree.map(lambda p: jnp.asarray(p, jnp.float64), params)
        flights = []
        for k, (width, steps) in enumerate([(3.0, 800), (1.0, 600)]):
            s0, targets, kicks, box = Flights(steps=steps, inverted=1.0, gust=1.0).draw(
                drone, jax.random.key(args.seed + k), 1
            )
            box = box.at[:, 0].set(width)
            states, us, spikes = fly(net, params, drone, s0, targets, kicks, box)
            flights.append(
                {
                    "box": box[0].tolist(),
                    "start": s0[0].tolist(),
                    "targets": targets[:, 0].tolist(),
                    "kicks": kicks[:, 0].tolist(),
                    "states": states[:, 0].tolist(),
                    "readout": us[:, 0].tolist(),
                    "spikes": [np.flatnonzero(row).tolist() for row in np.asarray(spikes[:, 0])],
                }
            )
    Path(args.out).write_text(
        json.dumps(
            {"dtype": "float64", "jax": jax.__version__, "sparx": sparx.__version__, "flights": flights}
        )
    )


def lesions(args: argparse.Namespace) -> None:
    """How often the pilot still arrives with neurons silenced: each count of silenced neurons, drawn five
    times from both layers. Silencing a neuron zeroes its outgoing weights, which leaves the rest of the
    network as the browser's silenced neuron does, sending nothing."""
    net, params, drone = from_json(json.loads(Path(args.model).read_text()))
    rng = np.random.default_rng(args.seed)
    rows = []
    for count in (0, 8, 16, 24, 32, 48, 64, 80, 96, 112):
        arrived = []
        for draw in range(5):
            cut = rng.choice(128, count, replace=False)
            silenced = jax.tree.map(np.array, params)
            for neuron in cut:
                layer = "layers_2" if neuron < 64 else "layers_4"
                silenced[layer]["kernel"][neuron % 64] = 0.0
            arrived.append(evaluation(net, silenced, drone, args.seed + draw, flights=500)["arrived"])
        rows.append({"silenced": int(count), "arrived": arrived})
        print(json.dumps(rows[-1]), flush=True)
    Path(args.out).write_text(json.dumps({"flights": 500, "draws": 5, "rows": rows}))


def describe_nir(args: argparse.Namespace) -> None:
    """The pilot's NIR graph node by node, and the network sparx reads back from it, run beside ours."""
    import nir

    from sparx.nir import from_nir

    model = json.loads(Path(args.model).read_text())
    net, params, drone = from_json(model)
    graph = nir.read(args.nir)
    nodes = []
    for name, node in graph.nodes.items():
        fields = {}
        for key, value in vars(node).items():
            if isinstance(value, np.ndarray) and value.size and key not in ("input_type", "output_type"):
                fields[key] = {
                    "shape": list(value.shape),
                    "mean": float(np.mean(value)),
                    "min": float(np.min(value)),
                    "max": float(np.max(value)),
                }
        nodes.append({"name": name, "type": type(node).__name__, "fields": fields})
    imported, variables = from_nir(graph, dt=drone.dt)
    x = jax.random.normal(jax.random.key(args.seed), (200, 8, 7))
    ours = net.apply({"params": params}, x)
    theirs = imported.apply(variables, x)
    Path(args.out).write_text(
        json.dumps(
            {
                "nodes": nodes,
                "edges": [list(e) for e in graph.edges],
                "metadata": {k: str(v) for k, v in graph.metadata.items()},
                "bytes": Path(args.nir).stat().st_size,
                "nir": nir.__version__ if hasattr(nir, "__version__") else None,
                "round_trip": {
                    "steps": 200,
                    "batch": 8,
                    "largest_difference": float(jnp.max(jnp.abs(ours - theirs))),
                },
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(required=True)
    t = commands.add_parser("train")
    t.add_argument("--out", default="site/public/pilot")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--horizon", type=int, default=300)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--tau", type=float, default=3.0)
    t.add_argument("--readout-tau", type=float, default=5.0)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--rate-low", type=float, default=0.02)
    t.add_argument("--rate-high", type=float, default=0.3)
    t.add_argument("--speed", type=float, default=1.5, help="initial speeds' spread (m/s)")
    t.add_argument("--spin", type=float, default=3.0, help="initial spins' spread (rad/s)")
    t.add_argument("--gust-speed", type=float, default=2.0, help="gusts' spread of speed (m/s)")
    t.add_argument("--gust-spin", type=float, default=8.0, help="gusts' spread of spin (rad/s)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=50)
    t.set_defaults(func=train)
    e = commands.add_parser("evaluate")
    e.add_argument("--model", default="site/public/pilot/pilot.json")
    e.add_argument("--seed", type=int, default=7)
    e.set_defaults(func=evaluate)
    r = commands.add_parser("record")
    r.add_argument("--model", default="site/public/pilot/pilot.json")
    r.add_argument("--out", default="site/test/fixtures/pilot.json")
    r.add_argument("--seed", type=int, default=11)
    r.set_defaults(func=record)
    les = commands.add_parser("lesions")
    les.add_argument("--model", default="site/public/pilot/pilot.json")
    les.add_argument("--out", default="site/public/pilot/lesions.json")
    les.add_argument("--seed", type=int, default=21)
    les.set_defaults(func=lesions)
    n = commands.add_parser("nir")
    n.add_argument("--model", default="site/public/pilot/pilot.json")
    n.add_argument("--nir", default="site/public/pilot/pilot.nir")
    n.add_argument("--out", default="site/public/pilot/nir.json")
    n.add_argument("--seed", type=int, default=5)
    n.set_defaults(func=describe_nir)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
