"""The bound on how far the browser's car may drift from sparx's, derived from each one's rounding.

The car's update (`drive` in site/lab/racer.py, `Racer.step` in src/engines/racer.ts) is computed exactly in
mpmath at 60 digits. For each fixture drive and each implementation, every step is redone exactly from the
state the implementation itself held, on its own readouts: the difference from the state it then held is
that step's own rounding error. The two readouts' difference is carried through the exact update the same
way. A first-order propagation of those per-step errors through the car's dynamics,

    b_v <- (1 - dt / tau) b_v + e_v
    b_psi <- b_psi + dt |tan steer| / L b_v + e_psi (+ dt v / L |d tan steer| from the readouts)
    b_x <- b_x + dt (|cos psi| b_v + v |sin psi| b_psi) + e_x, and b_y likewise,

bounds each implementation's distance from the exact car, and their sum bounds the browser's distance from
sparx. This is where the browser's car is held to sparx's: every step's distance must be within that
step's bound, in units in the last place of each of x, y, heading and speed's largest value over the
drive. It exits 1 if any is not. The parity tests (test/racer.test.ts, test/racer-conv.test.ts) hold
events, spikes and readouts exactly or to 1e-15, and the Site workflow runs this after them:

    bun lab/racer-drives.ts drives.json
    uv run --no-project --with mpmath==1.3.0 python lab/car_exact.py test/fixtures drives.json parity.json
"""

import json
import math
import sys
from pathlib import Path

import mpmath

mpmath.mp.dps = 60
NAMES = ("x", "y", "heading", "speed")


class Car:
    def __init__(self, world: dict):
        self.dt, self.tau, self.length = (mpmath.mpf(world[k]) for k in ("dt", "speed_tau", "wheelbase"))
        self.steer_max, self.speed_max = mpmath.mpf(world["max_steer"]), mpmath.mpf(world["max_speed"])

    def steer(self, u: float):
        return self.steer_max * mpmath.tanh(mpmath.mpf(u))

    def step(self, car: list[float], u: list[float]) -> list:
        """One exact step from the float64 state `car` on the readouts `u`."""
        x, y, psi, v = (mpmath.mpf(c) for c in car)
        target = self.speed_max / (1 + mpmath.exp(-mpmath.mpf(u[1])))
        v = v + self.dt * (target - v) / self.tau
        psi = psi + self.dt * v * mpmath.tan(self.steer(u[0])) / self.length
        return [x + self.dt * v * mpmath.cos(psi), y + self.dt * v * mpmath.sin(psi), psi, v]


def local_errors(car: Car, start: list[float], cars: list[list[float]], readout: list[list[float]]) -> list:
    """Each step's own rounding error, `|held - exact step from the previous held state|`, per component."""
    errors, before = [], start
    for held, u in zip(cars, readout, strict=True):
        exact = car.step(before, u)
        errors.append([abs(mpmath.mpf(h) - e) for h, e in zip(held, exact, strict=True)])
        before = held
    return errors


def readout_errors(car: Car, start: list[float], cars: list[list[float]], ours: list[list[float]],
                   theirs: list[list[float]]) -> list:
    """Each step's difference from taking the exact step on the other readouts, from the same state."""
    errors, before = [], start
    for held, u, w in zip(cars, ours, theirs, strict=True):
        a, b = car.step(before, u), car.step(before, w)
        errors.append([abs(p - q) for p, q in zip(a, b, strict=True)])
        before = held
    return errors


def propagate(car: Car, cars: list[list[float]], readout: list[list[float]], errors: list) -> list:
    """The first-order bound on each component's distance after every step, from the per-step errors."""
    bx = by = bpsi = bv = mpmath.mpf(0)
    bounds = []
    for held, u, e in zip(cars, readout, errors, strict=True):
        _, _, psi, v = (mpmath.mpf(c) for c in held)
        tan = abs(mpmath.tan(car.steer(u[0])))
        bv = (1 - car.dt / car.tau) * bv + e[3]
        bpsi = bpsi + car.dt * tan / car.length * bv + e[2]
        bx = bx + car.dt * (abs(mpmath.cos(psi)) * bv + abs(v) * abs(mpmath.sin(psi)) * bpsi) + e[0]
        by = by + car.dt * (abs(mpmath.sin(psi)) * bv + abs(v) * abs(mpmath.cos(psi)) * bpsi) + e[1]
        bounds.append([bx, by, bpsi, bv])
    return bounds


def main() -> None:
    fixtures, browser, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
    public = json.loads((Path(__file__).resolve().parents[1] / "public/racer/racer.json").read_text())
    report = {}
    for name, drives in browser.items():
        recorded = json.loads((fixtures / f"{name}.json").read_text())
        car = Car((recorded.get("model") or public)["world"])
        rows = []
        for lap, mine in zip(recorded["laps"], drives, strict=True):
            scale = [math.ulp(max(abs(c[i]) for c in lap["cars"])) for i in range(4)]
            sparx = propagate(car, lap["cars"], lap["readout"],
                              local_errors(car, lap["start"], lap["cars"], lap["readout"]))
            ours = propagate(car, mine["cars"], mine["readout"],
                             local_errors(car, lap["start"], mine["cars"], mine["readout"]))
            differences = readout_errors(car, lap["start"], lap["cars"], lap["readout"], mine["readout"])
            readouts = propagate(car, lap["cars"], lap["readout"], differences)
            # Each step's bound and distance, in units in the last place of each component's largest value.
            bounds = [[float(s[i] + o[i] + r[i]) / scale[i] for i in range(4)]
                      for s, o, r in zip(sparx, ours, readouts, strict=True)]
            distances = [[abs(p[i] - q[i]) / scale[i] for i in range(4)]
                         for p, q in zip(mine["cars"], lap["cars"], strict=True)]
            covered = all(d[i] <= b[i] for d, b in zip(distances, bounds, strict=True) for i in range(4))
            largest = [max(b[i] for b in bounds) for i in range(4)]
            seen = [max(d[i] for d in distances) for i in range(4)]
            rows.append({"steps": len(lap["cars"]), "bound": dict(zip(NAMES, largest, strict=True)),
                         "observed": dict(zip(NAMES, seen, strict=True)), "covered": covered})
            shown = {k: ({n: round(x, 2) for n, x in v.items()} if isinstance(v, dict) else v)
                     for k, v in rows[-1].items()}
            print(name, json.dumps(shown), flush=True)
        report[name] = rows
    out.write_text(json.dumps(report, indent=1) + "\n")
    if not all(row["covered"] for rows in report.values() for row in rows):
        sys.exit("the browser's car left its rounding bound")


if __name__ == "__main__":
    main()
