"""The bound on how far the browser's car may drift from sparx's, derived from float64's rounding.

The car's update (`drive` in site/lab/racer.py, `Racer.step` in src/engines/racer.ts) is computed exactly in
mpmath at 60 digits, and beside it, by running error analysis, a budget for each result: every basic
operation may round by half a unit in the last place of its result, every transcendental (tanh, exp, tan,
cos, sin) by `TRANSCENDENTAL` units, and errors already in an operation's inputs pass through it by its
derivative. The readouts enter with the error the parity tests allow them, 1e-15 of their largest size.

For each fixture drive and each implementation, every step is redone exactly from the state that
implementation held, on sparx's readouts. The difference from the state it then held is that step's
rounding, and it must be within the step's budget: a wrong formula fails here. The budgets, carried
through the car's dynamics to first order,

    b_v <- (1 - dt / tau) b_v + e_v
    b_psi <- b_psi + dt |tan steer| / L b_v + e_psi
    b_x <- b_x + dt (|cos psi| b_v + v |sin psi| b_psi) + e_x, and b_y likewise,

bound each implementation's distance from the exact car, and their sum bounds the browser's distance from
sparx at every step, in units in the last place of each of x, y, heading and speed's largest value. The
script exits 1 if a step's rounding exceeds its budget or the browser's distance exceeds its bound. The
parity tests (test/racer.test.ts, test/racer-conv.test.ts) hold events and spikes exactly and readouts to
1e-15; the Site workflow runs this after them:

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
UNIT = mpmath.mpf(2) ** -53
"""Float64's unit roundoff: a correctly rounded result is within this much of the exact one, relatively."""
TRANSCENDENTAL = 4
"""The units in the last place a library's tanh, exp, tan, cos or sin may be off by: fdlibm's, which the
browser uses, promise 1, and XLA's on the CPU are taken to be within 4."""
READOUT = mpmath.mpf("1e-15")
"""The readouts' error the parity tests allow, relative to their largest size."""


class Value:
    """An exact value and a bound on how far a float64 computation of it may be."""

    def __init__(self, value, error=0):
        self.value, self.error = mpmath.mpf(value), mpmath.mpf(error)

    def rounded(self, error):
        return Value(self.value, error + UNIT * abs(self.value))

    def __add__(self, other):
        return Value(self.value + other.value, self.error + other.error).rounded(0)

    def __sub__(self, other):
        return Value(self.value - other.value, self.error + other.error).rounded(0)

    def __mul__(self, other):
        value = self.value * other.value
        error = abs(self.value) * other.error + abs(other.value) * self.error + self.error * other.error
        return Value(value).rounded(error)

    def __truediv__(self, other):
        value = self.value / other.value
        error = (self.error + abs(value) * other.error) / (abs(other.value) - other.error)
        return Value(value).rounded(error)

    def apply(self, f, slope):
        """`f` of this value, its input's error carried by the largest slope nearby."""
        value = f(self.value)
        near = max(abs(slope(self.value + d)) for d in (-self.error, 0, self.error))
        return Value(value, near * self.error + TRANSCENDENTAL * 2 * UNIT * abs(value))


class Car:
    def __init__(self, world: dict):
        self.dt, self.tau, self.length = (Value(world[k]) for k in ("dt", "speed_tau", "wheelbase"))
        self.steer_max, self.speed_max = Value(world["max_steer"]), Value(world["max_speed"])

    def step(self, car: list[float], u: list[float], readout_error) -> list[Value]:
        """One step from the float64 state `car` on the readouts `u`, exactly, with each result's budget."""
        x, y, psi, v = (Value(c) for c in car)
        u0, u1 = Value(u[0], readout_error), Value(u[1], readout_error)
        steer = self.steer_max * u0.apply(mpmath.tanh, lambda a: 1 / mpmath.cosh(a) ** 2)
        e = (Value(0) - u1).apply(mpmath.exp, mpmath.exp)
        target = self.speed_max * (Value(1) / (Value(1) + e))
        v = v + self.dt * (target - v) / self.tau
        psi = psi + self.dt * v * steer.apply(mpmath.tan, lambda a: 1 / mpmath.cos(a) ** 2) / self.length
        x = x + self.dt * v * psi.apply(mpmath.cos, lambda a: mpmath.sin(a))
        y = y + self.dt * v * psi.apply(mpmath.sin, lambda a: mpmath.cos(a))
        return [x, y, psi, v]


def drive(car: Car, start: list[float], cars: list[list[float]], readout: list[list[float]],
          readout_error) -> tuple[list, list[float]]:
    """Each step's budget, and each component's largest rounding as a share of its budget, at most 1 when
    every step stayed within it."""
    budgets, worst, before = [], [0.0] * 4, start
    for held, u in zip(cars, readout, strict=True):
        exact = car.step(before, u, readout_error)
        for i, (h, e) in enumerate(zip(held, exact, strict=True)):
            worst[i] = max(worst[i], float(abs(mpmath.mpf(h) - e.value) / e.error) if e.error else math.inf)
        budgets.append([e.error for e in exact])
        before = held
    return budgets, worst


def propagate(car: Car, cars: list[list[float]], readout: list[list[float]], budgets: list) -> list:
    """The first-order bound on each component's distance from the exact car after every step."""
    bx = by = bpsi = bv = mpmath.mpf(0)
    dt, tau, length = car.dt.value, car.tau.value, car.length.value
    bounds = []
    for held, u, e in zip(cars, readout, budgets, strict=True):
        _, _, psi, v = (mpmath.mpf(c) for c in held)
        tan = abs(mpmath.tan(car.steer_max.value * mpmath.tanh(mpmath.mpf(u[0]))))
        bv = (1 - dt / tau) * bv + e[3]
        bpsi = bpsi + dt * tan / length * bv + e[2]
        bx = bx + dt * (abs(mpmath.cos(psi)) * bv + abs(v) * abs(mpmath.sin(psi)) * bpsi) + e[0]
        by = by + dt * (abs(mpmath.sin(psi)) * bv + abs(v) * abs(mpmath.cos(psi)) * bpsi) + e[1]
        bounds.append([bx, by, bpsi, bv])
    return bounds


def wrong_car(world: dict, start: list[float], readout: list[list[float]]) -> list[list[float]]:
    """The car in float64 with its speed's time constant off by one part in 1e12: a wrong formula the
    budgets must catch."""
    x, y, psi, v = start
    tau = world["speed_tau"] * (1 + 1e-12)
    cars = []
    for u in readout:
        steer = world["max_steer"] * math.tanh(u[0])
        target = world["max_speed"] * (1 / (1 + math.exp(-u[1])))
        v = v + world["dt"] * (target - v) / tau
        psi = psi + world["dt"] * v * math.tan(steer) / world["wheelbase"]
        x = x + world["dt"] * v * math.cos(psi)
        y = y + world["dt"] * v * math.sin(psi)
        cars.append([x, y, psi, v])
    return cars


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
            readout_error = READOUT * max(abs(v) for step in lap["readout"] for v in step)
            sparx_budgets, sparx_within = drive(car, lap["start"], lap["cars"], lap["readout"], 0)
            ours_budgets, ours_within = drive(car, lap["start"], mine["cars"], lap["readout"], readout_error)
            sparx = propagate(car, lap["cars"], lap["readout"], sparx_budgets)
            ours = propagate(car, mine["cars"], lap["readout"], ours_budgets)
            bounds = [[float(s[i] + o[i]) / scale[i] for i in range(4)]
                      for s, o in zip(sparx, ours, strict=True)]
            distances = [[abs(p[i] - q[i]) / scale[i] for i in range(4)]
                         for p, q in zip(mine["cars"], lap["cars"], strict=True)]
            covered = all(d[i] <= b[i] for d, b in zip(distances, bounds, strict=True) for i in range(4))
            largest = [max(b[i] for b in bounds) for i in range(4)]
            seen = [max(d[i] for d in distances) for i in range(4)]
            rows.append({"steps": len(lap["cars"]),
                         "sparx_rounding_of_budget": dict(zip(NAMES, sparx_within, strict=True)),
                         "browser_rounding_of_budget": dict(zip(NAMES, ours_within, strict=True)),
                         "bound": dict(zip(NAMES, largest, strict=True)),
                         "observed": dict(zip(NAMES, seen, strict=True)), "covered": covered})
            shown = {k: ({n: round(x, 2) for n, x in v.items()} if isinstance(v, dict) else v)
                     for k, v in rows[-1].items()}
            print(name, json.dumps(shown), flush=True)
        report[name] = rows
    world = (recorded.get("model") or public)["world"]
    lap = recorded["laps"][0]
    _, shares = drive(car, lap["start"], wrong_car(world, lap["start"], lap["readout"]), lap["readout"], 0)
    caught = max(shares) > 1
    report["wrong_car_caught"] = caught
    print("a car with its speed's time constant off by 1e-12 is caught:", caught, round(max(shares), 1),
          flush=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    if not caught:
        sys.exit("the budgets did not catch a wrong car")
    fits = all(max(row[k].values()) <= 1 and row["covered"] for name, rows in report.items()
               if name != "wrong_car_caught" for row in rows
               for k in ("sparx_rounding_of_budget", "browser_rounding_of_budget"))
    if not fits:
        sys.exit("a step's rounding exceeded its budget, or the browser's car left its bound")


if __name__ == "__main__":
    main()
