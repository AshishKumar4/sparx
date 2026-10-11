"""How far each implementation's car drifts from an exact one, and so how far apart the two can be.

The car's update (`drive` in site/lab/racer.py, `Racer.step` in src/engines/racer.ts) is replayed in
mpmath at 60 digits from each drive's start, on each implementation's own readouts. For every fixture
drive and each of x, y, heading and speed, in units in the last place of the component's largest value:

- sparx: sparx's float64 car against the exact car on sparx's readouts, its own accumulated rounding;
- browser: the browser's car against the exact car on the browser's readouts;
- readouts: the two exact cars against each other, what the readouts' difference does to the car;
- bound: their sum, which bounds the browser's distance from sparx by the triangle inequality;
- observed: the browser's distance from sparx.

    python car_exact.py <fixtures dir> <browser drives.json> <out.json>    (pip install mpmath)
"""

import json
import math
import sys
from pathlib import Path

import mpmath

mpmath.mp.dps = 60
NAMES = ("x", "y", "heading", "speed")


def exact_drive(world: dict, start: list[float], readout: list[list[float]]) -> list[list[float]]:
    """The car's states after each step, computed exactly from `readout` and rounded to float64 at the end."""
    x, y, psi, v = (mpmath.mpf(c) for c in start)
    dt, tau, length = (mpmath.mpf(world[k]) for k in ("dt", "speed_tau", "wheelbase"))
    steer_max, speed_max = mpmath.mpf(world["max_steer"]), mpmath.mpf(world["max_speed"])
    states = []
    for u in readout:
        steer = steer_max * mpmath.tanh(mpmath.mpf(u[0]))
        target = speed_max / (1 + mpmath.exp(-mpmath.mpf(u[1])))
        v = v + dt * (target - v) / tau
        psi = psi + dt * v * mpmath.tan(steer) / length
        x = x + dt * v * mpmath.cos(psi)
        y = y + dt * v * mpmath.sin(psi)
        states.append([float(x), float(y), float(psi), float(v)])
    return states


def ulps(a: list[list[float]], b: list[list[float]], scale: list[float]) -> list[float]:
    return [max(abs(p[i] - q[i]) for p, q in zip(a, b, strict=True)) / math.ulp(scale[i]) for i in range(4)]


def main() -> None:
    fixtures, browser, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
    public = json.loads((Path(__file__).resolve().parents[2] / "public/racer/racer.json").read_text())
    report = {}
    for name, drives in browser.items():
        recorded = json.loads((fixtures / f"{name}.json").read_text())
        world = (recorded.get("model") or public)["world"]
        rows = []
        for lap, mine in zip(recorded["laps"], drives, strict=True):
            scale = [max(abs(car[i]) for car in lap["cars"]) for i in range(4)]
            on_sparx = exact_drive(world, lap["start"], lap["readout"])
            on_browser = exact_drive(world, lap["start"], mine["readout"])
            terms = {"sparx": ulps(lap["cars"], on_sparx, scale),
                     "browser": ulps(mine["cars"], on_browser, scale),
                     "readouts": ulps(on_sparx, on_browser, scale)}
            bound = [sum(terms[k][i] for k in terms) for i in range(4)]
            observed = ulps(mine["cars"], lap["cars"], scale)
            named = {k: dict(zip(NAMES, v, strict=True)) for k, v in terms.items()}
            rows.append({"steps": len(lap["cars"]), **named,
                         "bound": dict(zip(NAMES, bound, strict=True)),
                         "observed": dict(zip(NAMES, observed, strict=True)),
                         "covered": all(o <= b for o, b in zip(observed, bound, strict=True))})
            print(name, json.dumps({k: (round(max(v.values()), 2) if isinstance(v, dict) else v)
                                    for k, v in rows[-1].items()}), flush=True)
        report[name] = rows
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
