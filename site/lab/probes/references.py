"""How far two recordings of sparx's float64 racer runs differ: the same fixtures (site/lab/racer.py record
and fixture) made on two machines. For each drive, the car's x, y, heading and speed each give their
largest difference in units in the last place of the largest value they take, the measure the browser's
parity tests bound; the readouts, where recorded, their largest difference relative to their largest size.

    python3 site/lab/probes/references.py <fixtures dir A> <fixtures dir B>

Standard library only.
"""

import json
import math
import sys
from pathlib import Path


def ulp(x: float) -> float:
    return math.ulp(abs(x))


def compare(a: dict, b: dict) -> list[dict]:
    rows = []
    for k, (one, two) in enumerate(zip(a["laps"], b["laps"], strict=True)):
        row = {"lap": k, "steps": len(one["cars"])}
        for i, name in enumerate(("x", "y", "heading", "speed")):
            size = max(abs(car[i]) for car in one["cars"])
            error = max(abs(p[i] - q[i]) for p, q in zip(one["cars"], two["cars"], strict=True))
            row[name] = error / ulp(size)
        row["car"] = max(row[name] for name in ("x", "y", "heading", "speed"))
        if "readout" in one and "readout" in two:
            size = max(abs(v) for step in one["readout"] for v in step)
            error = max(abs(p - q) for s, t in zip(one["readout"], two["readout"], strict=True)
                        for p, q in zip(s, t, strict=True))
            # A drive whose readouts stay at zero has only an absolute difference to give.
            row["readout_max_relative" if size else "readout_max"] = error / size if size else error
        row["events_equal"] = one.get("events") == two.get("events")
        row["spikes_equal"] = one.get("spikes") == two.get("spikes")
        rows.append(row)
    return rows


def main() -> None:
    first, second = Path(sys.argv[1]), Path(sys.argv[2])
    for path in sorted(first.glob("racer*.json")):
        other = second / path.name
        if not other.exists():
            continue
        for row in compare(json.loads(path.read_text()), json.loads(other.read_text())):
            shown = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
            print(path.name, json.dumps(shown))


if __name__ == "__main__":
    main()
