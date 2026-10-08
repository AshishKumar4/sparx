"""Izhikevich (2004) Figure 1, the twenty firing patterns, as his own MATLAB code runs them.

    uv pip sync tools/environments/brian2.txt                   # and GNU Octave 8.4.0
    python tools/make_izhikevich_2004_fixtures.py figure1.m

`figure1.m` is the script published with "Which model to use for cortical
spiking neurons?" (IEEE Trans. Neural Netw. 2004), from izhikevich.org
(archived at web.archive.org/web/2015id_/http://www.izhikevich.org/publications/figure1.m).
Its plotting is removed and nothing else changed: each panel's loop runs as
written, and the current it applied each step (`I`), its step (`tau`), its
parameters, its starting `V` and `u`, and the voltage it recorded (`VV`, 30
on a spike step) are saved per panel to `tests/fixtures/izhikevich_2004.npz`.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from references import require_file, require_program

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "izhikevich_2004.npz"


def instrumented(source: str) -> str:
    out, open_parens, records = [], 0, False
    for line in source.splitlines():
        stripped = line.strip()
        if open_parens or re.match(r"^(subplot|plot|axis|set|hold|line|text)\b", stripped):
            # A plotting call can continue over lines until its parentheses close.
            open_parens += stripped.count("(") - stripped.count(")")
            continue
        title = re.match(r"^title\('\((\w)\)", stripped)
        if title:
            panel = title.group(1)
            names = "'VV', 'II', 'tau', 'a', 'b', 'c', 'd', 'V0', 'u0'"
            out.append(f"save('-v7', fullfile(OUT, '{panel}.mat'), {names});")
            continue
        if re.match(r"^V\s*=\s*V\s*\+\s*tau", stripped) and not records:
            out.append("II(end+1)=I;")
        out.append(line)
        if re.match(r"^VV\s*=\s*\[\]", stripped):
            # Panel (R) keeps its own record of the current.
            records = "II=[]" in stripped.replace(" ", "")
            out.append("V0=V; u0=u;" + ("" if records else " II=[];"))
    return "\n".join(out)


def main(figure1: str) -> None:
    require_program("octave")
    require_file(Path(figure1), "figure1.m")
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "run.m"
        script.write_text(f"OUT='{directory}';\n" + instrumented(Path(figure1).read_text()))
        subprocess.run(["octave", "--no-gui", "--quiet", str(script)], check=True)
        from scipy.io import loadmat
        arrays = {}
        for mat in sorted(Path(directory).glob("*.mat")):
            panel = loadmat(mat)
            for key in ("VV", "II"):
                arrays[f"{mat.stem}/{key}"] = np.asarray(panel[key], np.float64).ravel()
            for key in ("tau", "a", "b", "c", "d", "V0", "u0"):
                arrays[f"{mat.stem}/{key}"] = np.float64(panel[key].ravel()[0])
    assert len({k.split("/")[0] for k in arrays}) == 20, sorted(arrays)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
