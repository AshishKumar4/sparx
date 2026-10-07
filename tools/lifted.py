"""Definitions lifted out of a reference script, for fixture tools whose references run when imported.

Some reference repositories keep the classes a parity test needs in scripts
that train, plot, or reach for a GPU or a cluster's tools at import.
`lift` parses such a script and executes only the named top-level classes
and functions, over the globals the caller hands them, so a tool runs the
reference's own code and nothing around it.
"""

import ast
from pathlib import Path


def lift(path: Path, *names: str, **namespace: object) -> dict[str, object]:
    """The top-level definitions `names` of the script at `path`, executed over the globals `namespace`."""
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body
            if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in body}
    if missing:
        raise LookupError(f"{path} defines no {sorted(missing)}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}
