"""The API reference, read from sparx's source with griffe, so the build imports neither JAX nor sparx.

One page per public module below. A page documents every name its module's `__all__` exports, in
that order: the signature, the docstring as written, a class's fields and public methods, and a link
to the source line. A name a page exports from a submodule is grouped under that submodule. The build
fails when an exported name cannot be found.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import griffe

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "site"
CONTENT = SITE / "src/content/docs/docs/api"
SOURCE = "https://github.com/AshishKumar4/sparx/blob/main/"

PAGES = [
    "sparx", "sparx.nn", "sparx.models", "sparx.surrogate", "sparx.encode", "sparx.losses", "sparx.rates",
    "sparx.dynamics", "sparx.graph", "sparx.spiketrains", "sparx.learn", "sparx.objectives", "sparx.metrics",
    "sparx.tasks", "sparx.config", "sparx.datasets", "sparx.serve", "sparx.nir",
]


def final(obj: griffe.Object | griffe.Alias) -> griffe.Object:
    return obj.final_target if obj.is_alias else obj


def summary(obj: griffe.Object) -> str:
    text = obj.docstring.value.strip() if obj.docstring else ""
    first = re.split(r"\n\s*\n", text, maxsplit=1)[0].replace("\n", " ")
    return first.replace("|", "\\|")


def anchor(name: str) -> str:
    return re.sub(r"[^\w\- ]", "", name.lower()).replace(" ", "-")


def parameters(function: griffe.Function, skip_self: bool = True) -> str:
    params = []
    for p in function.parameters:
        if skip_self and p.name in ("self", "cls"):
            continue
        text = p.name
        if p.kind == griffe.ParameterKind.var_positional:
            text = f"*{p.name}"
        elif p.kind == griffe.ParameterKind.var_keyword:
            text = f"**{p.name}"
        if p.annotation is not None:
            text += f": {p.annotation}"
        if p.default is not None:
            text += f" = {p.default}" if p.annotation is not None else f"={p.default}"
        params.append(text)
    return ", ".join(params)


def signature(name: str, obj: griffe.Object) -> str:
    if obj.is_function:
        returns = f" -> {obj.returns}" if obj.returns is not None else ""
        return f"def {name}({parameters(obj)}){returns}"
    if obj.is_class:
        bases = ", ".join(str(base) for base in obj.bases)
        return f"class {name}({bases})" if bases else f"class {name}"
    if obj.is_attribute:
        annotation = f": {obj.annotation}" if obj.annotation is not None else ""
        value = f" = {obj.value}" if obj.value is not None and len(str(obj.value)) < 120 else ""
        return f"{name}{annotation}{value}"
    return name


def source(obj: griffe.Object) -> str:
    path = Path(obj.filepath).relative_to(REPO).as_posix()
    return f"{SOURCE}{path}#L{obj.lineno}" if obj.lineno else f"{SOURCE}{path}"


def docstring(obj: griffe.Object) -> str:
    """The docstring as Markdown, its indented examples fenced as Python."""
    text = obj.docstring.value.strip() if obj.docstring else ""
    out: list[str] = []
    fenced = False
    lines = text.split("\n")
    for k, line in enumerate(lines):
        indented = line.startswith("    ") and line.strip()
        if not fenced and indented and (k == 0 or not lines[k - 1].strip()):
            out.append("```python")
            fenced = True
        if fenced and line.strip() and not line.startswith("    "):
            while out and not out[-1].strip():
                out.pop()
            out += ["```", ""]
            fenced = False
        out.append(line[4:] if fenced else line)
    if fenced:
        while out and not out[-1].strip():
            out.pop()
        out.append("```")
    return "\n".join(out)


def fields(cls: griffe.Class) -> list[griffe.Attribute]:
    return [m for m in cls.members.values() if not m.is_alias and m.is_attribute and not m.name.startswith("_")
            and m.annotation is not None and "ClassVar" not in str(m.annotation)]


def methods(cls: griffe.Class) -> list[griffe.Function]:
    return [m for m in cls.members.values() if not m.is_alias and m.is_function
            and (not m.name.startswith("_") or m.name == "__call__")]


def render(module: griffe.Module) -> str:
    exported = [str(name) for name in module.exports or []]
    if not exported:
        raise SystemExit(f"{module.path} declares no __all__")
    lines: list[str] = []
    intro = docstring(module)
    if intro:
        lines += [intro, ""]
    entries = []
    for name in exported:
        if name == "__version__":
            continue
        member = module.members.get(name)
        if member is None:
            raise SystemExit(f"{module.path}.__all__ names {name}, which griffe cannot find")
        obj = final(member)
        home = obj.module.path if not obj.is_module else module.path
        entries.append((name, obj, home))
    submodules = [(name, obj) for name, obj, _ in entries if obj.is_module]
    if submodules:
        lines += ["## Modules", "", "| Module | |", "| --- | --- |"]
        for name, obj in submodules:
            link = f"/docs/api/{obj.path}/" if obj.path in PAGES else source(obj)
            lines.append(f"| [`{obj.path}`]({link}) | {summary(obj)} |")
        lines.append("")
    objects = [(name, obj, home) for name, obj, home in entries if not obj.is_module]
    if objects:
        lines += ["## Contents", "", "| Name | |", "| --- | --- |"]
        lines += [f"| [`{name}`](#{anchor(name)}) | {summary(obj)} |" for name, obj, _ in objects]
        lines.append("")
    for name, obj, home in objects:
        lines += [f"## `{name}`", ""]
        lines += ["```python", signature(name, obj), "```", ""]
        lines += [f"<a class=\"api-source\" href=\"{source(obj)}\"><code>{home}</code> on GitHub</a>", ""]
        text = docstring(obj)
        if text:
            lines += [text, ""]
        if obj.is_class:
            table = fields(obj)
            if table:
                lines += ["| Field | Type | Default |", "| --- | --- | --- |"]
                for f in table:
                    default = f"`{f.value}`" if f.value is not None and len(str(f.value)) < 60 else ""
                    lines.append(f"| `{f.name}` | `{str(f.annotation).replace('|', chr(92) + '|')}` | {default} |")
                lines.append("")
            for method in methods(obj):
                lines += [f"### `{name}.{method.name}`", "", "```python",
                          f"def {method.name}({parameters(method)})"
                          + (f" -> {method.returns}" if method.returns is not None else ""), "```", ""]
                if docstring(method):
                    lines += [docstring(method), ""]
    return "\n".join(lines)


def main() -> None:
    package = griffe.load("sparx", search_paths=[REPO / "src"], resolve_aliases=True,
                          resolve_external=False, docstring_parser=None)
    CONTENT.mkdir(parents=True, exist_ok=True)
    for old in CONTENT.glob("*.md"):
        old.unlink()
    sidebar = [{"label": "Overview", "slug": "docs/api"}]
    index = ["The public modules of sparx, each documented from its source.", "",
             "| Module | |", "| --- | --- |"]
    for path in PAGES:
        module = package if path == "sparx" else package[path.removeprefix("sparx.")]
        body = render(module)
        description = summary(module).replace('"', "'")
        front = f'---\ntitle: "{path}"\nslug: docs/api/{path}\ndescription: "{description}"\neditUrl: false\n---\n\n'
        (CONTENT / f"{path}.md").write_text(front + body + "\n")
        sidebar.append({"label": path, "slug": f"docs/api/{path}"})
        index.append(f"| [`{path}`](/docs/api/{path}/) | {summary(module)} |")
    (CONTENT / "index.md").write_text('---\ntitle: "API reference"\ndescription: "Every public module of '
                                      'sparx, documented from its source."\neditUrl: false\n---\n\n'
                                      + "\n".join(index) + "\n")
    generated = SITE / "src/generated"
    generated.mkdir(exist_ok=True)
    (generated / "api.json").write_text(json.dumps(sidebar, indent=1) + "\n")
    print(f"gen_api: {len(PAGES)} modules")


if __name__ == "__main__":
    main()
