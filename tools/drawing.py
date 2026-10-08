"""SVG drawing for sparx's banner, diagrams and clips: one palette, one typeface, light and dark.

`Canvas` lays out shapes and text in SVG and measures text with the fonts'
own advance widths, so labels sit centered in their boxes. Each SVG embeds
subsets of Inter and JetBrains Mono (both under the SIL Open Font License)
holding only the glyphs it uses, so it renders the same in any browser,
and on GitHub, whose README images load no outside fonts. The palette is
the dataviz reference palette's blue, orange, aqua and violet, checked
for color-vision separation on GitHub's light (#ffffff) and dark
(#0d1117) pages; each color has a role across every figure:

    blue    state that runs continuously: membranes, activity, the training half
    orange  spikes and events
    aqua    the biological half: physical units, circuits, connectomes
    violet  learning signals: gradients, eligibilities, rewards

Fonts come from the system (Inter under /usr/share/fonts/opentype/inter)
and from `npm pack @fontsource/jetbrains-mono@5.1.1`, unpacked under
`../refs/fonts`; `fontTools` and `brotli` subset them.
"""

from __future__ import annotations

import base64
import functools
import html
import io
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
INTER = Path("/usr/share/fonts/opentype/inter")
MONO = ROOT.parent / "refs" / "fonts" / "package" / "files"
FONTS = {
    ("sans", 400): INTER / "Inter-Regular.otf",
    ("sans", 500): INTER / "Inter-Medium.otf",
    ("sans", 600): INTER / "Inter-SemiBold.otf",
    ("display", 600): INTER / "InterDisplay-SemiBold.otf",
    ("display", 700): INTER / "InterDisplay-Bold.otf",
    ("mono", 400): MONO / "jetbrains-mono-latin-400-normal.woff2",
    ("mono", 500): MONO / "jetbrains-mono-latin-500-normal.woff2",
}
FAMILY = {"sans": "SxSans", "display": "SxDisplay", "mono": "SxMono"}
FALLBACK = {"sans": "Inter, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif",
            "display": "Inter, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif",
            "mono": "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace"}


@dataclass(frozen=True)
class Theme:
    name: str
    page: str
    ink: str
    ink2: str
    muted: str
    hairline: str
    panel: str
    panel_line: str
    blue: str
    orange: str
    aqua: str
    violet: str

    def soft(self, color: str, alpha: float = 0.12) -> str:
        """`color` at `alpha`, as an rgba fill over the page."""
        r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{alpha})"


THEMES = {
    "light": Theme("light", "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fbfbfa", "#dcdbd4",
                   "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"),
    "dark": Theme("dark", "#0d1117", "#f0f6fc", "#c3c2b7", "#8b949e", "#2c2c2a", "#161b22", "#30363d",
                  "#3987e5", "#d95926", "#199e70", "#9085e9"),
}


@functools.cache
def _font(key: tuple[str, int]) -> TTFont:
    return TTFont(FONTS[key])


@functools.cache
def _advances(key: tuple[str, int]) -> tuple[dict[int, int], int, int]:
    font = _font(key)
    cmap = font.getBestCmap()
    metrics = font["hmtx"].metrics
    widths = {code: metrics[name][0] for code, name in cmap.items()}
    return widths, font["head"].unitsPerEm, metrics[cmap[ord("x")]][0]


def text_width(s: str, size: float, weight: int = 400, family: str = "sans") -> float:
    """The advance width of `s` set in `family` at `weight` and `size` px, without kerning."""
    widths, em, fallback = _advances((family, weight))
    return sum(widths.get(ord(c), fallback) for c in s) * size / em


@functools.cache
def _subset(key: tuple[str, int], chars: frozenset[str]) -> str:
    """A WOFF2 of the font `key` holding `chars`, base64-encoded."""
    font = TTFont(FONTS[key])
    options = subset.Options()
    options.flavor = "woff2"
    options.layout_features = ["kern", "liga", "calt", "tnum"]
    options.name_IDs = []
    subsetter = subset.Subsetter(options)
    subsetter.populate(text="".join(sorted(chars | {" "})))
    subsetter.subset(font)
    buffer = io.BytesIO()
    font.flavor = "woff2"
    font.save(buffer)
    return base64.b64encode(buffer.getvalue()).decode()


class Canvas:
    """An SVG of `width` by `height` px drawn in `theme`'s colors."""

    def __init__(self, width: float, height: float, theme: Theme, *, background: bool = False):
        self.width, self.height, self.theme = width, height, theme
        self.parts: list[str] = []
        self.defs: list[str] = []
        self.used: dict[tuple[str, int], set[str]] = defaultdict(set)
        self.markers: set[str] = set()
        if background:
            self.rect(0, 0, width, height, fill=theme.page)

    def add(self, element: str) -> None:
        self.parts.append(element)

    def rect(self, x: float, y: float, w: float, h: float, *, r: float = 0, fill: str = "none",
             stroke: str = "none", width: float = 1, dash: str | None = None,
             opacity: float | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{width}"{extra}/>')

    def circle(self, x: float, y: float, r: float, *, fill: str = "none", stroke: str = "none",
               width: float = 1, opacity: float | None = None) -> None:
        extra = f' opacity="{opacity:.3f}"' if opacity is not None else ""
        self.add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{width}"{extra}/>')

    def path(self, d: str, *, stroke: str = "none", width: float = 1.5, fill: str = "none",
             dash: str | None = None, arrow: str | None = None, opacity: float | None = None,
             cap: str = "round") -> None:
        """A path; `arrow` is the color of an arrowhead at its end."""
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' opacity="{opacity:.3f}"' if opacity is not None else ""
        if arrow:
            extra += f' marker-end="url(#{self._marker(arrow)})"'
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
                 f'stroke-linecap="{cap}" stroke-linejoin="round"{extra}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, **kwargs) -> None:
        self.path(f"M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}", **kwargs)

    def polyline(self, points: list[tuple[float, float]], **kwargs) -> None:
        self.path("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points), **kwargs)

    def _marker(self, color: str) -> str:
        name = "arrow" + color.strip("#")
        if name not in self.markers:
            self.markers.add(name)
            self.defs.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
                             f'markerHeight="7" orient="auto-start-reverse"><path d="M1,1.2 L9,5 L1,8.8 '
                             f'C2.6,6.4 2.6,3.6 1,1.2 Z" fill="{color}"/></marker>')
        return name

    def text(self, x: float, y: float, s: str, *, size: float = 14, weight: int = 400, family: str = "sans",
             fill: str | None = None, anchor: str = "start", spacing: float = 0,
             opacity: float | None = None) -> float:
        """Text with its baseline at `y`; returns its width."""
        self.used[(family, weight)].update(s)
        extra = f' letter-spacing="{spacing}"' if spacing else ""
        extra += f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" class="{family}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill or self.theme.ink}" text-anchor="{anchor}"{extra}>{html.escape(s)}</text>')
        return text_width(s, size, weight, family) + spacing * len(s)

    def rich(self, x: float, y: float, runs: list[tuple[str, dict]], *, anchor: str = "start") -> float:
        """Runs of text, each with its own `text` options, set one after another on one baseline."""
        widths = [text_width(s, o.get("size", 14), o.get("weight", 400), o.get("family", "sans"))
                  for s, o in runs]
        total = sum(widths)
        start = x - (total / 2 if anchor == "middle" else total if anchor == "end" else 0)
        for (s, options), w in zip(runs, widths, strict=True):
            self.text(start, y, s, **options)
            start += w
        return total

    def inline(self, x: float, y: float, s: str, *, size: float = 15, weight: int = 400,
               fill: str | None = None, code: str | None = None, anchor: str = "start") -> float:
        """`s` with its `backticked` spans set in the monospace face, a size smaller; returns its width."""
        runs = []
        for i, part in enumerate(s.split("`")):
            if not part:
                continue
            if i % 2:
                runs.append((part, {"size": size - 1, "weight": 500 if weight >= 500 else 400,
                                    "family": "mono", "fill": code or fill or self.theme.ink}))
            else:
                runs.append((part, {"size": size, "weight": weight, "fill": fill or self.theme.ink}))
        return self.rich(x, y, runs, anchor=anchor)

    def inline_width(self, s: str, *, size: float = 15, weight: int = 400) -> float:
        total = 0.0
        for i, part in enumerate(s.split("`")):
            total += text_width(part, size - 1, 500 if weight >= 500 else 400, "mono") if i % 2 else \
                text_width(part, size, weight)
        return total

    def wrap(self, s: str, width: float, *, size: float = 15, weight: int = 400) -> list[str]:
        """`s` broken into lines no wider than `width`, each `backticked` span kept whole on one line."""
        lines, line = [], ""
        for word in re.findall(r"[^`\s]*`[^`]*`\S*|\S+", s):
            trial = f"{line} {word}" if line else word
            if line and self.inline_width(trial, size=size, weight=weight) > width:
                lines.append(line)
                line = word
            else:
                line = trial
        return [*lines, line] if line else lines

    def box(self, x: float, y: float, w: float, h: float, *, color: str | None = None, r: float = 12,
            fill: str | None = None, alpha: float = 0.08) -> None:
        """A rounded card, tinted with `color` or the theme's panel."""
        tint = self.theme.soft(color, alpha) if color else self.theme.panel
        self.rect(x, y, w, h, r=r, fill=fill or tint, stroke=color or self.theme.panel_line, width=1.25)

    def svg(self, *, title: str, description: str, glyphs: str = "") -> str:
        """The SVG, its fonts holding the glyphs it uses and `glyphs`; frames of one clip pass the same
        `glyphs` so their subsets match and are made once."""
        faces = []
        for (family, weight), chars in sorted(self.used.items()):
            subset_ = _subset((family, weight), frozenset(chars | set(glyphs)))
            faces.append(f"@font-face{{font-family:{FAMILY[family]};font-weight:{weight};"
                         f"src:url(data:font/woff2;base64,{subset_}) format('woff2');}}")
        classes = "".join(f".{f}{{font-family:{FAMILY[f]},{FALLBACK[f]};}}" for f in FAMILY)
        style = f'{"".join(faces)}{classes}text{{font-kerning:normal;white-space:pre;}}'
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
                f'width="{self.width}" height="{self.height}" role="img" aria-labelledby="title desc">'
                f'<title id="title">{html.escape(title)}</title>'
                f'<desc id="desc">{html.escape(description)}</desc>'
                f'<defs><style>{style}</style>{"".join(self.defs)}</defs>{"".join(self.parts)}</svg>')
