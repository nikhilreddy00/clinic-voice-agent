"""Minimal dependency-free SVG line chart for the load-test results.

Deliberately not matplotlib. The output is a committed artifact that has to render in a README
and in a browser on any machine, and adding a plotting stack to a voice-agent repo to draw two
lines is not a trade worth making.
"""

from __future__ import annotations

from pathlib import Path

W, H = 860, 460
PAD_L, PAD_R, PAD_T, PAD_B = 78, 150, 46, 62
PLOT_W, PLOT_H = W - PAD_L - PAD_R, H - PAD_T - PAD_B

# Colors chosen to stay legible on both light and dark backgrounds.
SERIES_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    step = 10 ** (len(str(int(value))) - 1)
    return step * (int(value / step) + 1)


def render(
    path: str | Path,
    *,
    title: str,
    subtitle: str,
    x_values: list[float],
    series: dict[str, list[float | None]],
    y_label: str,
    knee: float | None = None,
    knee_label: str = "",
) -> Path:
    """Write a log-x line chart. ``series`` maps a label to one y per x (None = missing)."""
    y_max = _nice_ceiling(
        max((v for vals in series.values() for v in vals if v is not None), default=1.0) * 1.12
    )
    import math

    lo, hi = math.log10(max(1e-9, min(x_values))), math.log10(max(x_values))
    span = (hi - lo) or 1.0

    def sx(x: float) -> float:
        return PAD_L + (math.log10(x) - lo) / span * PLOT_W

    def sy(y: float) -> float:
        return PAD_T + PLOT_H - (y / y_max) * PLOT_H

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif">',
        # Explicit light background: an SVG with no fill is transparent, and these strokes
        # vanish against a dark README.
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{PAD_L}" y="24" font-size="16" font-weight="600" fill="#0f172a">{title}</text>',
        f'<text x="{PAD_L}" y="40" font-size="11" fill="#64748b">{subtitle}</text>',
    ]

    for i in range(6):  # horizontal gridlines + y ticks
        y = y_max * i / 5
        py = sy(y)
        parts.append(
            f'<line x1="{PAD_L}" y1="{py:.1f}" x2="{PAD_L + PLOT_W}" y2="{py:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 10}" y="{py + 4:.1f}" font-size="11" fill="#64748b" '
            f'text-anchor="end">{y:,.0f}</text>'
        )

    for x in x_values:  # x ticks
        px = sx(x)
        parts.append(
            f'<text x="{px:.1f}" y="{PAD_T + PLOT_H + 20:.1f}" font-size="11" fill="#64748b" '
            f'text-anchor="middle">{int(x)}</text>'
        )

    if knee is not None:
        px = sx(knee)
        parts.append(
            f'<line x1="{px:.1f}" y1="{PAD_T}" x2="{px:.1f}" y2="{PAD_T + PLOT_H}" '
            f'stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5,4"/>'
        )
        parts.append(
            f'<text x="{px + 6:.1f}" y="{PAD_T + 14}" font-size="11" fill="#dc2626" '
            f'font-weight="600">{knee_label}</text>'
        )

    for idx, (label, values) in enumerate(series.items()):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        points = [
            f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(x_values, values) if v is not None
        ]
        if points:
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
                f'stroke-width="2.5" stroke-linejoin="round"/>'
            )
            for point in points:
                cx, cy = point.split(",")
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="{color}"/>')
        ly = PAD_T + 8 + idx * 20
        parts.append(
            f'<line x1="{PAD_L + PLOT_W + 16}" y1="{ly}" x2="{PAD_L + PLOT_W + 40}" y2="{ly}" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        parts.append(
            f'<text x="{PAD_L + PLOT_W + 46}" y="{ly + 4}" font-size="11" '
            f'fill="#334155">{label}</text>'
        )

    parts.append(
        f'<text x="{PAD_L + PLOT_W / 2:.0f}" y="{H - 18}" font-size="12" fill="#334155" '
        f'text-anchor="middle">concurrent sessions (log scale)</text>'
    )
    parts.append(
        f'<text transform="translate(20,{PAD_T + PLOT_H / 2:.0f}) rotate(-90)" font-size="12" '
        f'fill="#334155" text-anchor="middle">{y_label}</text>'
    )
    parts.append("</svg>")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
