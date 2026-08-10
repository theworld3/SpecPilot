"""Small statistics helpers (stdlib only)."""

from __future__ import annotations

from typing import Dict, List, Sequence

__all__ = ["percentile", "summarize", "format_table"]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` in [0, 100]."""
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    frac = pos - lo
    return data[lo] * (1 - frac) + data[hi] * frac


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def format_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    """Render a Markdown table. Used by the benchmark CLI so results can be
    pasted straight into the README without a plotting dependency."""
    if not rows:
        return ""
    widths = {c: len(c) for c in columns}
    cells = []
    for row in rows:
        rendered = {}
        for c in columns:
            v = row.get(c, "")
            s = f"{v:.3f}" if isinstance(v, float) else str(v)
            rendered[c] = s
            widths[c] = max(widths[c], len(s))
        cells.append(rendered)

    header = "| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |"
    sep = "| " + " | ".join("-" * widths[c] for c in columns) + " |"
    body = [
        "| " + " | ".join(r[c].ljust(widths[c]) for c in columns) + " |" for r in cells
    ]
    return "\n".join([header, sep, *body])
