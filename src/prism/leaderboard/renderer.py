"""Pure-Python HTML leaderboard renderer."""
from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

from prism.leaderboard import templates as tmpl


def _esc(s: Any) -> str:
    return _html.escape(str(s), quote=True)


def _score_class(score: float) -> str:
    if score >= 0.8:
        return "score-high"
    if score >= 0.5:
        return "score-mid"
    return "score-low"


def _render_main(main_rows: list[dict[str, Any]]) -> str:
    if not main_rows:
        return ""
    models = sorted({r["model_id"] for r in main_rows})
    benchmarks = sorted({r["benchmark"] for r in main_rows})
    by_cell: dict[tuple[str, str], dict[str, Any]] = {
        (r["model_id"], r["benchmark"]): r for r in main_rows
    }

    header_cells = "".join(f"<th>{_esc(b)}</th>" for b in benchmarks)
    row_html_parts = []
    for m in models:
        cells = [f'<td class="model">{_esc(m)}</td>']
        for b in benchmarks:
            c = by_cell.get((m, b))
            if c is None:
                cells.append('<td class="empty">—</td>')
            else:
                pct = int(round(c["mean_score"] * 100))
                cls = _score_class(c["mean_score"])
                cells.append(f'<td class="{cls}">{pct}%</td>')
        row_html_parts.append("<tr>" + "".join(cells) + "</tr>")
    rows = "\n".join(row_html_parts)
    return tmpl.MAIN_HEADER.format(benchmark_headers=header_cells, rows=rows)


def _render_staircase(staircase_rows: list[dict[str, Any]]) -> str:
    if not staircase_rows:
        return ""
    models = sorted({r["model_id"] for r in staircase_rows})
    lengths = sorted({r["context_tokens"] for r in staircase_rows})
    by_cell: dict[tuple[str, int], dict[str, Any]] = {
        (r["model_id"], r["context_tokens"]): r for r in staircase_rows
    }
    header_cells = "".join(f"<th>{_esc(ctx_len)}</th>" for ctx_len in lengths)
    row_html_parts = []
    for m in models:
        cells = [f'<td class="model">{_esc(m)}</td>']
        for ctx_len in lengths:
            c = by_cell.get((m, ctx_len))
            if c is None:
                cells.append('<td class="empty">—</td>')
            else:
                pct = int(round(c["mean_score"] * 100))
                cls = _score_class(c["mean_score"])
                cells.append(f'<td class="{cls}">{pct}%</td>')
        row_html_parts.append("<tr>" + "".join(cells) + "</tr>")
    rows = "\n".join(row_html_parts)
    return tmpl.STAIRCASE_HEADER.format(length_headers=header_cells, rows=rows)


def _render_sweep(sweep_groups: list[dict[str, Any]], main_rows: list[dict[str, Any]]) -> str:
    if not sweep_groups:
        return ""
    score_by_model: dict[str, list[float]] = {}
    cost_by_model: dict[str, float] = {}
    for r in main_rows:
        score_by_model.setdefault(r["model_id"], []).append(r["mean_score"])
        cost_by_model[r["model_id"]] = cost_by_model.get(r["model_id"], 0.0) + r["total_cost"]

    row_parts = []
    for g in sweep_groups:
        base = g["base"]
        for variant in g["variants"]:
            effort = g["efforts"].get(variant) or "?"
            scores = score_by_model.get(variant, [])
            mean = sum(scores) / len(scores) if scores else 0.0
            cost = cost_by_model.get(variant, 0.0)
            pct = int(round(mean * 100))
            cls = _score_class(mean)
            row_parts.append(
                f"<tr><td>{_esc(base)}</td>"
                f"<td>{_esc(variant)}</td>"
                f"<td>{_esc(effort)}</td>"
                f'<td class="{cls}">{pct}%</td>'
                f"<td>${cost:.4f}</td></tr>"
            )
    rows = "\n".join(row_parts)
    return tmpl.SWEEP_HEADER.format(rows=rows)


def render_leaderboard_html(data: dict[str, Any]) -> str:
    main = _render_main(data.get("main", []))
    staircase = _render_staircase(data.get("staircase", []))
    sweep = _render_sweep(data.get("sweep_groups", []), data.get("main", []))
    if not any([main, staircase, sweep]):
        main = tmpl.EMPTY_STATE
    return tmpl.PAGE_TEMPLATE.format(
        main_section=main,
        staircase_section=staircase,
        sweep_section=sweep,
    )


def render_leaderboard(data: dict[str, Any], *, output_dir: Path | str) -> Path:
    """Write index.html + data.json to output_dir. Returns the HTML path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    html = render_leaderboard_html(data)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out / "index.html"
