"""Build a single-file HTML dashboard from a Prism run directory.

Reads `<work_dir>/prism.db` + optionally walks `<work_dir>/artifacts/` for
per-response reasoning text, emits `<work_dir>/dashboard.html` —
self-contained (all data + JS inlined, Chart.js via CDN).

Usage:
    uv run python scripts/build_dashboard.py /tmp/pandora-run
    uv run python scripts/build_dashboard.py /tmp/pandora-run --output /tmp/custom.html
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def _benchmark_of(prompt_id: str) -> str:
    """`ceval-advanced_mathematics-0` -> `ceval`; `ruler_mk-len1024-...` -> `ruler_mk`."""
    for prefix in ("ruler_mk", "mmlu_pro", "toy_agent"):
        if prompt_id.startswith(prefix + "-"):
            return prefix
    return prompt_id.split("-", 1)[0]


def _ceval_subject(prompt_id: str) -> str | None:
    if not prompt_id.startswith("ceval-"):
        return None
    rest = prompt_id[len("ceval-"):]
    return rest.rsplit("-", 1)[0] if "-" in rest else None


def _niah_length(prompt_id: str) -> int | None:
    for marker in ("-len",):
        i = prompt_id.find(marker)
        if i != -1:
            tail = prompt_id[i + len(marker):]
            n = ""
            for ch in tail:
                if ch.isdigit():
                    n += ch
                else:
                    break
            return int(n) if n else None
    return None


def build(db_path: Path, artifacts_root: Path | None) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT r.id, r.run_id, r.prompt_id, r.seed, r.text, r.reasoning_text,
               r.tokens_in, r.tokens_out, r.latency_ms, r.cost_usd, r.finish_reason,
               r.model_id, s.score, s.judge, s.reasoning AS judge_reasoning
        FROM responses r LEFT JOIN scores s ON s.response_id = r.id
        ORDER BY r.id
    """)
    rows = [dict(r) for r in cur.fetchall()]

    # `expected` is nullable (column may not exist in old DBs, so tolerate).
    try:
        cur.execute("SELECT id, text AS prompt_text, expected FROM prompts")
        prompt_info = {r["id"]: (r["prompt_text"], r["expected"]) for r in cur.fetchall()}
    except sqlite3.OperationalError:
        cur.execute("SELECT id, text AS prompt_text FROM prompts")
        prompt_info = {r["id"]: (r["prompt_text"], None) for r in cur.fetchall()}

    conn.close()

    import re as _re
    _letter_re = _re.compile(r"Answer:\s*([A-Za-z])", _re.IGNORECASE)
    for r in rows:
        r["benchmark"] = _benchmark_of(r["prompt_id"])
        pt, exp = prompt_info.get(r["prompt_id"], ("", None))
        r["prompt_text"] = pt
        r["expected"] = exp
        # Best-effort extract model's chosen letter for MCQ benchmarks
        m = _letter_re.search(r["text"] or "") if r["text"] else None
        r["model_letter"] = m.group(1).upper() if m else None
        if r["score"] is None:
            r["score"] = None
            r["pass"] = None
        else:
            r["pass"] = bool(r["score"] >= 1.0)

    per_bm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_bm[r["benchmark"]].append(r)

    bm_stats = []
    for bm, lst in sorted(per_bm.items()):
        n = len(lst)
        scored = [r for r in lst if r["score"] is not None]
        passed = sum(1 for r in scored if r["pass"])
        total_cost = sum(r["cost_usd"] or 0.0 for r in lst)
        total_in = sum(r["tokens_in"] or 0 for r in lst)
        total_out = sum(r["tokens_out"] or 0 for r in lst)
        avg_lat = sum(r["latency_ms"] or 0.0 for r in lst) / n if n else 0.0
        pass_rate = passed / len(scored) if scored else None
        bm_stats.append({
            "benchmark": bm,
            "n": n,
            "passed": passed,
            "scored": len(scored),
            "pass_rate": pass_rate,
            "cost": total_cost,
            "tokens_in": total_in,
            "tokens_out": total_out,
            "avg_latency_ms": avg_lat,
        })

    ceval_subjects: dict[str, dict] = defaultdict(lambda: {"n": 0, "passed": 0})
    for r in per_bm.get("ceval", []):
        subj = _ceval_subject(r["prompt_id"])
        if not subj:
            continue
        ceval_subjects[subj]["n"] += 1
        if r["pass"]:
            ceval_subjects[subj]["passed"] += 1
    ceval_breakdown = sorted(
        (
            {
                "subject": k,
                "n": v["n"],
                "passed": v["passed"],
                "pass_rate": v["passed"] / v["n"] if v["n"] else 0.0,
            }
            for k, v in ceval_subjects.items()
        ),
        key=lambda x: -x["pass_rate"],
    )

    lc_points: list[dict] = []
    for bm in ("niah", "ruler_mk"):
        for r in per_bm.get(bm, []):
            length = _niah_length(r["prompt_id"])
            if length is None:
                continue
            lc_points.append({
                "benchmark": bm,
                "context": length,
                "latency_ms": r["latency_ms"],
                "pass": r["pass"],
                "prompt_id": r["prompt_id"],
                "cost": r["cost_usd"],
            })
    lc_points.sort(key=lambda x: (x["benchmark"], x["context"]))

    total_n = len(rows)
    total_cost = sum(r["cost_usd"] or 0.0 for r in rows)
    total_in = sum(r["tokens_in"] or 0 for r in rows)
    total_out = sum(r["tokens_out"] or 0 for r in rows)
    scored_all = [r for r in rows if r["score"] is not None]
    passed_all = sum(1 for r in scored_all if r["pass"])

    compact_rows = [
        {
            "id": r["id"],
            "run_id": r["run_id"],
            "benchmark": r["benchmark"],
            "prompt_id": r["prompt_id"],
            "score": r["score"],
            "pass": r["pass"],
            "cost_usd": round(r["cost_usd"] or 0.0, 4),
            "tokens_in": r["tokens_in"],
            "tokens_out": r["tokens_out"],
            "latency_ms": round(r["latency_ms"] or 0.0, 1),
            "text": r["text"] or "",
            "reasoning_text": r["reasoning_text"] or "",
            "judge": r["judge"] or "",
            "judge_reasoning": r["judge_reasoning"] or "",
            "prompt_text": r["prompt_text"],
            "finish_reason": r["finish_reason"] or "",
            "expected": r["expected"],
            "model_letter": r["model_letter"],
        }
        for r in rows
    ]

    return {
        "summary": {
            "total_responses": total_n,
            "total_scored": len(scored_all),
            "total_passed": passed_all,
            "overall_pass_rate": (passed_all / len(scored_all)) if scored_all else 0.0,
            "total_cost_usd": total_cost,
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
        },
        "benchmarks": bm_stats,
        "ceval": ceval_breakdown,
        "long_context": lc_points,
        "rows": compact_rows,
    }


def render_html(data: dict, title: str) -> str:
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Segoe UI", sans-serif;
    margin: 0; padding: 0; background: #0f1419; color: #e6edf3;
    font-size: 14px; line-height: 1.5;
}}
header {{
    padding: 24px 32px; background: linear-gradient(135deg, #1f2937, #111827);
    border-bottom: 1px solid #2d3748;
}}
h1 {{ margin: 0 0 8px 0; font-size: 22px; font-weight: 600; }}
h2 {{ margin: 24px 0 12px 0; font-size: 16px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }}
.subtitle {{ color: #9ca3af; font-size: 13px; }}
main {{ padding: 24px 32px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 8px; }}
.kpi {{ background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; padding: 16px; }}
.kpi-label {{ color: #9ca3af; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
.kpi-value {{ font-size: 22px; font-weight: 600; margin-top: 6px; color: #f3f4f6; }}
.kpi-sub {{ color: #6b7280; font-size: 11px; margin-top: 4px; }}
.bm-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; margin-bottom: 24px; }}
.bm-card {{ background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; padding: 14px; cursor: pointer; transition: all 0.15s; }}
.bm-card:hover {{ border-color: #4a5568; transform: translateY(-1px); }}
.bm-card.selected {{ border-color: #60a5fa; box-shadow: 0 0 0 1px #60a5fa; }}
.bm-name {{ font-weight: 600; font-size: 13px; margin-bottom: 6px; }}
.bm-bar {{ height: 6px; background: #374151; border-radius: 3px; margin: 8px 0; overflow: hidden; }}
.bm-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
.bm-stats {{ display: flex; justify-content: space-between; font-size: 12px; color: #9ca3af; }}
.bm-pct {{ font-size: 20px; font-weight: 700; }}
.chart-row {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 24px; }}
.chart-card {{ background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; padding: 16px; min-height: 320px; }}
@media (max-width: 1024px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
.filter-bar {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }}
.filter-bar select, .filter-bar input, .filter-bar button {{
    background: #1a202c; border: 1px solid #2d3748; color: #e6edf3; padding: 6px 10px; border-radius: 6px; font-size: 13px;
}}
.filter-bar button {{ cursor: pointer; }}
.filter-bar button.active {{ background: #60a5fa; color: #0f1419; border-color: #60a5fa; }}
table {{ width: 100%; border-collapse: collapse; background: #1a202c; border-radius: 8px; overflow: hidden; font-size: 12.5px; }}
th {{ text-align: left; padding: 10px 12px; background: #2d3748; font-weight: 600; color: #9ca3af; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; position: sticky; top: 0; }}
td {{ padding: 10px 12px; border-top: 1px solid #2d3748; }}
tr.row-pass td {{ border-left: 2px solid #22c55e; }}
tr.row-fail td {{ border-left: 2px solid #ef4444; }}
tr.row-unscored td {{ border-left: 2px solid #6b7280; }}
tr[data-idx] {{ cursor: pointer; }}
tr[data-idx]:hover {{ background: #242d3d; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
.badge-pass {{ background: #14532d; color: #86efac; }}
.badge-fail {{ background: #7f1d1d; color: #fca5a5; }}
.badge-none {{ background: #374151; color: #9ca3af; }}
.bm-chip {{ display: inline-block; padding: 2px 6px; background: #374151; border-radius: 3px; font-size: 11px; color: #d1d5db; }}
.mono {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 11.5px; color: #d1d5db; }}
.modal-backdrop {{
    position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7);
    display: none; z-index: 100; align-items: center; justify-content: center;
}}
.modal-backdrop.show {{ display: flex; }}
.modal {{
    background: #1a202c; border: 1px solid #4a5568; border-radius: 10px;
    max-width: 1000px; width: 92%; max-height: 88vh; overflow: hidden;
    display: flex; flex-direction: column;
}}
.modal-header {{ padding: 16px 20px; border-bottom: 1px solid #2d3748; display: flex; justify-content: space-between; align-items: center; }}
.modal-body {{ padding: 20px; overflow-y: auto; flex: 1; }}
.modal-close {{ background: none; border: none; color: #9ca3af; font-size: 22px; cursor: pointer; padding: 0 8px; }}
.modal-section {{ margin-bottom: 20px; }}
.modal-section h4 {{ color: #9ca3af; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 6px 0; }}
.modal-section pre {{
    background: #0f1419; border: 1px solid #2d3748; border-radius: 6px;
    padding: 10px; overflow-x: auto; white-space: pre-wrap; word-break: break-word;
    font-family: "SF Mono", Menlo, monospace; font-size: 12px; color: #e6edf3;
    max-height: 400px; overflow-y: auto; margin: 0;
}}
.meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-bottom: 16px; }}
.meta-item {{ background: #0f1419; padding: 8px 10px; border-radius: 4px; }}
.meta-label {{ font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }}
.meta-value {{ font-size: 13px; color: #e6edf3; margin-top: 2px; font-weight: 500; }}
.row-count {{ color: #9ca3af; font-size: 12px; margin: 8px 0; }}
</style>
</head>
<body>
<header>
    <h1>{title}</h1>
    <div class="subtitle" id="subtitle"></div>
</header>
<main>
    <div class="kpi-grid" id="kpi-grid"></div>
    <h2>按 Benchmark 成绩</h2>
    <div class="bm-grid" id="bm-grid"></div>

    <div class="chart-row">
        <div class="chart-card"><canvas id="chart-bm"></canvas></div>
        <div class="chart-card"><canvas id="chart-ceval"></canvas></div>
    </div>

    <div class="chart-card" style="margin-bottom: 24px;"><canvas id="chart-lc"></canvas></div>

    <h2>单题明细</h2>
    <div class="filter-bar">
        <select id="filter-bm"><option value="">全部 benchmark</option></select>
        <button data-f="all" class="active">全部</button>
        <button data-f="pass">通过</button>
        <button data-f="fail">失败</button>
        <input id="filter-search" placeholder="搜 prompt_id / 答案 / reasoning..." style="flex: 1; min-width: 200px;">
        <span class="row-count" id="row-count"></span>
    </div>
    <div style="max-height: 70vh; overflow: auto; border-radius: 8px;">
        <table id="rows-table">
            <thead>
                <tr><th>Benchmark</th><th>Prompt ID</th><th>Score</th><th>模型 / 正确</th><th>Cost</th><th>Tokens</th><th>Latency</th><th>Preview</th></tr>
            </thead>
            <tbody id="rows-body"></tbody>
        </table>
    </div>
</main>

<div class="modal-backdrop" id="modal-bg">
    <div class="modal">
        <div class="modal-header">
            <div>
                <strong id="modal-title"></strong>
                <span id="modal-badge" style="margin-left: 8px;"></span>
            </div>
            <button class="modal-close" onclick="closeModal()">×</button>
        </div>
        <div class="modal-body" id="modal-body"></div>
    </div>
</div>

<script>
const DATA = {data_json};

// ---------- Utility ----------
const fmtPct = x => x == null ? "—" : (x * 100).toFixed(1) + "%";
const fmtCost = x => "$" + (x || 0).toFixed(2);
const fmtMs = x => x == null ? "—" : (x >= 1000 ? (x / 1000).toFixed(1) + "s" : Math.round(x) + "ms");
const fmtTokens = x => x == null ? "—" : x >= 1e6 ? (x/1e6).toFixed(2) + "M" : x >= 1e3 ? (x/1e3).toFixed(1) + "k" : String(x);

function colorForRate(r) {{
    if (r == null) return "#6b7280";
    if (r >= 0.9) return "#22c55e";
    if (r >= 0.8) return "#84cc16";
    if (r >= 0.7) return "#eab308";
    if (r >= 0.5) return "#f97316";
    return "#ef4444";
}}

// ---------- KPIs ----------
const s = DATA.summary;
document.getElementById("subtitle").textContent =
    `${{s.total_responses}} 题 · ${{fmtPct(s.overall_pass_rate)}} 通过 · ${{fmtCost(s.total_cost_usd)}} · ${{fmtTokens(s.total_tokens_in + s.total_tokens_out)}} tokens`;
document.getElementById("kpi-grid").innerHTML = [
    {{label: "总题数", value: s.total_responses}},
    {{label: "通过率", value: fmtPct(s.overall_pass_rate), sub: `${{s.total_passed}} / ${{s.total_scored}}`}},
    {{label: "累计花费", value: fmtCost(s.total_cost_usd)}},
    {{label: "Tokens 输入", value: fmtTokens(s.total_tokens_in)}},
    {{label: "Tokens 输出", value: fmtTokens(s.total_tokens_out)}},
    {{label: "Benchmark", value: DATA.benchmarks.length}},
].map(k => `<div class="kpi"><div class="kpi-label">${{k.label}}</div><div class="kpi-value">${{k.value}}</div>${{k.sub ? `<div class="kpi-sub">${{k.sub}}</div>` : ""}}</div>`).join("");

// ---------- Benchmark cards ----------
const bmGrid = document.getElementById("bm-grid");
DATA.benchmarks.forEach(b => {{
    const rate = b.pass_rate;
    const card = document.createElement("div");
    card.className = "bm-card";
    card.dataset.bm = b.benchmark;
    card.innerHTML = `
        <div class="bm-name">${{b.benchmark}}</div>
        <div class="bm-pct" style="color: ${{colorForRate(rate)}}">${{fmtPct(rate)}}</div>
        <div class="bm-bar"><div class="bm-bar-fill" style="width: ${{((rate||0)*100).toFixed(1)}}%; background: ${{colorForRate(rate)}}"></div></div>
        <div class="bm-stats">
            <span>${{b.passed}}/${{b.scored}} pass · ${{b.n}} 题</span>
            <span>${{fmtCost(b.cost)}}</span>
        </div>
    `;
    card.onclick = () => {{
        const sel = document.getElementById("filter-bm");
        sel.value = sel.value === b.benchmark ? "" : b.benchmark;
        document.querySelectorAll(".bm-card").forEach(c => c.classList.remove("selected"));
        if (sel.value) card.classList.add("selected");
        renderTable();
    }};
    bmGrid.appendChild(card);
}});

// ---------- Chart: pass rate per benchmark ----------
new Chart(document.getElementById("chart-bm"), {{
    type: "bar",
    data: {{
        labels: DATA.benchmarks.map(b => b.benchmark),
        datasets: [{{
            label: "pass@1",
            data: DATA.benchmarks.map(b => (b.pass_rate || 0) * 100),
            backgroundColor: DATA.benchmarks.map(b => colorForRate(b.pass_rate)),
            borderRadius: 4,
        }}]
    }},
    options: {{
        indexAxis: "y",
        plugins: {{
            title: {{display: true, text: "各 Benchmark pass@1 (%)", color: "#e6edf3", font: {{size: 14}}}},
            legend: {{display: false}}
        }},
        scales: {{
            x: {{min: 0, max: 100, grid: {{color: "#2d3748"}}, ticks: {{color: "#9ca3af"}}}},
            y: {{grid: {{color: "#2d3748"}}, ticks: {{color: "#e6edf3", font: {{size: 11}}}}}}
        }},
    }}
}});

// ---------- Chart: C-Eval subjects ----------
if (DATA.ceval && DATA.ceval.length) {{
    new Chart(document.getElementById("chart-ceval"), {{
        type: "bar",
        data: {{
            labels: DATA.ceval.map(c => c.subject.replace(/_/g, " ")),
            datasets: [{{
                data: DATA.ceval.map(c => c.pass_rate * 100),
                backgroundColor: DATA.ceval.map(c => colorForRate(c.pass_rate)),
                borderRadius: 4,
            }}]
        }},
        options: {{
            indexAxis: "y",
            plugins: {{title: {{display: true, text: "C-Eval 8 学科", color: "#e6edf3"}}, legend: {{display: false}}}},
            scales: {{
                x: {{min: 0, max: 100, grid: {{color: "#2d3748"}}, ticks: {{color: "#9ca3af"}}}},
                y: {{grid: {{color: "#2d3748"}}, ticks: {{color: "#e6edf3", font: {{size: 10}}}}}}
            }}
        }}
    }});
}}

// ---------- Chart: latency vs context length ----------
if (DATA.long_context && DATA.long_context.length) {{
    const byBm = {{}};
    DATA.long_context.forEach(p => {{
        if (!byBm[p.benchmark]) byBm[p.benchmark] = [];
        byBm[p.benchmark].push({{x: p.context, y: p.latency_ms / 1000, pass: p.pass}});
    }});
    const colors = {{niah: "#60a5fa", ruler_mk: "#f472b6"}};
    new Chart(document.getElementById("chart-lc"), {{
        type: "scatter",
        data: {{
            datasets: Object.entries(byBm).map(([bm, pts]) => ({{
                label: bm,
                data: pts,
                backgroundColor: pts.map(p => p.pass ? colors[bm] : "#ef4444"),
                borderColor: colors[bm],
                pointRadius: 6,
                pointHoverRadius: 8,
            }}))
        }},
        options: {{
            plugins: {{title: {{display: true, text: "长上下文 latency × context (秒 × tokens)", color: "#e6edf3"}}, legend: {{labels: {{color: "#e6edf3"}}}}}},
            scales: {{
                x: {{type: "logarithmic", title: {{display: true, text: "上下文 tokens", color: "#9ca3af"}}, grid: {{color: "#2d3748"}}, ticks: {{color: "#9ca3af"}}}},
                y: {{title: {{display: true, text: "延迟 (s)", color: "#9ca3af"}}, grid: {{color: "#2d3748"}}, ticks: {{color: "#9ca3af"}}}}
            }}
        }}
    }});
}}

// ---------- Table + filters ----------
const filterBmSel = document.getElementById("filter-bm");
DATA.benchmarks.forEach(b => {{
    const o = document.createElement("option");
    o.value = b.benchmark; o.textContent = `${{b.benchmark}} (${{b.n}})`;
    filterBmSel.appendChild(o);
}});

let filterPass = "all";
document.querySelectorAll(".filter-bar button[data-f]").forEach(btn => {{
    btn.onclick = () => {{
        document.querySelectorAll(".filter-bar button[data-f]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        filterPass = btn.dataset.f;
        renderTable();
    }};
}});

const searchInput = document.getElementById("filter-search");
searchInput.addEventListener("input", () => renderTable());
filterBmSel.addEventListener("change", () => {{
    document.querySelectorAll(".bm-card").forEach(c => c.classList.toggle("selected", c.dataset.bm === filterBmSel.value));
    renderTable();
}});

function renderTable() {{
    const bmFilter = filterBmSel.value;
    const q = searchInput.value.toLowerCase().trim();
    const body = document.getElementById("rows-body");
    body.innerHTML = "";
    let n = 0;
    DATA.rows.forEach((r, idx) => {{
        if (bmFilter && r.benchmark !== bmFilter) return;
        if (filterPass === "pass" && !r.pass) return;
        if (filterPass === "fail" && (r.pass !== false)) return;
        if (q) {{
            const hay = (r.prompt_id + " " + r.text + " " + r.reasoning_text + " " + r.judge_reasoning).toLowerCase();
            if (!hay.includes(q)) return;
        }}
        n += 1;
        const tr = document.createElement("tr");
        tr.dataset.idx = idx;
        tr.className = r.pass === true ? "row-pass" : r.pass === false ? "row-fail" : "row-unscored";
        const badge = r.pass === true ? '<span class="badge badge-pass">PASS</span>' :
                      r.pass === false ? '<span class="badge badge-fail">FAIL</span>' :
                      '<span class="badge badge-none">—</span>';
        const preview = (r.text || "").replace(/\\n/g, " ").slice(0, 80);
        // Show model-letter vs expected-letter for quick MCQ diagnosis
        let answerCell = "—";
        if (r.model_letter && r.expected) {{
            const expLetter = r.expected.length === 1 ? r.expected.toUpperCase() : r.expected;
            const match = r.model_letter === expLetter;
            answerCell = `<span class="mono" style="color: ${{match ? '#86efac' : '#fca5a5'}}">${{r.model_letter}}</span> / <span class="mono" style="color: #93c5fd">${{expLetter}}</span>`;
        }} else if (r.expected) {{
            // Free-form expected (math, simpleqa, etc.)
            const exp = r.expected.length > 20 ? r.expected.slice(0, 18) + "…" : r.expected;
            answerCell = `<span class="mono" style="color: #93c5fd">${{exp}}</span>`;
        }}
        tr.innerHTML = `
            <td><span class="bm-chip">${{r.benchmark}}</span></td>
            <td class="mono">${{r.prompt_id}}</td>
            <td>${{badge}} ${{r.score != null ? '<span class="mono">' + r.score.toFixed(2) + '</span>' : ""}}</td>
            <td>${{answerCell}}</td>
            <td class="mono">${{fmtCost(r.cost_usd)}}</td>
            <td class="mono">${{fmtTokens(r.tokens_in)}}/${{fmtTokens(r.tokens_out)}}</td>
            <td class="mono">${{fmtMs(r.latency_ms)}}</td>
            <td style="color: #9ca3af;">${{preview}}</td>
        `;
        tr.onclick = () => openModal(idx);
        body.appendChild(tr);
    }});
    document.getElementById("row-count").textContent = `显示 ${{n}} / ${{DATA.rows.length}}`;
}}

// ---------- Modal ----------
function openModal(idx) {{
    const r = DATA.rows[idx];
    document.getElementById("modal-title").textContent = r.prompt_id;
    document.getElementById("modal-badge").innerHTML =
        r.pass === true ? '<span class="badge badge-pass">PASS</span>' :
        r.pass === false ? '<span class="badge badge-fail">FAIL</span>' :
        '<span class="badge badge-none">未判分</span>';
    const body = document.getElementById("modal-body");
    const meta = `
        <div class="meta-grid">
            <div class="meta-item"><div class="meta-label">Benchmark</div><div class="meta-value">${{r.benchmark}}</div></div>
            <div class="meta-item"><div class="meta-label">Score</div><div class="meta-value">${{r.score != null ? r.score.toFixed(3) : "—"}}</div></div>
            <div class="meta-item"><div class="meta-label">模型答案</div><div class="meta-value" style="color: ${{r.pass ? '#86efac' : r.pass === false ? '#fca5a5' : '#e6edf3'}}">${{r.model_letter || "—"}}</div></div>
            <div class="meta-item"><div class="meta-label">正确答案</div><div class="meta-value" style="color: #93c5fd">${{r.expected != null ? (r.expected.length > 40 ? r.expected.slice(0,38)+"…" : r.expected) : "—"}}</div></div>
            <div class="meta-item"><div class="meta-label">Judge</div><div class="meta-value">${{r.judge || "—"}}</div></div>
            <div class="meta-item"><div class="meta-label">Cost</div><div class="meta-value">${{fmtCost(r.cost_usd)}}</div></div>
            <div class="meta-item"><div class="meta-label">Tokens in / out</div><div class="meta-value">${{fmtTokens(r.tokens_in)}} / ${{fmtTokens(r.tokens_out)}}</div></div>
            <div class="meta-item"><div class="meta-label">Latency</div><div class="meta-value">${{fmtMs(r.latency_ms)}}</div></div>
            <div class="meta-item"><div class="meta-label">Finish</div><div class="meta-value">${{r.finish_reason || "—"}}</div></div>
            <div class="meta-item"><div class="meta-label">Run</div><div class="meta-value mono" style="font-size:11px;">${{r.run_id}}</div></div>
        </div>`;
    const sections = [
        {{title: "Prompt（题面，前 2000 字）", content: (r.prompt_text || "").slice(0, 2000)}},
        {{title: "Model Text Response", content: r.text || "(空)"}},
        {{title: "Reasoning / Thinking", content: r.reasoning_text || "(无)"}},
        {{title: "Judge Reasoning", content: r.judge_reasoning || "(无)"}},
    ];
    body.innerHTML = meta + sections.map(s => `<div class="modal-section"><h4>${{s.title}}</h4><pre>${{escapeHtml(s.content)}}</pre></div>`).join("");
    document.getElementById("modal-bg").classList.add("show");
}}
function closeModal() {{ document.getElementById("modal-bg").classList.remove("show"); }}
function escapeHtml(s) {{
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}}
document.getElementById("modal-bg").addEventListener("click", (e) => {{
    if (e.target.id === "modal-bg") closeModal();
}});
document.addEventListener("keydown", (e) => {{ if (e.key === "Escape") closeModal(); }});

renderTable();
</script>
</body>
</html>""".format(title=title, data_json=data_json)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a single-file HTML eval dashboard.")
    ap.add_argument("work_dir", help="Prism work directory (contains prism.db + artifacts/)")
    ap.add_argument("--output", help="Output HTML file path (default: <work_dir>/dashboard.html)")
    ap.add_argument("--title", default="Prism 评测看板", help="Dashboard title")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    db_path = work_dir / "prism.db"
    if not db_path.exists():
        raise SystemExit(f"Can't find {db_path}")
    artifacts_root = work_dir / "artifacts"
    output = Path(args.output) if args.output else (work_dir / "dashboard.html")

    data = build(db_path, artifacts_root if artifacts_root.exists() else None)
    html = render_html(data, title=args.title)
    output.write_text(html, encoding="utf-8")
    s = data["summary"]
    print(f"OK: {output}")
    print(f"  {s['total_responses']} responses, pass_rate={s['overall_pass_rate']:.1%}, cost=${s['total_cost_usd']:.2f}")


if __name__ == "__main__":
    main()
