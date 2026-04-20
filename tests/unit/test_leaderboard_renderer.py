from pathlib import Path

from prism.leaderboard.renderer import render_leaderboard, render_leaderboard_html


def test_render_main_table_contains_models_and_benchmarks():
    data = {
        "main": [
            {"model_id": "gpt-5@high", "benchmark": "mmlu_pro",
             "mean_score": 0.85, "count": 100, "total_cost": 1.5},
            {"model_id": "gpt-5@high", "benchmark": "niah",
             "mean_score": 0.92, "count": 30, "total_cost": 0.3},
            {"model_id": "claude-opus@max", "benchmark": "mmlu_pro",
             "mean_score": 0.88, "count": 100, "total_cost": 2.0},
            {"model_id": "claude-opus@max", "benchmark": "niah",
             "mean_score": 0.95, "count": 30, "total_cost": 0.4},
        ],
        "staircase": [],
        "sweep_groups": [],
    }
    html = render_leaderboard_html(data)
    assert "<table" in html
    assert "gpt-5@high" in html
    assert "claude-opus@max" in html
    assert "mmlu_pro" in html
    assert "niah" in html
    assert "85" in html
    assert "95" in html


def test_render_escapes_html_in_model_id():
    data = {
        "main": [
            {"model_id": "<script>alert('xss')</script>", "benchmark": "mmlu_pro",
             "mean_score": 0.5, "count": 1, "total_cost": 0.0},
        ],
        "staircase": [],
        "sweep_groups": [],
    }
    html = render_leaderboard_html(data)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html or "&lt;" in html


def test_render_staircase_section_when_present():
    data = {
        "main": [],
        "staircase": [
            {"model_id": "m1", "context_tokens": 1024, "mean_score": 1.0, "count": 3},
            {"model_id": "m1", "context_tokens": 4096, "mean_score": 0.66, "count": 3},
            {"model_id": "m1", "context_tokens": 16384, "mean_score": 0.33, "count": 3},
        ],
        "sweep_groups": [],
    }
    html = render_leaderboard_html(data)
    assert "Context Length Staircase" in html or "staircase" in html.lower()
    assert "1024" in html
    assert "16384" in html


def test_render_sweep_section_when_present():
    data = {
        "main": [
            {"model_id": "gpt-5@high", "benchmark": "mmlu_pro",
             "mean_score": 0.8, "count": 100, "total_cost": 1.0},
            {"model_id": "gpt-5@max", "benchmark": "mmlu_pro",
             "mean_score": 0.9, "count": 100, "total_cost": 2.0},
        ],
        "staircase": [],
        "sweep_groups": [
            {"base": "openai/gpt-5", "variants": ["gpt-5@high", "gpt-5@max"],
             "efforts": {"gpt-5@high": "high", "gpt-5@max": "max"}},
        ],
    }
    html = render_leaderboard_html(data)
    assert "Reasoning Effort Sweep" in html or "sweep" in html.lower()
    assert "openai/gpt-5" in html


def test_render_empty_data_produces_valid_html():
    data = {"main": [], "staircase": [], "sweep_groups": []}
    html = render_leaderboard_html(data)
    assert "<html" in html
    assert "</html>" in html
    assert (
        "no data" in html.lower()
        or "empty" in html.lower()
        or "0 benchmarks" in html.lower()
        or "prism run" in html.lower()
    )


def test_render_writes_files(tmp_path: Path):
    """render_leaderboard should write index.html + data.json to output_dir."""
    data = {"main": [], "staircase": [], "sweep_groups": []}
    html_path = render_leaderboard(data, output_dir=tmp_path / "out")
    assert html_path.exists()
    assert (tmp_path / "out" / "data.json").exists()
