import asyncio
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from prism import __version__
from prism.adapters.litellm_adapter import LiteLLMAdapter
from prism.agent import agent_registry
from prism.benchmarks import default_registry
from prism.config.loader import load_model_profile
from prism.leaderboard import (
    aggregate_by_model_benchmark,
    aggregate_staircase,
    list_thinking_variants,
    render_leaderboard,
)
from prism.runners.agent import AgentRunner
from prism.runners.limit import LimitRunner
from prism.service import RunService
from prism.storage.database import Database

app = typer.Typer(
    name="prism",
    help="Prism — benchmark frontier LLMs to their limits.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print version."""
    console.print(f"prism {__version__}")


@app.command()
def doctor() -> None:
    """Check runtime environment."""
    table = Table(title="Prism Doctor")
    table.add_column("Check")
    table.add_column("Value")
    table.add_column("Status")

    # Python
    ok_py = sys.version_info >= (3, 11)
    table.add_row("python", sys.version.split()[0], "OK" if ok_py else "FAIL")

    # litellm
    try:
        litellm_version = importlib.metadata.version("litellm")
        table.add_row("litellm", litellm_version, "OK")
    except importlib.metadata.PackageNotFoundError:
        table.add_row("litellm", "-", "FAIL")
        ok_py = False

    # API keys for all supported providers
    _PROVIDER_ENV_KEYS = [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("xai", "XAI_API_KEY"),
        ("kimi", "KIMI_API_KEY"),
        ("qwen", "DASHSCOPE_API_KEY"),  # Qwen's DashScope API
    ]
    for provider_name, env in _PROVIDER_ENV_KEYS:
        value = os.environ.get(env)
        if value:
            table.add_row(f"{provider_name} ({env})", "set", "OK")
        else:
            table.add_row(
                f"{provider_name} ({env})",
                f"(unset — run: export {env}=...)",
                "WARN",
            )

    # HuggingFace login — required for gated datasets like GPQA / C-Eval
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    hf_env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token_path.exists() or hf_env_token:
        table.add_row("huggingface", "logged in", "OK")
    else:
        table.add_row(
            "huggingface",
            "(not logged in — run: huggingface-cli login)",
            "WARN",
        )

    # Working dir / artifacts dir
    artifacts = Path.cwd() / "artifacts"
    table.add_row("artifacts dir", str(artifacts), "OK")

    # Benchmarks registered
    from prism.benchmarks import default_registry
    bench_names = default_registry().names()
    table.add_row("benchmarks", ", ".join(bench_names), "OK" if bench_names else "WARN")

    console.print(table)
    raise typer.Exit(code=0 if ok_py else 1)


_DEFAULT_WORK_DIR = ".prism_runs"


@app.command(name="run")
def run_cmd(
    track: str = typer.Option(
        ..., "--track", help="Track: limit|agent|taste (P2a supports limit)"
    ),
    benchmark: str = typer.Option(
        ..., "--benchmark", help="Benchmark name (see registry)"
    ),
    model: Path = typer.Option(  # noqa: B008
        ..., "--model", exists=True, help="Path to model profile YAML"
    ),
    judge_model: Path | None = typer.Option(  # noqa: B008
        None,
        "--judge-model",
        exists=True,
        help="Path to LLM judge model profile YAML (required for benchmarks that use LLMJudge).",
    ),
    work_dir: Path = typer.Option(  # noqa: B008
        _DEFAULT_WORK_DIR,
        "--work-dir",
        help="Directory for SQLite DB + artifacts + checkpoint",
    ),
    subset: str | None = typer.Option(
        "quick", "--subset", help="Benchmark subset (quick|standard|full)"
    ),
    seeds: str = typer.Option("0", "--seeds", help="Comma-separated integer seeds"),
    max_concurrency: int = typer.Option(8, "--max-concurrency"),
    benchmark_source: str | None = typer.Option(
        None,
        "--benchmark-source",
        help="Override benchmark source (e.g., local jsonl path for testing)",
    ),
    benchmark_format: str | None = typer.Option(
        None, "--benchmark-format", help="jsonl|hf"
    ),
) -> None:
    """Run a benchmark against a model, producing scored results."""
    if track not in ("limit", "agent"):
        console.print(f"[red]Unknown track: {track!r}. Supported: limit | agent[/red]")
        raise typer.Exit(code=2)

    if track == "agent":
        try:
            bm_cls = agent_registry().get_class(benchmark)
        except KeyError:
            console.print(
                f"[red]Unknown agent benchmark: {benchmark!r}. "
                f"Known: {agent_registry().names()}[/red]"
            )
            raise typer.Exit(code=2) from None
        bm = bm_cls()
        profile = load_model_profile(model)
        adapter = LiteLLMAdapter(profile)

        work_dir.mkdir(parents=True, exist_ok=True)
        svc = RunService(
            db_path=work_dir / "prism.db",
            artifacts_root=work_dir / "artifacts",
            checkpoint_path=work_dir / "checkpoint.db",
        )

        async def _run_agent() -> dict[str, Any]:
            await svc.init()
            agent_runner = AgentRunner(service=svc)
            return await agent_runner.run(
                benchmark=bm,
                profile=profile,
                adapter=adapter,
                subset=subset,
            )

        result = asyncio.run(_run_agent())
        console.print(json.dumps(result, indent=2))
        return

    try:
        limit_bm_cls = default_registry().get_class(benchmark)
    except KeyError:
        known = default_registry().names()
        console.print(
            f"[red]Unknown benchmark: {benchmark!r}. Known: {known}[/red]"
        )
        raise typer.Exit(code=2) from None

    # Instantiate benchmark (allow override for source/format in tests)
    kwargs: dict[str, str] = {}
    if benchmark_source is not None:
        kwargs["source"] = benchmark_source
    if benchmark_format is not None:
        kwargs["source_format"] = benchmark_format
    limit_bm = limit_bm_cls(**kwargs)

    profile = load_model_profile(model)
    adapter = LiteLLMAdapter(profile)

    judge_adapter: LiteLLMAdapter | None = None
    if judge_model is not None:
        judge_profile = load_model_profile(judge_model)
        judge_adapter = LiteLLMAdapter(judge_profile)

    work_dir.mkdir(parents=True, exist_ok=True)
    svc = RunService(
        db_path=work_dir / "prism.db",
        artifacts_root=work_dir / "artifacts",
        checkpoint_path=work_dir / "checkpoint.db",
    )

    seeds_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]

    async def _run() -> dict[str, object]:
        await svc.init()
        limit = LimitRunner(service=svc)
        return await limit.run(
            benchmark=limit_bm,
            profile=profile,
            adapter=adapter,
            judge_adapter=judge_adapter,
            seeds=seeds_list,
            subset=subset,
            max_concurrency=max_concurrency,
        )

    result = asyncio.run(_run())
    console.print(json.dumps(result, indent=2))


leaderboard_app = typer.Typer(help="Leaderboard generation")
app.add_typer(leaderboard_app, name="leaderboard")


@leaderboard_app.command("publish")
def leaderboard_publish_cmd(
    workdir: Annotated[Path, typer.Argument(exists=True, help="Path containing prism.db")],
    output: Annotated[
        Path, typer.Option("--output", help="Directory to write index.html + data.json")
    ],
) -> None:
    """Render the leaderboard HTML from a Prism workdir's SQLite DB."""
    db_path = workdir / "prism.db"
    if not db_path.exists():
        console.print(f"[red]No prism.db found in {workdir}[/red]")
        raise typer.Exit(code=2)

    async def _build() -> dict[str, Any]:
        db = Database(db_path)
        main = await aggregate_by_model_benchmark(db=db)
        sweep_groups = await list_thinking_variants(db=db)
        staircase = []
        for bm in ("niah", "ruler_mk"):
            staircase.extend(await aggregate_staircase(db=db, benchmark=bm))
        return {"main": main, "staircase": staircase, "sweep_groups": sweep_groups}

    data = asyncio.run(_build())
    html_path = render_leaderboard(data, output_dir=output)
    console.print(f"Wrote leaderboard → {html_path}")


@app.command(name="list-benchmarks")
def list_benchmarks_cmd() -> None:
    """List all registered benchmarks with metadata."""
    from prism.agent import agent_registry
    from prism.benchmarks import default_registry

    table = Table(title="Prism Benchmarks")
    table.add_column("Name", style="cyan")
    table.add_column("Track")
    table.add_column("Judge")
    table.add_column("Source")

    # Limit track
    limit_reg = default_registry()
    for name in limit_reg.names():
        cls = limit_reg.get_class(name)
        needs_judge = getattr(cls, "needs_llm_judge", False)
        # Instantiate with no args to read default source (some benchmarks accept
        # source kwargs; defaults reflect the "real" HF path)
        try:
            bm = cls()
            source = getattr(bm, "source", None) or "builtin"
        except Exception:
            source = "(construction failed)"
        table.add_row(
            name,
            "limit",
            "LLM judge" if needs_judge else "rules",
            str(source),
        )

    # Agent track
    agent_reg = agent_registry()
    for name in agent_reg.names():
        cls = agent_reg.get_class(name)
        try:
            bm = cls()
            track = getattr(bm, "track", "agent")
        except Exception:
            track = "agent"
        table.add_row(name, track, "hard judge (subprocess)", "builtin")

    console.print(table)


# Sensible default rate_limit and cost presets per provider.
# Users can hand-edit the generated YAML afterwards.
_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "rate_limit": {"rpm": 50, "tpm": 400000},
        "cost": {"input_per_mtok": 15.0, "output_per_mtok": 75.0},
        "thinking": {"enabled": True, "effort": "high"},
    },
    "openai": {
        "rate_limit": {"rpm": 100, "tpm": 500000},
        "cost": {"input_per_mtok": 10.0, "output_per_mtok": 40.0},
        "reasoning_effort": "high",
    },
    "google": {
        "rate_limit": {"rpm": 60, "tpm": 400000},
        "cost": {"input_per_mtok": 7.0, "output_per_mtok": 21.0},
    },
    "deepseek": {
        "rate_limit": {"rpm": 100, "tpm": 500000},
        "cost": {"input_per_mtok": 0.5, "output_per_mtok": 2.0},
    },
    "xai": {
        "rate_limit": {"rpm": 60, "tpm": 400000},
        "cost": {"input_per_mtok": 5.0, "output_per_mtok": 15.0},
    },
    "kimi": {
        "rate_limit": {"rpm": 60, "tpm": 400000},
        "cost": {"input_per_mtok": 1.0, "output_per_mtok": 3.0},
    },
    "qwen": {
        "rate_limit": {"rpm": 60, "tpm": 400000},
        "cost": {"input_per_mtok": 1.0, "output_per_mtok": 3.0},
    },
    "custom": {
        "rate_limit": {"rpm": 60, "tpm": 200000},
        "cost": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
    },
}


@app.command(name="init-config")
def init_config_cmd(
    provider: str = typer.Option(..., "--provider", help="anthropic|openai|google|deepseek|xai|kimi|qwen|custom"),
    model: str = typer.Option(..., "--model", help="Provider-native model name (e.g., gpt-5, claude-opus-4-7)"),
    output: Path = typer.Option(..., "--output", help="Destination YAML path"),
    id_: str | None = typer.Option(None, "--id", help="Profile id (default: <model>@<effort>)"),
    display_name: str | None = typer.Option(None, "--display-name"),
    api_base: str | None = typer.Option(None, "--api-base", help="Custom endpoint URL (e.g., a self-hosted Anthropic-compatible proxy)"),
    effort: str | None = typer.Option(None, "--effort", help="off|low|medium|high|max (overrides provider default)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing file"),
) -> None:
    """Generate a model profile YAML with sensible defaults for a provider."""
    import yaml

    if provider not in _PROVIDER_DEFAULTS:
        console.print(f"[red]Unknown provider: {provider!r}. Known: {list(_PROVIDER_DEFAULTS)}[/red]")
        raise typer.Exit(code=2)

    if output.exists() and not force:
        console.print(f"[red]{output} already exists. Use --force to overwrite.[/red]")
        raise typer.Exit(code=2)

    defaults = _PROVIDER_DEFAULTS[provider]

    effective_effort = effort
    profile_id = id_ or (f"{model}@{effective_effort}" if effective_effort else f"{model}-default")

    config: dict[str, Any] = {
        "id": profile_id,
        "display_name": display_name or f"{model} via {provider}",
        "provider": provider,
        "model": model,
    }
    if api_base:
        config["api_base"] = api_base

    # Apply provider-specific thinking / reasoning_effort defaults.
    if effective_effort:
        if provider == "anthropic":
            config["thinking"] = {"enabled": effective_effort != "off", "effort": effective_effort}
        else:
            config["reasoning_effort"] = effective_effort
    else:
        for key in ("thinking", "reasoning_effort"):
            if key in defaults:
                config[key] = defaults[key]

    config["rate_limit"] = defaults["rate_limit"]
    config["cost"] = defaults["cost"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    console.print(f"Wrote {output}")


if __name__ == "__main__":
    app()
