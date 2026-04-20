import asyncio
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from prism import __version__
from prism.adapters.litellm_adapter import LiteLLMAdapter
from prism.benchmarks import default_registry
from prism.config.loader import load_model_profile
from prism.runners.limit import LimitRunner
from prism.service import RunService

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

    # API keys (just reports presence, not validates)
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        present = bool(os.environ.get(env))
        table.add_row(env, "set" if present else "(unset)", "OK" if present else "WARN")

    # Working dir / artifacts dir
    artifacts = Path.cwd() / "artifacts"
    table.add_row("artifacts dir", str(artifacts), "OK")

    console.print(table)
    raise typer.Exit(code=0 if ok_py else 1)


@app.command(name="run")
def run_cmd(
    track: str = typer.Option(..., "--track", help="Track: limit|agent|taste (P2a supports limit)"),
    benchmark: str = typer.Option(..., "--benchmark", help="Benchmark name (see registry)"),
    model: Path = typer.Option(..., "--model", exists=True, help="Path to model profile YAML"),
    work_dir: Path = typer.Option(
        Path.cwd() / ".prism_runs",
        "--work-dir",
        help="Directory for SQLite DB + artifacts + checkpoint",
    ),
    subset: str | None = typer.Option("quick", "--subset", help="Benchmark subset (quick|standard|full)"),
    seeds: str = typer.Option("0", "--seeds", help="Comma-separated integer seeds"),
    max_concurrency: int = typer.Option(8, "--max-concurrency"),
    benchmark_source: str | None = typer.Option(
        None, "--benchmark-source", help="Override benchmark source (e.g., local jsonl path for testing)"
    ),
    benchmark_format: str | None = typer.Option(
        None, "--benchmark-format", help="jsonl|hf"
    ),
) -> None:
    """Run a benchmark against a model, producing scored results."""
    if track != "limit":
        console.print(f"[red]Only --track limit is implemented in P2a[/red]")
        raise typer.Exit(code=2)

    try:
        bm_cls = default_registry()._by_name[benchmark]
    except KeyError:
        console.print(f"[red]Unknown benchmark: {benchmark!r}. Known: {default_registry().names()}[/red]")
        raise typer.Exit(code=2)

    # Instantiate benchmark (allow override for source/format in tests)
    kwargs: dict = {}
    if benchmark_source is not None:
        kwargs["source"] = benchmark_source
    if benchmark_format is not None:
        kwargs["source_format"] = benchmark_format
    bm = bm_cls(**kwargs)

    profile = load_model_profile(model)
    adapter = LiteLLMAdapter(profile)

    work_dir.mkdir(parents=True, exist_ok=True)
    svc = RunService(
        db_path=work_dir / "prism.db",
        artifacts_root=work_dir / "artifacts",
        checkpoint_path=work_dir / "checkpoint.db",
    )

    seeds_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]

    async def _run() -> dict:
        await svc.init()
        limit = LimitRunner(service=svc)
        return await limit.run(
            benchmark=bm,
            profile=profile,
            adapter=adapter,
            seeds=seeds_list,
            subset=subset,
            max_concurrency=max_concurrency,
        )

    result = asyncio.run(_run())
    console.print(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
