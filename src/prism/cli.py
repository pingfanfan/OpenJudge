import importlib.metadata
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from prism import __version__

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


if __name__ == "__main__":
    app()
