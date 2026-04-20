from pathlib import Path

from typer.testing import CliRunner

from prism.cli import app

runner = CliRunner()


def test_leaderboard_publish_help():
    result = runner.invoke(app, ["leaderboard", "publish", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.stdout
    assert "workdir" in result.stdout.lower() or "work-dir" in result.stdout.lower()


def test_leaderboard_publish_missing_workdir_errors(tmp_path: Path):
    nonexistent = tmp_path / "does-not-exist"
    result = runner.invoke(app, [
        "leaderboard", "publish", str(nonexistent), "--output", str(tmp_path / "out"),
    ])
    assert result.exit_code != 0


def test_leaderboard_publish_empty_workdir_still_produces_html(tmp_path: Path):
    """Running publish against a workdir with empty prism.db should produce
    an 'empty state' leaderboard HTML without crashing."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    # Initialize an empty DB so the publish command has something to query.
    from prism.storage.database import Database
    import asyncio
    async def _init():
        db = Database(workdir / "prism.db")
        await db.init()
    asyncio.run(_init())

    out_dir = tmp_path / "out"
    result = runner.invoke(app, [
        "leaderboard", "publish", str(workdir), "--output", str(out_dir),
    ])
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "index.html").exists()
    assert (out_dir / "data.json").exists()
    html = (out_dir / "index.html").read_text()
    assert "<html" in html
