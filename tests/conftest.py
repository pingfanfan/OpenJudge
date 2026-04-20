import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def cassette_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "cassettes"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip API keys during unit tests to prevent accidental real calls."""
    for key in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY", "XAI_API_KEY", "KIMI_API_KEY", "QWEN_API_KEY",
    ):
        # Keep them if integration/e2e marker is set, otherwise strip.
        if os.environ.get("PRISM_ALLOW_REAL_CALLS") != "1":
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def vcr_config(cassette_dir: Path) -> dict:
    return {
        "cassette_library_dir": str(cassette_dir),
        "filter_headers": ["authorization", "x-api-key", "x-goog-api-key"],
        "record_mode": "none",  # tests must use existing cassettes only
    }
