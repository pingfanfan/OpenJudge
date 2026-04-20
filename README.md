# Prism

> The open benchmark for testing frontier LLMs to their limits.

**Status:** pre-alpha (P1 Foundation)

See `docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md` for the full design.

## Install

```bash
uv sync --extra dev
```

## Commands

```bash
uv run prism doctor       # check environment
uv run prism version
```

## Develop

```bash
make test
make lint
make typecheck
make all
```

## Architecture (P1)

P1 establishes the foundation:

- `prism.adapters` — LiteLLM-based model adapter with thinking/reasoning_effort translation
- `prism.storage` — SQLite schema + async session + JSON artifact store
- `prism.orchestrator` — execution matrix, rate limiter, checkpoint, async runner
- `prism.judges` — Tier 1 rule judges (exact/numeric/regex) + Tier 2 LLM judge
- `prism.service` — top-level orchestration service
- `prism.cli` — Typer CLI entry point

P2 (Limit Runner), P3 (Agent Runner), P4 (Meta-Ability), P5 (Web UI) are planned separately.

## License

Apache-2.0
