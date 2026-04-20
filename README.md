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

## Architecture

**P1 Foundation (complete, tag `p1-foundation`):**

- `prism.adapters` — LiteLLM-based model adapter with thinking/reasoning_effort translation
- `prism.storage` — SQLite schema + async session + JSON artifact store
- `prism.orchestrator` — execution matrix, rate limiter, checkpoint, async runner
- `prism.judges` — Tier 1 rule judges + Tier 2 LLM judge + pytest judge
- `prism.service` — top-level orchestration service
- `prism.cli` — Typer CLI entry point

**P2a Limit Runner (complete):**

- `prism.benchmarks` — Benchmark ABC + PromptSpec + Registry
- `prism.benchmarks.mmlu_pro` / `aime` / `humaneval` — 3 initial benchmarks
- `prism.runners.limit` — LimitRunner: benchmark → adapter → judge → score

**Example:**

```bash
uv run prism run --track limit \
    --benchmark mmlu_pro \
    --model configs/models/gpt-5-high.example.yaml
```

P2b (more benchmarks), P2c (special views + leaderboard), P3 (Agent Runner), P4 (Meta-Ability), P5 (Web UI) are planned.

## License

Apache-2.0
