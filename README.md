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

**P2a–P2b Limit Runner (complete):**

- `prism.benchmarks` — Benchmark ABC + PromptSpec + Registry with LLM-judge adapter wiring
- 17 benchmarks across 10 dimensions: `mmlu_pro`, `gpqa` (knowledge); `aime`, `math500` (math); `humaneval`, `livecodebench` (code); `ifeval` (instruction following); `ceval`, `superclue` (Chinese); `simpleqa`, `truthfulqa` (hallucination); `harmbench`, `xstest` (safety); `mmmu`, `mathvista` (multimodal); `niah`, `ruler_mk` (long context)
- `prism.judges.ifeval` — IFEvalJudge with 12 constraint checkers
- `prism.runners.limit` — LimitRunner: benchmark → adapter → judge → score (optionally with separate judge model)

**Example:**

```bash
# Rule-based benchmark
uv run prism run --track limit --benchmark mmlu_pro \
    --model configs/models/gpt-5-high.example.yaml

# LLM-judge benchmark (requires --judge-model)
uv run prism run --track limit --benchmark simpleqa \
    --model configs/models/gpt-5-high.example.yaml \
    --judge-model configs/models/claude-opus-4-7-max.example.yaml
```

P2g (contamination probe), P3 (Agent Runner), P4 (Meta-Ability), P5 (Web UI) are planned.

## Safety benchmarks

HarmBench and XSTest test model safety and over-refusal behavior. See
[`docs/safety-considerations.md`](docs/safety-considerations.md) for data
handling and ethical usage notes before running them.

## Publishing a leaderboard

After running one or more benchmarks, render a static HTML leaderboard:

````bash
uv run prism leaderboard publish ./.prism_runs --output ./leaderboard
````

This produces `leaderboard/index.html` (human-readable) and `leaderboard/data.json`
(machine-readable). The HTML has three sections:

- **Main Leaderboard** — model × benchmark pass@1 grid
- **Context Length Staircase** — per-length accuracy for NIAH/RULER runs
- **Reasoning Effort Sweep** — same base model compared across thinking=off/high/max

To publish to GitHub Pages, commit the `leaderboard/` directory to a `gh-pages`
branch, or copy its contents to your Pages source.

## License

Apache-2.0
