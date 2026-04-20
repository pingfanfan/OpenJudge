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
- 10 benchmarks: `mmlu_pro`, `gpqa` (knowledge); `aime`, `math500` (math); `humaneval`, `livecodebench` (code); `ifeval` (instruction following); `ceval` (Chinese); `simpleqa`, `truthfulqa` (hallucination/truthfulness)
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

P2c (safety + SuperCLUE), P2d (multimodal), P2e (long context), P3 (Agent Runner), P4 (Meta-Ability), P5 (Web UI) are planned.

## Safety benchmarks

HarmBench and XSTest test model safety and over-refusal behavior. See
[`docs/safety-considerations.md`](docs/safety-considerations.md) for data
handling and ethical usage notes before running them.

## License

Apache-2.0
