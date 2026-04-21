# Prism Quickstart

Get your model scored on a benchmark in under 5 minutes.

## 1. Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if you don't have uv
git clone <your-prism-repo> && cd prism
uv sync --extra dev
```

## 2. Set an API key

Pick one provider — Prism supports 7 out of the box.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."      # or
export OPENAI_API_KEY="sk-..."             # or
export GOOGLE_API_KEY="..."                # Gemini
export DEEPSEEK_API_KEY="..."
export XAI_API_KEY="..."                   # Grok
export KIMI_API_KEY="..."                  # Moonshot
export DASHSCOPE_API_KEY="..."             # Qwen
```

Verify with:

```bash
uv run prism doctor
```

The "Status" column will flag any missing keys with a copy-paste `export ...` command.

## 3. Generate a model config

```bash
uv run prism init-config \
  --provider openai \
  --model gpt-5 \
  --output configs/models/my-gpt-5.yaml
```

Anthropic-compatible custom endpoint? Add `--api-base`:

```bash
uv run prism init-config \
  --provider anthropic \
  --model claude-opus-4-7 \
  --api-base "https://your-proxy.example.com/v1" \
  --id "my-proxy-claude" \
  --output configs/models/my-proxy.yaml
```

Open the YAML and hand-edit `rate_limit` and `cost` to match your provider's
actual limits/prices if the defaults are off.

## 4. Pick a benchmark

```bash
uv run prism list-benchmarks
```

Shows all 18 benchmarks (17 Limit + 1 Agent) with:
- **Track** — `limit` (single-turn Q&A) or `agent` (multi-turn tool use)
- **Judge** — `rules` (regex/numeric/pytest) or `LLM judge` (needs a second model)
- **Source** — HuggingFace path, or `builtin` for synthetic benchmarks

## 5. Run

Rule-based benchmark (no LLM judge needed):

```bash
uv run prism run --track limit \
  --benchmark mmlu_pro \
  --model configs/models/my-gpt-5.yaml \
  --subset quick
```

LLM-judge benchmark (needs `--judge-model`):

```bash
uv run prism run --track limit \
  --benchmark simpleqa \
  --model configs/models/my-gpt-5.yaml \
  --judge-model configs/models/claude-opus-4-7-max.example.yaml \
  --subset quick
```

Agent benchmark:

```bash
uv run prism run --track agent \
  --benchmark toy_agent \
  --model configs/models/my-gpt-5.yaml
```

Results land in `./.prism_runs/prism.db` (SQLite) + `./.prism_runs/artifacts/`.

## 6. View the leaderboard

After running one or more benchmarks:

```bash
uv run prism leaderboard publish ./.prism_runs --output ./leaderboard
open ./leaderboard/index.html
```

The HTML has three sections:
- **Main** — model × benchmark pass@1 grid
- **Context Length Staircase** — NIAH / RULER MK-NIAH drop-off curves
- **Reasoning Effort Sweep** — same model at thinking=high/max side-by-side

To publish: commit `./leaderboard/` to a `gh-pages` branch, or copy its contents to your Pages source.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` | Run the install one-liner in step 1, then `source ~/.zshrc` or reopen terminal |
| `ANTHROPIC_API_KEY not set` | Run the `export` hint from `prism doctor` |
| HF `401 Unauthorized` on GPQA / C-Eval | These are gated. Run `huggingface-cli login` |
| Rate-limit errors mid-run | Edit your model YAML: lower `rate_limit.rpm` / `tpm` |
| Want to test without hitting HF | Pass `--benchmark-source ./local-sample.jsonl --benchmark-format jsonl` |

## Where to go next

- **Published leaderboard**: commit `./leaderboard/` to `gh-pages` and enable GitHub Pages.
- **Custom benchmark**: see `src/prism/benchmarks/mmlu_pro/benchmark.py` as a template. Sub-class `Benchmark`, set `name` / `subset_caps`, implement `load_prompts` + `make_judge`. Register in `src/prism/benchmarks/__init__.py`.
- **Agent track**: `src/prism/benchmarks/toy_agent/benchmark.py` shows the `AgentTask` shape. P3b–P3d will add Claude Code integration, SWE-Bench, and Prism Real Tasks.
- **Safety benchmarks**: see `docs/safety-considerations.md` before running `harmbench` or `xstest`.
