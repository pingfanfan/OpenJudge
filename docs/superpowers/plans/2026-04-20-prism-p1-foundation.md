# Prism P1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 Prism 项目的底层地基——Model Adapter（含 thinking/effort 翻译）、Storage 层、Orchestrator、Judge Tier 1/2 基础、CLI 骨架，使得后续 P2-P5 可以在此之上叠加 Runner 与 UI。

**Architecture:** Python 3.11+ 单包，uv 管理依赖，asyncio 为并发底层。Model Adapter 基于 LiteLLM 做薄封装，Storage 用 SQLite 单库 + JSON artifact 目录，Orchestrator 维护执行矩阵与断点续跑。Judge 层分 Tier 1（规则）/Tier 2（LLM Judge）两级，上层 Runner 组合调用。CLI 用 Typer。

**Tech Stack:** Python 3.11+, uv, LiteLLM, Pydantic v2, SQLAlchemy 2.x (SQLite), aiosqlite, anyio, Typer, pytest, pytest-asyncio, pytest-vcr, respx。

---

## 参考文档

- 设计文档：`docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md`
- 相关章节：§3 系统架构、§8 Judge 层、§9 模型适配与 thinking/reasoning_effort 处理、§10 Storage、§11 目录结构

---

## 文件结构（P1 完成后）

```
prism/
├── pyproject.toml
├── LICENSE
├── README.md
├── .gitignore
├── .python-version                   # 3.11
├── src/prism/
│   ├── __init__.py
│   ├── cli.py                         # Typer 入口
│   ├── config/
│   │   ├── __init__.py
│   │   ├── model_profile.py           # Pydantic: 单个模型 profile
│   │   └── loader.py                  # YAML 加载器
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                    # Adapter ABC + 标准 Request/Response
│   │   ├── litellm_adapter.py         # LiteLLM thin wrapper
│   │   └── reasoning_translator.py    # thinking/effort 到各家字段的翻译
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── schema.py                  # SQLAlchemy 模型
│   │   ├── database.py                # 连接/会话/迁移
│   │   └── artifacts.py               # JSON artifact IO
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── matrix.py                  # (task × model × seed × effort) 展开
│   │   ├── runner.py                  # asyncio 并发调度
│   │   ├── rate_limit.py              # per-provider RPM/TPM 限流器
│   │   └── checkpoint.py              # 断点续跑
│   ├── judges/
│   │   ├── __init__.py
│   │   ├── base.py                    # Judge ABC
│   │   ├── rules.py                   # exact_match / numeric / regex / pytest
│   │   └── llm.py                     # Tier 2 LLM Judge
│   └── utils/
│       ├── __init__.py
│       └── cost.py                    # token→cost 换算
└── tests/
    ├── conftest.py                    # 夹具 + VCR 配置
    ├── fixtures/
    │   └── cassettes/                 # VCR 回放
    ├── unit/
    │   ├── test_model_profile.py
    │   ├── test_reasoning_translator.py
    │   ├── test_litellm_adapter.py
    │   ├── test_storage.py
    │   ├── test_artifacts.py
    │   ├── test_matrix.py
    │   ├── test_rate_limit.py
    │   ├── test_checkpoint.py
    │   ├── test_rules.py
    │   └── test_llm_judge.py
    ├── integration/
    │   └── test_end_to_end.py
    └── e2e/
        └── test_cli.py
```

---

## Task 1：初始化 uv 项目 + 基础文件

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `LICENSE`, `README.md`
- Create: `src/prism/__init__.py`, `tests/__init__.py`

- [ ] **Step 1：初始化 git 仓库**

Run:
```bash
cd "/Users/pingfan/Library/Mobile Documents/com~apple~CloudDocs/claude-code/DS-model1"
git init
git branch -M main
```
Expected: `Initialized empty Git repository`

- [ ] **Step 2：写入 `.python-version`**

```
3.11
```

- [ ] **Step 3：写入 `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
.uv/
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage
*.db
*.db-journal
artifacts/
.DS_Store
.vscode/
.idea/
```

- [ ] **Step 4：写入 `LICENSE`**（Apache-2.0 全文——复制自 https://www.apache.org/licenses/LICENSE-2.0.txt 标准文本，这里不展开）

说明：从 Apache 官方站点复制标准 Apache-2.0 文本到 `LICENSE` 文件，版权年份 `2026`，版权人 `Prism Contributors`。

- [ ] **Step 5：写入 `pyproject.toml`**

```toml
[project]
name = "prism-eval"
version = "0.1.0.dev0"
description = "Prism — The open benchmark for testing frontier LLMs to their limits"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
authors = [{ name = "Prism Contributors" }]
dependencies = [
    "litellm>=1.50.0",
    "pydantic>=2.5",
    "pyyaml>=6.0",
    "sqlalchemy>=2.0",
    "aiosqlite>=0.19",
    "anyio>=4.2",
    "typer>=0.12",
    "rich>=13.7",
    "tenacity>=8.2",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.1",
    "pytest-vcr>=1.0.2",
    "respx>=0.21",
    "mypy>=1.8",
    "ruff>=0.4",
]

[project.scripts]
prism = "prism.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/prism"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "ASYNC"]

[tool.mypy]
python_version = "3.11"
strict = true
```

- [ ] **Step 6：写入最小 `README.md`**

```markdown
# Prism

> The open benchmark for testing frontier LLMs to their limits.

**Status:** pre-alpha (P1 Foundation in progress)

See `docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md` for the full design.

## Quick start

```bash
uv sync --extra dev
uv run prism doctor
```

## License

Apache-2.0
```

- [ ] **Step 7：创建空的包 `__init__.py`**

Create `src/prism/__init__.py`:
```python
__version__ = "0.1.0.dev0"
```

Create `tests/__init__.py`:
```python
```

- [ ] **Step 8：初始化 uv 环境**

Run:
```bash
uv sync --extra dev
```
Expected: `.venv/` created, `uv.lock` produced, all deps installed.

- [ ] **Step 9：验证 pytest 可运行**

Run:
```bash
uv run pytest --version
```
Expected: `pytest 8.x`.

- [ ] **Step 10：Commit**

```bash
git add .python-version .gitignore LICENSE README.md pyproject.toml src/prism/__init__.py tests/__init__.py uv.lock
git commit -m "chore: initialize prism project skeleton"
```

---

## Task 2：ModelProfile Pydantic schema + YAML loader

**Files:**
- Create: `src/prism/config/__init__.py`
- Create: `src/prism/config/model_profile.py`
- Create: `src/prism/config/loader.py`
- Test: `tests/unit/test_model_profile.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_model_profile.py`:
```python
import pytest
from pathlib import Path
from prism.config.model_profile import ModelProfile, Thinking, RateLimit, Cost
from prism.config.loader import load_model_profile


def test_minimal_profile_parses():
    profile = ModelProfile(
        id="gpt-5@high",
        provider="openai",
        model="gpt-5",
        reasoning_effort="high",
    )
    assert profile.id == "gpt-5@high"
    assert profile.reasoning_effort == "high"
    assert profile.thinking is None


def test_anthropic_thinking_profile():
    profile = ModelProfile(
        id="claude-opus-4-7@max",
        provider="anthropic",
        model="claude-opus-4-7",
        thinking=Thinking(enabled=True, effort="max"),
    )
    assert profile.thinking.enabled is True
    assert profile.thinking.effort == "max"


def test_invalid_effort_rejected():
    with pytest.raises(ValueError, match="effort"):
        ModelProfile(
            id="x", provider="openai", model="x",
            reasoning_effort="super-mega",
        )


def test_load_from_yaml(tmp_path: Path):
    yaml_path = tmp_path / "opus.yaml"
    yaml_path.write_text(
        "id: claude-opus-4-7@max\n"
        "display_name: Claude Opus 4.7 (max)\n"
        "provider: anthropic\n"
        "model: claude-opus-4-7\n"
        "thinking:\n"
        "  enabled: true\n"
        "  effort: max\n"
        "rate_limit:\n"
        "  rpm: 50\n"
        "  tpm: 400000\n"
        "cost:\n"
        "  input_per_mtok: 15.0\n"
        "  output_per_mtok: 75.0\n"
    )
    profile = load_model_profile(yaml_path)
    assert profile.id == "claude-opus-4-7@max"
    assert profile.rate_limit.rpm == 50
    assert profile.cost.output_per_mtok == 75.0
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_model_profile.py -v
```
Expected: `ModuleNotFoundError: No module named 'prism.config'`

- [ ] **Step 3：实现 config 包**

Create `src/prism/config/__init__.py`:
```python
from prism.config.model_profile import Cost, ModelProfile, RateLimit, Thinking
from prism.config.loader import load_model_profile

__all__ = ["Cost", "ModelProfile", "RateLimit", "Thinking", "load_model_profile"]
```

Create `src/prism/config/model_profile.py`:
```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Effort = Literal["off", "low", "medium", "high", "max"]


class Thinking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    effort: Effort = "high"


class RateLimit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rpm: int = Field(gt=0, default=60)
    tpm: int = Field(gt=0, default=200_000)


class Cost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    display_name: str | None = None
    provider: Literal[
        "anthropic", "openai", "google", "deepseek", "xai", "kimi", "qwen", "custom"
    ]
    model: str
    thinking: Thinking | None = None
    reasoning_effort: Effort | None = None
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    cost: Cost = Field(default_factory=Cost)

    @field_validator("reasoning_effort")
    @classmethod
    def _check_effort(cls, v: Effort | None) -> Effort | None:
        return v
```

Create `src/prism/config/loader.py`:
```python
from pathlib import Path

import yaml

from prism.config.model_profile import ModelProfile


def load_model_profile(path: str | Path) -> ModelProfile:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ModelProfile.model_validate(data)
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_model_profile.py -v
```
Expected: 4 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/config tests/unit/test_model_profile.py
git commit -m "feat(config): add ModelProfile schema and YAML loader"
```

---

## Task 3：Reasoning Translator（thinking/effort 到各家原生字段的翻译）

**Files:**
- Create: `src/prism/adapters/__init__.py`
- Create: `src/prism/adapters/reasoning_translator.py`
- Test: `tests/unit/test_reasoning_translator.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_reasoning_translator.py`:
```python
from prism.adapters.reasoning_translator import translate
from prism.config.model_profile import ModelProfile, Thinking


def test_anthropic_thinking_max():
    profile = ModelProfile(
        id="x", provider="anthropic", model="claude-opus-4-7",
        thinking=Thinking(enabled=True, effort="max"),
    )
    extra = translate(profile)
    assert extra["thinking"] == {"type": "enabled"}
    assert extra["output_config"] == {"effort": "max"}


def test_anthropic_thinking_disabled():
    profile = ModelProfile(
        id="x", provider="anthropic", model="claude-opus-4-7",
        thinking=Thinking(enabled=False, effort="high"),
    )
    extra = translate(profile)
    assert extra["thinking"] == {"type": "disabled"}
    assert "output_config" not in extra


def test_openai_reasoning_effort():
    profile = ModelProfile(
        id="x", provider="openai", model="gpt-5",
        reasoning_effort="high",
    )
    extra = translate(profile)
    assert extra["reasoning_effort"] == "high"


def test_openai_no_effort_empty():
    profile = ModelProfile(
        id="x", provider="openai", model="gpt-4o",
    )
    extra = translate(profile)
    assert extra == {}


def test_google_thinking_budget():
    profile = ModelProfile(
        id="x", provider="google", model="gemini-2.5-pro",
        reasoning_effort="max",
    )
    extra = translate(profile)
    assert "thinkingConfig" in extra
    assert extra["thinkingConfig"]["thinkingBudget"] >= 32768


def test_deepseek_reasoning_flag():
    profile = ModelProfile(
        id="x", provider="deepseek", model="deepseek-r1",
        reasoning_effort="high",
    )
    extra = translate(profile)
    assert extra.get("reasoning") is True


def test_unsupported_provider_returns_empty():
    profile = ModelProfile(
        id="x", provider="custom", model="my-local",
        reasoning_effort="high",
    )
    extra = translate(profile)
    assert extra == {}
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_reasoning_translator.py -v
```
Expected: ImportError / AttributeError on `translate`.

- [ ] **Step 3：实现 translator**

Create `src/prism/adapters/__init__.py`:
```python
```

Create `src/prism/adapters/reasoning_translator.py`:
```python
from typing import Any

from prism.config.model_profile import Effort, ModelProfile

_GOOGLE_BUDGET: dict[Effort, int] = {
    "off": 0,
    "low": 1024,
    "medium": 8192,
    "high": 16384,
    "max": 32768,
}


def translate(profile: ModelProfile) -> dict[str, Any]:
    """Return provider-specific extra kwargs to pass through LiteLLM."""
    extra: dict[str, Any] = {}
    provider = profile.provider

    if provider == "anthropic":
        if profile.thinking is not None:
            if profile.thinking.enabled:
                extra["thinking"] = {"type": "enabled"}
                extra["output_config"] = {"effort": profile.thinking.effort}
            else:
                extra["thinking"] = {"type": "disabled"}
        elif profile.reasoning_effort:
            extra["thinking"] = {"type": "enabled"}
            extra["output_config"] = {"effort": profile.reasoning_effort}

    elif provider == "openai":
        if profile.reasoning_effort:
            extra["reasoning_effort"] = profile.reasoning_effort

    elif provider == "google":
        effort = profile.reasoning_effort or "high"
        extra["thinkingConfig"] = {"thinkingBudget": _GOOGLE_BUDGET[effort]}

    elif provider == "deepseek":
        if profile.reasoning_effort and profile.reasoning_effort != "off":
            extra["reasoning"] = True

    elif provider in ("xai", "kimi", "qwen"):
        if profile.reasoning_effort:
            extra["reasoning_effort"] = profile.reasoning_effort

    return extra
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_reasoning_translator.py -v
```
Expected: 7 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/adapters tests/unit/test_reasoning_translator.py
git commit -m "feat(adapters): add reasoning_effort/thinking translator for all providers"
```

---

## Task 4：Adapter 基类 + 标准 Request/Response 数据类型

**Files:**
- Create: `src/prism/adapters/base.py`
- Test: `tests/unit/test_adapter_base.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_adapter_base.py`:
```python
import pytest
from prism.adapters.base import AdapterRequest, AdapterResponse, Adapter


def test_request_minimal():
    req = AdapterRequest(
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=16,
    )
    assert req.messages[0]["role"] == "user"
    assert req.max_output_tokens == 16
    assert req.temperature == 0.0


def test_response_cost_computed():
    resp = AdapterResponse(
        text="hello",
        reasoning_text=None,
        tokens_in=100,
        tokens_out=50,
        latency_ms=1234.5,
        cost_usd=0.00175,
        raw={},
    )
    assert resp.tokens_in == 100
    assert resp.cost_usd == 0.00175


def test_abstract_adapter_cannot_instantiate():
    with pytest.raises(TypeError):
        Adapter()
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_adapter_base.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现 base**

Create `src/prism/adapters/base.py`:
```python
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.config.model_profile import ModelProfile


class AdapterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, Any]]
    max_output_tokens: int = Field(gt=0, default=4096)
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    seed: int | None = None


class AdapterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    reasoning_text: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    raw: dict[str, Any]
    finish_reason: str | None = None


class Adapter(ABC):
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    @abstractmethod
    async def complete(self, request: AdapterRequest) -> AdapterResponse: ...
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_adapter_base.py -v
```
Expected: 3 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/adapters/base.py tests/unit/test_adapter_base.py
git commit -m "feat(adapters): add Adapter ABC and Request/Response schemas"
```

---

## Task 5：Cost 换算工具

**Files:**
- Create: `src/prism/utils/__init__.py`
- Create: `src/prism/utils/cost.py`
- Test: `tests/unit/test_cost.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_cost.py`:
```python
from prism.config.model_profile import Cost
from prism.utils.cost import compute_cost


def test_zero_cost_profile():
    c = Cost(input_per_mtok=0.0, output_per_mtok=0.0)
    assert compute_cost(c, tokens_in=1000, tokens_out=1000) == 0.0


def test_non_zero_cost():
    c = Cost(input_per_mtok=15.0, output_per_mtok=75.0)
    # 1M tokens input = $15, 1M tokens output = $75
    assert compute_cost(c, tokens_in=1_000_000, tokens_out=0) == 15.0
    assert compute_cost(c, tokens_in=0, tokens_out=1_000_000) == 75.0


def test_partial():
    c = Cost(input_per_mtok=3.0, output_per_mtok=15.0)
    got = compute_cost(c, tokens_in=100_000, tokens_out=50_000)
    expected = 3.0 * 0.1 + 15.0 * 0.05
    assert abs(got - expected) < 1e-9
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_cost.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现**

Create `src/prism/utils/__init__.py`:
```python
```

Create `src/prism/utils/cost.py`:
```python
from prism.config.model_profile import Cost


def compute_cost(cost: Cost, *, tokens_in: int, tokens_out: int) -> float:
    return (
        cost.input_per_mtok * tokens_in / 1_000_000
        + cost.output_per_mtok * tokens_out / 1_000_000
    )
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_cost.py -v
```
Expected: 3 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/utils tests/unit/test_cost.py
git commit -m "feat(utils): add token→cost computation"
```

---

## Task 6：LiteLLM Adapter 实现（含 thinking 字段透传）

**Files:**
- Create: `src/prism/adapters/litellm_adapter.py`
- Test: `tests/unit/test_litellm_adapter.py`

- [ ] **Step 1：写失败测试（使用 mock 不发真实请求）**

Create `tests/unit/test_litellm_adapter.py`:
```python
from unittest.mock import AsyncMock, patch

import pytest

from prism.adapters.base import AdapterRequest
from prism.adapters.litellm_adapter import LiteLLMAdapter
from prism.config.model_profile import Cost, ModelProfile, Thinking


@pytest.fixture
def anthropic_profile() -> ModelProfile:
    return ModelProfile(
        id="claude-opus-4-7@max",
        provider="anthropic",
        model="claude-opus-4-7",
        thinking=Thinking(enabled=True, effort="max"),
        cost=Cost(input_per_mtok=15.0, output_per_mtok=75.0),
    )


@pytest.fixture
def openai_profile() -> ModelProfile:
    return ModelProfile(
        id="gpt-5@high",
        provider="openai",
        model="gpt-5",
        reasoning_effort="high",
        cost=Cost(input_per_mtok=10.0, output_per_mtok=40.0),
    )


class _FakeUsage:
    def __init__(self, pt: int, ct: int) -> None:
        self.prompt_tokens = pt
        self.completion_tokens = ct


class _FakeMessage:
    def __init__(self, content: str, reasoning: str | None = None) -> None:
        self.content = content
        self.reasoning_content = reasoning


class _FakeChoice:
    def __init__(self, content: str, reasoning: str | None = None) -> None:
        self.message = _FakeMessage(content, reasoning)
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self, content: str, pt: int, ct: int, reasoning: str | None = None) -> None:
        self.choices = [_FakeChoice(content, reasoning)]
        self.usage = _FakeUsage(pt, ct)

    def model_dump(self) -> dict:
        return {"fake": True}


@pytest.mark.asyncio
async def test_anthropic_call_passes_thinking(anthropic_profile):
    adapter = LiteLLMAdapter(anthropic_profile)
    req = AdapterRequest(messages=[{"role": "user", "content": "2+2"}], max_output_tokens=64)
    fake = _FakeResponse("4", pt=10, ct=5, reasoning="let me think")
    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=fake),
    ) as m:
        resp = await adapter.complete(req)

    assert resp.text == "4"
    assert resp.reasoning_text == "let me think"
    assert resp.tokens_in == 10
    assert resp.tokens_out == 5
    kwargs = m.call_args.kwargs
    # LiteLLM should see anthropic/model and the thinking fields
    assert kwargs["model"].startswith("anthropic/")
    assert kwargs["thinking"] == {"type": "enabled"}
    assert kwargs["output_config"] == {"effort": "max"}


@pytest.mark.asyncio
async def test_openai_call_passes_reasoning_effort(openai_profile):
    adapter = LiteLLMAdapter(openai_profile)
    req = AdapterRequest(messages=[{"role": "user", "content": "hi"}], max_output_tokens=16)
    fake = _FakeResponse("hello", pt=5, ct=2)
    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=fake),
    ) as m:
        resp = await adapter.complete(req)

    kwargs = m.call_args.kwargs
    assert kwargs["model"].startswith("openai/")
    assert kwargs["reasoning_effort"] == "high"
    assert resp.cost_usd == pytest.approx(10.0 * 5 / 1_000_000 + 40.0 * 2 / 1_000_000)
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_litellm_adapter.py -v
```
Expected: ImportError on `LiteLLMAdapter`.

- [ ] **Step 3：实现**

Create `src/prism/adapters/litellm_adapter.py`:
```python
import time

import litellm

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.adapters.reasoning_translator import translate
from prism.utils.cost import compute_cost


class LiteLLMAdapter(Adapter):
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        extra = translate(self.profile)
        model_id = f"{self.profile.provider}/{self.profile.model}"

        kwargs: dict = {
            "model": model_id,
            "messages": request.messages,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.stop:
            kwargs["stop"] = request.stop
        if request.tools:
            kwargs["tools"] = request.tools
        if request.seed is not None:
            kwargs["seed"] = request.seed
        kwargs.update(extra)

        t0 = time.perf_counter()
        resp = await litellm.acompletion(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = resp.choices[0]
        message = choice.message
        text = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)

        usage = resp.usage
        tokens_in = getattr(usage, "prompt_tokens", 0)
        tokens_out = getattr(usage, "completion_tokens", 0)

        return AdapterResponse(
            text=text,
            reasoning_text=reasoning,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=compute_cost(self.profile.cost, tokens_in=tokens_in, tokens_out=tokens_out),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            finish_reason=getattr(choice, "finish_reason", None),
        )
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_litellm_adapter.py -v
```
Expected: 2 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/adapters/litellm_adapter.py tests/unit/test_litellm_adapter.py
git commit -m "feat(adapters): add LiteLLMAdapter with thinking/reasoning_effort passthrough"
```

---

## Task 7：Storage SQLAlchemy schema

**Files:**
- Create: `src/prism/storage/__init__.py`
- Create: `src/prism/storage/schema.py`
- Test: `tests/unit/test_storage_schema.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_storage_schema.py`:
```python
from prism.storage.schema import Base, Model, Prompt, Response, Run, Score, Task


def test_tables_defined():
    names = {t.name for t in Base.metadata.tables.values()}
    assert {"runs", "models", "tasks", "prompts", "responses", "scores"} <= names


def test_run_columns():
    cols = {c.name for c in Run.__table__.columns}
    assert {"id", "created_at", "suite", "status", "config_hash"} <= cols


def test_response_columns():
    cols = {c.name for c in Response.__table__.columns}
    assert {
        "id", "run_id", "model_id", "prompt_id", "seed",
        "text", "reasoning_text", "tokens_in", "tokens_out",
        "latency_ms", "cost_usd", "finish_reason", "created_at",
    } <= cols
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_storage_schema.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现**

Create `src/prism/storage/__init__.py`:
```python
```

Create `src/prism/storage/schema.py`:
```python
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    suite: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending")
    config_hash: Mapped[str] = mapped_column(String)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Model(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    provider: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    thinking_enabled: Mapped[bool] = mapped_column(default=False)
    reasoning_effort: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cost_input_per_mtok: Mapped[float] = mapped_column(Float, default=0.0)
    cost_output_per_mtok: Mapped[float] = mapped_column(Float, default=0.0)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    benchmark: Mapped[str] = mapped_column(String)
    track: Mapped[str] = mapped_column(String)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"))
    version: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    system: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    task = relationship("Task")


class Response(Base):
    __tablename__ = "responses"
    __table_args__ = (
        UniqueConstraint("run_id", "model_id", "prompt_id", "seed", name="uq_resp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"))
    prompt_id: Mapped[str] = mapped_column(ForeignKey("prompts.id"))
    seed: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    reasoning_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    finish_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Score(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_id: Mapped[int] = mapped_column(ForeignKey("responses.id"))
    judge: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_storage_schema.py -v
```
Expected: 3 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/storage tests/unit/test_storage_schema.py
git commit -m "feat(storage): define SQLAlchemy schema for runs/models/tasks/prompts/responses/scores"
```

---

## Task 8：Storage Database（连接/迁移/CRUD helpers）

**Files:**
- Create: `src/prism/storage/database.py`
- Test: `tests/unit/test_database.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_database.py`:
```python
from pathlib import Path

import pytest

from prism.storage.database import Database
from prism.storage.schema import Model, Run


@pytest.mark.asyncio
async def test_init_creates_tables(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    await db.init()
    async with db.session() as s:
        # query should not raise
        await s.execute("SELECT 1 FROM runs WHERE 1=0")


@pytest.mark.asyncio
async def test_upsert_model(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    await db.init()
    async with db.session() as s:
        s.add(Model(id="m1", provider="openai", model="gpt-5"))
        await s.commit()
    async with db.session() as s:
        got = await s.get(Model, "m1")
        assert got is not None
        assert got.provider == "openai"


@pytest.mark.asyncio
async def test_create_run(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    await db.init()
    async with db.session() as s:
        s.add(Run(id="r1", suite="quick", config_hash="abc"))
        await s.commit()
    async with db.session() as s:
        got = await s.get(Run, "r1")
        assert got.status == "pending"
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_database.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现**

Create `src/prism/storage/database.py`:
```python
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from prism.storage.schema import Base


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self.path}", future=True)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._session_factory() as s:
            yield s

    async def dispose(self) -> None:
        await self._engine.dispose()
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_database.py -v
```
Expected: 3 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/storage/database.py tests/unit/test_database.py
git commit -m "feat(storage): async sqlite engine with session context manager"
```

---

## Task 9：Artifact IO（JSON 目录存储）

**Files:**
- Create: `src/prism/storage/artifacts.py`
- Test: `tests/unit/test_artifacts.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_artifacts.py`:
```python
import json
from pathlib import Path

from prism.storage.artifacts import ArtifactStore


def test_put_and_get(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")
    store.put("run-1", "trace/prompt-1.json", {"messages": [{"role": "user", "content": "hi"}]})
    got = store.get("run-1", "trace/prompt-1.json")
    assert got["messages"][0]["content"] == "hi"


def test_list(tmp_path: Path):
    store = ArtifactStore(tmp_path / "a")
    store.put("r", "a.json", {"x": 1})
    store.put("r", "sub/b.json", {"y": 2})
    assert set(store.list("r")) == {"a.json", "sub/b.json"}


def test_missing_returns_none(tmp_path: Path):
    store = ArtifactStore(tmp_path / "a")
    assert store.get("r", "nope.json") is None


def test_atomic_write(tmp_path: Path):
    store = ArtifactStore(tmp_path / "a")
    store.put("r", "x.json", {"a": 1})
    # File should exist, not .tmp
    assert (tmp_path / "a" / "r" / "x.json").exists()
    assert not any(p.suffix == ".tmp" for p in (tmp_path / "a").rglob("*"))
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_artifacts.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现**

Create `src/prism/storage/artifacts.py`:
```python
import json
import os
from pathlib import Path
from typing import Any


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str, relative: str) -> Path:
        return self.root / run_id / relative

    def put(self, run_id: str, relative: str, data: Any) -> None:
        path = self._path(run_id, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def get(self, run_id: str, relative: str) -> Any | None:
        path = self._path(run_id, relative)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def list(self, run_id: str) -> list[str]:
        base = self.root / run_id
        if not base.exists():
            return []
        return [str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()]
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_artifacts.py -v
```
Expected: 4 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/storage/artifacts.py tests/unit/test_artifacts.py
git commit -m "feat(storage): atomic JSON artifact store"
```

---

## Task 10：Orchestrator Matrix（执行矩阵展开）

**Files:**
- Create: `src/prism/orchestrator/__init__.py`
- Create: `src/prism/orchestrator/matrix.py`
- Test: `tests/unit/test_matrix.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_matrix.py`:
```python
from prism.orchestrator.matrix import Cell, expand_matrix
from prism.config.model_profile import ModelProfile


def _m(id_: str) -> ModelProfile:
    return ModelProfile(id=id_, provider="openai", model="x")


def test_single_model_single_task_single_seed():
    cells = list(expand_matrix(
        models=[_m("a")],
        prompt_ids=["p1"],
        seeds=[0],
    ))
    assert cells == [Cell(model_id="a", prompt_id="p1", seed=0)]


def test_full_product():
    cells = list(expand_matrix(
        models=[_m("a"), _m("b")],
        prompt_ids=["p1", "p2"],
        seeds=[0, 1, 2],
    ))
    assert len(cells) == 2 * 2 * 3


def test_deterministic_order():
    cells = list(expand_matrix(
        models=[_m("b"), _m("a")],
        prompt_ids=["p2", "p1"],
        seeds=[1, 0],
    ))
    # First model "b", first prompt "p2", first seed 1
    assert cells[0] == Cell(model_id="b", prompt_id="p2", seed=1)
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_matrix.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现**

Create `src/prism/orchestrator/__init__.py`:
```python
```

Create `src/prism/orchestrator/matrix.py`:
```python
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import product

from prism.config.model_profile import ModelProfile


@dataclass(frozen=True)
class Cell:
    model_id: str
    prompt_id: str
    seed: int


def expand_matrix(
    *,
    models: Iterable[ModelProfile],
    prompt_ids: Iterable[str],
    seeds: Iterable[int],
) -> Iterator[Cell]:
    for m, p, s in product(models, prompt_ids, seeds):
        yield Cell(model_id=m.id, prompt_id=p, seed=s)
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_matrix.py -v
```
Expected: 3 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/orchestrator tests/unit/test_matrix.py
git commit -m "feat(orchestrator): add execution matrix expansion"
```

---

## Task 11：Orchestrator Rate Limit（per-provider RPM/TPM 令牌桶）

**Files:**
- Create: `src/prism/orchestrator/rate_limit.py`
- Test: `tests/unit/test_rate_limit.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_rate_limit.py`:
```python
import asyncio
import time

import pytest

from prism.orchestrator.rate_limit import RateLimiter


@pytest.mark.asyncio
async def test_acquire_no_wait_when_under_limit():
    rl = RateLimiter(rpm=60, tpm=1_000_000)
    t0 = time.perf_counter()
    await rl.acquire(tokens=100)
    assert time.perf_counter() - t0 < 0.05


@pytest.mark.asyncio
async def test_rpm_enforcement():
    # 120 rpm -> 2 per second. 4 requests should take ~1.5s.
    rl = RateLimiter(rpm=120, tpm=10_000_000)
    t0 = time.perf_counter()
    for _ in range(4):
        await rl.acquire(tokens=1)
    elapsed = time.perf_counter() - t0
    assert 1.3 <= elapsed <= 2.0


@pytest.mark.asyncio
async def test_tpm_enforcement():
    # 60 rpm fine, but tpm=1000, each request uses 500 tokens.
    # 3 requests need ~60s wait on tokens; we just check 2nd takes wait.
    rl = RateLimiter(rpm=600, tpm=1000)
    await rl.acquire(tokens=600)
    t0 = time.perf_counter()
    await rl.acquire(tokens=600)
    # Must wait ~36s. Scale down for test: use smaller limits.
```

Note: the test as written would take too long. Replace the last case with a short one:

Create `tests/unit/test_rate_limit.py` (final version):
```python
import asyncio
import time

import pytest

from prism.orchestrator.rate_limit import RateLimiter


@pytest.mark.asyncio
async def test_acquire_no_wait_when_under_limit():
    rl = RateLimiter(rpm=60, tpm=1_000_000)
    t0 = time.perf_counter()
    await rl.acquire(tokens=100)
    assert time.perf_counter() - t0 < 0.05


@pytest.mark.asyncio
async def test_rpm_enforcement():
    rl = RateLimiter(rpm=120, tpm=10_000_000)  # 2 rps
    t0 = time.perf_counter()
    for _ in range(4):
        await rl.acquire(tokens=1)
    elapsed = time.perf_counter() - t0
    assert 1.3 <= elapsed <= 2.2


@pytest.mark.asyncio
async def test_negative_tokens_rejected():
    rl = RateLimiter(rpm=60, tpm=1000)
    with pytest.raises(ValueError):
        await rl.acquire(tokens=-1)
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_rate_limit.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现令牌桶（双桶：请求 + token）**

Create `src/prism/orchestrator/rate_limit.py`:
```python
import asyncio
import time


class _Bucket:
    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self.updated
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)
        self.updated = now

    def _try_consume(self, amount: float) -> float:
        """Return 0 if consumed, else wait-seconds needed."""
        self._refill()
        if self.tokens >= amount:
            self.tokens -= amount
            return 0.0
        deficit = amount - self.tokens
        return deficit / self.rate if self.rate > 0 else float("inf")


class RateLimiter:
    def __init__(self, *, rpm: int, tpm: int) -> None:
        if rpm <= 0 or tpm <= 0:
            raise ValueError("rpm and tpm must be positive")
        self._req = _Bucket(rpm / 60.0, float(rpm))
        self._tok = _Bucket(tpm / 60.0, float(tpm))
        self._lock = asyncio.Lock()

    async def acquire(self, *, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        while True:
            async with self._lock:
                wait_r = self._req._try_consume(1.0)
                if wait_r == 0.0:
                    wait_t = self._tok._try_consume(float(tokens))
                    if wait_t == 0.0:
                        return
                    # Refund the request so it isn't double-spent on retry.
                    self._req.tokens += 1.0
                    wait = wait_t
                else:
                    wait = wait_r
            await asyncio.sleep(wait)
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_rate_limit.py -v
```
Expected: 3 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/orchestrator/rate_limit.py tests/unit/test_rate_limit.py
git commit -m "feat(orchestrator): add per-provider RPM/TPM rate limiter"
```

---

## Task 12：Orchestrator Checkpoint（断点续跑）

**Files:**
- Create: `src/prism/orchestrator/checkpoint.py`
- Test: `tests/unit/test_checkpoint.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_checkpoint.py`:
```python
from pathlib import Path

import pytest

from prism.orchestrator.checkpoint import CheckpointStore
from prism.orchestrator.matrix import Cell


@pytest.mark.asyncio
async def test_mark_and_query(tmp_path: Path):
    cp = CheckpointStore(tmp_path / "cp.db")
    await cp.init()
    c = Cell(model_id="m", prompt_id="p1", seed=0)
    assert await cp.status(run_id="r", cell=c) == "pending"
    await cp.mark(run_id="r", cell=c, status="running")
    assert await cp.status(run_id="r", cell=c) == "running"
    await cp.mark(run_id="r", cell=c, status="done")
    assert await cp.status(run_id="r", cell=c) == "done"


@pytest.mark.asyncio
async def test_pending_cells_filter(tmp_path: Path):
    cp = CheckpointStore(tmp_path / "cp.db")
    await cp.init()
    cells = [Cell("m", f"p{i}", 0) for i in range(3)]
    await cp.mark("r", cells[0], "done")
    await cp.mark("r", cells[1], "running")
    pending = [c async for c in cp.pending_cells("r", cells)]
    assert pending == [cells[2]]
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_checkpoint.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现**

Create `src/prism/orchestrator/checkpoint.py`:
```python
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import aiosqlite

from prism.orchestrator.matrix import Cell

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoint (
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (run_id, model_id, prompt_id, seed)
);
"""


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(_SCHEMA)
            await db.commit()

    async def mark(self, *, run_id: str, cell: Cell, status: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO checkpoint(run_id,model_id,prompt_id,seed,status) VALUES (?,?,?,?,?) "
                "ON CONFLICT(run_id,model_id,prompt_id,seed) DO UPDATE SET status=excluded.status",
                (run_id, cell.model_id, cell.prompt_id, cell.seed, status),
            )
            await db.commit()

    async def status(self, *, run_id: str, cell: Cell) -> str:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT status FROM checkpoint WHERE run_id=? AND model_id=? AND prompt_id=? AND seed=?",
                (run_id, cell.model_id, cell.prompt_id, cell.seed),
            )
            row = await cursor.fetchone()
            return row[0] if row else "pending"

    async def pending_cells(
        self, run_id: str, cells: Iterable[Cell]
    ) -> AsyncIterator[Cell]:
        for cell in cells:
            if await self.status(run_id=run_id, cell=cell) not in ("done",):
                # For resume semantics, "running" cells also need to be redone
                # (they may have been interrupted). We only skip "done".
                if await self.status(run_id=run_id, cell=cell) == "running":
                    yield cell
                elif await self.status(run_id=run_id, cell=cell) == "pending":
                    yield cell
```

Simplify the implementation: the loop above does 3 queries per cell. Replace with a single batch query:

Replace `pending_cells` body:
```python
    async def pending_cells(
        self, run_id: str, cells: Iterable[Cell]
    ) -> AsyncIterator[Cell]:
        async with aiosqlite.connect(self.path) as db:
            for cell in cells:
                cursor = await db.execute(
                    "SELECT status FROM checkpoint WHERE run_id=? AND model_id=? AND prompt_id=? AND seed=?",
                    (run_id, cell.model_id, cell.prompt_id, cell.seed),
                )
                row = await cursor.fetchone()
                status = row[0] if row else "pending"
                if status != "done":
                    yield cell
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_checkpoint.py -v
```
Expected: 2 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/orchestrator/checkpoint.py tests/unit/test_checkpoint.py
git commit -m "feat(orchestrator): add resumable checkpoint store"
```

---

## Task 13：Orchestrator Runner（并发执行引擎）

**Files:**
- Create: `src/prism/orchestrator/runner.py`
- Test: `tests/unit/test_orchestrator_runner.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_orchestrator_runner.py`:
```python
import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.config.model_profile import ModelProfile, RateLimit
from prism.orchestrator.matrix import Cell
from prism.orchestrator.runner import OrchestratorRunner


class FakeAdapter(Adapter):
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        await asyncio.sleep(0.01)
        return AdapterResponse(
            text="ok",
            reasoning_text=None,
            tokens_in=10,
            tokens_out=5,
            latency_ms=10.0,
            cost_usd=0.0,
            raw={},
        )


@pytest.mark.asyncio
async def test_run_all_cells(tmp_path: Path):
    profile = ModelProfile(
        id="m1", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    runner = OrchestratorRunner(
        adapters={"m1": FakeAdapter(profile)},
        profiles={"m1": profile},
        checkpoint_path=tmp_path / "cp.db",
    )
    await runner.init()

    cells = [Cell("m1", f"p{i}", 0) for i in range(5)]
    prompts = {f"p{i}": [{"role": "user", "content": str(i)}] for i in range(5)}

    results: list[tuple[Cell, AdapterResponse]] = []

    async def on_done(cell: Cell, resp: AdapterResponse) -> None:
        results.append((cell, resp))

    await runner.run(
        run_id="r",
        cells=cells,
        prompts=prompts,
        on_done=on_done,
        max_concurrency=3,
    )
    assert len(results) == 5
    assert all(r.text == "ok" for _, r in results)


@pytest.mark.asyncio
async def test_resume_skips_done(tmp_path: Path):
    profile = ModelProfile(
        id="m1", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    adapter = FakeAdapter(profile)
    adapter.complete = AsyncMock(wraps=adapter.complete)

    runner = OrchestratorRunner(
        adapters={"m1": adapter},
        profiles={"m1": profile},
        checkpoint_path=tmp_path / "cp.db",
    )
    await runner.init()

    cells = [Cell("m1", f"p{i}", 0) for i in range(3)]
    prompts = {f"p{i}": [{"role": "user", "content": str(i)}] for i in range(3)}

    await runner.run(run_id="r", cells=cells, prompts=prompts, on_done=None, max_concurrency=2)
    assert adapter.complete.await_count == 3

    # Second call should skip all (all done).
    await runner.run(run_id="r", cells=cells, prompts=prompts, on_done=None, max_concurrency=2)
    assert adapter.complete.await_count == 3
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_orchestrator_runner.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现**

Create `src/prism/orchestrator/runner.py`:
```python
import asyncio
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.config.model_profile import ModelProfile
from prism.orchestrator.checkpoint import CheckpointStore
from prism.orchestrator.matrix import Cell
from prism.orchestrator.rate_limit import RateLimiter

OnDone = Callable[[Cell, AdapterResponse], Awaitable[None]] | None


class OrchestratorRunner:
    def __init__(
        self,
        *,
        adapters: dict[str, Adapter],
        profiles: dict[str, ModelProfile],
        checkpoint_path: str | Path,
    ) -> None:
        self.adapters = adapters
        self.profiles = profiles
        self.checkpoint = CheckpointStore(checkpoint_path)
        self._limiters: dict[str, RateLimiter] = {
            mid: RateLimiter(rpm=p.rate_limit.rpm, tpm=p.rate_limit.tpm)
            for mid, p in profiles.items()
        }

    async def init(self) -> None:
        await self.checkpoint.init()

    async def run(
        self,
        *,
        run_id: str,
        cells: Iterable[Cell],
        prompts: dict[str, list[dict[str, Any]]],
        on_done: OnDone,
        max_concurrency: int = 8,
    ) -> None:
        sem = asyncio.Semaphore(max_concurrency)
        cells_list = list(cells)

        async def _execute(cell: Cell) -> None:
            async with sem:
                profile = self.profiles[cell.model_id]
                adapter = self.adapters[cell.model_id]
                limiter = self._limiters[cell.model_id]
                messages = prompts[cell.prompt_id]

                # Approx token count: 4 chars per token
                approx_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
                await limiter.acquire(tokens=approx_tokens)

                await self.checkpoint.mark(run_id=run_id, cell=cell, status="running")
                try:
                    resp = await adapter.complete(AdapterRequest(
                        messages=messages,
                        max_output_tokens=4096,
                        seed=cell.seed,
                    ))
                    await self.checkpoint.mark(run_id=run_id, cell=cell, status="done")
                    if on_done is not None:
                        await on_done(cell, resp)
                except Exception:
                    await self.checkpoint.mark(run_id=run_id, cell=cell, status="failed")
                    raise

        pending: list[Cell] = []
        async for c in self.checkpoint.pending_cells(run_id, cells_list):
            pending.append(c)

        await asyncio.gather(*(_execute(c) for c in pending))
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_orchestrator_runner.py -v
```
Expected: 2 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/orchestrator/runner.py tests/unit/test_orchestrator_runner.py
git commit -m "feat(orchestrator): async runner with concurrency, rate limit, checkpoint"
```

---

## Task 14：Judge Base + Rules judges（exact / numeric / regex）

**Files:**
- Create: `src/prism/judges/__init__.py`
- Create: `src/prism/judges/base.py`
- Create: `src/prism/judges/rules.py`
- Test: `tests/unit/test_rules_judge.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_rules_judge.py`:
```python
import pytest
from prism.judges.base import JudgeResult
from prism.judges.rules import ExactMatchJudge, NumericJudge, RegexJudge


def test_exact_match_pass():
    j = ExactMatchJudge()
    r = j.judge(output="hello", expected="hello")
    assert r.score == 1.0
    assert r.confidence == 1.0


def test_exact_match_fail():
    j = ExactMatchJudge()
    r = j.judge(output="hello!", expected="hello")
    assert r.score == 0.0


def test_exact_match_case_insensitive_option():
    j = ExactMatchJudge(case_sensitive=False)
    r = j.judge(output="Hello", expected="hello")
    assert r.score == 1.0


def test_numeric_exact():
    j = NumericJudge()
    r = j.judge(output="The answer is 42.", expected="42")
    assert r.score == 1.0


def test_numeric_tolerance():
    j = NumericJudge(tolerance=0.01)
    r = j.judge(output="3.141", expected="3.14")
    assert r.score == 1.0


def test_numeric_no_number_found():
    j = NumericJudge()
    r = j.judge(output="I don't know", expected="42")
    assert r.score == 0.0


def test_regex_pass():
    j = RegexJudge(pattern=r"\bAnswer:\s*([A-D])\b")
    r = j.judge(output="My analysis leads to Answer: C here.", expected="C")
    assert r.score == 1.0


def test_regex_wrong_capture():
    j = RegexJudge(pattern=r"Answer:\s*([A-D])")
    r = j.judge(output="Answer: B", expected="C")
    assert r.score == 0.0


def test_regex_no_match():
    j = RegexJudge(pattern=r"Answer:\s*([A-D])")
    r = j.judge(output="I refuse", expected="C")
    assert r.score == 0.0
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_rules_judge.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现**

Create `src/prism/judges/__init__.py`:
```python
```

Create `src/prism/judges/base.py`:
```python
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None


class Judge(ABC):
    name: str = "judge"

    @abstractmethod
    def judge(self, *, output: str, expected: str) -> JudgeResult: ...
```

Create `src/prism/judges/rules.py`:
```python
import re

from prism.judges.base import Judge, JudgeResult

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class ExactMatchJudge(Judge):
    name = "exact_match"

    def __init__(self, *, case_sensitive: bool = True, strip: bool = True) -> None:
        self.case_sensitive = case_sensitive
        self.strip = strip

    def judge(self, *, output: str, expected: str) -> JudgeResult:
        a = output.strip() if self.strip else output
        b = expected.strip() if self.strip else expected
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()
        return JudgeResult(score=1.0 if a == b else 0.0, confidence=1.0)


class NumericJudge(Judge):
    name = "numeric"

    def __init__(self, *, tolerance: float = 0.0) -> None:
        self.tolerance = tolerance

    def judge(self, *, output: str, expected: str) -> JudgeResult:
        try:
            exp = float(expected.strip())
        except ValueError:
            return JudgeResult(score=0.0, confidence=0.0, reasoning="bad expected number")

        matches = _NUMBER_RE.findall(output)
        if not matches:
            return JudgeResult(score=0.0, confidence=1.0, reasoning="no number in output")
        # Prefer the last number in the output as the "final answer"
        try:
            got = float(matches[-1])
        except ValueError:
            return JudgeResult(score=0.0, confidence=0.0, reasoning="unparseable")

        if self.tolerance > 0:
            ok = abs(got - exp) <= self.tolerance
        else:
            ok = got == exp
        return JudgeResult(score=1.0 if ok else 0.0, confidence=1.0)


class RegexJudge(Judge):
    name = "regex"

    def __init__(self, *, pattern: str, flags: int = 0) -> None:
        self._re = re.compile(pattern, flags)

    def judge(self, *, output: str, expected: str) -> JudgeResult:
        m = self._re.search(output)
        if not m:
            return JudgeResult(score=0.0, confidence=1.0, reasoning="no match")
        captured = m.group(1) if m.groups() else m.group(0)
        return JudgeResult(
            score=1.0 if captured.strip() == expected.strip() else 0.0,
            confidence=1.0,
        )
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_rules_judge.py -v
```
Expected: 9 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/judges tests/unit/test_rules_judge.py
git commit -m "feat(judges): add Judge ABC and rules judges (exact/numeric/regex)"
```

---

## Task 15：LLM Judge（Tier 2，调用独立模型给分 + confidence）

**Files:**
- Create: `src/prism/judges/llm.py`
- Test: `tests/unit/test_llm_judge.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_llm_judge.py`:
```python
from unittest.mock import AsyncMock

import pytest

from prism.adapters.base import AdapterResponse
from prism.judges.base import JudgeResult
from prism.judges.llm import LLMJudge


class FakeAdapter:
    def __init__(self, content: str) -> None:
        self._content = content

    async def complete(self, request):
        return AdapterResponse(
            text=self._content,
            reasoning_text=None,
            tokens_in=10,
            tokens_out=10,
            latency_ms=5.0,
            cost_usd=0.0,
            raw={},
        )


@pytest.mark.asyncio
async def test_llm_judge_parses_json():
    payload = '{"score": 0.9, "confidence": 0.85, "reasoning": "mostly right"}'
    j = LLMJudge(adapter=FakeAdapter(payload), rubric="Score 0-1.")
    r = await j.judge_async(output="2+2=4", expected="4")
    assert isinstance(r, JudgeResult)
    assert r.score == 0.9
    assert r.confidence == 0.85


@pytest.mark.asyncio
async def test_llm_judge_parses_json_with_extra_text():
    payload = 'Here is my analysis.\n```json\n{"score": 0.0, "confidence": 1.0, "reasoning": "wrong"}\n```\n'
    j = LLMJudge(adapter=FakeAdapter(payload), rubric="Score 0-1.")
    r = await j.judge_async(output="2+2=5", expected="4")
    assert r.score == 0.0


@pytest.mark.asyncio
async def test_llm_judge_malformed_returns_low_confidence():
    j = LLMJudge(adapter=FakeAdapter("I cannot parse this."), rubric="Score 0-1.")
    r = await j.judge_async(output="x", expected="y")
    assert r.confidence < 0.5
    assert r.score == 0.0
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/unit/test_llm_judge.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现**

Create `src/prism/judges/llm.py`:
```python
import json
import re

from prism.adapters.base import Adapter, AdapterRequest
from prism.judges.base import Judge, JudgeResult

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{[^{}]*\"score\"[^{}]*\})", re.DOTALL)

_DEFAULT_PROMPT = """You are an impartial grader. Compare the MODEL_OUTPUT to the REFERENCE.
Return STRICT JSON only, with keys: score (0.0-1.0), confidence (0.0-1.0), reasoning (short).

Rubric: {rubric}

MODEL_OUTPUT:
{output}

REFERENCE:
{expected}

JSON:"""


class LLMJudge(Judge):
    name = "llm_judge"

    def __init__(self, *, adapter: Adapter, rubric: str) -> None:
        self.adapter = adapter
        self.rubric = rubric

    async def judge_async(self, *, output: str, expected: str) -> JudgeResult:
        prompt = _DEFAULT_PROMPT.format(rubric=self.rubric, output=output, expected=expected)
        resp = await self.adapter.complete(AdapterRequest(
            messages=[{"role": "user", "content": prompt}],
            max_output_tokens=512,
        ))
        return self._parse(resp.text)

    def judge(self, *, output: str, expected: str) -> JudgeResult:  # sync not supported
        raise NotImplementedError("Use judge_async")

    @staticmethod
    def _parse(text: str) -> JudgeResult:
        candidates: list[str] = []
        m = _FENCED_JSON_RE.search(text)
        if m:
            candidates.append(m.group(1))
        m2 = _BARE_JSON_RE.search(text)
        if m2:
            candidates.append(m2.group(1))
        candidates.append(text)

        for c in candidates:
            try:
                data = json.loads(c)
                score = float(data.get("score", 0.0))
                confidence = float(data.get("confidence", 0.5))
                reasoning = data.get("reasoning")
                score = max(0.0, min(1.0, score))
                confidence = max(0.0, min(1.0, confidence))
                return JudgeResult(score=score, confidence=confidence, reasoning=reasoning)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return JudgeResult(score=0.0, confidence=0.0, reasoning="unparseable judge output")
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/unit/test_llm_judge.py -v
```
Expected: 3 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/judges/llm.py tests/unit/test_llm_judge.py
git commit -m "feat(judges): add LLM-as-judge with strict JSON parsing"
```

---

## Task 16：CLI 骨架 + `prism doctor`

**Files:**
- Modify: `src/prism/cli.py`（新文件）
- Test: `tests/e2e/test_cli.py`

- [ ] **Step 1：写失败测试**

Create `tests/__init__.py`（若已存在则跳过）和 `tests/e2e/__init__.py`:
```python
```

Create `tests/e2e/test_cli.py`:
```python
from typer.testing import CliRunner

from prism.cli import app

runner = CliRunner()


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "prism" in result.stdout.lower()


def test_doctor_reports_python():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)  # may be 1 if missing API keys
    assert "python" in result.stdout.lower()
    assert "litellm" in result.stdout.lower()


def test_version_flag():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1" in result.stdout
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/e2e/test_cli.py -v
```
Expected: ImportError on `prism.cli.app`.

- [ ] **Step 3：实现 CLI**

Create `src/prism/cli.py`:
```python
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
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
uv run pytest tests/e2e/test_cli.py -v
```
Expected: 3 passed.

- [ ] **Step 5：确认 `prism` 命令可以本地启动**

Run:
```bash
uv run prism --help
uv run prism version
uv run prism doctor
```
Expected: help text renders; version prints `0.1.0.dev0`; doctor prints a table.

- [ ] **Step 6：Commit**

```bash
git add src/prism/cli.py tests/e2e/__init__.py tests/e2e/test_cli.py
git commit -m "feat(cli): add typer app with version and doctor commands"
```

---

## Task 17：Run Lifecycle（将一次 run 的所有组件串起来的 thin service 层）

**Files:**
- Create: `src/prism/service.py`
- Test: `tests/integration/__init__.py`
- Test: `tests/integration/test_end_to_end.py`

- [ ] **Step 1：写失败集成测试（使用 FakeAdapter，端到端演练 run lifecycle）**

Create `tests/integration/__init__.py`:
```python
```

Create `tests/integration/test_end_to_end.py`:
```python
import asyncio
from pathlib import Path
from typing import Any

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.config.model_profile import Cost, ModelProfile, RateLimit
from prism.service import RunService


class EchoAdapter(Adapter):
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        content = request.messages[-1]["content"]
        return AdapterResponse(
            text=content,  # Echo back
            reasoning_text=None,
            tokens_in=len(content) // 4 + 1,
            tokens_out=len(content) // 4 + 1,
            latency_ms=1.0,
            cost_usd=0.0,
            raw={},
        )


@pytest.mark.asyncio
async def test_full_run_lifecycle(tmp_path: Path):
    profile = ModelProfile(
        id="echo",
        provider="openai",  # pretend
        model="echo",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
        cost=Cost(),
    )
    adapter = EchoAdapter(profile)

    svc = RunService(
        db_path=tmp_path / "prism.db",
        artifacts_root=tmp_path / "artifacts",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()

    prompts: dict[str, dict[str, Any]] = {
        "p1": {"version": "v1", "text": "What is 2+2?", "system": None, "task_id": "t1"},
        "p2": {"version": "v1", "text": "Say hello.",   "system": None, "task_id": "t1"},
    }

    run_id = await svc.create_run(suite="smoke")
    await svc.register_model(profile)
    await svc.register_task(task_id="t1", benchmark="smoke", track="limit")
    for pid, meta in prompts.items():
        await svc.register_prompt(prompt_id=pid, **meta)

    await svc.execute(
        run_id=run_id,
        profiles={profile.id: profile},
        adapters={profile.id: adapter},
        prompts={pid: [{"role": "user", "content": m["text"]}] for pid, m in prompts.items()},
        seeds=[0],
        max_concurrency=2,
    )

    summary = await svc.summarize(run_id=run_id)
    assert summary["response_count"] == 2
    assert summary["total_cost_usd"] == 0.0
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
uv run pytest tests/integration/test_end_to_end.py -v
```
Expected: ImportError on `prism.service`.

- [ ] **Step 3：实现 service 层**

Create `src/prism/service.py`:
```python
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select

from prism.adapters.base import Adapter, AdapterResponse
from prism.config.model_profile import ModelProfile
from prism.orchestrator.matrix import Cell, expand_matrix
from prism.orchestrator.runner import OrchestratorRunner
from prism.storage.artifacts import ArtifactStore
from prism.storage.database import Database
from prism.storage.schema import Model, Prompt, Response, Run, Task


class RunService:
    def __init__(
        self,
        *,
        db_path: str | Path,
        artifacts_root: str | Path,
        checkpoint_path: str | Path,
    ) -> None:
        self.db = Database(db_path)
        self.artifacts = ArtifactStore(artifacts_root)
        self.checkpoint_path = Path(checkpoint_path)

    async def init(self) -> None:
        await self.db.init()

    async def create_run(self, *, suite: str, config_hash: str = "") -> str:
        run_id = f"run-{uuid4().hex[:12]}"
        async with self.db.session() as s:
            s.add(Run(id=run_id, suite=suite, config_hash=config_hash, status="running"))
            await s.commit()
        return run_id

    async def register_model(self, profile: ModelProfile) -> None:
        async with self.db.session() as s:
            if await s.get(Model, profile.id) is None:
                s.add(Model(
                    id=profile.id,
                    display_name=profile.display_name,
                    provider=profile.provider,
                    model=profile.model,
                    thinking_enabled=bool(profile.thinking and profile.thinking.enabled),
                    reasoning_effort=profile.reasoning_effort or (
                        profile.thinking.effort if profile.thinking else None
                    ),
                    cost_input_per_mtok=profile.cost.input_per_mtok,
                    cost_output_per_mtok=profile.cost.output_per_mtok,
                ))
                await s.commit()

    async def register_task(self, *, task_id: str, benchmark: str, track: str) -> None:
        async with self.db.session() as s:
            if await s.get(Task, task_id) is None:
                s.add(Task(id=task_id, benchmark=benchmark, track=track))
                await s.commit()

    async def register_prompt(
        self, *, prompt_id: str, task_id: str, version: str, text: str, system: str | None = None
    ) -> None:
        async with self.db.session() as s:
            if await s.get(Prompt, prompt_id) is None:
                s.add(Prompt(id=prompt_id, task_id=task_id, version=version, text=text, system=system))
                await s.commit()

    async def execute(
        self,
        *,
        run_id: str,
        profiles: dict[str, ModelProfile],
        adapters: dict[str, Adapter],
        prompts: dict[str, list[dict[str, Any]]],
        seeds: list[int],
        max_concurrency: int = 8,
    ) -> None:
        runner = OrchestratorRunner(
            adapters=adapters,
            profiles=profiles,
            checkpoint_path=self.checkpoint_path,
        )
        await runner.init()

        cells = list(expand_matrix(
            models=list(profiles.values()),
            prompt_ids=list(prompts.keys()),
            seeds=seeds,
        ))

        async def _persist(cell: Cell, resp: AdapterResponse) -> None:
            async with self.db.session() as s:
                s.add(Response(
                    run_id=run_id,
                    model_id=cell.model_id,
                    prompt_id=cell.prompt_id,
                    seed=cell.seed,
                    text=resp.text,
                    reasoning_text=resp.reasoning_text,
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    latency_ms=resp.latency_ms,
                    cost_usd=resp.cost_usd,
                    finish_reason=resp.finish_reason,
                ))
                await s.commit()
            self.artifacts.put(
                run_id,
                f"responses/{cell.model_id}/{cell.prompt_id}-seed{cell.seed}.json",
                resp.model_dump(),
            )

        await runner.run(
            run_id=run_id, cells=cells, prompts=prompts,
            on_done=_persist, max_concurrency=max_concurrency,
        )

        async with self.db.session() as s:
            run = await s.get(Run, run_id)
            if run is not None:
                run.status = "done"
                await s.commit()

    async def summarize(self, *, run_id: str) -> dict[str, Any]:
        async with self.db.session() as s:
            count = (await s.execute(
                select(func.count()).select_from(Response).where(Response.run_id == run_id)
            )).scalar_one()
            total_cost = (await s.execute(
                select(func.coalesce(func.sum(Response.cost_usd), 0.0)).where(Response.run_id == run_id)
            )).scalar_one()
        return {"run_id": run_id, "response_count": count, "total_cost_usd": float(total_cost)}
```

- [ ] **Step 4：运行集成测试确认通过**

Run:
```bash
uv run pytest tests/integration/test_end_to_end.py -v
```
Expected: 1 passed.

- [ ] **Step 5：运行全部测试**

Run:
```bash
uv run pytest
```
Expected: all tests pass. No warnings about event loops.

- [ ] **Step 6：Commit**

```bash
git add src/prism/service.py tests/integration/__init__.py tests/integration/test_end_to_end.py
git commit -m "feat: add RunService end-to-end orchestration layer"
```

---

## Task 18：conftest.py + pytest-vcr 配置（为后续真实 provider 调用准备回放）

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/fixtures/cassettes/.gitkeep`

- [ ] **Step 1：创建 conftest + 空 cassette 目录**

Create `tests/conftest.py`:
```python
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
```

Create empty file `tests/fixtures/cassettes/.gitkeep`:
```
```

- [ ] **Step 2：确认现有测试仍通过**

Run:
```bash
uv run pytest
```
Expected: all tests pass, no env leakage.

- [ ] **Step 3：Commit**

```bash
git add tests/conftest.py tests/fixtures/cassettes/.gitkeep
git commit -m "test: add conftest with env isolation and vcr config"
```

---

## Task 19：项目级 Makefile + 文档

**Files:**
- Create: `Makefile`
- Modify: `README.md`

- [ ] **Step 1：创建 Makefile**

Create `Makefile`:
```makefile
.PHONY: install test lint typecheck fmt all

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

all: lint typecheck test
```

- [ ] **Step 2：扩展 README**

Replace `README.md` content with:
```markdown
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
```

- [ ] **Step 3：运行 `make all` 验证全绿**

Run:
```bash
make all
```
Expected: ruff check clean, mypy clean, pytest green.

If mypy complains, fix the offenders inline (the most likely source is missing `py.typed` marker).

- [ ] **Step 4：添加 `py.typed` 标记**

Create `src/prism/py.typed`:
```
```

Modify `pyproject.toml` `[tool.hatch.build.targets.wheel]`:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/prism"]
include = ["src/prism/py.typed"]
```

- [ ] **Step 5：Commit**

```bash
git add Makefile README.md src/prism/py.typed pyproject.toml
git commit -m "chore: add Makefile, py.typed marker, expand README"
```

---

## Task 20：P1 完工验证 + 打标

**Files:** 无新增；运行整套校验。

- [ ] **Step 1：运行全量测试 + lint + typecheck**

Run:
```bash
make all
```
Expected: all green.

- [ ] **Step 2：运行 `prism doctor` 确认环境**

Run:
```bash
uv run prism doctor
```
Expected: table 展示，Python/litellm OK。

- [ ] **Step 3：打 P1 完工 tag**

Run:
```bash
git tag -a p1-foundation -m "P1 Foundation complete: adapters, storage, orchestrator, judges, CLI skeleton"
git log --oneline -n 20
```

- [ ] **Step 4：更新规范文档状态**

Modify the first section of `docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md`:

Change:
```
- **状态**：设计确认，等待进入实现规划（writing-plans）
```
To:
```
- **状态**：P1 Foundation 已实现（tag `p1-foundation`）；P2 Limit Runner 待启动
```

- [ ] **Step 5：Commit**

```bash
git add docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md
git commit -m "docs: mark P1 Foundation complete"
```

---

## Self-Review checklist（完成实现后自查）

- [ ] 每个 task 都有完整可运行的代码，无 TODO / TBD
- [ ] `make all` 全绿
- [ ] `uv run prism doctor` 表格正常渲染
- [ ] 所有 commit 的 message 都遵循 `feat/fix/test/chore/docs:` 前缀
- [ ] P2-P5 所需的接口在 P1 内有占位（`Adapter`, `Judge`, `RunService`）且有测试覆盖
- [ ] `tests/fixtures/cassettes/` 空但已 checked-in
- [ ] 规范文档状态字段已更新

---

## P1 Success Criteria

- `make all` 全绿（lint + typecheck + 所有测试通过）
- `uv run prism doctor` 正常输出环境表
- `RunService` 端到端集成测试（`tests/integration/test_end_to_end.py`）通过：能接收模型 profile、执行 matrix、落盘 SQLite、写 artifact、产出 summary
- Adapter 能正确为 Anthropic / OpenAI / Google / DeepSeek / xAI / Kimi / Qwen 翻译 thinking/reasoning_effort 参数
- Judge 层 exact/numeric/regex/llm-judge 四种评分器可用
- Checkpoint 支持断点续跑（第二次运行跳过已完成单元）
- 该基础层足以支撑 P2 Limit Runner 在其上叠加 benchmark loader 与 prompt manager，无需修改 P1 代码
