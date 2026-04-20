# Prism — 大模型极限能力评测框架 设计文档

- **项目代号**：Prism（*Probe, Rank, Inspect, Score Models*）
- **日期**：2026-04-20
- **作者**：pingfan.work@gmail.com
- **许可证**：Apache-2.0
- **状态**：设计确认，等待进入实现规划（writing-plans）

---

## 1. 项目定位

**一句话定位**：
> *The open benchmark for testing frontier LLMs to their limits — where a model's raw answer, agent skill, and taste are scored side by side.*

**目标受众**：
- 关注前沿模型能力对比的开发者与研究者
- 希望复现厂商宣发指标并独立验证的技术决策者
- 关心中文场景表现、agent 真实可用性、模型 taste 的实用主义用户

**定位相对于已有框架（OpenCompass / HELM / lm-evaluation-harness / Inspect AI）**：
- 以 **Agent 能力为一等公民**（不仅是一次性问答）
- 以 **Claude Code 作为统一 Agent 载体**，通过 Router 让任意模型接入，实现跨厂商对比
- 以 **思考强度（thinking / reasoning_effort）作为一等配置维度**
- 以 **人工 Taste 盲审** 补齐自动 benchmark 的盲区
- 以 **反复核验 + canary + self-verification** 提供抗污染、可复现的"Verified Score"

---

## 2. 三大赛道

| 赛道 | 全称 | 测什么 | 评分方式 |
|---|---|---|---|
| ① **Limit** | One-shot Prompt→Response | 长上下文、知识、数学、代码、指令遵循、中文、安全、幻觉、多模态、成本/延迟 | 规则优先 + LLM Judge 兜底 |
| ② **Agent** | Claude Code–hosted task execution | SWE-Bench、Terminal-Bench、Tau-Bench、Aider Polyglot、CORE-Bench + 自研 Prism Real Tasks | 硬性可执行判定 + LLM Judge |
| ③ **Taste** | Human pairwise blind preference | 代码品味、设计感、思考深度 | 人工盲审 + ELO + tag |

**Prism Score（综合分）**：三赛道加权合成，默认权重 Limit 40 / Agent 40 / Taste 20，权重可在配置中覆写。每个模型条目在 leaderboard 必须标注 `thinking` / `reasoning_effort` 配置，否则分数无意义。

---

## 3. 系统架构

```
┌───────────────────────────────────────────────────────────────┐
│                        Prism CLI  (Typer)                     │
│    prism run / view / review / leaderboard / task / doctor    │
└──────────────────┬──────────────────────────┬─────────────────┘
                   │                          │
     ┌─────────────▼──────────┐   ┌───────────▼────────────┐
     │   Orchestrator (asyncio)│   │  Web UI (FastAPI+React)│
     │   - run planning        │   │  - leaderboard         │
     │   - concurrency/ratelim │   │  - trace viewer        │
     │   - checkpoint/resume   │   │  - blind review        │
     └─┬────────┬─────────┬────┘   └────────────▲───────────┘
       │        │         │                     │
  ┌────▼──┐ ┌───▼────┐ ┌──▼──────┐        ┌────┴──────┐
  │ Limit │ │ Agent  │ │  Taste  │        │  Storage  │
  │Runner │ │Runner  │ │ Sampler │        │ SQLite +  │
  └──┬────┘ └───┬────┘ └────┬────┘        │ JSON art. │
     │          │           │             └───────────┘
     │     ┌────▼───────────────┐
     │     │  Claude Code Router│   ← 把任意模型伪装成 Claude API
     │     └────▲───────────────┘
     │          │
  ┌──▼──────────┴──────────┐
  │     Model Adapter      │   ← LiteLLM + thinking/effort 翻译层
  │  Anthropic · OpenAI ·  │
  │  Google · DeepSeek ·   │
  │  xAI · Kimi · Qwen · … │
  └────────────┬───────────┘
               │
  ┌────────────▼───────────┐
  │      Judge Layer       │
  │ rules → LLM-judge →    │
  │ canary → human audit   │
  └────────────────────────┘
```

### 3.1 关键模块职责

- **Orchestrator**：把 `(task × model × seed × thinking_effort)` 展开为执行矩阵，并发调度（asyncio + 每 provider 独立限流），支持断点续跑（SQLite 中记录每条的 `pending / running / done / failed`）。
- **Model Adapter**：基于 LiteLLM，薄封装处理 `thinking` / `reasoning_effort` 到各家原生字段的翻译（Anthropic `thinking.type` + `output_config.effort`、OpenAI `reasoning_effort`、DeepSeek `reasoning`、Gemini `thinkingConfig` 等）。统一返回 `{text, reasoning_text, tokens_in, tokens_out, latency, cost}`。
- **Claude Code Router**：独立子进程，提供 Anthropic-compatible endpoint，把 Claude Code 的工具调用转发到任意模型。Agent Runner 启动 Claude Code 时设置 `ANTHROPIC_BASE_URL` 指向它。
- **Runner**：三个 Runner 各自独立，共享 Orchestrator 与 Judge。
- **Judge Layer**：四层流水线（规则 → LLM Judge → Canary → Human）。
- **Storage**：SQLite（核心表：`runs`, `models`, `tasks`, `prompts`, `responses`, `scores`, `traces`, `pairwise_votes`, `meta_ability`）+ `artifacts/<run_id>/` 存原始 trace JSON 与 Judge 的完整 reasoning。
- **Web UI**：FastAPI serve + 静态 React 打包，支持 leaderboard / trace viewer / blind review 三个视图；产物可直接发布到 GitHub Pages。

### 3.2 典型数据流（一次 run）

```
prism run --suite quick --models configs/models/*.yaml
  → Orchestrator 展开执行矩阵
  → 投递到对应 Runner
  → Runner 调用模型 / 执行任务，记录 trace
  → Judge Layer 评分（并行规则 + LLM 共识 + perturbation）
  → 写入 SQLite + artifact
  → prism view 本地启动 Web UI
  → prism leaderboard publish 产出静态 HTML
```

---

## 4. 赛道 ① Limit（极限问答）

### 4.1 十个维度与 v0.1 题库

| 维度 | 选定 benchmark | Quick 题量 |
|---|---|---|
| a. 长上下文 | Needle-in-Haystack 阶梯（8K/32K/128K/256K/512K/1M）+ RULER subset | 120 |
| b. 知识 | MMLU-Pro subset + GPQA-Diamond 全量 | 400 |
| c. 数学 | AIME 2024/2025 全量 + MATH-500 subset | 230 |
| d. 代码（一口气写对） | HumanEval+ 全量 + LiveCodeBench 最新月 | 250 |
| e. 指令遵循 | IFEval subset | 200 |
| f. 中文 / 多语言 | C-Eval subset + SuperCLUE subset | 400 |
| g. 安全 / 越狱 | HarmBench subset + XSTest 全量 | 400 |
| h. 幻觉 / 事实性 | SimpleQA subset + TruthfulQA 全量 | 400 |
| i. 多模态 | MMMU subset + MathVista subset | 400 |
| j. 成本 / 延迟 | 派生指标，从上述 benchmark 自动汇总 | — |
| 专项：思考强度 sweep | AIME 30 + MMLU-Pro 100 | 130 |
| **合计** | | **~2930** |

**三档套件**（通过 `--suite` 切换）：
- **Quick**（默认）~2930 题
- **Standard** ~8000 题（subset 加大、RULER 全量、CodeForces 加入）
- **Full**（研究者用）全量题库

**采样**：默认 `n=3`，上报 pass@1 / pass@3 / std。

### 4.2 "极限探测"三大专项视图

1. **Context Length Staircase**：同一 NIAH 任务在 8K/32K/128K/256K/512K/1M 六档下各跑一次，产出能力-长度曲线，暴露"宣称 1M 但 128K 就掉分"的情况。
2. **Reasoning Effort Sweep**：同一模型在 `thinking=off / high / max` 三档上跑 130 题专项集，产出"准确率 × 成本 × 延迟"的 Pareto 曲线。项目的杀手级可视化之一。
3. **Contamination Probe**：每个 benchmark 混入 20 条 canary（题面改写 / 扰动），若原题分数显著高于 canary 则标记"疑似污染"。

### 4.3 Prompt 版本化

每个 benchmark 的 prompt 模板、system message、few-shot 示例都存在 `prompts/<benchmark>/<version>.yaml`；版本号写进结果。改 prompt 就不是同一场比赛，这是可复现的硬纪律。

---

## 5. 赛道 ② Agent（代理能力）

### 5.1 学术基线

| Benchmark | Quick 子集 | Full | 评分 |
|---|---|---|---|
| SWE-Bench Verified | 100 | 500 | 官方 harness + pytest |
| Terminal-Bench | 40 | 80 | 官方 harness |
| Tau-Bench（retail+airline） | 60 | 230 | 官方 policy + tool check |
| Aider Polyglot | 75 | 225 | diff + test |
| CORE-Bench | 60 | 270 | 输出精确匹配 |

### 5.2 Prism Real Tasks（PRT）— 自研差异化

v0.1 起步 **30 个任务**（v0.2 扩到更多），按类别分层：

| 类别 | 数量 | 任务示例 |
|---|---|---|
| 新特性开发 | 8 | 给 FastAPI 应用加 JWT 中间件 + 单元测试 |
| Bug 定位修复 | 6 | flaky test 在 CI 上随机失败，定位并修复 |
| 重构 | 4 | 把 Express 路由重构成 Fastify，保持行为等价 |
| 迁移升级 | 4 | Python 3.9 → 3.12 迁移 + 修复所有 warning |
| 数据处理 | 4 | 读 CSV，按业务规则聚合，输出可视化 HTML |
| 工具脚本 | 4 | 从 OpenAPI spec 生成 TypeScript SDK 的 CLI |

**每个任务规格**：
- 初始 repo 快照（git tarball，内容哈希固定）
- 任务描述（模拟真实用户指令）
- 验收 rubric：
  - **硬判**：必须跑过的 pytest / lint / build 命令
  - **软判**：LLM Judge 按 rubric 打分（清晰度、是否过度改动、是否引入新依赖）
  - **Taste 采样点**：是否进入 Taste 赛道的 pairwise 候选池

### 5.3 执行流程

```
prism run --track agent --task prt-001 --model claude-opus-4-7@max
  → 拉取 task 快照到临时工作目录
  → 启动 Claude Code，ANTHROPIC_BASE_URL 指向 Router
  → Router 把请求路由到 --model 指定的被测模型
  → 注入用户指令到 Claude Code
  → 记录完整 trace（每轮 tool call、file diff、最终 output）
  → 超时 / step-limit 保护（默认 20 min, 30 steps）
  → 跑验收脚本 + LLM Judge → 得分
  → trace 存入 artifact，可在 Web UI 里逐步回放
```

### 5.4 "极限"探测（Agent 赛道）

- **工具调用深度曲线**：同一任务在 step-limit ∈ {10, 30, 100} 下分别跑，看成功率与轮数的关系
- **大 repo 上下文压力**：在任务 repo 里塞入 500K tokens 的干扰代码，测模型定位能力
- **错误恢复**：任务中故意 mock 一次工具失败，看模型是否死循环、是否切换策略

---

## 6. 赛道 ③ Taste（品味人工盲审）

### 6.1 执行规格

- **候选池**：每次 run 完成后自动从 Agent 与 Limit 中挑选"开放式题目"（PRT、自由文本 QA、代码重构）作为 pairwise 候选
- **评审 UI**：`prism review` 启动本地页面
  - 随机呈现 A/B 两个去标识输出，随机顺序，可选长度归一化（pad 到等长或截断到等长）
  - 评审人输入：二选一 / 平手 + 可选 tag（多选）
- **最少样本**：每模型 30 pair 才入榜
- **聚合**：Bradley-Terry 模型产出 ELO 与 95% 置信区间
- **多评审**：本地用户名 + 可选导出 JSON；v0.3 之后考虑 HF Spaces 众包

### 6.2 Tag 体系（预设）

- `好的代码品味` / `更克制` / `过度工程`
- `更清晰的解释` / `思考更深入`
- `AI 味重` / `其它（自定义）`

Tag 统计用于"定性分析"：比如"评审者选 A 时最常打的 tag 是 '更克制、不过度抽象'"作为 leaderboard 的叙事素材。

### 6.3 在 Leaderboard 中的呈现

Taste Leaderboard 与 Limit / Agent 自动榜单**并列独立展示**，强调 taste 是独立维度。

同时，Taste 以默认 **20% 权重**参与 Prism Score 合成（Limit 40 / Agent 40 / Taste 20，可在配置中覆写）。**若某模型的人工 pairwise 样本不足 30 对**，该模型的 Taste 权重自动归零，Limit 与 Agent 权重按比例重新归一化（即 50/50）。这保证了新模型在拿到足够人工评审前也能有合理的综合分，且不会因缺评审被不公平拉低。

---

## 7. 元能力（Meta-Ability）机制 — 横跨所有赛道

### 7.1 Self-Verification Loop

每条题目的执行：

```
Round 1: 模型回答 → raw_answer, raw_score
Round 2: (question + raw_answer) 重新交给同一模型：
         "请检查以上答案是否正确，若不正确请修正。"
         → verified_answer, verified_score
Round 3 (可选): 元检查：
         "你上一次的修正是否改进了答案？"
         → meta_score
```

**指标**：
- **Self-Correction Rate** = 自检后分数提升的比例
- **False-Retraction Rate** = 原本对却被自己改错的比例
- **Metacognition Score** = Correction − False-Retraction

### 7.2 反复核验（Cross-Verification）

三条独立通道，全部通过才计入 **Verified Score**：

1. **Self-consistency @ n=3**：同一模型三次采样投票
2. **Cross-model judge**：用另一厂商的强模型复核（禁止模型做自己的裁判）
3. **Perturbation robustness**：题面语义等价改写 3 个变体，答案应保持一致

Verified Score 与 Raw Score 在 leaderboard 上独立展示。Verified Score 更抗污染、更严格，是"硬榜"。

---

## 8. Judge 层

四层流水线：

- **Tier 1 — Rules**：unit test、exact match、正则、SymPy 数值归一、编译、lint。优先使用。
- **Tier 2 — LLM Judge**：
  - 多 Judge（≥2 个不同厂商）共识
  - 每次 Judge 必须输出 `{score, confidence ∈ [0,1], reasoning}`
  - 禁止"模型做自己的裁判"（避免自评偏差），Judge 与被测模型不能同厂商
  - Judge 的 prompt、rubric、reasoning 全部开源并存入 artifact，支持第三方复核
- **Tier 3 — Canary**：每 benchmark 混入 20 条私有 canary 题，检测污染
- **Tier 4 — Human Audit**：低置信（confidence < 0.6）或 Judge 分歧大的样本自动入队，由人工抽检复核

---

## 9. 模型适配与 thinking / reasoning_effort 处理

### 9.1 统一配置 schema

```yaml
# configs/models/claude-opus-4-7-max.yaml
id: claude-opus-4-7@thinking-max
display_name: "Claude Opus 4.7 (thinking=max)"
provider: anthropic
model: claude-opus-4-7
thinking: { enabled: true, effort: max }   # 一等参数
rate_limit: { rpm: 50, tpm: 400000 }
cost: { input_per_mtok: 15.0, output_per_mtok: 75.0 }
```

### 9.2 适配器翻译规则

| 目标 API | thinking 开关 | effort 字段 |
|---|---|---|
| Anthropic API | `thinking.type: enabled/disabled` | `output_config.effort: high/max` |
| Chat Completion API | （无独立开关，由 model name 或 `reasoning_effort` 决定） | `reasoning_effort: high/max` |
| OpenAI reasoning（o-family） | model ID 即开关 | `reasoning_effort: low/medium/high` |
| Gemini | `thinkingConfig.thinkingBudget` | 按预算数值映射 |
| DeepSeek | `reasoning` 字段 | 模型变体决定 |
| 其它 | 适配器兜底 | 适配器兜底 |

**Leaderboard 展示规则**：每个模型条目必须带 `thinking=off/high/max` 后缀，否则不允许上榜。

---

## 10. 数据与产出

### 10.1 Storage

- **SQLite** 单库（`prism.db`）：核心表 `runs / models / tasks / prompts / responses / scores / traces / pairwise_votes / meta_ability`
- **Artifact 目录** `artifacts/<run_id>/`：原始 trace JSON、LLM Judge 完整 reasoning、canary 结果、perturbation 配对、模型输出文本
- 支持 `prism replay <run-zip> --judge-only` 第三方完整复核

### 10.2 Leaderboard 产出

- **静态 HTML**：`leaderboard/index.html`，可发布到 GitHub Pages
- 排序 / 筛选 / 思考强度分层展示
- 每个条目：分数 / Verified Score / 成本 / 延迟 / thinking 配置 / run 时间戳 / artifact 下载链接

### 10.3 Web UI（本地）

- `prism view`：交互式 trace viewer（逐步回放 Agent 执行、Judge 理由、模型逐 token 输出）
- `prism review`：盲审页
- `prism leaderboard`：本地 leaderboard 预览

---

## 11. 目录结构

```
prism/
├── README.md                    # 英文 + 中文双语
├── pyproject.toml               # uv + hatch
├── LICENSE                      # Apache-2.0
├── src/prism/
│   ├── cli.py                   # Typer 入口
│   ├── orchestrator/            # 并发 / 限流 / 断点续跑
│   ├── adapters/
│   │   ├── model.py             # LiteLLM wrapper + thinking/effort 翻译
│   │   └── router.py            # Claude Code Router 子进程管理
│   ├── runners/
│   │   ├── limit.py
│   │   ├── agent.py
│   │   └── taste.py
│   ├── judges/
│   │   ├── rules.py
│   │   ├── llm.py
│   │   ├── meta.py              # self-verification loop
│   │   └── cross.py             # cross-model verification
│   ├── storage/                 # SQLite schema + artifact IO
│   ├── webui/                   # FastAPI + 静态 React 构建
│   └── leaderboard/             # 静态 HTML 生成器
├── benchmarks/
│   ├── mmlu_pro/
│   ├── gpqa_diamond/
│   ├── swe_bench_verified/
│   ├── prt/                     # Prism Real Tasks
│   └── ...
├── configs/
│   ├── models/
│   ├── suites/
│   └── example.yaml
├── prompts/
├── tests/                       # pytest + VCR replay
├── docs/
└── .github/workflows/
```

---

## 12. 运行与发布

### 12.1 CLI

```
prism doctor                        # 检查 API key / Claude Code / Router 就绪
prism run --suite quick --models configs/models/*.yaml
prism view                          # 本地 Web UI
prism review                        # 盲审
prism leaderboard publish           # 生成并 push gh-pages
prism replay <run-zip>              # 复现历史 run（judge-only 模式支持）
```

### 12.2 CI / 自动化

- **PR CI**：跑 Quick suite 的 smoke 子集（~100 题）+ 架构 lint + schema 校验
- **每周定时 run**：GitHub Actions 拉取最新模型，跑 Quick，自动更新官方 leaderboard
- **每月 Standard run**：半年跑一次 Full run

### 12.3 可复现保证

每次 run 产出 `run-<ts>.zip`：模型配置、prompt 版本哈希、完整 trace、judge reasoning、SQLite 快照。第三方 `prism replay` 可完整复核所有 LLM Judge 评分。

---

## 13. 里程碑

- **v0.1（MVP，2 个月）**：
  - Limit Quick（10 维 ~2930 题）
  - Agent：5 个学术 benchmark subset + 5 个 PRT 自研任务
  - Taste：Pairwise UI + ELO
  - 6 个 provider（Anthropic / OpenAI / Google / DeepSeek / xAI / Kimi）
  - 本地 Web UI + GitHub Pages leaderboard
  - Self-verification + Cross-verification + Perturbation 三条元能力通道
- **v0.2**：
  - Agent 扩到 30 PRT 任务
  - 中文 leaderboard 独立视图
  - Perturbation 自动化
  - Canary 池建设
- **v0.3**：
  - 社区贡献 benchmark 机制（`prism benchmarks add`）
  - 众包盲审工作流（HF Spaces / 社区）
  - Full 多模态支持
- **v1.0**：
  - 稳定 API / 文档
  - 学术引用（preprint）
  - 成为独立可复现的评测基线

---

## 14. Non-goals（明确不做的）

- **不做**商业闭源模型的"逆向"API 适配（要求 provider 提供官方可用 endpoint）
- **v1.0 前不做**云端多用户 leaderboard（Chatbot Arena / HELM 已占据该生态位）
- **不做**模型微调或训练数据清洗
- **不做**与"训练数据去污染"相关的工具链（仅做 canary 检测）
- **v0.1 不做**自建 Router，复用成熟的 Claude Code Router / LiteLLM Proxy
- **不做**"原生 Agent 载体"的多 Agent 对比（坚持统一载体，避免比较产品组合）

---

## 15. Open Questions（待实现阶段决议）

1. **Claude Code Router 的具体选型**：自研薄路由 vs 直接用社区 `claude-code-router` vs LiteLLM proxy——需在实现阶段做技术验证。
2. **多模态评测的成本预算**：v0.1 默认开启还是作为 opt-in？
3. **PRT 30 个任务的具体清单**：需单独写 `docs/prt-catalog.md` 列出每个任务的 repo snapshot、指令、rubric。
4. **Leaderboard 防刷**：社区 fork 后可能产出"乱跑结果"——是否需要"官方 run"与"社区 run"两种徽章。
5. **盲审伦理**：人工评审者是否匿名、是否支付；v0.3 众包时需设计 consent 流程。
6. **Canary 题的来源**：需要私藏池（不公开），但又要 reproducible——考虑"加密发布 + 版本回滚时解密"的方案。

这些问题在 writing-plans 阶段的前两周技术验证中处理。
