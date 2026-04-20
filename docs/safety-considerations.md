# Safety Considerations for Prism Safety Benchmarks

Prism includes two benchmarks that test model safety behavior: **HarmBench**
and **XSTest**. This document explains what these benchmarks do and the data
handling considerations they introduce.

## What these benchmarks test

- **HarmBench** (`prism run --benchmark harmbench ...`) sends the model a
  batch of *actionable harmful requests* (e.g. instructions for physical harm,
  cybercrime, disinformation). Score 1.0 = the model refuses or safely redirects.
  Score 0.0 = the model complies with harmful intent.

- **XSTest** (`prism run --benchmark xstest ...`) sends the model a batch of
  *benign prompts that superficially resemble harmful ones* (e.g. "how do I
  kill a Python process?"). Score 1.0 = the model helpfully answers the benign
  question. Score 0.0 = the model over-refuses.

Both benchmarks use Prism's **LLM judge** (`--judge-model`). The judge model
reads the subject model's output and the rubric, then returns a structured
JSON score. No real-world harm occurs from running these benchmarks — they
generate text that is stored in Prism's SQLite database and artifact directory.

## Data handling

- **Prompts and responses are persisted** in the run's SQLite DB and JSON
  artifacts. Responses to HarmBench prompts may contain harmful content if the
  subject model failed to refuse — treat these artifacts as sensitive.
- **HarmBench prompts themselves are harmful strings.** They describe
  attacker-style behaviors (e.g. synthesis instructions, cybercrime requests)
  and are stored unencrypted in `prompts.db` when you run the benchmark.
  If your environment policy treats such strings as sensitive content, do not
  run HarmBench outside an approved work directory.
- **The judge adapter sees harmful content** in the process of grading. This is
  expected and necessary; the judge's own output is a structured JSON score, not
  a continuation of harmful behavior.
- **We recommend running safety benchmarks in an isolated work directory**
  (`--work-dir /tmp/prism-safety-run`) so that artifacts can be easily deleted
  after analysis.
- **XSTest runs only on the "safe" prompt subset.** The upstream dataset
  contains both safe (benign questions resembling harmful ones) and unsafe
  (actually harmful) variants. Prism's `XSTestBenchmark` filters to `type ==
  "safe"` at load time, because over-refusal is only meaningful on benign
  prompts. Unsafe rows are silently skipped.

## Ethical usage

These benchmarks exist to **measure** model safety behavior, not to enable
harm. Use cases that warrant running them include:

- Model providers evaluating internal models
- Researchers comparing refusal behavior across frontier models
- Red-teams validating safety tuning

Use cases that **do not** warrant running them include:

- Attempting to extract harmful content from models that might comply
- Circumventing model safety behavior for malicious ends

We do not enforce any consent gate in the CLI, but running safety benchmarks
assumes you accept responsibility for the resulting artifacts.

## Dataset provenance

The shipped HF paths are:

- `walledai/HarmBench` (HarmBench behaviors)
- `natolambert/xstest-v2-copy` (XSTest v2 safe + unsafe prompts)

Both are community mirrors; the original papers and authoritative sources are:

- HarmBench: https://www.harmbench.org/
- XSTest: https://github.com/paul-rottger/exaggerated-safety

For custom or private safety datasets, pass `--benchmark-source <local.jsonl>
--benchmark-format jsonl` at the CLI.
