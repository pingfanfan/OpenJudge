from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.rules import RegexJudge
from prism.utils.image import image_to_data_url

if TYPE_CHECKING:
    from prism.adapters.base import Adapter

_PROMPT_TEXT_TEMPLATE = (
    "Study the image above, then answer the question."
    ' Respond with ONLY the letter (A/B/C/D) on the last line, prefixed by "Answer:".'
    "\n\nQuestion: {question}\n\nChoices:\n{choices_block}"
    '\n\nGive your final answer as "Answer: X".'
)

_JUDGE_PATTERN = r"Answer:\s*([A-D])\b"


class MMMUBenchmark(Benchmark):
    """MMMU — Massive Multi-discipline Multimodal Understanding.

    Prompts are multimodal: content is a list of [text, image_url] parts.
    The image is either a PIL.Image (from HF) or a local path (from test fixtures).
    """

    name = "mmmu"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 100, "standard": 300, "full": None}

    def __init__(
        self,
        *,
        source: str = "MMMU/MMMU",
        source_format: str = "hf",
        split: str = "validation",
        subset_name: str | None = None,
        fixture_root: Path | str | None = None,
    ) -> None:
        """fixture_root: if set, image_path fields in JSONL rows are resolved relative to it."""
        self.source = source
        self.source_format = source_format
        self.split = split
        self.subset_name = subset_name
        self.fixture_root = Path(fixture_root) if fixture_root else None

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        cap = self._cap_for(subset)
        yielded = 0
        load_kwargs: dict[str, Any] = {}
        if self.source_format == "hf" and self.subset_name:
            load_kwargs["name"] = self.subset_name
        for row in load_dataset_cached(
            source=self.source, format=self.source_format, split=self.split, **load_kwargs
        ):
            if cap is not None and yielded >= cap:
                break
            yield self._row_to_prompt(row)
            yielded += 1

    def make_judge(
        self,
        prompt: PromptSpec,
        *,
        llm_judge_adapter: Adapter | None = None,
    ) -> Judge:
        return RegexJudge(pattern=_JUDGE_PATTERN)

    def _row_to_prompt(self, row: dict[str, Any]) -> PromptSpec:
        qid = str(row.get("id") or row["question"][:32])
        options = row["options"]
        choices_block = "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options))
        text = _PROMPT_TEXT_TEMPLATE.format(question=row["question"], choices_block=choices_block)

        # Resolve image: HF returns PIL.Image under 'image'; JSONL fixtures use 'image_path'.
        if "image" in row and row["image"] is not None:
            image_url = image_to_data_url(row["image"])
        elif "image_path" in row:
            path = Path(row["image_path"])
            if not path.is_absolute() and self.fixture_root is not None:
                candidate = self.fixture_root / path
                if not candidate.exists():
                    stripped = Path(*path.parts[1:]) if len(path.parts) > 1 else path
                    candidate = self.fixture_root / stripped
                path = candidate
            image_url = image_to_data_url(path)
        else:
            raise ValueError(f"MMMU row {qid!r} missing 'image' or 'image_path'")

        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        return PromptSpec(
            prompt_id=f"mmmu-{qid}",
            task_id="mmmu",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=row["answer"],
            metadata={},
        )
