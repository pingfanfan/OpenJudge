from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.rules import NumericJudge
from prism.utils.image import image_to_data_url

if TYPE_CHECKING:
    from prism.adapters.base import Adapter

_PROMPT_TEXT_TEMPLATE = (
    "Study the image above, then answer the math question."
    " Give your final numeric answer on the last line.\n\nQuestion: {question}"
)


class MathVistaBenchmark(Benchmark):
    """MathVista — multimodal math reasoning.

    v0.1 focuses on `question_type=free_form` with numeric `answer_type`.
    Multi-choice rows are skipped (not yet supported in v0.1).
    """

    name = "mathvista"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 100, "standard": 300, "full": None}

    def __init__(
        self,
        *,
        source: str = "AI4Math/MathVista",
        source_format: str = "hf",
        split: str = "testmini",
        fixture_root: Path | str | None = None,
    ) -> None:
        self.source = source
        self.source_format = source_format
        self.split = split
        self.fixture_root = Path(fixture_root) if fixture_root else None

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        cap = self._cap_for(subset)
        yielded = 0
        for row in load_dataset_cached(
            source=self.source, format=self.source_format, split=self.split
        ):
            if row.get("question_type") != "free_form":
                continue
            if row.get("answer_type") not in ("integer", "float"):
                continue
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
        return NumericJudge(tolerance=1e-3)

    def _row_to_prompt(self, row: dict[str, Any]) -> PromptSpec:
        pid = str(row.get("pid") or row.get("id") or row["question"][:32])
        text = _PROMPT_TEXT_TEMPLATE.format(question=row["question"])

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
            raise ValueError(f"MathVista row {pid!r} missing 'image' or 'image_path'")

        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        return PromptSpec(
            prompt_id=f"mathvista-{pid}",
            task_id="mathvista",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=str(row["answer"]),
            metadata={"answer_type": row.get("answer_type")},
        )
