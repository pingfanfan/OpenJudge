from pathlib import Path

from prism.benchmarks.ifeval.benchmark import IFEvalBenchmark
from prism.judges.ifeval import IFEvalJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "ifeval_sample.jsonl"
    bm = IFEvalBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    first = prompts[0]
    assert first.task_id == "ifeval"
    assert first.prompt_id == "ifeval-ifeval-1"
    assert "climate change" in first.messages[0]["content"]
    # metadata carries zipped (id, kwargs) pairs
    assert first.metadata["constraints"] == [
        {"id": "length_constraints:number_words", "kwargs": {"relation": "at least", "num_words": 100}},
        {"id": "punctuation:no_comma", "kwargs": {}},
    ]


def test_judge_is_ifeval():
    fixture = Path(__file__).parent.parent / "fixtures" / "ifeval_sample.jsonl"
    bm = IFEvalBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, IFEvalJudge)
    assert len(judge.constraints) == 2
