from prism.benchmarks.ruler_mk.benchmark import RulerMKBenchmark
from prism.judges.rules import RegexJudge


def test_emits_staircase_times_depths_prompts():
    bm = RulerMKBenchmark(lengths=[1024, 4096], depths=[0.25, 0.75])
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 4
    for p in prompts:
        assert p.task_id == "ruler_mk"
        assert "context_tokens" in p.metadata
        assert "queried_key" in p.metadata


def test_prompt_contains_3_keys_and_asks_for_one():
    bm = RulerMKBenchmark(lengths=[1024], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    content = prompt.messages[0]["content"]
    assert "Question" in content
    queried = prompt.metadata["queried_key"]
    assert queried in content


def test_judge_is_regex():
    bm = RulerMKBenchmark(lengths=[1024], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)


def test_expected_value_matches_queried_key():
    bm = RulerMKBenchmark(lengths=[1024], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    content = prompt.messages[0]["content"]
    assert prompt.expected is not None
    assert prompt.expected in content
