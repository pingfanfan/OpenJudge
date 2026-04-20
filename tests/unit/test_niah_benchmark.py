from prism.benchmarks.niah.benchmark import NIAHBenchmark
from prism.judges.rules import RegexJudge


def test_default_lengths_and_depths_produce_prompt_matrix():
    bm = NIAHBenchmark(
        lengths=[1024, 4096],
        depths=[0.0, 0.5, 1.0],
    )
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 6
    for p in prompts:
        assert "context_tokens" in p.metadata
        assert "needle_depth" in p.metadata
        assert p.metadata["context_tokens"] in (1024, 4096)
        assert p.metadata["needle_depth"] in (0.0, 0.5, 1.0)


def test_prompt_contains_needle():
    bm = NIAHBenchmark(lengths=[512], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    content = prompt.messages[0]["content"]
    assert "special passcode" in content.lower()
    assert prompt.expected is not None


def test_judge_is_regex():
    bm = NIAHBenchmark(lengths=[512], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)


def test_prompt_id_encodes_length_and_depth():
    bm = NIAHBenchmark(lengths=[1024], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    assert "niah-" in prompt.prompt_id
    assert "len1024" in prompt.prompt_id
    assert "depth50" in prompt.prompt_id


def test_needle_is_deterministic_per_position():
    bm1 = NIAHBenchmark(lengths=[1024], depths=[0.5], seed=7)
    bm2 = NIAHBenchmark(lengths=[1024], depths=[0.5], seed=7)
    p1 = next(iter(bm1.load_prompts(subset="full")))
    p2 = next(iter(bm2.load_prompts(subset="full")))
    assert p1.expected == p2.expected
