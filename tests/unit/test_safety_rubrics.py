from prism.judges.safety_rubrics import HARM_REFUSAL_RUBRIC, OVER_REFUSAL_RUBRIC


def test_harm_refusal_rubric_scores_refusal_as_1():
    r = HARM_REFUSAL_RUBRIC.lower()
    assert "refus" in r
    assert "1.0" in r or "1" in r
    assert "harmful" in r or "harm" in r
    assert "0.0" in r or "0" in r


def test_over_refusal_rubric_scores_helpful_response_as_1():
    r = OVER_REFUSAL_RUBRIC.lower()
    assert "help" in r or "comply" in r
    assert "1.0" in r or "1" in r
    assert "refus" in r
    assert "0.0" in r or "0" in r
