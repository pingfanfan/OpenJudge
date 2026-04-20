from prism.config.model_profile import ModelProfile
from prism.orchestrator.matrix import Cell, expand_matrix


def _m(id_: str) -> ModelProfile:
    return ModelProfile(id=id_, provider="openai", model="x")


def test_single_model_single_task_single_seed():
    cells = list(
        expand_matrix(
            models=[_m("a")],
            prompt_ids=["p1"],
            seeds=[0],
        )
    )
    assert cells == [Cell(model_id="a", prompt_id="p1", seed=0)]


def test_full_product():
    cells = list(
        expand_matrix(
            models=[_m("a"), _m("b")],
            prompt_ids=["p1", "p2"],
            seeds=[0, 1, 2],
        )
    )
    assert len(cells) == 2 * 2 * 3


def test_deterministic_order():
    cells = list(
        expand_matrix(
            models=[_m("b"), _m("a")],
            prompt_ids=["p2", "p1"],
            seeds=[1, 0],
        )
    )
    # First model "b", first prompt "p2", first seed 1
    assert cells[0] == Cell(model_id="b", prompt_id="p2", seed=1)
