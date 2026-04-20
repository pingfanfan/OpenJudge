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
