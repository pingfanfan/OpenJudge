from prism.config.model_profile import Cost


def compute_cost(cost: Cost, *, tokens_in: int, tokens_out: int) -> float:
    return (
        cost.input_per_mtok * tokens_in / 1_000_000
        + cost.output_per_mtok * tokens_out / 1_000_000
    )
