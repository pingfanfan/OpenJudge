from pathlib import Path

import yaml

from prism.config.model_profile import ModelProfile


def load_model_profile(path: str | Path) -> ModelProfile:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ModelProfile.model_validate(data)
