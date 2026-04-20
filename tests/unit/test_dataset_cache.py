from pathlib import Path
from unittest.mock import patch

import pytest

from prism.benchmarks.dataset_cache import load_dataset_cached


def test_load_dataset_cached_with_local_jsonl_fixture(tmp_path: Path):
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    rows = list(load_dataset_cached(source=str(fixture), format="jsonl"))
    assert len(rows) == 2
    assert rows[0]["question"] == "What is 2+2?"
    assert rows[1]["answer"] == "C"


def test_load_dataset_cached_falls_back_to_hf(tmp_path: Path):
    """When format is 'hf', we delegate to the datasets library."""
    with patch("prism.benchmarks.dataset_cache.datasets") as mock_datasets:
        mock_datasets.load_dataset.return_value = [
            {"question_id": "q1"}, {"question_id": "q2"}
        ]
        rows = list(load_dataset_cached(source="TIGER-Lab/MMLU-Pro", format="hf", split="test"))
        assert len(rows) == 2
        mock_datasets.load_dataset.assert_called_once_with("TIGER-Lab/MMLU-Pro", split="test")


def test_load_dataset_cached_rejects_unknown_format():
    with pytest.raises(ValueError, match="unknown format"):
        list(load_dataset_cached(source="x", format="xml"))
