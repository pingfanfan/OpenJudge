from prism.storage.schema import Base, Response, Run


def test_tables_defined():
    names = {t.name for t in Base.metadata.tables.values()}
    assert {"runs", "models", "tasks", "prompts", "responses", "scores"} <= names


def test_run_columns():
    cols = {c.name for c in Run.__table__.columns}
    assert {"id", "created_at", "suite", "status", "config_hash"} <= cols


def test_response_columns():
    cols = {c.name for c in Response.__table__.columns}
    assert {
        "id",
        "run_id",
        "model_id",
        "prompt_id",
        "seed",
        "text",
        "reasoning_text",
        "tokens_in",
        "tokens_out",
        "latency_ms",
        "cost_usd",
        "finish_reason",
        "created_at",
    } <= cols
