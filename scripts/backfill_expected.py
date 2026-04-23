"""Backfill `prompts.expected` in an existing Prism DB.

For DBs created before the `expected` column existed, this:
  1. Adds the column if missing (idempotent).
  2. Iterates registered benchmarks and re-loads their prompts from the
     configured source, writing the `expected` field for every prompt_id
     that matches.

Usage:
    uv run python scripts/backfill_expected.py ~/Documents/prism-runs/pandora-2026-04-22
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from prism.benchmarks import default_registry


def _ensure_column(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(prompts)")}
    if "expected" not in cols:
        conn.execute("ALTER TABLE prompts ADD COLUMN expected TEXT")
        conn.commit()
        print("  added prompts.expected column")
    else:
        print("  prompts.expected already exists")


def _current_benchmarks(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT task_id FROM prompts").fetchall()
    return [r[0] for r in rows]


def backfill(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    print(f"DB: {db_path}")
    _ensure_column(conn)

    registry = default_registry()
    present = set(_current_benchmarks(conn))
    print(f"  benchmarks in DB: {sorted(present)}")

    for bm_name in sorted(present):
        if bm_name not in registry.names():
            print(f"  [skip] {bm_name}: not in registry")
            continue
        bm_cls = registry.get_class(bm_name)
        try:
            bm = bm_cls()
        except Exception as e:
            print(f"  [skip] {bm_name}: can't instantiate: {e}")
            continue

        updated = 0
        missing = 0
        try:
            for subset in ("full", "standard", "quick"):
                # Walk the widest subset we can load so we cover every persisted prompt
                try:
                    specs = list(bm.load_prompts(subset=subset))
                    break
                except Exception:
                    continue
            else:
                specs = list(bm.load_prompts(subset="quick"))
        except Exception as e:
            print(f"  [fail] {bm_name}: load_prompts raised: {e}")
            continue

        spec_map = {s.prompt_id: s.expected for s in specs}
        rows = conn.execute(
            "SELECT id FROM prompts WHERE task_id = ?", (bm_name,)
        ).fetchall()
        for (pid,) in rows:
            if pid in spec_map and spec_map[pid] is not None:
                conn.execute(
                    "UPDATE prompts SET expected = ? WHERE id = ?",
                    (str(spec_map[pid]), pid),
                )
                updated += 1
            else:
                missing += 1
        conn.commit()
        print(f"  {bm_name}: updated {updated}, missing {missing} (total {updated + missing})")

    # Final count
    total = conn.execute("SELECT COUNT(*) FROM prompts WHERE expected IS NOT NULL").fetchone()[0]
    grand = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
    print(f"\nDone. {total}/{grand} prompts now have an expected value.")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir", help="Prism work dir (contains prism.db)")
    args = ap.parse_args()
    db = Path(args.work_dir) / "prism.db"
    if not db.exists():
        raise SystemExit(f"Not found: {db}")
    backfill(db)


if __name__ == "__main__":
    main()
