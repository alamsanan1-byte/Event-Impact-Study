from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(path: str) -> Path:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect_db():
    # Imported lazily so pure helper/unit tests do not require DuckDB just to import modules.
    import duckdb

    cfg = load_config()
    db_path = project_path(cfg["paths"]["db"])
    con = duckdb.connect(str(db_path))
    schema = (ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    con.execute(schema)
    return con


def ensure_dirs() -> None:
    cfg = load_config()
    for key in ("raw", "curated", "outputs"):
        (ROOT / cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
