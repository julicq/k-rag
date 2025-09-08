import os
import re
from pathlib import Path
from typing import Iterable, Dict, Any, List
import orjson
import pickle

from app.config import settings


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\u0400-\u04FF]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row))
            f.write(b"\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(orjson.loads(line))
    return items


def write_pickle(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def data_path(*parts: str) -> Path:
    return settings.data_dir.joinpath(*parts)
