from __future__ import annotations
from typing import List, Dict
import numpy as np

from app.utils.io import data_path, read_jsonl
from app.embed.ollama_client import OllamaClient
from app.index.faiss_store import build_hnsw_index, save_index
from app.config import settings


def load_chunks() -> List[Dict]:
    chunks_dir = data_path("chunks")
    items: List[Dict] = []
    for f in chunks_dir.glob("*.jsonl"):
        items.extend(read_jsonl(f))
    return items


def embed_texts(texts: List[str], batch: int = 32) -> np.ndarray:
    client = OllamaClient()
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i : i + batch]
        emb = client.embed(batch_texts)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        vecs.append(emb.astype(np.float32))
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)


def build_from_chunks() -> None:
    rows = load_chunks()
    if not rows:
        raise RuntimeError("No chunks found. Run crawler and preprocess first.")
    texts = [r["text"] for r in rows]
    metas = []
    for r in rows:
        meta = dict(r.get("meta", {}))
        meta["text"] = r["text"]
        meta["id"] = r.get("id")
        metas.append(meta)
    embeddings = embed_texts(texts)
    index = build_hnsw_index(embeddings, m=settings.hnsw_m, ef_construction=settings.hnsw_ef_construction)
    save_index(index, metas)


if __name__ == "__main__":
    build_from_chunks()
