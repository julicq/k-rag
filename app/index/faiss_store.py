from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import faiss

from app.utils.io import data_path, write_pickle, read_pickle, ensure_dir


INDEX_DIR = data_path("index")
INDEX_FILE = INDEX_DIR / "chunks.index"
META_FILE = INDEX_DIR / "meta.pkl"


def build_hnsw_index(embeddings: np.ndarray, m: int = 32, ef_construction: int = 200) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, metas: List[Dict[str, Any]]) -> None:
    ensure_dir(INDEX_DIR)
    faiss.write_index(index, str(INDEX_FILE))
    write_pickle(META_FILE, metas)


def load_index(ef_search: int = 64) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("FAISS index or meta not found")
    index = faiss.read_index(str(INDEX_FILE))
    try:
        index.hnsw.efSearch = ef_search
    except Exception:
        pass
    metas: List[Dict[str, Any]] = read_pickle(META_FILE)
    return index, metas


def search(index: faiss.Index, query_vec: np.ndarray, topk: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    sims, ids = index.search(query_vec, topk)
    return sims, ids
