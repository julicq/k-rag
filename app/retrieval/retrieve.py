from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

from app.config import settings
from app.embed.ollama_client import OllamaClient
from app.index import faiss_store


class Retriever:
    def __init__(self):
        self.ollama = OllamaClient()
        self.cross_encoder: CrossEncoder | None = None
        self.index = None
        self.metas: List[Dict[str, Any]] = []

    def _ensure_loaded(self) -> None:
        if self.index is None:
            self.index, self.metas = faiss_store.load_index(settings.hnsw_ef_search)

    def _ensure_reranker(self) -> None:
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder(settings.rerank_model)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.ollama.embed([query])
        vec = vec.astype(np.float32)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        return vec

    def ann_search(self, query: str, topk: int | None = None) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        self._ensure_loaded()
        qv = self.embed_query(query)
        sims, ids = faiss_store.search(self.index, qv, topk or settings.topk)
        ids_list = ids[0].tolist()
        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ids_list):
            if idx < 0 or idx >= len(self.metas):
                continue
            meta = self.metas[idx]
            meta = dict(meta)
            meta["_id"] = idx
            meta["_rank"] = rank
            meta["_sim"] = float(sims[0][rank])
            hits.append(meta)
        return hits, qv

    def rerank(self, query: str, hits: List[Dict[str, Any]], topn: int | None = None) -> List[Dict[str, Any]]:
        if not hits:
            return []
        self._ensure_reranker()
        assert self.cross_encoder is not None
        pairs = [(query, h["text"]) for h in hits]
        scores = self.cross_encoder.predict(pairs).tolist()
        for h, s in zip(hits, scores):
            h["_rerank"] = float(s)
        hits.sort(key=lambda x: x["_rerank"], reverse=True)
        return hits[: (topn or settings.topn_context)]
