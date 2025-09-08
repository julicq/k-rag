from __future__ import annotations
from typing import List, Dict, Tuple
import os
from collections import Counter

from app.retrieval.retrieve import Retriever
from app.generation.generate import generate_answer
from app.index.faiss_store import INDEX_FILE, META_FILE
from app.config import settings


class Pipeline:
    def __init__(self, auto_bootstrap: bool = True):
        if auto_bootstrap and not (INDEX_FILE.exists() and META_FILE.exists()):
            self.bootstrap()
        self.retriever = Retriever()

    def bootstrap(self) -> None:
        from app.crawler import crawl
        from app.preprocess import clean_and_chunk
        from app.index import build_index

        crawl.run()
        clean_and_chunk.process_raw_to_chunks()
        build_index.build_from_chunks()

    def _filter_by_majority_product(self, hits: List[Dict]) -> List[Dict]:
        if not hits:
            return hits
        products = [(h.get("product") or h.get("meta", {}).get("product")) for h in hits]
        products = [p for p in products if p]
        if not products:
            return hits
        majority, _ = Counter(products).most_common(1)[0]
        return [h for h in hits if (h.get("product") or h.get("meta", {}).get("product")) == majority]

    def ask(self, question: str) -> Tuple[str, List[Dict], List[int]]:
        hits, _ = self.retriever.ann_search(question, topk=settings.topk)
        reranked = self.retriever.rerank(question, hits, topn=settings.topn_context)
        reranked = self._filter_by_majority_product(reranked)
        contexts = reranked
        answer = generate_answer(question, contexts)
        seen = set()
        sources: List[Dict] = []
        used_ids: List[int] = []
        for h in contexts:
            url = h.get("url") or h.get("meta", {}).get("url")
            if url and url not in seen:
                seen.add(url)
                sources.append({
                    "url": url,
                    "h1": h.get("h1") or h.get("meta", {}).get("h1"),
                    "h2": h.get("h2") or h.get("meta", {}).get("h2"),
                })
            if "_id" in h:
                used_ids.append(int(h["_id"]))
        return answer, sources, used_ids


_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(auto_bootstrap=True)
    return _pipeline
