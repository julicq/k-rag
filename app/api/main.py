from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import ORJSONResponse
from typing import Dict, Any
import time

from app.config import settings

app = FastAPI(default_response_class=ORJSONResponse, title="K-RAG API")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/ask")
def ask(q: str = Query(..., min_length=1, max_length=512)) -> Dict[str, Any]:
    start = time.time()
    try:
        from app.pipeline import get_pipeline
        pipeline = get_pipeline()
        answer, sources, used_chunk_ids = pipeline.ask(q)
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "answer": answer,
            "sources": sources,
            "used_chunks": used_chunk_ids,
            "elapsed_ms": elapsed_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
