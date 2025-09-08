from typing import List, Dict, Any
import httpx
import numpy as np

from app.config import settings


class OllamaClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.ollama_url
        self.client = httpx.Client(base_url=self.base_url, timeout=300.0)

    def _embed_once(self, text: str, model_name: str) -> List[float]:
        resp = self.client.post("/api/embeddings", json={"model": model_name, "prompt": text})
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]
        if isinstance(data, dict) and "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
            return data["embeddings"][0]
        resp2 = self.client.post("/api/embeddings", json={"model": model_name, "prompt": text})
        resp2.raise_for_status()
        data2 = resp2.json()
        if isinstance(data2, dict) and "embedding" in data2 and isinstance(data2["embedding"], list):
            return data2["embedding"]
        if isinstance(data2, dict) and "embeddings" in data2 and isinstance(data2["embeddings"], list) and data2["embeddings"]:
            return data2["embeddings"][0]
        raise RuntimeError("Ollama embeddings response has no 'embedding' field")

    def embed(self, texts: List[str], model: str | None = None) -> np.ndarray:
        model_name = model or settings.embed_model
        vectors: List[List[float]] = []
        for t in texts:
            emb = self._embed_once(t, model_name)
            vectors.append(emb)
        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] == 0:
            raise RuntimeError(f"Invalid embeddings shape: {arr.shape}")
        return arr

    def chat(self, messages: List[Dict[str, str]], model: str | None = None, temperature: float = 0.2, max_tokens: int = 192) -> str:
        model_name = model or settings.llm_model
        try:
            resp = self.client.post(
                "/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                    "stream": False,
                },
            )
            if resp.status_code == 404:
                raise httpx.HTTPStatusError("Not Found", request=resp.request, response=resp)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
                return data["message"].get("content", "")
            if isinstance(data, dict) and "response" in data:
                return data.get("response", "")
            return ""
        except httpx.HTTPStatusError:
            system_content = "\n".join(m["content"] for m in messages if m.get("role") == "system")
            user_content = "\n\n".join(m["content"] for m in messages if m.get("role") in {"user", "assistant"})
            payload: Dict[str, Any] = {
                "model": model_name,
                "prompt": user_content,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False,
            }
            if system_content:
                payload["system"] = system_content
            r2 = self.client.post("/api/generate", json=payload)
            r2.raise_for_status()
            d2 = r2.json()
            return d2.get("response", "")
