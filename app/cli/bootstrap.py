from __future__ import annotations
import sys

from app.crawler import crawl
from app.preprocess import clean_and_chunk
from app.index import build_index
from app.embed.ollama_client import OllamaClient


def main() -> int:
    crawl.run()
    clean_and_chunk.process_raw_to_chunks()
    build_index.build_from_chunks()
    try:
        client = OllamaClient()
        client.embed(["warmup"])
        client.chat([
            {"role": "system", "content": "Отвечай кратко."},
            {"role": "user", "content": "Тест"},
        ], max_tokens=8)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
