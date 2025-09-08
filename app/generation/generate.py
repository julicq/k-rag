from typing import List, Dict
import re

from app.config import settings
from app.embed.ollama_client import OllamaClient

SYSTEM_PROMPT = (
    "Отвечай только на основании контекста; если нет ответа — скажи «не знаю». "
    "В конце ответа приводи ссылки с указанием источника (URL)."
)


def build_user_prompt(question: str, contexts: List[Dict]) -> str:
    ctx_lines: List[str] = []
    for i, c in enumerate(contexts, 1):
        header = f"[{i}] {c.get('h1') or ''} / {c.get('h2') or ''}"
        url = c.get("url", "")
        text = c.get("text", "")
        ctx_lines.append(f"Источник: {url}\n{header}\n{text}")
    ctx_block = "\n\n".join(ctx_lines)
    return f"Вопрос: {question}\n\nКонтекст:\n{ctx_block}\n\nОтвет:"


_cjk_re = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def sanitize_answer(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _cjk_re.search(line):
            continue
        lines.append(line)
    out = "\n".join(lines)
    out = re.sub(r"(\n\s*){3,}", "\n\n", out).strip()
    return out


def generate_answer(question: str, contexts: List[Dict]) -> str:
    client = OllamaClient()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, contexts)},
    ]
    raw = client.chat(messages, model=settings.llm_model)
    return sanitize_answer(raw)
