from __future__ import annotations
import time
import re
import json
import httpx
from pathlib import Path
from app.utils.io import data_path


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def ask(q: str, timeout_s: float = 600.0) -> dict:
    timeout = httpx.Timeout(connect=30.0, read=timeout_s, write=timeout_s, pool=timeout_s)
    with httpx.Client(timeout=timeout) as client:
        r = client.get("http://localhost:8000/ask", params={"q": q})
        r.raise_for_status()
        return r.json()


def load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_sequential(sleep_between: float = 3.0) -> dict:
    qpath = data_path("eval", "questions.jsonl")
    rows = load_questions(qpath)
    results: list[dict] = []
    recall_hits = 0
    exact_hits = 0

    for row in rows:
        q = row["question"]
        expected = row.get("expected", "")
        url = row.get("url") or row.get("source_url")
        t0 = time.time()
        try:
            out = ask(q)
            elapsed_ms = int((time.time() - t0) * 1000)
            answer = out.get("answer", "")
            sources = out.get("sources", [])
            # Recall@3: совпадение URL в первых трех источниках
            recall3 = 0
            for s in sources[:3]:
                if s.get("url") and url and str(s["url"]).startswith(url):
                    recall3 = 1
                    break
            recall_hits += recall3
            # exact
            try:
                exact = 1 if re.search(expected, answer, flags=re.I) else 0
            except re.error:
                exact = 1 if norm_text(expected) in norm_text(answer) else 0
            exact_hits += exact
            results.append({
                "question": q,
                "expected": expected,
                "source_url": url,
                "answer": answer,
                "sources": sources,
                "recall@3": recall3,
                "exact": exact,
                "elapsed_ms": elapsed_ms,
            })
        except Exception as e:
            results.append({
                "question": q,
                "expected": expected,
                "source_url": url,
                "error": str(e),
            })
        time.sleep(sleep_between)

    n = max(1, len(rows))
    report = {
        "total": n,
        "recall@3": recall_hits / n,
        "exact": exact_hits / n,
        "details": results,
    }
    out_path = data_path("eval", "eval_report.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_sequential(), ensure_ascii=False, indent=2))
