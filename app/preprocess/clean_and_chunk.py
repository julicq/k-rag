from __future__ import annotations
from typing import List, Dict, Iterable, DefaultDict
from collections import defaultdict
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString
import re
import hashlib

from app.utils.io import data_path, write_jsonl, read_jsonl
from app.config import settings


REMOVE_SELECTORS = [
    "nav", "header", "footer", "aside", "script", "style", "noscript", "form",
]


def is_allowed_url(url: str | None, product: str, version: str) -> bool:
    if not url:
        return False
    pattern = f"^https://support\\.kaspersky\\.com/(?:help/)?{re.escape(product)}/{re.escape(version)}/ru-RU/"
    m = re.match(pattern, url)
    return m is not None


def clean_html(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for sel in REMOVE_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    main = soup.select_one("main") or soup.select_one("article") or soup.select_one("div#content") or soup
    return main


def tables_to_plaintext(soup: BeautifulSoup) -> None:
    tables = list(soup.find_all("table"))
    for tbl in tables:
        lines: List[str] = []
        for tr in tbl.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            parts = [c.get_text(" ", strip=True) for c in cells]
            line = " | ".join(filter(None, parts)).strip()
            if line:
                lines.append(line)
        replacement = "\n".join(lines)
        tbl.replace_with(NavigableString(replacement if replacement else ""))


def extract_text(soup: BeautifulSoup) -> str:
    tables_to_plaintext(soup)
    text = soup.get_text("\n")
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_headers(soup: BeautifulSoup) -> Dict[str, str | None]:
    h1 = soup.select_one("h1")
    h2 = soup.select_one("h2")
    return {"h1": h1.get_text(strip=True) if h1 else None, "h2": h2.get_text(strip=True) if h2 else None}


def extract_canonical_url(full_doc_html: str) -> str | None:
    try:
        doc = BeautifulSoup(full_doc_html, "lxml")
        link = doc.select_one("link[rel=canonical]")
        if link and link.get("href"):
            return link.get("href")
    except Exception:
        return None
    return None


def chunk_text(text: str, words_per_chunk: int, overlap_words: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    i = 0
    n = len(words)
    while i < n:
        j = min(i + words_per_chunk, n)
        chunk = " ".join(words[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap_words)
    return chunks


def process_raw_to_chunks() -> List[Dict]:
    manifest_path = data_path("raw", "manifest.jsonl")
    entries: List[Dict[str, object]]
    if manifest_path.exists():
        entries = read_jsonl(manifest_path)
    else:
        entries = []
        raw_root = data_path("raw")
        for product_dir in raw_root.glob("*_*"):
            product, version = product_dir.name.split("_", 1)
            for html_file in product_dir.glob("*.html"):
                entries.append({"product": product, "version": version, "url": None, "path": str(html_file)})

    acc: DefaultDict[str, List[Dict]] = defaultdict(list)
    out_all: List[Dict] = []

    words_per_chunk = max(50, int(0.75 * settings.chunk_size))
    overlap_words = max(10, int(0.75 * settings.chunk_overlap))

    for e in entries:
        product = str(e["product"])
        version = str(e["version"])
        url_from_manifest = e.get("url") if isinstance(e, dict) else None
        html_file = Path(str(e["path"]))
        if not html_file.exists():
            continue
        full_html = html_file.read_text(encoding="utf-8", errors="ignore")
        canonical = extract_canonical_url(full_html)
        soup = clean_html(full_html)
        headers = extract_headers(soup)
        text = extract_text(soup)
        if not text:
            continue
        final_url = canonical or (url_from_manifest if isinstance(url_from_manifest, str) else None)
        if not is_allowed_url(final_url, product, version):
            # пропускаем страницы вне нужной ветки
            continue
        chunks = chunk_text(text, words_per_chunk, overlap_words)
        for idx, ch in enumerate(chunks):
            ch_norm = ch.strip()
            if len(ch_norm) < 80:
                continue
            sha = hashlib.sha256(ch_norm.encode("utf-8")).hexdigest()
            row = {
                "id": f"{product}_{version}:{html_file.stem}:{idx}",
                "text": ch_norm,
                "meta": {
                    "product": product,
                    "version": version,
                    "url": final_url,
                    "h1": headers.get("h1"),
                    "h2": headers.get("h2"),
                    "sha256": sha,
                },
            }
            pv = f"{product}_{version}"
            acc[pv].append(row)
            out_all.append(row)

    for pv, rows in acc.items():
        out_path = data_path("chunks", f"{pv}.jsonl")
        write_jsonl(out_path, rows)

    return out_all


if __name__ == "__main__":
    process_raw_to_chunks()
