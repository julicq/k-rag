from __future__ import annotations
from typing import Set, List, Dict
from urllib.parse import urljoin, urlparse, urlencode
from pathlib import Path
import re
import httpx
from bs4 import BeautifulSoup

from app.utils.io import ensure_dir, data_path, slugify, write_jsonl
from app.config import settings


SEEDS = [
    {"product": "KSC", "version": "15.1", "base": "https://support.kaspersky.com/KSC/15.1/ru-RU/"},
    {"product": "KATA", "version": "7.1", "base": "https://support.kaspersky.com/KATA/7.1/ru-RU/"},
]

SEARCH_QUERIES = [
    "установка", "устройства", "политика", "обновление", "агент", "инциденты", "песочница", "интеграция", "SIEM"
]


def path_matches_product(path: str, product: str, version: str) -> bool:
    pattern = rf"/(help/)?{re.escape(product)}/{re.escape(version)}/ru-RU/"
    return re.search(pattern, path) is not None


def is_allowed(href: str, base: str, product: str, version: str) -> bool:
    if not href:
        return False
    if href.startswith("#"):
        return False
    if href.startswith("mailto:") or href.startswith("javascript:"):
        return False
    try:
        base_parsed = urlparse(base)
        url_parsed = urlparse(urljoin(base, href))
        if base_parsed.netloc != url_parsed.netloc:
            return False
        if not path_matches_product(url_parsed.path, product, version):
            return False
        return True
    except Exception:
        return False


def extract_links_from_doc(html: str, base: str, product: str, version: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not is_allowed(href, base, product, version):
            continue
        url = urljoin(base, href)
        links.append(url)
    seen: Set[str] = set()
    uniq: List[str] = []
    for u in links:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def discover_via_search(product: str, version: str, limit: int = 50) -> List[str]:
    base_host = "https://support.kaspersky.com"
    found: List[str] = []
    seen: Set[str] = set()
    with httpx.Client(headers={"User-Agent": "K-RAG/0.1"}, timeout=30.0) as client:
        for q in SEARCH_QUERIES:
            params = {"q": f"{product} {version} {q}"}
            url = f"{base_host}/search?{urlencode(params)}"
            try:
                r = client.get(url)
                if r.status_code != 200:
                    continue
                soup = BeautifulSoup(r.text, "lxml")
                for a in soup.select("a[href]"):
                    href = a.get("href")
                    if not href:
                        continue
                    if not is_allowed(href, f"{base_host}/", product, version):
                        continue
                    full = urljoin(base_host + "/", href)
                    if full not in seen:
                        seen.add(full)
                        found.append(full)
                        if len(found) >= limit:
                            return found
            except Exception:
                continue
    return found


def discover_from_sitemap(product: str, version: str) -> List[str]:
    base_host = "https://support.kaspersky.com"
    urls: List[str] = []
    try:
        with httpx.Client(headers={"User-Agent": "K-RAG/0.1"}, timeout=30.0) as client:
            r = client.get(f"{base_host}/sitemap.xml")
            if r.status_code != 200:
                return []
            text = r.text
            pattern = re.compile(rf"https://support\\.kaspersky\\.com/(?:help/)?{re.escape(product)}/{re.escape(version)}/ru-RU/[^<\s]+")
            urls = sorted(set(pattern.findall(text)))
    except Exception:
        return []
    return urls[:100]


def fetch(url: str, client: httpx.Client) -> str | None:
    try:
        resp = client.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def crawl_from_seed(base: str, product: str, version: str, limit: int = 50) -> List[str]:
    urls: List[str] = []
    with httpx.Client(headers={"User-Agent": "K-RAG/0.1"}, timeout=settings.request_timeout) as client:
        html = fetch(base, client)
        if html:
            urls.append(base)
            inner = extract_links_from_doc(html, base, product, version)
            for u in inner:
                if len(urls) >= limit:
                    break
                urls.append(u)
    return urls


def crawl_depth1() -> List[Dict]:
    results: List[Dict] = []
    with httpx.Client(headers={"User-Agent": "K-RAG/0.1"}) as client:
        # Используем EXTRA_SEEDS как основные точки краулинга
        if settings.extra_seeds:
            for token in settings.extra_seeds.split(','):
                token = token.strip()
                if not token:
                    continue
                try:
                    product, version, url = token.split('|', 2)
                except ValueError:
                    continue
                
                collected: List[str] = []
                collected.extend(crawl_from_seed(url, product, version, limit=settings.crawl_seed_limit))
                
                if len(collected) < settings.crawl_seed_limit // 2:
                    collected.extend(discover_via_search(product, version, limit=settings.crawl_search_limit))
                if len(collected) < settings.crawl_seed_limit // 2:
                    collected.extend(discover_from_sitemap(product, version)[: settings.crawl_sitemap_limit])
                
                seen: Set[str] = set()
                urls: List[str] = []
                for u in collected:
                    if u not in seen:
                        seen.add(u)
                        urls.append(u)
                    if len(urls) >= settings.crawl_max_urls:
                        break
                
                for page_url in urls:
                    html = fetch(page_url, client)
                    if not html:
                        continue
                    path = save_raw(product, version, page_url, html)
                    results.append({"product": product, "version": version, "url": page_url, "path": str(path)})
        else:
            for seed in SEEDS:
                product = seed["product"]
                version = seed["version"]
                base = seed["base"]
                collected: List[str] = []
                collected.extend(crawl_from_seed(base, product, version, limit=20))
                if len(collected) < 20:
                    collected.extend(discover_via_search(product, version, limit=50))
                if len(collected) < 20:
                    collected.extend(discover_from_sitemap(product, version))
                seen: Set[str] = set()
                urls: List[str] = []
                for u in collected:
                    if u not in seen:
                        seen.add(u)
                        urls.append(u)
                    if len(urls) >= 60:
                        break
                for url in urls:
                    html = fetch(url, client)
                    if not html:
                        continue
                    path = save_raw(product, version, url, html)
                    results.append({"product": product, "version": version, "url": url, "path": str(path)})
    return results


def save_raw(product: str, version: str, url: str, html: str) -> Path:
    out_dir = data_path("raw", f"{product}_{version}")
    ensure_dir(out_dir)
    name = slugify(url)[:150] or "index"
    out_path = out_dir / f"{name}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def run() -> List[Dict]:
    results = crawl_depth1()
    if results:
        manifest_path = data_path("raw", "manifest.jsonl")
        write_jsonl(manifest_path, results)
    return results


if __name__ == "__main__":
    run()
