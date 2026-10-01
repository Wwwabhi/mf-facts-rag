from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.data.source_registry import SourceRecord


def slugify_for_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned.lower() if cleaned else "untitled"


def normalize_url(url: str) -> str:
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.scheme == "http":
        return f"https://{parsed.netloc}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")
    return url


def fetch_html(record: SourceRecord, raw_dir: Path) -> tuple[bool, Path | None, str | None]:
    url = normalize_url(record.source_url)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{record.id}_{slugify_for_filename(record.scheme)}_{slugify_for_filename(record.document_type)}.html"
    raw_path = raw_dir / file_name

    last_error = None
    for fetcher in ("requests", "playwright"):
        try:
            if fetcher == "requests":
                response = requests.get(
                    url,
                    timeout=60,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Upgrade-Insecure-Requests": "1",
                        "sec-fetch-site": "none",
                        "sec-fetch-mode": "navigate",
                        "sec-fetch-user": "?1",
                    },
                    allow_redirects=True,
                )
                response.raise_for_status()
                html = response.text
            else:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    launch_kwargs: dict[str, Any] = {"headless": True}
                    try:
                        browser = p.chromium.launch(channel="chrome", **launch_kwargs)
                    except Exception:
                        browser = p.chromium.launch(**launch_kwargs)
                    page = browser.new_page(viewport={"width": 1440, "height": 1800}, user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
                    page.goto(url, wait_until="networkidle", timeout=120000)
                    html = page.content()
                    browser.close()

            raw_path.write_bytes(html.encode("utf-8", errors="replace"))
            return True, raw_path, html
        except Exception as exc:  # pragma: no cover - operational guard
            last_error = exc

    return False, raw_path, str(last_error)


def _clean_html_fragment(container: Any) -> str:
    if container is None:
        return ""

    for selector in [
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "button",
        ".nav",
        ".header",
        ".footer",
        ".breadcrumb",
        ".breadcrumbs",
        ".cookie",
        ".modal",
        ".social",
        "[role='navigation']",
        "[role='banner']",
        "[role='contentinfo']",
    ]:
        for tag in container.select(selector):
            tag.decompose()

    pieces: list[str] = []
    for node in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "th", "td", "table"]):
        text = " ".join(node.get_text(" ", strip=True).split())
        if not text:
            continue
        if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            pieces.append(f"{node.name.upper()}: {text}")
        elif node.name == "table":
            rows = []
            for tr in node.find_all("tr"):
                cells = [" ".join(td.get_text(" ", strip=True).split()) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                pieces.append("Table: " + "\n".join(rows))
        else:
            pieces.append(text)

    return "\n".join(pieces)


def extract_text_from_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    target = None
    for selector in ["main", "article", "[role='main']", ".main-content", ".content-area", ".cms-content", "#content", ".content"]:
        target = soup.select_one(selector)
        if target is not None:
            break

    if target is None:
        target = soup.body or soup

    content = _clean_html_fragment(target)
    if not content.strip():
        for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
            tag.decompose()
        content = "\n".join(" ".join(line.split()) for line in soup.get_text("\n").splitlines() if " ".join(line.split()))

    lines = [line for line in content.splitlines() if line.strip()]
    if title:
        lines.insert(0, f"Title: {title}")
    return "\n".join(lines)


def process_html_source(record: SourceRecord, raw_dir: Path, processed_dir: Path) -> bool:
    success, raw_path, html_text = fetch_html(record, raw_dir)
    if not success or html_text is None:
        return False

    text = extract_text_from_html(html_text)
    if not text.strip():
        return False

    payload: dict[str, Any] = {
        "source_url": record.source_url,
        "scheme": record.scheme,
        "publisher": record.publisher,
        "document_type": record.document_type,
        "title": record.title,
        "date": record.date,
        "source_date": record.source_date,
        "verified_on": record.verified_on,
        "raw_file": str(raw_path),
        "content": text,
        "status": "success",
    }

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_file = processed_dir / f"{record.id}_{slugify_for_filename(record.scheme)}_{slugify_for_filename(record.document_type)}.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True
