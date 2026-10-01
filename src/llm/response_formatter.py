from __future__ import annotations

import re
from dataclasses import dataclass


URL_PATTERN = re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)
SOURCE_LINE = re.compile(r"^Source URL:\s*(https?://\S+)\s*$", re.MULTILINE)
DATE_LINE = re.compile(r"^Last updated from sources:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class FormattedAnswer:
    answer: str
    source_url: str
    source_date: str


def format_answer(answer: str, source_url: str, source_date: str) -> str:
    body = answer.strip()
    if not body:
        raise ValueError("Answer text must not be empty")
    if URL_PATTERN.search(body):
        raise ValueError("Answer text must not contain citations; provenance is appended separately")
    return f"{body}\n\nSource URL: {source_url}\nLast updated from sources: {source_date}"


def parse_formatted_answer(text: str) -> FormattedAnswer | None:
    sources = SOURCE_LINE.findall(text)
    dates = DATE_LINE.findall(text)
    urls = URL_PATTERN.findall(text)
    if len(sources) != 1 or len(dates) != 1 or len(urls) != 1:
        return None
    answer = SOURCE_LINE.sub("", text)
    answer = DATE_LINE.sub("", answer).strip()
    if not answer:
        return None
    return FormattedAnswer(answer, sources[0].rstrip(".,;"), dates[0].strip())