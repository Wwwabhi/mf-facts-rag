from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceRecord:
    id: str
    product: str
    amc: str
    scheme: str
    source_type: str
    title: str
    source_url: str
    source_date: str
    verified_on: str
    use_in_rag: str

    @property
    def publisher(self) -> str:
        return self.amc

    @property
    def document_type(self) -> str:
        return self.source_type

    @property
    def date(self) -> str:
        if self.verified_on and self.verified_on != "Live/current page":
            return self.verified_on
        if self.source_date and self.source_date != "Live/current page":
            return self.source_date
        return "unknown"

    @property
    def url(self) -> str:
        return self.source_url


def load_registry(csv_path: str | Path) -> list[SourceRecord]:
    csv_path = Path(csv_path)
    records: list[SourceRecord] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                SourceRecord(
                    id=row.get("id", "").strip(),
                    product=row.get("product", "").strip(),
                    amc=row.get("amc", "").strip(),
                    scheme=row.get("scheme", "").strip(),
                    source_type=row.get("source_type", "").strip(),
                    title=row.get("title", "").strip(),
                    source_url=row.get("url", "").strip(),
                    source_date=row.get("source_date", "").strip(),
                    verified_on=row.get("verified_on", "").strip(),
                    use_in_rag=row.get("use_in_rag", "").strip(),
                )
            )
    return records
