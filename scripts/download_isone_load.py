#!/usr/bin/env python3
"""Download official ISO-NE annual hourly system-load files.

The ISO Express page loads its document list from a public JSON endpoint. This
script queries that endpoint and downloads the available EEI files for the
requested year range without requiring an ISO-NE account.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.parse
from pathlib import Path


BASE_URL = "https://www.iso-ne.com"
LIST_URL = f"{BASE_URL}/isoexpress/web/reports/download/docWidgetGetMore"


def get_documents() -> list[dict]:
    documents: list[dict] = []
    start = 0
    while True:
        query = urllib.parse.urlencode(
            {"start": start, "treenode": "sys-load-eei-fmt"}
        )
        response = subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error", f"{LIST_URL}?{query}"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(response.stdout)
        batch = payload["data"]
        documents.extend(batch)
        if payload["count"] < 40:
            break
        start += 40
    return documents


def document_year(document: dict) -> int | None:
    words = document.get("descriptionFormatted", "").split()
    try:
        return int(words[-1])
    except (ValueError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/isone_load"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for document in get_documents():
        year = document_year(document)
        if year is None or not args.start_year <= year <= args.end_year:
            continue
        output_path = args.output_dir / f"{year}_eei_loads.txt"
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--output",
                str(output_path),
                f"{BASE_URL}{document['path']}",
            ],
            check=True,
        )
        selected.append(
            {
                "year": year,
                "source_url": f"{BASE_URL}{document['path']}",
                "iso_ne_publish_date": document.get("publishDate"),
                "local_file": str(output_path),
            }
        )
        print(f"Downloaded {year}: {output_path}")

    selected.sort(key=lambda row: row["year"])
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
