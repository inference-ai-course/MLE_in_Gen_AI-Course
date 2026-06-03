#!/usr/bin/env python3
"""Scrape 1-2 recent NHESS paper abstracts into arxiv_clean.json.

The assignment filename says arxiv_clean.json, but this script targets NHESS:
https://nhess.copernicus.org/recent_papers.html
"""

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

RECENT_URL = "https://nhess.copernicus.org/recent_papers.html"
OUTPUT_PATH = Path(__file__).resolve().parent / "arxiv_clean.json"
MAX_PAPERS = 2
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; course-abstract-scraper/1.0)"}


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.removeprefix("Abstract ").strip()


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    return BeautifulSoup(response.text, "html.parser")


def first_meta(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("meta", attrs={"name": name})
    return clean_text(tag["content"]) if tag and tag.get("content") else ""


def all_meta(soup: BeautifulSoup, name: str) -> list[str]:
    return [clean_text(tag["content"]) for tag in soup.find_all("meta", attrs={"name": name}) if tag.get("content")]


def recent_final_article_urls(limit: int = MAX_PAPERS) -> list[str]:
    soup = fetch_soup(RECENT_URL)
    urls: list[str] = []
    for item in soup.select(".paperList-final"):
        title_link = item.find("a", href=re.compile(r"/articles/\d+/\d+/\d+/$"))
        if not title_link:
            continue
        url = urljoin(RECENT_URL, title_link["href"])
        if url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def parse_article(url: str) -> dict[str, Any]:
    soup = fetch_soup(url)
    abstract_node = soup.select_one("#abstract, div.abstract, .abstract")
    abstract = clean_text(abstract_node.get_text(" ", strip=True)) if abstract_node else ""

    date = first_meta(soup, "citation_publication_date").replace("/", "-")
    return {
        "url": url,
        "title": first_meta(soup, "citation_title"),
        "abstract": abstract,
        "authors": all_meta(soup, "citation_author"),
        "date": date,
    }


def scrape_nhess_abstracts(limit: int = MAX_PAPERS) -> list[dict[str, Any]]:
    return [parse_article(url) for url in recent_final_article_urls(limit)]


def main() -> None:
    records = scrape_nhess_abstracts(MAX_PAPERS)
    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    size = OUTPUT_PATH.stat().st_size
    if size > 1_000_000:
        raise RuntimeError(f"{OUTPUT_PATH} is larger than 1 MB: {size} bytes")
    print(f"Wrote {len(records)} records to {OUTPUT_PATH} ({size} bytes)")


if __name__ == "__main__":
    main()
