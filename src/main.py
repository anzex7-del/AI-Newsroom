"""Command-line entry point for News Scout."""

from __future__ import annotations

import json
from pathlib import Path

from newsroom.config import load_sources
from newsroom.feeds import FeedFetchError, fetch_feed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_PATH = PROJECT_ROOT / "config" / "sources.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "news.json"


def collect_articles() -> list[dict[str, str | None]]:
    """Fetch articles from each valid, enabled RSS source."""
    articles: list[dict[str, str | None]] = []

    try:
        sources = load_sources(SOURCES_PATH)
    except (OSError, ValueError) as error:
        print(f"Could not load source configuration: {error}")
        return articles

    for source in sources:
        name = source.get("name")
        url = source.get("url")
        source_type = source.get("type")

        if source.get("enabled", True) is False:
            print(f"Skipping disabled source: {name or 'unnamed source'}")
            continue
        if not isinstance(name, str) or not name.strip():
            print("Skipping a source with no valid name.")
            continue
        if source_type != "rss":
            print(f"Skipping {name}: unsupported source type {source_type!r}.")
            continue
        if not isinstance(url, str) or not url.strip():
            print(f"Skipping {name}: add a real RSS URL in config/sources.yaml.")
            continue

        try:
            source_articles = fetch_feed(name=name, url=url)
        except FeedFetchError as error:
            print(f"Could not fetch {name}: {error}")
            continue

        articles.extend(source_articles)
        print(f"Fetched {len(source_articles)} article(s) from {name}.")

    return articles


def save_articles(articles: list[dict[str, str | None]]) -> None:
    """Save normalized articles as formatted JSON."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(articles, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the News Scout ingestion pipeline."""
    articles = collect_articles()
    save_articles(articles)

    print(f"\nNews Scout found {len(articles)} article(s).")
    for article in articles:
        print(f"- {article['title']} ({article['source']})")
        print(f"  {article['url']}")
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
