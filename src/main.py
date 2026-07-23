"""Command-line entry point for News Scout."""

from __future__ import annotations

import json
from pathlib import Path

from newsroom.config import load_editorial_config, load_sources
from newsroom.feeds import FeedFetchError, fetch_feed
from newsroom.ranking import RankingStats, rank_articles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_PATH = PROJECT_ROOT / "config" / "sources.yaml"
EDITORIAL_PATH = PROJECT_ROOT / "config" / "editorial.yaml"
NEWS_OUTPUT_PATH = PROJECT_ROOT / "data" / "news.json"
CANDIDATES_OUTPUT_PATH = PROJECT_ROOT / "data" / "candidates.json"


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


def save_json(path: Path, items: list[object]) -> None:
    """Save a list as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(items, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the News Scout ingestion pipeline."""
    articles = collect_articles()
    save_json(NEWS_OUTPUT_PATH, articles)

    try:
        editorial_config = load_editorial_config(EDITORIAL_PATH)
        candidates, stats = rank_articles(articles, editorial_config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Could not rank articles: {error}")
        candidates = []
        stats = RankingStats()

    save_json(CANDIDATES_OUTPUT_PATH, candidates)

    print("\nNews Scout v0.2 summary")
    print(f"Total articles fetched: {len(articles)}")
    print(f"Duplicates removed: {stats.duplicates_removed}")
    print(f"Rejected by recency: {stats.rejected_by_recency}")
    print(f"Rejected by relevance: {stats.rejected_by_relevance}")
    print(f"Final candidate count: {len(candidates)}")

    if candidates:
        print("\nTop candidates:")
        for candidate in candidates[:5]:
            print(f"- [{candidate['relevance_score']}] {candidate['title']}")
            print(f"  Topics: {', '.join(candidate['matched_topics'])}")
            print(f"  {candidate['url']}")

    print(f"\nSaved all articles to {NEWS_OUTPUT_PATH}")
    print(f"Saved candidates to {CANDIDATES_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
