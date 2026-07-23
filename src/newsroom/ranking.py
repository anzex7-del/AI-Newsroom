"""Deterministically filter, deduplicate, and rank news articles."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from newsroom.feeds import Article

Candidate = dict[str, str | int | list[str] | None]
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_NAMES = {"fbclid", "gclid"}


@dataclass
class RankingStats:
    """Counts produced by one ranking run."""

    duplicates_removed: int = 0
    rejected_by_recency: int = 0
    rejected_by_relevance: int = 0


def normalize_url(url: str) -> str:
    """Canonicalize a URL for deterministic duplicate checks."""
    if not url:
        return ""

    parts = urlsplit(url.strip())
    query_items = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if name.casefold() not in TRACKING_QUERY_NAMES
        and not name.casefold().startswith(TRACKING_QUERY_PREFIXES)
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            path,
            urlencode(sorted(query_items)),
            "",
        )
    )


def normalize_title(title: str) -> str:
    """Canonicalize title text for deterministic duplicate checks."""
    words = re.findall(r"\w+", title.casefold(), flags=re.UNICODE)
    return " ".join(words)


def deduplicate_articles(articles: list[Article]) -> tuple[list[Article], int]:
    """Remove articles with an already-seen normalized URL or title."""
    unique: list[Article] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    duplicates = 0

    for article in articles:
        url = normalize_url(str(article.get("url") or ""))
        title = normalize_title(str(article.get("title") or ""))
        is_duplicate = (bool(url) and url in seen_urls) or (
            bool(title) and title in seen_titles
        )
        if is_duplicate:
            duplicates += 1
            continue

        unique.append(article)
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)

    return unique, duplicates


def parse_published_at(value: str | None) -> datetime | None:
    """Parse common RSS or ISO date strings as timezone-aware datetimes."""
    if not value:
        return None

    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def article_age_days(article: Article, now: datetime) -> float | None:
    """Return article age in days, clamping future timestamps to zero."""
    published = parse_published_at(article.get("published_at"))
    if published is None:
        return None
    return max(0.0, (now - published).total_seconds() / 86_400)


def is_within_recency_window(
    article: Article,
    recency_window_days: int,
    now: datetime,
) -> bool:
    """Return whether an article has a date inside the configured window."""
    age = article_age_days(article, now)
    return age is not None and age <= recency_window_days


def match_topics(
    article: Article,
    topics: dict[str, list[str]],
) -> tuple[set[str], set[str]]:
    """Return topic names matched in the title and summary."""
    title = str(article.get("title") or "")
    summary = str(article.get("summary") or "")
    title_matches: set[str] = set()
    summary_matches: set[str] = set()

    for topic, keywords in topics.items():
        for keyword in keywords:
            pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
            if re.search(pattern, title, flags=re.IGNORECASE):
                title_matches.add(topic)
            if re.search(pattern, summary, flags=re.IGNORECASE):
                summary_matches.add(topic)

    return title_matches, summary_matches


def score_article(
    article: Article,
    config: dict[str, Any],
    now: datetime,
) -> tuple[int, list[str]]:
    """Calculate a relevance score and matched editorial topics."""
    scoring = config["scoring"]
    topics = config["topics"]
    title_matches, summary_matches = match_topics(article, topics)
    matched_topics = sorted(title_matches | summary_matches)

    score = len(title_matches) * int(scoring["title_topic_match"])
    score += len(summary_matches) * int(scoring["summary_topic_match"])

    age = article_age_days(article, now)
    window = int(config["recency_window_days"])
    maximum_bonus = int(scoring["maximum_recency_bonus"])
    if age is not None and window > 0:
        remaining_fraction = max(0.0, 1 - (age / window))
        score += math.ceil(maximum_bonus * remaining_fraction)

    priorities = config.get("source_priorities", {})
    score += int(priorities.get(article.get("source"), 0))
    return score, matched_topics


def rank_articles(
    articles: list[Article],
    config: dict[str, Any],
    now: datetime | None = None,
) -> tuple[list[Candidate], RankingStats]:
    """Deduplicate, filter, score, and sort normalized articles."""
    current_time = now or datetime.now(timezone.utc)
    unique, duplicate_count = deduplicate_articles(articles)
    stats = RankingStats(duplicates_removed=duplicate_count)
    candidates: list[Candidate] = []
    window = int(config["recency_window_days"])
    minimum_score = int(config["minimum_relevance_score"])

    for article in unique:
        if not is_within_recency_window(article, window, current_time):
            stats.rejected_by_recency += 1
            continue

        score, matched_topics = score_article(article, config, current_time)
        if not matched_topics or score < minimum_score:
            stats.rejected_by_relevance += 1
            continue

        candidate: Candidate = dict(article)
        candidate["relevance_score"] = score
        candidate["matched_topics"] = matched_topics
        candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            -int(candidate["relevance_score"]),  # type: ignore[arg-type]
            str(candidate["title"]).casefold(),
        )
    )
    return candidates, stats
