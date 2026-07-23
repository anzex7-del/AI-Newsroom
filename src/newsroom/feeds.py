"""Fetch and normalize RSS or Atom feeds."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import feedparser

Article = dict[str, str | None]


class FeedFetchError(Exception):
    """Raised when a feed cannot be fetched or parsed."""


def _optional_text(value: Any) -> str | None:
    """Return stripped text, or None when a value is absent."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def normalize_entry(entry: Mapping[str, Any], source_name: str) -> Article:
    """Normalize one feed entry into the common article schema."""
    title = _optional_text(entry.get("title")) or "(Untitled)"
    url = _optional_text(entry.get("link")) or ""
    published_at = _optional_text(entry.get("published"))
    if published_at is None:
        published_at = _optional_text(entry.get("updated"))

    return {
        "title": title,
        "url": url,
        "published_at": published_at,
        "source": source_name,
        "summary": _optional_text(entry.get("summary")),
    }


def fetch_feed(name: str, url: str) -> list[Article]:
    """Fetch one RSS/Atom feed and return normalized entries."""
    try:
        parsed = feedparser.parse(url)
    except Exception as error:
        raise FeedFetchError(str(error)) from error

    status = getattr(parsed, "status", None)
    if isinstance(status, int) and status >= 400:
        raise FeedFetchError(f"server returned HTTP {status}")

    if getattr(parsed, "bozo", False):
        error = getattr(parsed, "bozo_exception", "malformed feed")
        raise FeedFetchError(f"malformed feed: {error}")

    entries = getattr(parsed, "entries", None)
    if not isinstance(entries, list):
        raise FeedFetchError("feed contains no readable entries")

    return [normalize_entry(entry, name) for entry in entries]
