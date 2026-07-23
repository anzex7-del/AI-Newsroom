"""Tests for deterministic filtering and ranking."""

import unittest
from datetime import datetime, timezone

from newsroom.ranking import (
    deduplicate_articles,
    is_within_recency_window,
    match_topics,
    rank_articles,
    score_article,
)

NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)
CONFIG = {
    "recency_window_days": 30,
    "minimum_relevance_score": 4,
    "scoring": {
        "title_topic_match": 5,
        "summary_topic_match": 2,
        "maximum_recency_bonus": 3,
    },
    "source_priorities": {"Priority Source": 2},
    "topics": {
        "Microsoft Foundry": ["microsoft foundry"],
        "AI security": ["ai security"],
    },
}


def article(**overrides):
    item = {
        "title": "General technology update",
        "url": "https://example.invalid/article",
        "published_at": "2026-07-22T00:00:00Z",
        "source": "Regular Source",
        "summary": None,
    }
    item.update(overrides)
    return item


class KeywordMatchingTests(unittest.TestCase):
    def test_matches_topics_by_field_case_insensitively(self) -> None:
        item = article(
            title="Microsoft Foundry update",
            summary="New guidance for AI Security teams.",
        )

        title_matches, summary_matches = match_topics(item, CONFIG["topics"])

        self.assertEqual(title_matches, {"Microsoft Foundry"})
        self.assertEqual(summary_matches, {"AI security"})


class ScoringTests(unittest.TestCase):
    def test_score_combines_matches_recency_and_source_priority(self) -> None:
        item = article(
            title="Microsoft Foundry update",
            summary="Microsoft Foundry improves AI security.",
            source="Priority Source",
        )

        score, topics = score_article(item, CONFIG, NOW)

        self.assertEqual(score, 14)
        self.assertEqual(topics, ["AI security", "Microsoft Foundry"])

    def test_ranked_candidates_are_highest_score_first(self) -> None:
        lower = article(
            title="AI security guidance",
            url="https://example.invalid/lower",
        )
        higher = article(
            title="Microsoft Foundry and AI security",
            url="https://example.invalid/higher",
        )

        candidates, _ = rank_articles([lower, higher], CONFIG, NOW)

        self.assertEqual(candidates[0]["title"], higher["title"])


class DuplicateDetectionTests(unittest.TestCase):
    def test_removes_tracking_url_and_normalized_title_duplicates(self) -> None:
        original = article(url="https://example.invalid/story?utm_source=email")
        same_url = article(
            title="A different title",
            url="https://EXAMPLE.invalid/story#section",
        )
        same_title = article(
            title=" General—technology update! ",
            url="https://example.invalid/other",
        )

        unique, removed = deduplicate_articles([original, same_url, same_title])

        self.assertEqual(unique, [original])
        self.assertEqual(removed, 2)


class RecencyFilteringTests(unittest.TestCase):
    def test_accepts_recent_article(self) -> None:
        self.assertTrue(is_within_recency_window(article(), 30, NOW))

    def test_rejects_old_or_undated_articles(self) -> None:
        old = article(published_at="2026-06-01T00:00:00Z")
        undated = article(published_at=None)

        self.assertFalse(is_within_recency_window(old, 30, NOW))
        self.assertFalse(is_within_recency_window(undated, 30, NOW))


if __name__ == "__main__":
    unittest.main()
