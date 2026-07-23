"""Tests for feed entry normalization."""

import unittest

from newsroom.feeds import normalize_entry


class NormalizeEntryTests(unittest.TestCase):
    def test_normalizes_complete_entry(self) -> None:
        entry = {
            "title": "  Example article  ",
            "link": "https://example.invalid/article",
            "published": "2026-01-02T03:04:05Z",
            "summary": "  A short summary.  ",
        }

        article = normalize_entry(entry, "Example Source")

        self.assertEqual(
            article,
            {
                "title": "Example article",
                "url": "https://example.invalid/article",
                "published_at": "2026-01-02T03:04:05Z",
                "source": "Example Source",
                "summary": "A short summary.",
            },
        )

    def test_handles_missing_fields(self) -> None:
        article = normalize_entry({}, "Example Source")

        self.assertEqual(article["title"], "(Untitled)")
        self.assertEqual(article["url"], "")
        self.assertIsNone(article["published_at"])
        self.assertIsNone(article["summary"])

    def test_uses_updated_when_published_is_missing(self) -> None:
        article = normalize_entry(
            {"updated": "2026-01-02T03:04:05Z"},
            "Example Source",
        )

        self.assertEqual(article["published_at"], "2026-01-02T03:04:05Z")


if __name__ == "__main__":
    unittest.main()
