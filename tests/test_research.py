"""Tests for deterministic research dossier infrastructure."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from newsroom.research import (
    CandidateValidationError,
    build_user_input,
    clean_article_content,
    create_dossier,
    create_dossiers,
    create_story_id,
    prepare_article_content,
    retrieve_page_metadata,
)

CANDIDATE = {
    "title": "Microsoft Foundry update",
    "url": "https://example.invalid/story",
    "published_at": "2026-07-22T00:00:00Z",
    "source": "Example Publisher",
    "summary": None,
    "relevance_score": 10,
    "matched_topics": ["Microsoft Foundry"],
}


def successful_retrieval(url: str):
    return {
        "status": "success",
        "final_url": url,
        "http_status": 200,
        "retrieved_at": "2026-07-23T00:00:00Z",
        "content_type": "text/html",
        "content": "Example article content.",
        "original_content_chars": 24,
        "sent_to_model_chars": 0,
        "content_truncated": False,
        "error": None,
    }


class StoryIdTests(unittest.TestCase):
    def test_story_id_is_deterministic_and_ignores_tracking_parameters(self) -> None:
        variant = dict(
            CANDIDATE,
            url="https://EXAMPLE.invalid/story?utm_source=newsletter#top",
            title="Microsoft Foundry update!",
        )

        self.assertEqual(create_story_id(CANDIDATE), create_story_id(variant))


class DossierCreationTests(unittest.TestCase):
    def test_creates_empty_pending_dossier(self) -> None:
        dossier = create_dossier(CANDIDATE, retriever=successful_retrieval)

        self.assertEqual(dossier["story_id"], create_story_id(CANDIDATE))
        self.assertEqual(dossier["research_status"], "pending")
        self.assertEqual(dossier["claims"], [])
        self.assertEqual(dossier["key_takeaways"], [])
        self.assertEqual(dossier["open_questions"], [])
        self.assertEqual(dossier["primary_source"]["publisher"], "Example Publisher")
        self.assertEqual(dossier["retrieval"]["http_status"], 200)

    def test_batch_continues_after_malformed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths, malformed, retrieval_failures = create_dossiers(
                [{"title": "Missing fields"}, CANDIDATE],
                Path(temporary_directory),
                retriever=successful_retrieval,
            )

        self.assertEqual(len(paths), 1)
        self.assertEqual(malformed, 1)
        self.assertEqual(retrieval_failures, 0)


class MalformedCandidateTests(unittest.TestCase):
    def test_rejects_candidate_without_url(self) -> None:
        with self.assertRaises(CandidateValidationError):
            create_dossier(
                {"title": "Incomplete", "source": "Publisher"},
                retriever=successful_retrieval,
            )


class RetrievalFailureTests(unittest.TestCase):
    def test_network_failure_becomes_metadata_instead_of_exception(self) -> None:
        now = datetime(2026, 7, 23, tzinfo=timezone.utc)
        with patch(
            "newsroom.research.urlopen",
            side_effect=URLError("network unavailable"),
        ):
            metadata = retrieve_page_metadata(
                "https://example.invalid/story",
                now=now,
            )

        self.assertEqual(metadata["status"], "failed")
        self.assertIsNone(metadata["http_status"])
        self.assertIn("network unavailable", metadata["error"])

    def test_failed_retrieval_still_creates_dossier(self) -> None:
        def failed_retrieval(url: str):
            return {
                "status": "failed",
                "final_url": None,
                "http_status": None,
                "retrieved_at": "2026-07-23T00:00:00Z",
                "content_type": None,
                "content": None,
                "original_content_chars": 0,
                "sent_to_model_chars": 0,
                "content_truncated": False,
                "error": "offline",
            }

        dossier = create_dossier(CANDIDATE, retriever=failed_retrieval)

        self.assertEqual(dossier["research_status"], "pending")
        self.assertEqual(dossier["retrieval"]["status"], "failed")


class ArticleInputLimitTests(unittest.TestCase):
    def test_short_article_is_unchanged_except_duplicate_whitespace(self) -> None:
        content = "Heading\n\nFirst   paragraph.\n\nSecond paragraph."

        prepared, original_chars, truncated = prepare_article_content(content, 100)

        self.assertEqual(
            prepared,
            "Heading\n\nFirst paragraph.\n\nSecond paragraph.",
        )
        self.assertEqual(original_chars, len(prepared))
        self.assertFalse(truncated)

    def test_long_article_truncates_at_paragraph_boundary(self) -> None:
        content = (
            "First paragraph contains useful source text.\n\n"
            "Second paragraph also contains useful evidence.\n\n"
            "Third paragraph must not be cut in the middle of its sentence."
        )
        limit = content.index("Third paragraph") + 20

        prepared, original_chars, truncated = prepare_article_content(content, limit)

        self.assertEqual(
            prepared,
            "First paragraph contains useful source text.\n\n"
            "Second paragraph also contains useful evidence.",
        )
        self.assertEqual(original_chars, len(content))
        self.assertTrue(truncated)

    def test_build_user_input_records_truncation_metadata(self) -> None:
        item = create_dossier(CANDIDATE, retriever=successful_retrieval)
        item["retrieval"]["content"] = "Paragraph one.\n\nParagraph two is longer."

        with self.assertLogs("newsroom.research", level="WARNING"):
            user_input, _ = build_user_input(item, max_article_chars=20)

        self.assertIn("Paragraph one.", user_input)
        self.assertEqual(item["retrieval"]["original_content_chars"], 40)
        self.assertEqual(item["retrieval"]["sent_to_model_chars"], 14)
        self.assertTrue(item["retrieval"]["content_truncated"])

    def test_cleaning_preserves_paragraph_boundaries(self) -> None:
        cleaned = clean_article_content("  Heading  \n\n  A   paragraph.  ")
        self.assertEqual(cleaned, "Heading\n\nA paragraph.")


if __name__ == "__main__":
    unittest.main()
