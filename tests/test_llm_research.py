"""Mocked tests for controlled Researcher inference."""

import json
import os
import unittest
from io import BytesIO
from urllib.error import HTTPError
from unittest.mock import patch

from newsroom.github_models import (
    DEFAULT_MODEL,
    GITHUB_MODELS_ENDPOINT,
    GitHubModelsProvider,
)
from newsroom.providers import (
    InferenceResult,
    InferenceUsage,
    ProviderError,
    ResearchProvider,
)
from newsroom.research import apply_research

SOURCE_URL = "https://example.invalid/story"


def dossier():
    return {
        "story_id": "story_example",
        "title": "Example",
        "candidate_url": SOURCE_URL,
        "primary_source": {
            "url": SOURCE_URL,
            "publisher": "Example Publisher",
            "published_at": "2026-07-22T00:00:00Z",
        },
        "retrieval": {
            "status": "success",
            "final_url": SOURCE_URL,
            "http_status": 200,
            "retrieved_at": "2026-07-23T00:00:00Z",
            "content_type": "text/html",
            "content": "The supplied article states that the feature launched.",
            "original_content_chars": 54,
            "sent_to_model_chars": 0,
            "content_truncated": False,
            "error": None,
        },
        "claims": [],
        "key_takeaways": [],
        "open_questions": [],
        "research_status": "pending",
    }


class FakeProvider(ResearchProvider):
    def __init__(self, content: str):
        self.content = content

    def complete(
        self,
        system_instructions: str,
        user_input: str,
        response_schema=None,
    ):
        self.system_instructions = system_instructions
        self.user_input = user_input
        self.response_schema = response_schema
        return InferenceResult(
            content=self.content,
            usage=InferenceUsage("test/model", 100, 25, 125),
        )


class FailingProvider(ResearchProvider):
    def complete(
        self,
        system_instructions: str,
        user_input: str,
        response_schema=None,
    ):
        raise ProviderError("API unavailable")


class ResearchOutputTests(unittest.TestCase):
    def test_accepts_valid_structured_response(self) -> None:
        response = json.dumps(
            {
                "claims": [
                    {
                        "claim": "The feature launched.",
                        "evidence": "The article says the feature launched.",
                        "source_url": SOURCE_URL,
                        "confidence": "high",
                    }
                ],
                "key_takeaways": ["A feature was launched."],
                "open_questions": [],
            }
        )

        result = apply_research(dossier(), FakeProvider(response))

        self.assertEqual(result["research_status"], "researched")
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["inference"]["total_tokens"], 125)

    def test_invalid_json_preserves_empty_research_fields(self) -> None:
        result = apply_research(dossier(), FakeProvider("not JSON"))

        self.assertEqual(result["research_status"], "failed")
        self.assertEqual(result["claims"], [])
        self.assertIn("invalid JSON", result["research_error"])
        self.assertEqual(result["inference"]["total_tokens"], 125)

    def test_rejects_unsupported_source_url(self) -> None:
        response = json.dumps(
            {
                "claims": [
                    {
                        "claim": "Unsupported claim.",
                        "evidence": "Unsupported evidence.",
                        "source_url": "https://unsupported.invalid/story",
                        "confidence": "low",
                    }
                ],
                "key_takeaways": [],
                "open_questions": [],
            }
        )

        result = apply_research(dossier(), FakeProvider(response))

        self.assertEqual(result["research_status"], "failed")
        self.assertEqual(result["claims"], [])
        self.assertIn("unsupported source URL", result["research_error"])

    def test_provider_failure_marks_dossier_failed(self) -> None:
        result = apply_research(dossier(), FailingProvider())

        self.assertEqual(result["research_status"], "failed")
        self.assertEqual(result["research_error"], "API unavailable")
        self.assertEqual(result["claims"], [])


class AuthenticationTests(unittest.TestCase):
    def test_missing_github_token_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ProviderError, "GITHUB_TOKEN"):
                GitHubModelsProvider.from_environment()

    def test_authorization_header_uses_stripped_runtime_token(self) -> None:
        response_body = json.dumps(
            {
                "model": "test/model",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            }
        ).encode()

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return response_body

        def inspect_request(request, timeout):
            self.assertEqual(request.full_url, GITHUB_MODELS_ENDPOINT)
            self.assertEqual(
                request.get_header("Authorization"),
                "Bearer current-token",
            )
            self.assertEqual(request.get_header("Content-type"), "application/json")
            body = json.loads(request.data.decode())
            self.assertEqual(body["model"], "test/model")
            self.assertEqual(
                body["messages"],
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
            )
            self.assertEqual(timeout, 30)
            return FakeResponse()

        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "  current-token\n",
                "AI_MODEL": "  test/model  ",
            },
            clear=True,
        ):
            provider = GitHubModelsProvider.from_environment()
            with patch(
                "newsroom.github_models.urlopen",
                side_effect=inspect_request,
            ) as mocked:
                provider.complete("system", "user")

        mocked.assert_called_once()

    def test_environment_is_read_for_each_new_provider(self) -> None:
        with patch.dict(
            os.environ,
            {"GITHUB_TOKEN": "first-token"},
            clear=True,
        ):
            first = GitHubModelsProvider.from_environment()
            os.environ["GITHUB_TOKEN"] = "second-token-longer"
            second = GitHubModelsProvider.from_environment()

        with self.assertLogs("newsroom.github_models", level="INFO") as first_logs:
            first.log_diagnostics()
        with self.assertLogs("newsroom.github_models", level="INFO") as second_logs:
            second.log_diagnostics()

        self.assertIn("token_length: 11", "\\n".join(first_logs.output))
        self.assertIn("token_length: 19", "\\n".join(second_logs.output))
        self.assertNotIn("first-token", "\\n".join(first_logs.output))
        self.assertNotIn("second-token", "\\n".join(second_logs.output))

    def test_blank_model_uses_default(self) -> None:
        provider = GitHubModelsProvider(token="token", model="  ")
        self.assertEqual(provider.model, DEFAULT_MODEL)


class GitHubModelsErrorTests(unittest.TestCase):
    def test_http_413_has_clear_error_and_is_not_retried(self) -> None:
        error = HTTPError(
            url="https://models.github.ai/inference/chat/completions",
            code=413,
            msg="Payload Too Large",
            hdrs=None,
            fp=BytesIO(),
        )
        provider = GitHubModelsProvider(token="test-token")

        with patch("newsroom.github_models.urlopen", side_effect=error) as mocked:
            with self.assertRaisesRegex(
                ProviderError,
                "request exceeded provider payload limit",
            ):
                provider.complete("system", "user")

        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
