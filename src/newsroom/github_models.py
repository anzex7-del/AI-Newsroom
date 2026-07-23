"""GitHub Models implementation of the research provider interface."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from newsroom.providers import (
    InferenceResult,
    InferenceUsage,
    ProviderError,
    ResearchProvider,
)

GITHUB_MODELS_ENDPOINT = "https://models.github.ai/inference/chat/completions"
DEFAULT_MODEL = "openai/gpt-4.1-mini"
LOGGER = logging.getLogger(__name__)

class GitHubModelsProvider(ResearchProvider):
    """Send constrained chat-completion requests to GitHub Models."""

    def __init__(
        self,
        token: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 30,
    ) -> None:
        normalized_token = token.strip()
        normalized_model = model.strip()
        if not normalized_token:
            raise ProviderError("GITHUB_TOKEN is required")
        if not normalized_model:
            normalized_model = DEFAULT_MODEL
        self._token = normalized_token
        self.model = normalized_model
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> GitHubModelsProvider:
        """Read the current process environment and build a fresh provider."""
        token = os.environ.get("GITHUB_TOKEN", "")
        model = os.environ.get("AI_MODEL") or DEFAULT_MODEL
        return cls(token=token, model=model)

    def log_diagnostics(self) -> None:
        """Log authentication diagnostics without exposing credentials."""
        LOGGER.info("token_present: %s", str(bool(self._token)).lower())
        LOGGER.info("token_length: %d", len(self._token))
        LOGGER.info("endpoint: %s", GITHUB_MODELS_ENDPOINT)
        LOGGER.info("model: %s", self.model)

    def complete(
        self,
        system_instructions: str,
        user_input: str,
        response_schema: dict[str, Any] | None = None,
    ) -> InferenceResult:
        """Request one non-streaming, schema-constrained completion."""
        self.log_diagnostics()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_input},
            ],
            "temperature": 0,
            "max_tokens": 2000,
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }
        request = Request(
            GITHUB_MODELS_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2026-03-10",
                "User-Agent": "AI-Newsroom-Researcher/0.2",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code == 413:
                raise ProviderError(
                    "GitHub Models request exceeded provider payload limit"
                ) from error
            raise ProviderError(f"GitHub Models returned HTTP {error.code}") from error
        except (URLError, OSError, ValueError, json.JSONDecodeError) as error:
            raise ProviderError(f"GitHub Models request failed: {error}") from error

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError("GitHub Models response contained no completion") from error
        if not isinstance(content, str):
            raise ProviderError("GitHub Models completion was not text")

        usage = response_data.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        return InferenceResult(
            content=content,
            usage=InferenceUsage(
                model=str(response_data.get("model") or self.model),
                prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                completion_tokens=_optional_int(usage.get("completion_tokens")),
                total_tokens=_optional_int(usage.get("total_tokens")),
            ),
        )


def _optional_int(value: Any) -> int | None:
    """Return an integer usage value when provided."""
    return value if isinstance(value, int) else None
