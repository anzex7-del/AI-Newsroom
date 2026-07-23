"""Provider-neutral interfaces for controlled research inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ProviderError(RuntimeError):
    """Raised when a model provider cannot complete a request."""


@dataclass(frozen=True)
class InferenceUsage:
    """Non-secret token usage returned by a provider."""

    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class InferenceResult:
    """Raw structured-output text and its usage metadata."""

    content: str
    usage: InferenceUsage


class ResearchProvider(ABC):
    """Interface implemented by controlled research model providers."""

    @abstractmethod
    def complete(
        self,
        system_instructions: str,
        user_input: str,
        response_schema: dict[str, Any] | None = None,
    ) -> InferenceResult:
        """Return one model response without taking application actions."""
