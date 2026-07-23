"""Create one grounded social-media draft from a fact-checked dossier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from newsroom.editorial import editorial_gate
from newsroom.github_models import GitHubModelsProvider
from newsroom.providers import ProviderError
from newsroom.writer import apply_writer, save_output

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIRECTORY = PROJECT_ROOT / "data" / "research"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write exactly one eligible story.")
    parser.add_argument("--story-id", help="Story ID; defaults to the first eligible.")
    return parser.parse_args()


def select_dossier(story_id: str | None) -> dict:
    """Select exactly one dossier, preferring an eligible story."""
    paths = (
        [RESEARCH_DIRECTORY / f"{story_id}.json"]
        if story_id
        else sorted(RESEARCH_DIRECTORY.glob("story_*.json"))
    )
    fallback = None
    for path in paths:
        if not path.is_file():
            continue
        dossier = json.loads(path.read_text(encoding="utf-8"))
        if story_id:
            return dossier
        if editorial_gate(dossier) == "eligible_for_writing":
            return dossier
        fallback = fallback or dossier
    if fallback is not None:
        return fallback
    raise ValueError("no fact-checked dossier is available")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    try:
        dossier = select_dossier(args.story_id)
        gate_status = editorial_gate(dossier)
        if gate_status == "eligible_for_writing":
            provider = GitHubModelsProvider.from_environment()
        else:
            provider = _BlockedProvider()
        output = apply_writer(dossier, provider)
        path = save_output(output, OUTPUT_DIRECTORY)
    except (OSError, json.JSONDecodeError, ValueError, ProviderError) as error:
        print(f"Could not write story: {error}")
        return

    print(f"Story: {output['story_id']}")
    print(f"Editorial gate: {output['editorial_gate']}")
    print(f"Output status: {output['output_status']}")
    if output.get("error"):
        print(f"Error: {output['error']}")
    if output.get("inference"):
        usage = output["inference"]
        print(
            "Writer usage: "
            f"model={usage['model']}, prompt_tokens={usage['prompt_tokens']}, "
            f"completion_tokens={usage['completion_tokens']}, "
            f"total_tokens={usage['total_tokens']}"
        )
    print(f"Saved output: {path}")


class _BlockedProvider:
    """Provider placeholder that must never be called for blocked stories."""

    def complete(self, system_instructions, user_input, response_schema=None):
        raise AssertionError("blocked stories must not call a model")


if __name__ == "__main__":
    main()
