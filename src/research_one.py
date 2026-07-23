"""Optional one-candidate integration command for Researcher v0.2."""

from __future__ import annotations

import argparse
import logging
from json import JSONDecodeError
from pathlib import Path

from newsroom.github_models import GitHubModelsProvider
from newsroom.providers import ProviderError
from newsroom.research import (
    apply_research,
    create_dossier,
    create_story_id,
    load_candidates,
    save_dossier,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = PROJECT_ROOT / "data" / "candidates.json"
RESEARCH_DIRECTORY = PROJECT_ROOT / "data" / "research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research exactly one existing News Scout candidate.",
    )
    parser.add_argument(
        "--story-id",
        help="Stable story ID to process; defaults to the first candidate.",
    )
    return parser.parse_args()


def main() -> None:
    """Retrieve and research exactly one selected candidate."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    try:
        candidates = load_candidates(CANDIDATES_PATH)
        if not candidates:
            raise ValueError("no candidates are available")

        if args.story_id:
            matches = [
                candidate
                for candidate in candidates
                if create_story_id(candidate) == args.story_id
            ]
            if not matches:
                raise ValueError(f"candidate {args.story_id!r} was not found")
            candidate = matches[0]
        else:
            candidate = candidates[0]

        provider = GitHubModelsProvider.from_environment()
        dossier = create_dossier(candidate)
        dossier = apply_research(dossier, provider)
        output_path = save_dossier(dossier, RESEARCH_DIRECTORY)
    except (OSError, JSONDecodeError, ValueError, ProviderError) as error:
        print(f"Could not research candidate: {error}")
        return

    print(f"Story: {dossier['story_id']}")
    print(f"Status: {dossier['research_status']}")
    if dossier.get("research_error"):
        print(f"Error: {dossier['research_error']}")
    if dossier.get("inference"):
        usage = dossier["inference"]
        print(
            "Usage: "
            f"model={usage['model']}, "
            f"prompt_tokens={usage['prompt_tokens']}, "
            f"completion_tokens={usage['completion_tokens']}, "
            f"total_tokens={usage['total_tokens']}"
        )
    print(f"Saved dossier: {output_path}")


if __name__ == "__main__":
    main()
