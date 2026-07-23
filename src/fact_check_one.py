"""Process exactly one researched dossier with the Fact Checker."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from newsroom.fact_check import apply_fact_check
from newsroom.github_models import GitHubModelsProvider
from newsroom.providers import ProviderError
from newsroom.research import save_dossier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIRECTORY = PROJECT_ROOT / "data" / "research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fact-check exactly one researched dossier.",
    )
    parser.add_argument(
        "--story-id",
        help="Story ID to process; defaults to the first researched dossier.",
    )
    return parser.parse_args()


def select_dossier(story_id: str | None) -> dict:
    """Load exactly one researched dossier from the research directory."""
    if story_id:
        paths = [RESEARCH_DIRECTORY / f"{story_id}.json"]
    else:
        paths = sorted(RESEARCH_DIRECTORY.glob("story_*.json"))

    for path in paths:
        if not path.is_file():
            continue
        dossier = json.loads(path.read_text(encoding="utf-8"))
        if dossier.get("research_status") == "researched":
            return dossier
    if story_id:
        raise ValueError(f"researched dossier {story_id!r} was not found")
    raise ValueError("no researched dossier is available")


def main() -> None:
    """Run one grounded fact check and save its dossier."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    try:
        dossier = select_dossier(args.story_id)
        provider = GitHubModelsProvider.from_environment()
        dossier = apply_fact_check(dossier, provider)
        output_path = save_dossier(dossier, RESEARCH_DIRECTORY)
    except (OSError, json.JSONDecodeError, ValueError, ProviderError) as error:
        print(f"Could not fact-check dossier: {error}")
        return

    result = dossier["fact_check"]
    print(f"Story: {dossier['story_id']}")
    print(f"Fact-check status: {result['fact_check_status']}")
    print(f"Verified: {result['verified_claim_count']}")
    print(f"Partially verified: {result['partial_claim_count']}")
    print(f"Unsupported: {result['unsupported_claim_count']}")
    print(f"Contradicted: {result['contradicted_claim_count']}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    if result.get("inference"):
        usage = result["inference"]
        print(
            "Fact Checker usage: "
            f"model={usage['model']}, "
            f"prompt_tokens={usage['prompt_tokens']}, "
            f"completion_tokens={usage['completion_tokens']}, "
            f"total_tokens={usage['total_tokens']}"
        )
    print(f"Saved dossier: {output_path}")


if __name__ == "__main__":
    main()
