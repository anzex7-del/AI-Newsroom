"""Command-line entry point for Researcher v0.1 infrastructure."""

from __future__ import annotations

from json import JSONDecodeError
from pathlib import Path

from newsroom.research import create_dossiers, load_candidates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = PROJECT_ROOT / "data" / "candidates.json"
RESEARCH_DIRECTORY = PROJECT_ROOT / "data" / "research"


def main() -> None:
    """Create initial research dossiers from ranked candidate stories."""
    try:
        candidates = load_candidates(CANDIDATES_PATH)
    except (OSError, JSONDecodeError, ValueError) as error:
        print(f"Could not load candidates: {error}")
        return

    paths, malformed_count, retrieval_failure_count = create_dossiers(
        candidates,
        RESEARCH_DIRECTORY,
    )

    print("\nResearcher v0.1 summary")
    print(f"Candidates loaded: {len(candidates)}")
    print(f"Dossiers created: {len(paths)}")
    print(f"Malformed candidates skipped: {malformed_count}")
    print(f"Retrieval failures: {retrieval_failure_count}")
    print(f"Research directory: {RESEARCH_DIRECTORY}")


if __name__ == "__main__":
    main()
