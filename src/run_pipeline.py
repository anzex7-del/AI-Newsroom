"""Run the bounded AI Newsroom v0.1 pipeline for one candidate."""

from __future__ import annotations

import logging

from main import (
    CANDIDATES_OUTPUT_PATH,
    EDITORIAL_PATH,
    NEWS_OUTPUT_PATH,
    collect_articles,
    save_json,
)
from newsroom.config import load_editorial_config
from newsroom.editorial import editorial_gate
from newsroom.fact_check import apply_fact_check
from newsroom.github_models import GitHubModelsProvider
from newsroom.providers import ProviderError
from newsroom.ranking import rank_articles
from newsroom.research import apply_research, create_dossier, save_dossier
from newsroom.writer import apply_writer, save_output

RESEARCH_DIRECTORY = NEWS_OUTPUT_PATH.parent / "research"
OUTPUT_DIRECTORY = NEWS_OUTPUT_PATH.parent / "output"


def main() -> None:
    """Run fetch, rank, research, fact-check, gate, and write once."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("[1/6] Fetching configured feeds")
    articles = collect_articles()
    save_json(NEWS_OUTPUT_PATH, articles)
    print(f"Fetched: {len(articles)}")

    print("\n[2/6] Ranking candidate stories")
    try:
        config = load_editorial_config(EDITORIAL_PATH)
        candidates, stats = rank_articles(articles, config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Ranking failed: {error}")
        return
    save_json(CANDIDATES_OUTPUT_PATH, candidates)
    print(
        f"Candidates: {len(candidates)} "
        f"(duplicates={stats.duplicates_removed}, "
        f"old={stats.rejected_by_recency}, "
        f"irrelevant={stats.rejected_by_relevance})"
    )
    if not candidates:
        print("Pipeline stopped: no eligible candidate.")
        return
    candidate = candidates[0]
    print(f"Selected one candidate: {candidate['title']}")

    try:
        provider = GitHubModelsProvider.from_environment()
    except ProviderError as error:
        print(f"Provider setup failed: {error}")
        return

    print("\n[3/6] Researching selected candidate")
    dossier = create_dossier(candidate)
    dossier = apply_research(dossier, provider)
    save_dossier(dossier, RESEARCH_DIRECTORY)
    print(f"Research status: {dossier['research_status']}")
    if dossier["research_status"] != "researched":
        print(f"Pipeline stopped: {dossier.get('research_error')}")
        return

    print("\n[4/6] Fact-checking Researcher claims")
    dossier = apply_fact_check(dossier, provider)
    save_dossier(dossier, RESEARCH_DIRECTORY)
    fact_check = dossier["fact_check"]
    print(
        f"Fact-check status: {fact_check['fact_check_status']} "
        f"(verified={fact_check['verified_claim_count']}, "
        f"partial={fact_check['partial_claim_count']}, "
        f"unsupported={fact_check['unsupported_claim_count']}, "
        f"contradicted={fact_check['contradicted_claim_count']})"
    )

    print("\n[5/6] Applying deterministic editorial gate")
    gate_status = editorial_gate(dossier)
    print(f"Editorial gate: {gate_status}")

    print("\n[6/6] Producing Writer output")
    output = apply_writer(dossier, provider)
    path = save_output(output, OUTPUT_DIRECTORY)
    print(f"Output status: {output['output_status']}")
    if output.get("error"):
        print(f"Writer error: {output['error']}")
    print(f"Saved output: {path}")


if __name__ == "__main__":
    main()
