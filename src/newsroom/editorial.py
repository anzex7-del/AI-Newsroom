"""Deterministic editorial gate between fact-checking and writing."""

from __future__ import annotations

from typing import Any

GATE_RESULTS = {
    "passed": "eligible_for_writing",
    "needs_review": "human_review_required",
    "failed": "rejected",
}


def editorial_gate(dossier: dict[str, Any]) -> str:
    """Map the fact-check status to a deterministic editorial decision."""
    fact_check = dossier.get("fact_check")
    if not isinstance(fact_check, dict):
        return "rejected"
    return GATE_RESULTS.get(fact_check.get("fact_check_status"), "rejected")
