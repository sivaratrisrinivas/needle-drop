"""Phase 8: guardrails and human-in-the-loop flags.

Rules:
 1. A track is shown as cleared only when licensing.check_license returned status "cleared",
    which itself requires a verified license source. Nothing the LLM writes can change it.
 2. Generated text may not claim clearance for a track that is not cleared; such sentences are
    removed and the item is flagged.
 3. Anything uncertain (snapshot-only verification, ShareAlike conditions, missing license,
    weak match) carries a visible flag so a human reviews it before publishing.
"""
from __future__ import annotations

import re

CLAIM = re.compile(r"(\bcleared\b|royalty[- ]free|free to use|safe to use|copyright[- ]free|no copyright|"
                   r"licensed for (commercial|youtube|monetiz\w*)|can be used commercially|good to go)", re.I)
WEAK_MATCH = 0.45


def scrub_claims(text: str, cleared: bool) -> tuple[str, bool]:
    """Remove sentences claiming clearance when the track is not cleared."""
    if cleared or not text:
        return text, False
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not CLAIM.search(s)]
    return " ".join(kept).strip(), len(kept) != len(sentences)


def apply(items: list[dict]) -> list[str]:
    """Mutates shortlist items in place; returns shortlist-level warnings."""
    warnings = []
    for it in items:
        v = it["license"]
        it["cleared"] = bool(v["status"] == "cleared" and v["verification"] in ("dataset_snapshot", "live_api")
                             and v["source"])
        flags = list(v.get("flags", []))
        if v["status"] == "needs_review":
            flags.append("Needs human review: " + v["reason"])
        if v["status"] == "unverified":
            flags.append("No verified license source: never use without checking with the rights holder.")
        if it["score"] < WEAK_MATCH:
            flags.append("Weak match: listen before shortlisting.")
        text, removed = scrub_claims(it.get("blurb", ""), it["cleared"])
        if removed:
            it["blurb"] = text
            flags.append("Removed an unsupported licensing claim from the generated explanation.")
        it["flags"] = flags
        it["needs_review"] = (not it["cleared"]) or bool(flags)
    if not any(i["cleared"] for i in items):
        warnings.append("No track in this shortlist is cleared for the selected use.")
    return warnings
