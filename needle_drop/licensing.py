"""Phase 4 + 8: license retrieval (RAG) and the clearance guardrail.

Clearance is decided by rules over a *verified license record*, never by the LLM:
  - the record must come from a named source (dataset snapshot or live Jamendo API),
  - the license URL must parse to a known Creative Commons variant,
  - the variant must permit the intended use (video sync is always an adaptation).
Anything else is "unverified" and can never be shown as cleared.

The knowledge base (data/licenses/*.md) is chunked by section, embedded, and retrieved to
cite *why* a verdict holds and what it costs.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Literal

import httpx
import numpy as np

from . import config
from .catalog import Track
from .tracing import span

Use = Literal["personal", "monetized", "client"]
USE_LABEL = {
    "personal": "personal, non-monetized video",
    "monetized": "monetized YouTube / social video (commercial)",
    "client": "client or advertising work (commercial)",
}

_CC = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)/([\d.]+)?", re.I)


@dataclass
class Citation:
    doc_id: str
    title: str
    section: str
    url: str
    text: str


@dataclass
class LicenseVerdict:
    track_id: str
    use: str
    status: Literal["cleared", "needs_review", "not_cleared", "unverified"]
    cleared: bool
    license_id: str | None
    license_name: str
    license_url: str
    version: str | None
    verification: Literal["dataset_snapshot", "live_api", "none"]
    source: str
    reason: str
    conditions: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    attribution: str = ""
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_license_url(url: str) -> tuple[str | None, str | None]:
    m = _CC.search(url or "")
    if not m:
        return None, None
    kind, code, ver = m.group(1), m.group(2).lower(), m.group(3)
    if kind == "publicdomain":
        return ("cc0" if code == "zero" else None), ver
    lid = "cc-" + code
    return (lid if lid in {"cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "cc-by-nd", "cc-by-nc-nd"} else None), ver


# (status, reason, conditions) by license and use
RULES: dict[str, dict[str, tuple[str, str, list[str]]]] = {
    "cc-by": {u: ("cleared", "CC BY permits commercial use and adaptation (sync) with credit.", ["Credit the track (TASL)."])
              for u in ("personal", "monetized", "client")},
    "cc-by-sa": {
        "personal": ("cleared", "CC BY-SA permits sync; the video becomes BY-SA adapted material.",
                     ["Credit the track (TASL).", "Release your video under CC BY-SA."]),
        "monetized": ("cleared", "CC BY-SA permits commercial use and sync, with ShareAlike on the video.",
                      ["Credit the track (TASL).", "Release your video under CC BY-SA."]),
        "client": ("needs_review", "CC BY-SA would require licensing the client's video under BY-SA; most clients cannot accept that.",
                   ["Confirm the client accepts CC BY-SA on the final video."]),
    },
    "cc-by-nc": {
        "personal": ("cleared", "CC BY-NC permits non-commercial adaptation with credit.", ["Credit the track (TASL).", "No monetization."]),
        "monetized": ("not_cleared", "NonCommercial excludes monetized use; a commercial license is needed.", []),
        "client": ("not_cleared", "NonCommercial excludes client/advertising use; a commercial license is needed.", []),
    },
    "cc-by-nc-sa": {
        "personal": ("cleared", "CC BY-NC-SA permits non-commercial sync; the video must be BY-NC-SA.",
                     ["Credit the track (TASL).", "No monetization.", "Release your video under CC BY-NC-SA."]),
        "monetized": ("not_cleared", "NonCommercial excludes monetized use; a commercial license is needed.", []),
        "client": ("not_cleared", "NonCommercial excludes client/advertising use; a commercial license is needed.", []),
    },
    "cc-by-nd": {u: ("not_cleared", "NoDerivatives forbids adaptation, and syncing music to video is an adaptation.", [])
                 for u in ("personal", "monetized", "client")},
    "cc-by-nc-nd": {u: ("not_cleared", "NoDerivatives forbids adaptation, and syncing music to video is an adaptation.", [])
                    for u in ("personal", "monetized", "client")},
    "cc0": {u: ("cleared", "CC0 waives copyright; no conditions.", []) for u in ("personal", "monetized", "client")},
}


def _cost(status: str, lid: str | None) -> dict:
    if status == "cleared":
        return {"amount_usd": 0.0, "basis": "Free under the Creative Commons license terms (credit required)."}
    if status == "needs_review":
        return {"amount_usd": 0.0, "basis": "Free under CC terms only if the ShareAlike condition is acceptable; otherwise a commercial license (price not verified)."}
    if lid in ("cc-by-nd", "cc-by-nc-nd"):
        return {"amount_usd": None, "basis": "Not usable in video under CC; a sync license from the rights holder is needed (price not verified)."}
    if status == "not_cleared":
        return {"amount_usd": None, "basis": "Commercial license required from the rights holder, e.g. via Jamendo Licensing (price not verified)."}
    return {"amount_usd": None, "basis": "Unknown: no verified license record."}


def _live_record(track: Track) -> tuple[str, str] | None:
    """Live license lookup via the Jamendo API (needs JAMENDO_CLIENT_ID). Returns (name, url)."""
    if not config.JAMENDO_CLIENT_ID:
        return None
    try:
        r = httpx.get("https://api.jamendo.com/v3.0/tracks/", timeout=8, params={
            "client_id": config.JAMENDO_CLIENT_ID, "format": "json", "id": track.jamendo_id})
        res = r.json().get("results") or []
        if res and res[0].get("license_ccurl"):
            return "Creative Commons (live)", res[0]["license_ccurl"]
    except (httpx.HTTPError, ValueError):
        return None
    return None


def check_license(track: Track, use: Use = "monetized", live: bool = True) -> LicenseVerdict:
    with span("license.check", track=track.id, use=use) as s:
        name, url, source, verification = track.license_name, track.license_url, track.license_source, "dataset_snapshot"
        flags: list[str] = []
        rec = _live_record(track) if live else None
        if rec:
            if parse_license_url(rec[1])[0] != parse_license_url(url)[0]:
                flags.append(f"License changed since the 2019 snapshot ({url} -> {rec[1]}); using the live value.")
            name, url, verification = rec[0], rec[1], "live_api"
            source = f"https://api.jamendo.com/v3.0/tracks/?id={track.jamendo_id}"
        lid, ver = parse_license_url(url)
        if not source or not url or lid is None:
            v = LicenseVerdict(track.id, use, "unverified", False, None, name or "unknown", url, None, "none", source,
                               "No verified license record for this track, so it cannot be cleared.",
                               flags=flags + ["Unverified license: check the rights holder before any use."],
                               cost=_cost("unverified", None))
            s.set(status=v.status)
            return v
        status, reason, conditions = RULES[lid][use]
        if verification == "dataset_snapshot":
            flags.append("License verified against the 2019 MTG-Jamendo snapshot; reconfirm on the Jamendo track page before publishing.")
        short = lid.upper().replace("CC-", "CC ").replace("-", "-") + (f" {ver}" if ver else "")
        v = LicenseVerdict(
            track.id, use, status, status == "cleared", lid, name, url, ver, verification, source, reason,
            conditions, flags, _cost(status, lid),
            attribution=f"\"{track.title}\" by {track.artist} ({track.page_url}), licensed under {short} ({url})",
            citations=cite(lid, use))
        s.set(status=status, license=lid)
        return v


# ---------------------------------------------------------------- knowledge base + retrieval

@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str
    url: str
    text: str


@lru_cache(maxsize=1)
def kb() -> list[Chunk]:
    chunks = []
    for p in sorted(config.LICENSE_DOCS.glob("*.md")):
        raw = p.read_text()
        _, front, body = raw.split("---", 2)
        meta = dict(l.split(": ", 1) for l in front.strip().splitlines())
        for sec in re.split(r"\n## ", "\n" + body.strip()):
            sec = sec.strip()
            if not sec:
                continue
            head, _, text = sec.partition("\n")
            chunks.append(Chunk(meta["id"], meta["title"], head.strip("# "), meta.get("deed", ""), text.strip()))
    return chunks


@lru_cache(maxsize=1)
def kb_vectors() -> np.ndarray:
    from .embeddings import get_embedder

    return get_embedder().documents([f"{c.title}. {c.section}. {c.text}" for c in kb()])


def retrieve(question: str, k: int = 4) -> list[tuple[Chunk, float]]:
    from .embeddings import get_embedder

    with span("license.retrieve", q=question[:200]) as s:
        sims = kb_vectors() @ get_embedder().query(question)
        order = np.argsort(-sims)[:k]
        out = [(kb()[i], float(sims[i])) for i in order]
        s.set(hits=[(c.doc_id, c.section, round(x, 3)) for c, x in out])
        return out


def cite(lid: str, use: str) -> list[Citation]:
    """Citations for a verdict: the license's own video/cost sections, plus retrieved context."""
    want = []
    for c in kb():
        if c.doc_id == lid and c.section in ("Use in video", "Cost", "NonCommercial definition"):
            want.append(c)
    q = f"{lid} license {USE_LABEL[use]} sync music to video"
    for c, _ in retrieve(q, k=3):
        if c not in want and c.doc_id in (lid, "video-sync", "jamendo-source"):
            want.append(c)
    return [Citation(c.doc_id, c.title, c.section, c.url, c.text) for c in want[:4]]
