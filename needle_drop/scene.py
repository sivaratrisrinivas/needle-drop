"""Phase 1: scene description -> structured music brief (JSON).

The LLM is constrained to the catalog's own tag vocabulary (JSON-schema enums), so its
output can be matched against tracks directly instead of fuzzy-matched afterwards.
If no LLM is reachable we fall back to an embedding zero-shot parser.
"""
from __future__ import annotations

import collections
import re
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from . import llm
from .catalog import load_catalog
from .tracing import span


@lru_cache(maxsize=1)
def vocab(min_count: int = 20) -> dict[str, list[str]]:
    out = {}
    for field in ("moods", "genres", "instruments"):
        c = collections.Counter(x for t in load_catalog() for x in getattr(t, field))
        out[field] = sorted(k for k, v in c.items() if v >= min_count)
    return out


class TempoRange(BaseModel):
    min_bpm: int
    max_bpm: int

    @model_validator(mode="after")
    def _clamp(self):
        lo, hi = sorted((int(np.clip(self.min_bpm, 40, 220)), int(np.clip(self.max_bpm, 40, 220))))
        self.min_bpm, self.max_bpm = lo, hi
        return self


class SceneProfile(BaseModel):
    reasoning: str = Field("", description="At most 25 words: what the scene is about emotionally and what kind of song an editor would drop here.")
    summary: str = Field(description="At most 15 words: how the scene should feel.")
    relationship: Literal["match", "counterpoint"] = Field(
        description="match = music mirrors the on-screen emotion; counterpoint = deliberately contrasting")
    moods: list[str] = Field(min_length=1)
    genres: list[str] = Field(min_length=1)
    instruments: list[str] = Field(default_factory=list)
    energy: float = Field(description="0 = still and quiet, 1 = frantic and loud")
    tempo: TempoRange
    vocals: Literal["any", "instrumental", "vocal"] = "any"
    avoid: list[str] = Field(default_factory=list)
    music_description: str = Field(description="At most 20 words describing the ideal track in music terms.")
    source: str = "llm"

    @model_validator(mode="after")
    def _trim(self):
        self.energy = float(np.clip(self.energy, 0, 1))
        self.moods, self.genres = self.moods[:6], self.genres[:4]
        self.instruments, self.avoid = self.instruments[:4], self.avoid[:4]
        return self


def _schema() -> dict:
    # Structure only. Enumerating ~400 tags inside the grammar makes llama.cpp's constrained
    # sampler crawl on CPU, so the vocabulary goes in the prompt and we snap tags afterwards.
    s = SceneProfile.model_json_schema()
    s["properties"].pop("source", None)
    s["required"] = ["reasoning", "summary", "relationship", "moods", "genres", "instruments", "energy", "tempo",
                     "vocals", "avoid", "music_description"]
    return s


@lru_cache(maxsize=1)
def prompt_vocab() -> dict[str, list[str]]:
    """The most common tags, small enough to keep the cached system prompt short."""
    out = {}
    for field, n in (("moods", 100), ("genres", 100), ("instruments", 30)):
        c = collections.Counter(x for t in load_catalog() for x in getattr(t, field))
        out[field] = [k for k, _ in c.most_common(n)]
    return out


def _system() -> str:
    v = prompt_vocab()
    return (SYSTEM + "\n\nAllowed moods: " + ", ".join(v["moods"]) +
            "\nAllowed genres: " + ", ".join(v["genres"]) +
            "\nAllowed instruments: " + ", ".join(v["instruments"]))


# Prompt history (see evals/error_analysis.md):
#   v1  one JSON example, counterpoint-first wording -> genres collapsed onto the example
#       (indie/poprock), counterpoint over-predicted (30/50 vs 10/50 true), quiet scenes given
#       high energy.
#   v2  added era->genre and energy guidance plus two JSON examples -> genre recall 0.05->0.17,
#       but the 3B model copied the counterpoint example verbatim in 20/50 briefs.
#   v3  zero-shot: same guidance, no JSON examples to copy; a code eval checks for copying.
PROMPT_VERSION = "v3"
SYSTEM = """You are a music supervisor for film and video. Given a scene, write a short JSON brief
for the song that should play under it, the way an experienced editor would. Base every field on
THIS scene's specific details.

relationship: default to "match" (the song mirrors what the scene feels like). Use "counterpoint"
ONLY for violence, cruelty or horror shown in a stylised, darkly comic or swaggering way, where a
cheerful song plays against it. Grief, romance, nostalgia, triumph, sadness, chases and dances are
"match". When counterpoint, the moods and energy describe the SONG, not the violence.

genres: think about the setting and era. The 1950s-70s suggest 60s, 70s, rocknroll, soul, funk,
folk or classicrock; the 1980s suggest 80s, synthpop, newwave; grand, sacred, historic or space
imagery suggests classical or orchestral; neon night drives suggest synthwave or electronic; bikers,
brawls and car chases suggest rock, hardrock or punkrock. Pick 2-4 genres that fit this scene.

energy: 0.1 still, grieving, tender or awestruck; 0.3 slow and reflective; 0.5 walking pace,
cool or sensual; 0.7 dancing, upbeat; 0.9 only for chases, fights, explosions and frenzy.

Use tags from the allowed lists only. Keep summary and music_description short."""


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def snap_tags(tags: list[str], kind: str, limit: int) -> list[str]:
    """Map free-form tags onto the catalog vocabulary: exact match, else nearest embedding."""
    allowed = vocab()[kind]
    by_norm = {_norm(a): a for a in allowed}
    out: list[str] = []
    for t in tags:
        hit = by_norm.get(_norm(t))
        if hit is None:
            names, m = tag_vectors()[kind]
            from .embeddings import get_embedder
            sims = m @ get_embedder().documents([f"{t} music"])[0]
            i = int(np.argmax(sims))
            hit = names[i] if sims[i] > 0.8 else None
        if hit and hit not in out:
            out.append(hit)
    return out[:limit]


def parse_scene(text: str, hints: str = "") -> SceneProfile:
    """Text (optionally plus clip-analysis hints) -> SceneProfile."""
    with span("scene.parse", chars=len(text)) as s:
        if llm.available():
            try:
                prof = llm.structured(SceneProfile, [
                    {"role": "system", "content": _system()},
                    {"role": "user", "content": f"Scene:\n{text.strip()}\n" + (f"\nClip analysis:\n{hints}\n" if hints else "")},
                ], name="scene_profile", schema=_schema(), max_tokens=260)
                raw = {"moods": prof.moods, "genres": prof.genres, "instruments": prof.instruments}
                prof.moods = snap_tags(prof.moods, "moods", 6) or heuristic_profile(text).moods
                prof.genres = snap_tags(prof.genres, "genres", 4) or heuristic_profile(text).genres
                prof.instruments = snap_tags(prof.instruments, "instruments", 4)
                prof.avoid = snap_tags(prof.avoid, "moods", 4)
                s.set(source="llm", raw_tags=raw, profile=prof.model_dump())
                return prof
            except llm.LLMError as e:
                s.set(llm_error=str(e)[:300])
        prof = heuristic_profile(text + " " + hints)
        s.set(source="heuristic", profile=prof.model_dump())
        return prof


# ---------------------------------------------------------------- heuristic fallback

_FAST = re.compile(r"\b(chase|fight|run|race|explod\w*|frantic|fast|action|battle|heist|dance|party|car)\b", re.I)
_SLOW = re.compile(r"\b(quiet|slow|alone|grief|funeral|sleep|calm|rain|farewell|lonely|still|sunset|melanchol\w*)\b", re.I)


@lru_cache(maxsize=1)
def tag_vectors() -> dict[str, tuple[list[str], np.ndarray]]:
    from .embeddings import get_embedder

    emb = get_embedder()
    v = vocab()
    return {k: (tags, emb.documents([f"{t} music" for t in tags])) for k, tags in v.items()}


def heuristic_profile(text: str) -> SceneProfile:
    from .embeddings import get_embedder

    q = get_embedder().query(text)
    tv = tag_vectors()

    def top(kind, k):
        tags, m = tv[kind]
        return [tags[i] for i in np.argsort(-(m @ q))[:k]]

    fast, slow = len(_FAST.findall(text)), len(_SLOW.findall(text))
    energy = float(np.clip(0.5 + 0.15 * (fast - slow), 0.05, 0.95))
    bpm = int(70 + 80 * energy)
    return SceneProfile(
        reasoning="nearest catalog tags to the scene embedding", summary=text.strip()[:200], moods=top("moods", 4), genres=top("genres", 3), instruments=top("instruments", 2),
        energy=energy, tempo=TempoRange(min_bpm=max(40, bpm - 15), max_bpm=min(220, bpm + 15)),
        relationship="match", music_description=text.strip()[:200], source="heuristic")
