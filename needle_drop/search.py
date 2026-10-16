"""Phase 2: vibe search over the embedded catalog.

Hybrid ranking:
  semantic  cosine(query embedding, track vibe-document embedding)   [nomic-embed]
  tags      overlap between the brief's moods/genres/instruments and the track's tags
  energy    closeness of the brief's energy to the track's mood-derived energy
Candidates come from the vector index (top 600) plus any track sharing a mood AND a genre
with the brief, then everything is re-ranked and diversified (max 2 tracks per artist).

The "vector DB" is an exact in-process index (22k x 256 float32, one matmul, ~2 ms).
At this size exact search beats an ANN index on both recall and latency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import config
from .catalog import Track, load_catalog
from .scene import SceneProfile
from .tracing import span

WEIGHTS = dict(semantic=0.45, mood=0.25, genre=0.15, instrument=0.05, energy=0.10, avoid=0.20)


@dataclass
class Hit:
    track: Track
    score: float
    parts: dict = field(default_factory=dict)
    matched: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"track": self.track.to_dict(), "score": round(self.score, 4),
                "parts": {k: round(v, 3) for k, v in self.parts.items()}, "matched": self.matched}


class VectorIndex:
    def __init__(self, vectors: np.ndarray, tracks: list[Track]):
        self.m = vectors.astype(np.float32)
        self.m /= np.linalg.norm(self.m, axis=1, keepdims=True) + 1e-9
        self.tracks = tracks
        self.by_mood: dict[str, set[int]] = {}
        self.by_genre: dict[str, set[int]] = {}
        for i, t in enumerate(tracks):
            for x in t.moods:
                self.by_mood.setdefault(x, set()).add(i)
            for x in t.genres:
                self.by_genre.setdefault(x, set()).add(i)

    def knn(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims = self.m @ q.astype(np.float32)
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        return idx, sims


@lru_cache(maxsize=1)
def get_index() -> VectorIndex:
    return VectorIndex(np.load(config.VECTORS_PATH), load_catalog())


def query_text(p: SceneProfile) -> str:
    return (f"{p.music_description} Mood: {', '.join(p.moods)}. Genre: {', '.join(p.genres)}."
            + (f" Instruments: {', '.join(p.instruments)}." if p.instruments else ""))


def search(profile: SceneProfile, k: int = 10, *, query_vec: np.ndarray | None = None,
           allowed: set[str] | None = None, weights: dict | None = None, per_artist: int = 2) -> list[Hit]:
    """Rank catalog tracks for a scene profile. `allowed` restricts to these track ids."""
    from .embeddings import get_embedder

    w = {**WEIGHTS, **(weights or {})}
    idx = get_index()
    with span("search.catalog", k=k) as s:
        q = query_vec if query_vec is not None else get_embedder().query(query_text(profile))
        cand, sims = idx.knn(q, 600)
        cand = set(int(i) for i in cand)
        for mood in profile.moods:
            for genre in profile.genres:
                both = idx.by_mood.get(mood, set()) & idx.by_genre.get(genre, set())
                cand |= set(list(both)[:300])
        pm, pg, pi, pa = set(profile.moods), set(profile.genres), set(profile.instruments), set(profile.avoid)
        sem_vals = sims[list(cand)]
        lo, hi = float(np.percentile(sem_vals, 5)), float(sem_vals.max()) + 1e-9
        hits = []
        for i in cand:
            t = idx.tracks[i]
            if allowed is not None and t.id not in allowed:
                continue
            tm, tg, ti = set(t.moods), set(t.genres), set(t.instruments)
            parts = {
                "semantic": float(np.clip((sims[i] - lo) / (hi - lo), 0, 1)),
                "mood": len(pm & tm) / max(1, min(len(pm), 3)),
                "genre": len(pg & tg) / max(1, min(len(pg), 2)),
                "instrument": len(pi & ti) / max(1, len(pi)) if pi else 0.0,
                "energy": 1 - abs(profile.energy - t.energy),
                "avoid": float(bool(pa & tm)),
            }
            parts["mood"] = min(parts["mood"], 1.0)
            parts["genre"] = min(parts["genre"], 1.0)
            if profile.vocals == "instrumental" and "voice" in ti:
                parts["avoid"] = 1.0
            score = sum(w[k_] * v for k_, v in parts.items() if k_ != "avoid") - w["avoid"] * parts["avoid"]
            hits.append(Hit(t, score, parts, {"moods": sorted(pm & tm), "genres": sorted(pg & tg),
                                              "instruments": sorted(pi & ti)}))
        hits.sort(key=lambda h: -h.score)
        out, per = [], {}
        for h in hits:
            if per.get(h.track.artist, 0) >= per_artist:
                continue
            per[h.track.artist] = per.get(h.track.artist, 0) + 1
            out.append(h)
            if len(out) >= k:
                break
        s.set(candidates=len(cand), top=[(h.track.id, round(h.score, 3)) for h in out])
        return out
