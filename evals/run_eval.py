"""Phase 6: retrieval evals on 50 iconic needle-drop scenes.

Ground truth per scene: the original song plus its reference tags in the catalog vocabulary
(genres, moods, energy) and whether the drop was a match or a counterpoint.

Code evaluators (write-code-eval):
  original@10   the original song is in the top 10 (always 0 here: no major-label song is in a
                CC catalog; reported so nobody mistakes the other metrics for it)
  similar@10    at least one top-10 track "feels like" the original: shares >=1 genre AND >=1 mood
                with the reference and its energy is within 0.25
  P@10, MRR     precision of similar tracks in the top 10, reciprocal rank of the first one
  brief_*       phase-1 quality: tag recall of the LLM brief against the reference tags,
                counterpoint detection accuracy, energy error
  brief_valid   every brief parses, uses only catalog tags, energy in [0,1]

Usage: python evals/run_eval.py [--variants heuristic,llm_semantic,llm_hybrid,direct_baseline,direct_adapter] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from needle_drop import llm  # noqa: E402
from needle_drop.catalog import Track  # noqa: E402
from needle_drop.scene import SceneProfile, heuristic_profile, parse_scene, vocab  # noqa: E402
from needle_drop.search import query_text, search  # noqa: E402
from needle_drop.tracing import trace  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def load_scenes() -> list[dict]:
    v = vocab()
    scenes = [json.loads(l) for l in open(HERE / "scenes.jsonl") if l.strip()]
    for s in scenes:  # keep only reference tags that exist in the catalog vocabulary
        s["ref_genres"] = [g for g in s["ref_genres"] if g in v["genres"]]
        s["ref_moods"] = [m for m in s["ref_moods"] if m in v["moods"]]
    return scenes


def similar(t: Track, s: dict) -> bool:
    return (bool(set(t.genres) & set(s["ref_genres"])) and bool(set(t.moods) & set(s["ref_moods"]))
            and abs(t.energy - s["ref_energy"]) <= 0.25)


def is_original(t: Track, s: dict) -> bool:
    return t.title.lower() == s["song"].lower() and s["artist"].lower() in t.artist.lower()


def brief_valid(p: SceneProfile) -> tuple[bool, str]:
    v = vocab()
    bad = [m for m in p.moods + p.avoid if m not in v["moods"]] + [g for g in p.genres if g not in v["genres"]] \
        + [i for i in p.instruments if i not in v["instruments"]]
    if bad:
        return False, f"tags outside vocabulary: {bad}"
    if not (0 <= p.energy <= 1):
        return False, "energy out of range"
    if not p.moods or not p.genres:
        return False, "empty moods or genres"
    return True, "ok"


def score_ranking(tracks: list[Track], s: dict) -> dict:
    sims = [similar(t, s) for t in tracks]
    first = next((i for i, x in enumerate(sims) if x), None)
    return {"original@10": any(is_original(t, s) for t in tracks), "similar@10": any(sims),
            "P@10": sum(sims) / max(1, len(tracks)), "RR": 0 if first is None else 1 / (first + 1),
            "top": [(t.id, t.title, t.artist, sorted(set(t.moods) & set(s["ref_moods"])),
                     sorted(set(t.genres) & set(s["ref_genres"])), t.energy, sims[i]) for i, t in enumerate(tracks)]}


def brief_metrics(p: SceneProfile, s: dict) -> dict:
    rm, rg = set(s["ref_moods"]), set(s["ref_genres"])
    return {"mood_recall": len(rm & set(p.moods)) / max(1, len(rm)),
            "genre_recall": len(rg & set(p.genres)) / max(1, len(rg)),
            "relationship_ok": p.relationship == s["relationship"], "energy_err": abs(p.energy - s["ref_energy"])}


def profiles(scenes: list[dict], kind: str) -> dict[str, SceneProfile]:
    """LLM briefs are cached on disk so variants that share a brief don't re-pay for it."""
    from needle_drop.scene import PROMPT_VERSION
    path = OUT / (f"profiles_{kind}.jsonl" if kind == "heuristic" else f"profiles_{kind}_{PROMPT_VERSION}.jsonl")
    have = {}
    if path.exists():
        for l in open(path):
            d = json.loads(l)
            have[d["id"]] = SceneProfile(**d["profile"])
    with open(path, "a") as f:
        for s in scenes:
            if s["id"] in have:
                continue
            t0 = time.time()
            with trace(f"eval.brief.{kind}", {"scene": s["id"]}):
                p = heuristic_profile(s["scene"]) if kind == "heuristic" else parse_scene(s["scene"])
            have[s["id"]] = p
            f.write(json.dumps({"id": s["id"], "profile": p.model_dump(), "secs": round(time.time() - t0, 1)}) + "\n")
            f.flush()
            print(f"  brief {kind} {s['id']} {time.time() - t0:.1f}s", flush=True)
    return have


def run_variant(name: str, scenes: list[dict]) -> dict:
    # direct_*: no LLM brief; the raw scene text is embedded and searched semantically.
    # direct_baseline = off-the-shelf nomic, direct_adapter = the phase-7 fine-tuned map.
    kind = "llm" if name.startswith("llm") else "heuristic"
    profs = profiles(scenes, kind)
    weights = {"semantic": 1, "mood": 0, "genre": 0, "instrument": 0, "energy": 0, "avoid": 0} \
        if name in ("llm_semantic", "direct_baseline", "direct_adapter") else None
    adapter = None
    if name == "direct_adapter":
        from finetune.adapter import load_adapter
        adapter = load_adapter()
    rows = []
    for s in scenes:
        p = profs[s["id"]]
        qv = None
        if name == "direct_baseline":
            from needle_drop.embeddings import get_embedder
            qv = get_embedder().query(s["scene"])
        elif adapter:
            qv = adapter.query_vector_text(s["scene"])
        hits = search(p, k=10, weights=weights, query_vec=qv)
        r = score_ranking([h.track for h in hits], s)
        r.update(brief_metrics(p, s))
        r["valid"], r["valid_reason"] = brief_valid(p)
        r["id"], r["relationship"] = s["id"], s["relationship"]
        r["profile"] = p.model_dump()
        rows.append(r)

    def agg(rs):
        if not rs:
            return {}
        return {"n": len(rs), "original@10": float(np.mean([r["original@10"] for r in rs])),
                "similar@10": float(np.mean([r["similar@10"] for r in rs])),
                "P@10": float(np.mean([r["P@10"] for r in rs])), "MRR": float(np.mean([r["RR"] for r in rs])),
                "brief_mood_recall": float(np.mean([r["mood_recall"] for r in rs])),
                "brief_genre_recall": float(np.mean([r["genre_recall"] for r in rs])),
                "relationship_acc": float(np.mean([r["relationship_ok"] for r in rs])),
                "energy_mae": float(np.mean([r["energy_err"] for r in rs])),
                "brief_valid": float(np.mean([r["valid"] for r in rs]))}

    res = {"variant": name, "all": agg(rows), "match": agg([r for r in rows if r["relationship"] == "match"]),
           "counterpoint": agg([r for r in rows if r["relationship"] == "counterpoint"]), "rows": rows}
    from needle_drop.scene import PROMPT_VERSION
    if kind == "llm":
        res["variant"] = name = f"{name}_{PROMPT_VERSION}"
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.array(xs, dtype=float)
    m = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(results: list[dict]) -> str:
    lines = ["| variant | similar@10 (95% CI) | P@10 | MRR | brief mood recall | brief genre recall | counterpoint acc | energy MAE | original@10 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        a = r["all"]
        lo, hi = bootstrap_ci([x["similar@10"] for x in r["rows"]])
        brief = "| – | – | – | – " if r["variant"].startswith("direct") else (
            f"| {a['brief_mood_recall']:.2f} | {a['brief_genre_recall']:.2f} | {a['relationship_acc']:.2f} | {a['energy_mae']:.2f} ")
        lines.append(f"| {r['variant']} | {a['similar@10']:.2f} ({lo:.2f}-{hi:.2f}) | {a['P@10']:.2f} | {a['MRR']:.2f} "
                     + brief + f"| {a['original@10']:.0f} |")
    lines += ["", "By relationship (similar@10):", "", "| variant | match | counterpoint |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r['variant']} | {r['match'].get('similar@10', 0):.2f} (n={r['match'].get('n', 0)}) | "
                     f"{r['counterpoint'].get('similar@10', 0):.2f} (n={r['counterpoint'].get('n', 0)}) |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="heuristic,llm_semantic,llm_hybrid")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    scenes = load_scenes()[: a.limit or None]
    print("LLM available:", llm.available())
    results = [run_variant(v, scenes) for v in a.variants.split(",")]
    md = report(results)
    (OUT / f"summary_{a.variants.replace(',', '+')}.md").write_text(md + "\n")
    print(md)
