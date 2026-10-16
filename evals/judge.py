"""LLM-as-judge for one failure mode: "the recommended track does not fit the scene".

Built with the write-judge-prompt skill (binary Pass/Fail, critique before verdict, few-shot
examples from the training split only) and calibrated with validate-evaluator
(train/dev/test splits, TPR/TNR, Rogan-Gladen corrected pass rate with a bootstrap CI).

usage:
  python evals/judge.py build-set            # sample (scene, track) pairs -> web/static/review_set.json
  python evals/judge.py validate LABELS.json # splits, dev + test TPR/TNR, corrected top-10 fit rate
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from needle_drop import llm  # noqa: E402
from needle_drop.catalog import load_catalog  # noqa: E402
from needle_drop.tracing import trace  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


class Verdict(BaseModel):
    critique: str
    result: Literal["Pass", "Fail"]


TASK = """You are evaluating a music-supervision tool. Given a film scene and ONE recommended track
(described by its catalog tags and energy), decide whether the track fits the scene.

## Definitions
PASS: an editor could plausibly cut this scene to this track. Its mood and energy work with what is on
screen, either by matching it (a sad farewell with a melancholic ballad) or by deliberate, effective
counterpoint (a breezy upbeat pop song over stylised violence).
FAIL: the track would feel wrong under the scene: its mood or energy clashes with no purpose (a calm
meditative track under a frantic car chase), or its genre is jarring for the setting.
Judge the fit only. Ignore licensing and track popularity.
Editors have no perfect track: the tags are a rough description, so a track that is close in mood or
energy, or that shares a mood with the scene, is a PASS. Choose FAIL only when the clash is clear
(opposite mood, or energy far off, or a jarring genre). Do not fail a track just because it is not the
ideal one.

Write a detailed critique that names the specific tags and scene details you relied on, then the result."""


def describe_pair(p: dict) -> str:
    return (f"Scene: {p['scene']}\nTrack: \"{p['title']}\" by {p['artist']}\n"
            f"Track moods: {', '.join(p['moods'])}\nTrack genres: {', '.join(p['genres'])}\n"
            f"Track instruments: {', '.join(p['instruments']) or 'unknown'}\nTrack energy (0-1): {p['energy']:.2f}")


def judge(p: dict, shots: list[tuple[dict, str, str]]) -> Verdict:
    ex = "\n\n".join(f"### Example {i + 1}: {lab.upper()}\n{describe_pair(sp)}\nCritique: {crit}\nResult: {lab}"
                     for i, (sp, lab, crit) in enumerate(shots))
    msgs = [{"role": "system", "content": TASK + ("\n\n## Examples\n" + ex if ex else "")},
            {"role": "user", "content": describe_pair(p) + "\n\nReturn JSON with critique then result."}]
    return llm.structured(Verdict, msgs, name="judge_fit", max_tokens=220, temperature=0.0)


def build_set(variant: str = "llm_hybrid_v2", ranks=(1, 6), seed: int = 3) -> list[dict]:
    res = json.loads((RES / f"{variant}.json").read_text())
    scenes = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "scenes.jsonl")}
    cat = {t.id: t for t in load_catalog()}
    out = []
    for row in res["rows"]:
        s = scenes[row["id"]]
        for r in ranks:
            if r > len(row["top"]):
                continue
            tid = row["top"][r - 1][0]
            t = cat[tid]
            out.append({"pair_id": f"{s['id']}#{r}", "scene_id": s["id"], "film": s["film"], "year": s["year"],
                        "scene": s["scene"], "song": s["song"], "song_artist": s["artist"], "rank": r, "variant": variant,
                        "track_id": tid, "title": t.title, "artist": t.artist, "moods": t.moods, "genres": t.genres,
                        "instruments": t.instruments, "energy": t.energy, "page_url": t.page_url, "audio_url": t.audio_url,
                        "brief_moods": row["profile"]["moods"], "brief_genres": row["profile"]["genres"],
                        "brief_energy": row["profile"]["energy"]})
    random.Random(seed).shuffle(out)
    Path("web/static").mkdir(parents=True, exist_ok=True)
    Path("web/static/review_set.json").write_text(json.dumps(out, indent=1))
    return out


def rates(h, e):
    h, e = np.array(h), np.array(e)
    tp = ((h == "Pass") & (e == "Pass")).sum(); fn = ((h == "Pass") & (e == "Fail")).sum()
    tn = ((h == "Fail") & (e == "Fail")).sum(); fp = ((h == "Fail") & (e == "Pass")).sum()
    return (tp / (tp + fn) if tp + fn else float("nan")), (tn / (tn + fp) if tn + fp else float("nan"))


def rogan_gladen(p_obs, tpr, tnr):
    d = tpr + tnr - 1
    return float(np.clip((p_obs + tnr - 1) / d, 0, 1)) if abs(d) > 1e-6 else float("nan")


def bootstrap(h, e, p_obs, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    h, e = np.array(h), np.array(e)
    est = []
    for _ in range(n):
        idx = rng.integers(0, len(h), len(h))
        tpr, tnr = rates(h[idx], e[idx])
        if np.isnan(tpr) or np.isnan(tnr) or abs(tpr + tnr - 1) < 1e-6:
            continue
        est.append(rogan_gladen(p_obs, tpr, tnr))
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def validate(labels_path: str) -> dict:
    pairs = {p["pair_id"]: p for p in json.loads(Path("web/static/review_set.json").read_text())}
    labels = [l for l in json.loads(Path(labels_path).read_text()) if l.get("label") in ("Pass", "Fail")]
    rng = random.Random(42)
    by = {"Pass": [l for l in labels if l["label"] == "Pass"], "Fail": [l for l in labels if l["label"] == "Fail"]}
    split = {"train": [], "dev": [], "test": []}
    for lab, ls in by.items():  # stratified 15/45/40
        rng.shuffle(ls)
        a, b = round(len(ls) * .15), round(len(ls) * .60)
        split["train"] += ls[:a]; split["dev"] += ls[a:b]; split["test"] += ls[b:]
    shots = []
    for lab in ("Pass", "Fail"):
        for l in [x for x in split["train"] if x["label"] == lab and x.get("note")][:2]:
            shots.append((pairs[l["pair_id"]], lab, l["note"]))
    out = {"labels": len(labels), "pass": len(by["Pass"]), "fail": len(by["Fail"]),
           "split_sizes": {k: len(v) for k, v in split.items()}, "few_shot": [s[0]["pair_id"] for s in shots]}
    for name in ("dev", "test"):
        h, e, rows = [], [], []
        for l in split[name]:
            with trace("eval.judge", {"pair": l["pair_id"], "split": name}):
                try:
                    v = judge(pairs[l["pair_id"]], shots)
                except llm.LLMError:
                    continue
            h.append(l["label"]); e.append(v.result)
            rows.append({"pair_id": l["pair_id"], "human": l["label"], "judge": v.result, "critique": v.critique})
            print(name, l["pair_id"], l["label"], v.result, flush=True)
        tpr, tnr = rates(h, e)
        out[name] = {"n": len(h), "TPR": tpr, "TNR": tnr,
                     "disagreements": [r for r in rows if r["human"] != r["judge"]]}
        if name == "test":
            out["_test_h"], out["_test_e"] = h, e
    # production estimate: judge every top-10 pair of the main variant that was not labeled
    res = json.loads((RES / "llm_hybrid_v2.json").read_text())
    scenes = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "scenes.jsonl")}
    cat = {t.id: t for t in load_catalog()}
    labeled = {l["pair_id"] for l in labels}
    prod = []
    for row in res["rows"]:
        for r, top in enumerate(row["top"][:10], 1):
            pid = f"{row['id']}#{r}"
            if pid in labeled or r != 4:   # a fixed, unlabeled sample of top-10 slots
                continue
            t = cat[top[0]]
            p = {"scene": scenes[row["id"]]["scene"], "title": t.title, "artist": t.artist, "moods": t.moods,
                 "genres": t.genres, "instruments": t.instruments, "energy": t.energy}
            with trace("eval.judge", {"pair": pid, "split": "unlabeled"}):
                try:
                    prod.append(judge(p, shots).result)
                except llm.LLMError:
                    pass
    p_obs = float(np.mean([x == "Pass" for x in prod])) if prod else float("nan")
    tpr, tnr = out["test"]["TPR"], out["test"]["TNR"]
    lo, hi = bootstrap(out.pop("_test_h"), out.pop("_test_e"), p_obs)
    out["unlabeled"] = {"n": len(prod), "raw_pass_rate": p_obs, "corrected_pass_rate": rogan_gladen(p_obs, tpr, tnr),
                        "ci95": [lo, hi]}
    (RES / "judge_validation.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build-set":
        print(len(build_set()), "pairs -> web/static/review_set.json")
    elif cmd == "validate":
        r = validate(sys.argv[2])
        print(json.dumps({k: v for k, v in r.items() if k not in ("dev", "test")}, indent=1, default=float))
        for s in ("dev", "test"):
            print(s, {k: v for k, v in r[s].items() if k != "disagreements"})
