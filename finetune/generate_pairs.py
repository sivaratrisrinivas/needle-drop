"""Phase 7, step 1: synthetic (scene description -> track) training pairs.

Following the dimension/tuple recipe (generate-synthetic-data skill): each pair is a tuple of
  track      a catalog track, sampled stratified by its primary mood (the target)
  film genre thriller, romance, comedy, drama, horror, sci-fi, documentary, coming-of-age, western, ad
  setting    city at night, countryside, interior, vehicle, crowd/event, nature, workplace, home
then, in a separate LLM call, the tuple is turned into a scene description that never mentions
music. Eval scenes (evals/scenes.jsonl) are never used here.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pydantic import BaseModel  # noqa: E402

from needle_drop import llm  # noqa: E402
from needle_drop.catalog import load_catalog  # noqa: E402
from needle_drop.tracing import trace  # noqa: E402

OUT = Path(__file__).resolve().parent / "pairs.jsonl"
FILM = ["thriller", "romance", "comedy", "drama", "horror", "sci-fi", "documentary", "coming-of-age", "western", "commercial"]
SETTING = ["a city at night", "the countryside", "an interior room", "a moving vehicle", "a crowd or party",
           "the wilderness", "a workplace", "a family home", "a beach or coast", "a rooftop"]
BANNED = ("music", "song", "track", "soundtrack", "melody", "beat", "tune")


class Scene(BaseModel):
    scene: str


def sample_tracks(n: int, seed: int = 7) -> list:
    rng = random.Random(seed)
    by_mood: dict[str, list] = {}
    for t in load_catalog():
        if t.genres and t.license_source:
            by_mood.setdefault(t.moods[0], []).append(t)
    moods = [m for m, ts in by_mood.items() if len(ts) >= 15]
    out = []
    while len(out) < n:
        m = moods[len(out) % len(moods)]
        out.append(rng.choice(by_mood[m]))
    return out


def main(n: int = 400) -> None:
    rng = random.Random(11)
    done = {json.loads(l)["track_id"] for l in open(OUT)} if OUT.exists() else set()
    with open(OUT, "a") as f:
        for t in sample_tracks(n):
            if t.id in done:
                continue
            film, setting = rng.choice(FILM), rng.choice(SETTING)
            prompt = (f"Write one or two sentences describing a {film} film scene set in {setting}. Describe only what "
                      f"happens on screen and how it feels. The scene should feel {', '.join(t.moods[:3])}, with "
                      f"{'high' if t.energy > .65 else 'low' if t.energy < .35 else 'moderate'} energy, the kind of moment an "
                      f"editor would score with {'/'.join(t.genres[:2])} music. Never mention music, songs or sound. JSON only.")
            t0 = time.time()
            try:
                with trace("finetune.generate", {"track": t.id}):
                    s = llm.structured(Scene, [{"role": "user", "content": prompt}], name="synth_scene",
                                       max_tokens=90, temperature=0.9)
            except llm.LLMError:
                continue
            text = s.scene.strip()
            if len(text) < 40 or any(b in text.lower() for b in BANNED):
                continue  # quality filter
            f.write(json.dumps({"track_id": t.id, "film": film, "setting": setting, "scene": text}) + "\n")
            f.flush()
            done.add(t.id)
            print(len(done), f"{time.time() - t0:.1f}s", text[:90], flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
