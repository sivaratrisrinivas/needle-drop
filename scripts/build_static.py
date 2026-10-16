"""Build the static (GitHub Pages) assets: compact catalog + license rules, and the showcase
of precomputed backend results for the example scenes and sample clips."""
import base64
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from needle_drop.catalog import load_catalog  # noqa: E402
from needle_drop.licensing import RULES, _cost, parse_license_url  # noqa: E402
from needle_drop.catalog import LICENSE_SOURCE  # noqa: E402

OUT = Path("web/static")
OUT.mkdir(parents=True, exist_ok=True)


def lite():
    tracks = []
    for t in load_catalog():
        lid, _ = parse_license_url(t.license_url) if t.license_source else (None, None)
        tracks.append([t.id, t.title, t.artist, t.genres, t.moods, t.energy, lid])
    mc = collections.Counter(m for t in load_catalog() for m in t.moods)
    gc = collections.Counter(g for t in load_catalog() for g in t.genres)
    rules = {lid: {u: list(v) for u, v in r.items()} for lid, r in RULES.items()}
    costs = {"cleared": _cost("cleared", "cc-by")["basis"], "needs_review": _cost("needs_review", "cc-by-sa")["basis"],
             "not_cleared": _cost("not_cleared", "cc-by-nc")["basis"], "nd": _cost("not_cleared", "cc-by-nd")["basis"],
             "unverified": _cost("unverified", None)["basis"]}
    d = {"tracks": tracks, "moods": sorted(mc), "genres": sorted(gc),
         "top_moods": [m for m, _ in mc.most_common(36)], "top_genres": [g for g, _ in gc.most_common(30)],
         "rules": rules, "costs": costs, "license_source": LICENSE_SOURCE}
    (OUT / "catalog-lite.json").write_text(json.dumps(d, separators=(",", ":"), ensure_ascii=False))
    print("catalog-lite", len(tracks))


def inline_keyframes(res):
    a = res.get("analysis")
    if a:
        a["keyframes"] = ["data:image/jpeg;base64," + base64.b64encode(Path(k).read_bytes()).decode()
                          for k in a.get("keyframes", []) if Path(k).exists()]
    return res


def showcase():
    from needle_drop import agent
    from needle_drop.tracing import get_trace
    examples = json.loads(Path("scripts/showcase_inputs.json").read_text())
    out = {"scenes": [], "clips": {}}
    for text in examples["scenes"]:
        r = agent.run(text=text, use="monetized")
        r["cost"] = get_trace(r["trace_id"])["totals"]
        out["scenes"].append({"text": text, "result": r})
        print("scene done", text[:40], flush=True)
    for name in examples["clips"]:
        r = agent.run(clip_path=f"web/samples/{name}.mp4", srt_path=f"web/samples/{name}.srt", use="monetized")
        r["cost"] = get_trace(r["trace_id"])["totals"]
        out["clips"][name] = inline_keyframes(r)
        print("clip done", name, flush=True)
    (OUT / "showcase.json").write_text(json.dumps(out, default=str))


if __name__ == "__main__":
    lite()
    if "--showcase" in sys.argv:
        showcase()
