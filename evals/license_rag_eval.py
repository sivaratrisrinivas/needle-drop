"""Phase 4 retrieval eval (evaluate-rag skill): hand-written license questions mapped to the
knowledge-base sections that answer them. Reports Recall@k and MRR for the retriever alone;
generation is not graded here because verdicts are rule-based, not generated."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from needle_drop.licensing import retrieve  # noqa: E402

HERE = Path(__file__).resolve().parent
qs = [json.loads(l) for l in open(HERE / "license_questions.jsonl")]
res = {}
for k in (1, 3, 5):
    rec, rr, rows = [], [], []
    for q in qs:
        rel = {tuple(r) for r in q["relevant"]}
        got = [(c.doc_id, c.section) for c, _ in retrieve(q["q"], k=k)]
        rec.append(len(rel & set(got)) / len(rel))
        first = next((i for i, g in enumerate(got) if g in rel), None)
        rr.append(0 if first is None else 1 / (first + 1))
        rows.append({"q": q["q"], "got": got, "hit": first is not None})
    res[f"recall@{k}"] = sum(rec) / len(rec)
    res[f"hit@{k}"] = sum(r["hit"] for r in rows) / len(rows)
    res[f"mrr@{k}"] = sum(rr) / len(rr)
    if k == 3:
        res["misses@3"] = [r for r in rows if not r["hit"]]
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "license_rag.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if not k.startswith("misses")}, indent=1))
for m in res["misses@3"]:
    print("MISS", m["q"], m["got"])
