"""5-fold cross-validated sweep of adapter hyperparameters (similar@10 on held-out pairs)."""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finetune.adapter import load_pairs, train, recall_at  # noqa: E402

pairs, Q, ti = load_pairs()
folds = np.array_split(np.random.default_rng(0).permutation(len(Q)), 5)
I = np.eye(Q.shape[1], dtype=np.float32)
base = np.mean([recall_at(Q[f], ti[f], I) for f in folds])
print(f"identity {base:.3f}", flush=True)
rows = [{"cfg": "identity", "cv_similar@10": base}]
for lr, reg, ep in itertools.product((0.003, 0.01), (0.5, 2.0), (10, 30)):
    sc = []
    for k, f in enumerate(folds):
        tr = np.concatenate([g for j, g in enumerate(folds) if j != k])
        W, _ = train(Q[tr], ti[tr], epochs=ep, lr=lr, reg=reg)
        sc.append(recall_at(Q[f], ti[f], W))
    rows.append({"cfg": f"lr={lr} reg={reg} epochs={ep}", "lr": lr, "reg": reg, "epochs": ep, "cv_similar@10": float(np.mean(sc))})
    print(rows[-1], flush=True)
Path(__file__).with_name("sweep_report.json").write_text(json.dumps(rows, indent=1))
