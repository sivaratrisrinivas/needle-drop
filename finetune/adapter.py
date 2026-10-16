"""Phase 7, step 2: train a small adapter that maps scene embeddings into the track space.

Model: a linear map W (256x256, initialised to identity) applied to the nomic query embedding of
the raw scene text. Retrieval scores cos(W q, d) against the frozen track embeddings d, so the
adapter learns to translate "what's on screen" language into "what the track sounds like" language
without any LLM call at query time.

Target space: the track embeddings we have (vibe documents built from Jamendo's audio tags). The
code takes any target matrix, so real audio embeddings (e.g. CLAP over the Jamendo audio) can be
swapped in when the audio is available; we could not download audio in the build environment.

Loss: InfoNCE over in-batch tracks with soft labels (tracks that share most tags with the positive
are partially correct, so they are not punished as hard negatives), plus an L2 pull to identity.
Pure numpy; trains in seconds on CPU. Defaults (lr 0.003, reg 0.5, 30 epochs) were picked by the
5-fold sweep in sweep.py (sweep_report.json).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from needle_drop.catalog import load_catalog  # noqa: E402
from needle_drop.embeddings import get_embedder  # noqa: E402
from needle_drop.search import get_index  # noqa: E402

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "adapter_W.npy"


def _norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def load_pairs():
    idx = get_index()
    pos = {t.id: i for i, t in enumerate(idx.tracks)}
    pairs = [json.loads(l) for l in open(HERE / "pairs.jsonl")]
    pairs = [p for p in pairs if p["track_id"] in pos]
    emb = get_embedder()
    Q = np.stack([emb.query(p["scene"]) for p in pairs])
    ti = np.array([pos[p["track_id"]] for p in pairs])
    return pairs, Q, ti


def train(Q, ti, epochs=30, lr=0.003, tau=20.0, reg=0.5, batch=64, seed=0, val=None):
    idx = get_index()
    D = idx.m
    tags = [set(t.moods) | set(t.genres) for t in idx.tracks]
    dim = Q.shape[1]
    W = np.eye(dim, dtype=np.float32)
    m, v = np.zeros_like(W), np.zeros_like(W)
    rng = np.random.default_rng(seed)
    hist = []
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(Q))
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            q, d = Q[b], D[ti[b]]
            # soft labels from tag overlap between batch tracks
            Y = np.array([[jaccard(tags[ti[i]], tags[ti[j]]) for j in b] for i in b], dtype=np.float32)
            Y = np.where(np.eye(len(b)) > 0, 1.0, np.where(Y >= 0.6, Y * 0.5, 0.0))
            Y /= Y.sum(1, keepdims=True)
            z = q @ W.T                                  # (B, dim)
            zn = _norm(z)
            logits = tau * zn @ d.T                      # (B, B)
            logits -= logits.max(1, keepdims=True)
            P = np.exp(logits); P /= P.sum(1, keepdims=True)
            G = tau * (P - Y) @ d / len(b)               # dL/dzn
            nz = np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
            Gz = (G - zn * (G * zn).sum(1, keepdims=True)) / nz   # back through normalisation
            gW = Gz.T @ q + reg * (W - np.eye(dim))
            step += 1
            m = 0.9 * m + 0.1 * gW; v = 0.999 * v + 0.001 * gW ** 2
            W -= lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        if val is not None:
            hist.append((ep, recall_at(val[0], val[1], W)))
    return W, hist


def recall_at(Q, ti, W, k=10) -> float:
    """Fraction of held-out pairs with a "similar" track in the top k: the same criterion as the
    phase-6 eval (shares >=1 genre and >=1 mood with the source track, energy within 0.25).
    The exact source track among 22k is almost never in the top 10 for either model."""
    idx = get_index()
    S = _norm(Q @ W.T) @ idx.m.T
    top = np.argsort(-S, axis=1)[:, :k]
    T = idx.tracks

    def sim(a, b):
        return bool(set(a.genres) & set(b.genres)) and bool(set(a.moods) & set(b.moods)) and abs(a.energy - b.energy) <= .25
    ok = [any(j == ti[i] or sim(T[j], T[ti[i]]) for j in top[i]) for i in range(len(Q))]
    return float(np.mean(ok))


class Adapter:
    def __init__(self, W: np.ndarray):
        self.W = W

    def query_vector_text(self, text: str) -> np.ndarray:
        return _norm(self.W @ get_embedder().query(text))


@lru_cache(maxsize=1)
def load_adapter() -> Adapter:
    return Adapter(np.load(WEIGHTS))


if __name__ == "__main__":
    pairs, Q, ti = load_pairs()
    n = len(pairs)
    rng = np.random.default_rng(1)
    perm = rng.permutation(n)
    cut = int(n * 0.8)
    tr, va = perm[:cut], perm[cut:]
    base = recall_at(Q[va], ti[va], np.eye(Q.shape[1], dtype=np.float32))
    W, hist = train(Q[tr], ti[tr], val=(Q[va], ti[va]))
    best = max(hist, key=lambda x: x[1])
    print(f"pairs={n} train={len(tr)} val={len(va)}")
    print(f"held-out similar@10: identity (off-the-shelf) {base:.3f} -> adapter {hist[-1][1]:.3f} (best epoch {best[0]}: {best[1]:.3f})")
    # final model on all pairs, same number of epochs (not the best epoch, which would leak val)
    W_all, _ = train(Q, ti)
    np.save(WEIGHTS, W_all)
    (HERE / "train_report.json").write_text(json.dumps(
        {"pairs": n, "train": len(tr), "val": len(va), "val_similar@10_identity": base,
         "val_similar@10_adapter": hist[-1][1], "curve": hist}, indent=1))
    print("saved", WEIGHTS)
