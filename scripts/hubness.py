"""Hubness correction for vector search (CSLS, Conneau et al. 2018).

Short, generic vibe documents ("Happy sunset. happy, summer mood; pop music") sit close to
almost every query in embedding space, so they surface in the top 10 for unrelated scenes
(found in error analysis, F6). For each track we store r(t) = mean cosine to its 10 nearest
"probe" queries; search subtracts it: csls(q, t) = 2 cos(q, t) - r(t).
Probes are generic tag queries ("search_query: <tag> music") for every catalog tag, so no eval
scene is used.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from needle_drop import config  # noqa: E402
from needle_drop.embeddings import get_embedder  # noqa: E402
from needle_drop.scene import vocab  # noqa: E402
from needle_drop.search import get_index  # noqa: E402

v = vocab()
probes = [f"{t} music" for k in ("moods", "genres", "instruments") for t in v[k]]
emb = get_embedder()
P = np.stack([emb.query(p) for p in probes])
S = get_index().m @ P.T                       # tracks x probes
r = np.sort(S, axis=1)[:, -10:].mean(1)
np.save(config.DATA / "track_hub.npy", r.astype(np.float32))
idx = np.argsort(-r)[:10]
print("probes", len(probes), "hub score mean", r.mean(), "top hubs:", [get_index().tracks[i].title for i in idx])
