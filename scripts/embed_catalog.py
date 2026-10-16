"""Embed every catalog track's vibe description. Writes data/catalog_vectors.npy (float16)."""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from needle_drop import config  # noqa: E402
from needle_drop.catalog import load_catalog  # noqa: E402
from needle_drop.embeddings import get_embedder  # noqa: E402

tracks = load_catalog()
emb = get_embedder()
out = np.zeros((len(tracks), emb.dim), dtype=np.float32)
t0 = time.time()
for i, t in enumerate(tracks):
    out[i] = emb.documents([t.describe()])[0]
    if i % 1000 == 0:
        print(i, f"{time.time() - t0:.0f}s", flush=True)
np.save(config.VECTORS_PATH, out.astype(np.float16))
print("saved", out.shape, config.VECTORS_PATH)
