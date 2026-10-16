"""Download a GGUF model (and its multimodal projector, if any) from Docker Hub's ai/ namespace.

Docker publishes open-weight models as OCI artifacts (https://hub.docker.com/u/ai); this pulls
them anonymously with plain HTTPS, no Docker daemon needed.
usage: python scripts/pull_docker_model.py ai/qwen2.5 3B-Q4_K_M models/qwen2.5-3b-q4km.gguf [mmproj_out]
"""
import json
import sys
import urllib.request

repo, tag, out = sys.argv[1:4]
mmproj_out = sys.argv[4] if len(sys.argv) > 4 else None
tok = json.load(urllib.request.urlopen(
    f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"))["token"]


def get(url, accept="*/*"):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer " + tok, "Accept": accept}))


manifest = json.load(get(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}",
                         "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"))
for layer in manifest["layers"]:
    mt = layer["mediaType"]
    dest = out if mt.endswith("gguf.v3") else mmproj_out if mt.endswith("mmproj") else None
    if not dest:
        continue
    print("downloading", mt, layer["size"] // 2**20, "MB ->", dest, flush=True)
    with get(f"https://registry-1.docker.io/v2/{repo}/blobs/{layer['digest']}") as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
