"""FastAPI backend: queue-backed search + clip upload, traces, spend, ratings, static UI."""
from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, jobs, llm, tracing
from .catalog import load_catalog

UPLOADS = config.CACHE / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = 200 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop = jobs.start_workers()
    yield
    stop.set()


app = FastAPI(title="Needle Drop", lifespan=lifespan)


class SearchReq(BaseModel):
    text: str
    use: str = "monetized"
    k: int = 10
    explain: bool = True


def _check_use(use: str) -> None:
    if use not in ("personal", "monetized", "client"):
        raise HTTPException(400, "use must be personal, monetized or client")


@app.get("/api/health")
def health():
    return {"ok": True, "llm": llm.available(), "llm_model": config.LLM_MODEL, "embed_model": "nomic-embed-text-v1.5",
            "catalog_tracks": len(load_catalog()), **jobs.stats()}


@app.post("/api/search")
def post_search(req: SearchReq):
    _check_use(req.use)
    if not req.text.strip():
        raise HTTPException(400, "text is required")
    payload = req.model_dump()
    return jobs.submit("search", payload, jobs.cache_key("search", payload))


@app.post("/api/clips")
async def post_clip(file: UploadFile = File(...), srt: UploadFile | None = File(None), text: str = Form(""),
                    use: str = Form("monetized"), k: int = Form(10)):
    _check_use(use)
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "clip too large (max 200 MB); trim it to ~30 s")
    d = UPLOADS / uuid.uuid4().hex[:12]
    d.mkdir()
    clip = d / ("clip" + Path(file.filename or "clip.mp4").suffix.lower()[:6])
    clip.write_bytes(data)
    payload = {"text": text, "use": use, "k": k, "clip_path": str(clip), "explain": True}
    srt_bytes = b""
    if srt is not None and srt.filename:
        srt_bytes = await srt.read()
        (d / "sub.srt").write_bytes(srt_bytes)
        payload["srt_path"] = str(d / "sub.srt")
    return jobs.submit("clip", payload, jobs.cache_key("clip", payload, data + srt_bytes))


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "no such job")
    return j


@app.get("/api/media")
def media(path: str):
    p = Path(path).resolve()
    if not str(p).startswith(str(config.CACHE.resolve())) or not p.exists() or p.suffix != ".jpg":
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/traces")
def traces(limit: int = 50):
    return tracing.list_traces(limit)


@app.get("/api/traces/{tid}")
def trace_detail(tid: str):
    t = tracing.get_trace(tid)
    if not t:
        raise HTTPException(404)
    return t


@app.get("/api/spend")
def spend():
    return {**tracing.spend_summary(), **jobs.stats()}


class Rating(BaseModel):
    trace_id: str
    track_id: str
    rating: int
    note: str = ""
    scene: str = ""


@app.post("/api/ratings")
def post_rating(r: Rating):
    if not 1 <= r.rating <= 5:
        raise HTTPException(400, "rating 1-5")
    jobs.rate(r.trace_id, r.track_id, r.rating, r.note, r.scene)
    return {"ok": True}


@app.get("/api/ratings")
def get_ratings():
    return jobs.ratings()


WEB = config.ROOT / "web"
if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


def cleanup_uploads(max_age_s: float = 86400) -> None:
    import time
    for d in UPLOADS.iterdir():
        if time.time() - d.stat().st_mtime > max_age_s:
            shutil.rmtree(d, ignore_errors=True)
