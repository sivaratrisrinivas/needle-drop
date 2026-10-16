"""Phase 9: a durable job queue (SQLite) with worker threads, plus a result cache.

Video processing is slow on CPU, so requests are enqueued and processed by workers; the
client polls the job. Results are cached by content hash: the same clip (byte-identical),
scene text and use return instantly from cache, and the cache hit is still traced.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import traceback
import uuid

from . import config
from .tracing import span, trace

_lock = threading.Lock()


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY, kind TEXT, payload TEXT, status TEXT, result TEXT, error TEXT,
        cache_key TEXT, created REAL, started REAL, finished REAL, trace_id TEXT)""")
    con.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, result TEXT, created REAL, hits INT DEFAULT 0)")
    con.execute("""CREATE TABLE IF NOT EXISTS ratings(
        trace_id TEXT, track_id TEXT, rating INT, note TEXT, scene TEXT, created REAL,
        PRIMARY KEY(trace_id, track_id))""")
    return con


con = _con()


def _q(sql: str, args: tuple = ()) -> list:
    with _lock:
        cur = con.execute(sql, args)
        con.commit()
        return cur.fetchall()


def cache_key(kind: str, payload: dict, file_bytes: bytes | None = None) -> str:
    h = hashlib.sha256()
    h.update(kind.encode())
    h.update(json.dumps({k: v for k, v in payload.items() if k not in ("clip_path", "srt_path")},
                        sort_keys=True).encode())
    if file_bytes:
        h.update(hashlib.sha256(file_bytes).digest())
    return h.hexdigest()


def submit(kind: str, payload: dict, key: str) -> dict:
    jid = uuid.uuid4().hex[:12]
    tid = uuid.uuid4().hex[:16]
    _q("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
       (jid, kind, json.dumps(payload), "queued", None, None, key, time.time(), None, None, tid))
    return get(jid)


def get(jid: str) -> dict:
    rows = _q("SELECT id,kind,status,result,error,created,started,finished,trace_id,cache_key FROM jobs WHERE id=?", (jid,))
    if not rows:
        return {}
    r = rows[0]
    pos = _q("SELECT COUNT(*) FROM jobs WHERE status='queued' AND created < ?", (r[5],))[0][0] if r[2] == "queued" else 0
    return {"id": r[0], "kind": r[1], "status": r[2], "result": json.loads(r[3]) if r[3] else None, "error": r[4],
            "queued_at": r[5], "started_at": r[6], "finished_at": r[7], "trace_id": r[8], "queue_position": pos}


def _claim() -> tuple | None:
    with _lock:
        row = con.execute("SELECT id,kind,payload,cache_key,trace_id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if not row:
            return None
        con.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), row[0]))
        con.commit()
        return row


def _process(row: tuple) -> None:
    from . import agent

    jid, kind, payload, key, tid = row
    p = json.loads(payload)
    try:
        cached = _q("SELECT result FROM cache WHERE key=?", (key,))
        if cached:
            with trace(f"{kind}.cached", {"job": jid}, trace_id=tid):
                with span("cache.hit", key=key[:12]):
                    _q("UPDATE cache SET hits=hits+1 WHERE key=?", (key,))
            result = json.loads(cached[0][0])
            result["cache_hit"] = True
            result["trace_id"] = tid
        else:
            result = agent.run(text=p.get("text", ""), clip_path=p.get("clip_path"), srt_path=p.get("srt_path"),
                               use=p.get("use", "monetized"), k=int(p.get("k", 10)), trace_id=tid,
                               explain=p.get("explain", True))
            result["cache_hit"] = False
            _q("INSERT OR REPLACE INTO cache VALUES(?,?,?,0)", (key, json.dumps(result), time.time()))
        from .tracing import get_trace
        result["cost"] = get_trace(tid).get("totals", {})
        _q("UPDATE jobs SET status='done', result=?, finished=? WHERE id=?", (json.dumps(result), time.time(), jid))
    except Exception as e:  # job failure is recorded, worker keeps going
        _q("UPDATE jobs SET status='error', error=?, finished=? WHERE id=?",
           (f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", time.time(), jid))


def worker_loop(stop: threading.Event) -> None:
    while not stop.is_set():
        row = _claim()
        if row is None:
            stop.wait(0.3)
            continue
        _process(row)


def start_workers(n: int = config.WORKERS) -> threading.Event:
    # jobs left running by a crash go back to the queue
    _q("UPDATE jobs SET status='queued' WHERE status='running'")
    stop = threading.Event()
    for i in range(n):
        threading.Thread(target=worker_loop, args=(stop,), daemon=True, name=f"worker-{i}").start()
    return stop


def stats() -> dict:
    by = dict(_q("SELECT status, COUNT(*) FROM jobs GROUP BY status"))
    c = _q("SELECT COUNT(*), COALESCE(SUM(hits),0) FROM cache")[0]
    return {"jobs": by, "cache_entries": c[0], "cache_hits": c[1]}


def rate(trace_id: str, track_id: str, rating: int, note: str = "", scene: str = "") -> None:
    _q("INSERT OR REPLACE INTO ratings VALUES(?,?,?,?,?,?)", (trace_id, track_id, rating, note, scene, time.time()))


def ratings() -> list[dict]:
    return [dict(trace_id=r[0], track_id=r[1], rating=r[2], note=r[3], scene=r[4], created=r[5])
            for r in _q("SELECT * FROM ratings ORDER BY created")]
