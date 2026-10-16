"""Minimal tracing + spend tracking, stored in SQLite.

Every request gets a trace; every LLM call, tool call and pipeline step is a span with
duration, attributes, token counts and cost. Inspect them at /api/traces or in the UI.
Swap in OpenTelemetry later by replacing `span()`; the call sites stay the same.
"""
from __future__ import annotations

import contextvars
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

from . import config

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_parent: contextvars.ContextVar[str | None] = contextvars.ContextVar("parent", default=None)
_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS spans(
        id TEXT PRIMARY KEY, trace_id TEXT, parent_id TEXT, name TEXT,
        start REAL, end REAL, attrs TEXT, error TEXT,
        prompt_tokens INT DEFAULT 0, completion_tokens INT DEFAULT 0, cost_usd REAL DEFAULT 0)""")
    con.execute("""CREATE TABLE IF NOT EXISTS traces(
        id TEXT PRIMARY KEY, name TEXT, start REAL, end REAL, input TEXT, status TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS spans_trace ON spans(trace_id)")
    return con


_con = _db()


def _exec(sql: str, args: tuple) -> None:
    with _lock:
        _con.execute(sql, args)
        _con.commit()


def current_trace() -> str | None:
    return _trace_id.get()


@contextmanager
def trace(name: str, input: dict | None = None, trace_id: str | None = None):
    tid = trace_id or uuid.uuid4().hex[:16]
    tok = _trace_id.set(tid)
    ptok = _parent.set(None)
    _exec("INSERT OR REPLACE INTO traces VALUES(?,?,?,?,?,?)",
          (tid, name, time.time(), None, json.dumps(input or {})[:4000], "running"))
    status = "ok"
    try:
        yield tid
    except Exception:
        status = "error"
        raise
    finally:
        _exec("UPDATE traces SET end=?, status=? WHERE id=?", (time.time(), status, tid))
        _trace_id.reset(tok)
        _parent.reset(ptok)


class Span:
    def __init__(self, sid: str):
        self.id = sid
        self.attrs: dict = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0

    def set(self, **kw):
        self.attrs.update(kw)

    def usage(self, prompt: int, completion: int, model: str):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        pin, pout = config.PRICES.get(model, config.PRICES["local"])
        self.cost += (prompt * pin + completion * pout) / 1e6


@contextmanager
def span(name: str, **attrs):
    sid = uuid.uuid4().hex[:12]
    s = Span(sid)
    s.attrs.update(attrs)
    parent = _parent.get()
    tok = _parent.set(sid)
    start = time.time()
    err = None
    try:
        yield s
    except Exception as e:  # recorded, then re-raised
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        _parent.reset(tok)
        tid = _trace_id.get()
        if tid:
            _exec("INSERT INTO spans VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (sid, tid, parent, name, start, time.time(), json.dumps(s.attrs, default=str)[:20000],
                   err, s.prompt_tokens, s.completion_tokens, s.cost))


def get_trace(tid: str) -> dict:
    with _lock:
        t = _con.execute("SELECT id,name,start,end,input,status FROM traces WHERE id=?", (tid,)).fetchone()
        rows = _con.execute("SELECT id,parent_id,name,start,end,attrs,error,prompt_tokens,completion_tokens,cost_usd "
                            "FROM spans WHERE trace_id=? ORDER BY start", (tid,)).fetchall()
    if not t:
        return {}
    spans = [dict(id=r[0], parent=r[1], name=r[2], start=r[3], ms=round((r[4] - r[3]) * 1000),
                  attrs=json.loads(r[5] or "{}"), error=r[6], prompt_tokens=r[7], completion_tokens=r[8],
                  cost_usd=r[9]) for r in rows]
    return dict(id=t[0], name=t[1], start=t[2], ms=round(((t[3] or time.time()) - t[2]) * 1000),
                input=json.loads(t[4] or "{}"), status=t[5], spans=spans, totals=_totals(spans))


def _totals(spans: list[dict]) -> dict:
    pt = sum(s["prompt_tokens"] for s in spans)
    ct = sum(s["completion_tokens"] for s in spans)
    sin, sout = config.SHADOW_PRICE
    return dict(prompt_tokens=pt, completion_tokens=ct,
                llm_calls=sum(1 for s in spans if s["name"].startswith("llm.")),
                cost_usd=round(sum(s["cost_usd"] for s in spans), 6),
                shadow_cost_usd=round((pt * sin + ct * sout) / 1e6, 6))


def list_traces(limit: int = 50) -> list[dict]:
    with _lock:
        rows = _con.execute(
            "SELECT t.id,t.name,t.start,t.end,t.status,COALESCE(SUM(s.prompt_tokens),0),"
            "COALESCE(SUM(s.completion_tokens),0),COALESCE(SUM(s.cost_usd),0),COUNT(s.id) "
            "FROM traces t LEFT JOIN spans s ON s.trace_id=t.id GROUP BY t.id ORDER BY t.start DESC LIMIT ?",
            (limit,)).fetchall()
    return [dict(id=r[0], name=r[1], start=r[2], ms=round(((r[3] or r[2]) - r[2]) * 1000), status=r[4],
                 prompt_tokens=r[5], completion_tokens=r[6], cost_usd=round(r[7], 6), spans=r[8]) for r in rows]


def spend_summary() -> dict:
    with _lock:
        r = _con.execute("SELECT COUNT(DISTINCT trace_id), COALESCE(SUM(prompt_tokens),0), "
                         "COALESCE(SUM(completion_tokens),0), COALESCE(SUM(cost_usd),0) FROM spans").fetchone()
    sin, sout = config.SHADOW_PRICE
    return dict(requests=r[0], prompt_tokens=r[1], completion_tokens=r[2], cost_usd=round(r[3], 6),
                shadow_cost_usd=round((r[1] * sin + r[2] * sout) / 1e6, 6),
                shadow_price_per_mtok=dict(input=sin, output=sout))
