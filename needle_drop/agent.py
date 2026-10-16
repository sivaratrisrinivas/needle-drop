"""Phase 5: the music-supervisor agent.

The agent owns a small tool belt and decides, step by step, what to call next:
  analyze_clip     multimodal clip analysis (clip.py)
  parse_scene      scene/clip description -> structured music brief (scene.py)
  search_catalog   hybrid vibe search; can be restricted to tracks whose license allows the use
  check_licenses   rule-based license verdicts with retrieved citations (licensing.py)
  finish           rank, explain and return the shortlist
The LLM picks each action as constrained JSON. A 3B local model sometimes picks nonsense, so
every choice is validated against what is legal in the current state; an invalid choice is
replaced by the default next step and recorded in the trace.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from . import guardrails, llm
from .catalog import load_catalog
from .clip import ClipAnalysis, analyze_clip
from .licensing import USE_LABEL, check_license, parse_license_url, RULES
from .scene import SceneProfile, parse_scene
from .search import Hit, search
from .tracing import span, trace


class Action(BaseModel):
    thought: str = Field(description="One short sentence of reasoning.")
    tool: Literal["analyze_clip", "parse_scene", "search_catalog", "check_licenses", "finish"]
    only_cleared: bool = Field(False, description="search_catalog: restrict to tracks whose license allows the use")


@dataclass
class State:
    text: str
    clip_path: str | None
    srt_path: str | None
    use: str
    k: int
    min_cleared: int
    analysis: ClipAnalysis | None = None
    profile: SceneProfile | None = None
    hits: list[Hit] = field(default_factory=list)
    verdicts: dict = field(default_factory=dict)
    searches: int = 0
    steps: list[dict] = field(default_factory=list)

    def legal(self) -> list[str]:
        if self.clip_path and self.analysis is None:
            return ["analyze_clip"]
        if self.profile is None:
            return ["parse_scene"]
        if not self.hits:
            return ["search_catalog"]
        if len(self.verdicts) < len(self.hits):
            return ["check_licenses"]
        out = ["finish"]
        if self.searches < 2:
            out.insert(0, "search_catalog")
        return out

    def default(self) -> Action:
        legal = self.legal()
        if legal == ["search_catalog", "finish"]:
            enough = sum(v.cleared for v in self.verdicts.values()) >= self.min_cleared
            return Action(thought="default policy", tool="finish" if enough else "search_catalog",
                          only_cleared=not enough)
        return Action(thought="default policy", tool=legal[0])

    def observation(self) -> str:
        o = {"use": USE_LABEL[self.use], "have_clip": bool(self.clip_path),
             "clip_analyzed": self.analysis is not None, "brief_ready": self.profile is not None,
             "searches_done": self.searches, "candidates": len(self.hits),
             "cleared_candidates": sum(v.cleared for v in self.verdicts.values()),
             "min_cleared_wanted": self.min_cleared, "legal_next_tools": self.legal()}
        return json.dumps(o)


PLANNER = """You are the planning loop of a music-supervision agent. Choose the next tool.
Goal: a ranked shortlist of tracks that fit the scene, with at least `min_cleared_wanted` tracks
whose license is cleared for the intended use. If a search returned too few cleared tracks,
search again with only_cleared=true. Only choose from legal_next_tools. Reply with JSON."""


def _cleared_ids(use: str) -> set[str]:
    ok = {lid for lid, rules in RULES.items() if rules[use][0] == "cleared"}
    return {t.id for t in load_catalog() if t.license_source and parse_license_url(t.license_url)[0] in ok}


def _plan(st: State, use_llm: bool) -> Action:
    legal = st.legal()
    if len(legal) == 1 or not use_llm:
        return st.default() if len(legal) > 1 else Action(thought="only legal step", tool=legal[0])
    try:
        a = llm.structured(Action, [{"role": "system", "content": PLANNER},
                                    {"role": "user", "content": st.observation()}], name="agent_plan", max_tokens=80)
        if a.tool not in legal:
            st.steps.append({"note": f"planner chose illegal tool {a.tool}; using default"})
            return st.default()
        return a
    except llm.LLMError as e:
        st.steps.append({"note": f"planner failed ({str(e)[:80]}); using default"})
        return st.default()


def _evidence(h: Hit, p: SceneProfile) -> list[str]:
    why = []
    if h.matched["moods"]:
        why.append("mood match: " + ", ".join(h.matched["moods"]))
    if h.matched["genres"]:
        why.append("genre match: " + ", ".join(h.matched["genres"]))
    if h.matched["instruments"]:
        why.append("instruments: " + ", ".join(h.matched["instruments"]))
    why.append(f"energy {h.track.energy:.2f} vs scene {p.energy:.2f}")
    why.append(f"semantic similarity {h.parts['semantic']:.2f} (relative)")
    return why


class Blurbs(BaseModel):
    blurbs: list[str]


def _blurbs(items: list[dict], p: SceneProfile) -> None:
    """One LLM call writes a one-sentence 'why it fits' per track, from the evidence only."""
    lines = [f"{i + 1}. \"{it['track']['title']}\" by {it['track']['artist']} | tags: "
             f"{', '.join(it['track']['moods'] + it['track']['genres'])} | evidence: {'; '.join(it['why'])}"
             for i, it in enumerate(items)]
    msg = (f"Scene brief: {p.summary} Wanted: {', '.join(p.moods)}; {', '.join(p.genres)}; relationship={p.relationship}.\n"
           "For each track write one sentence on why it fits the scene, using only the tags and evidence given. "
           "Do not mention licensing.\n" + "\n".join(lines))
    try:
        out = llm.structured(Blurbs, [{"role": "user", "content": msg}], name="explain", max_tokens=60 * len(items))
        for it, b in zip(items, out.blurbs):
            it["blurb"] = b.strip()
    except llm.LLMError:
        pass


def run(text: str = "", clip_path: str | None = None, srt_path: str | None = None, use: str = "monetized",
        k: int = 10, min_cleared: int = 3, use_llm_planner: bool = True, explain: bool = True,
        use_vlm: bool = True, trace_id: str | None = None, max_steps: int = 8) -> dict:
    st = State(text, clip_path, srt_path, use, k, min_cleared)
    with trace("agent.run", {"text": text, "clip": bool(clip_path), "use": use}, trace_id=trace_id) as tid:
        llm_ok = llm.available()
        for _ in range(max_steps):
            a = _plan(st, use_llm_planner and llm_ok)
            with span(f"tool.{a.tool}", thought=a.thought, only_cleared=a.only_cleared):
                if a.tool == "analyze_clip":
                    st.analysis = analyze_clip(clip_path, srt_path, use_vlm=use_vlm)
                elif a.tool == "parse_scene":
                    desc = text.strip() or (st.analysis.description if st.analysis else "")
                    hints = st.analysis.description if (st.analysis and text.strip()) else ""
                    st.profile = parse_scene(desc, hints)
                elif a.tool == "search_catalog":
                    allowed = _cleared_ids(use) if a.only_cleared else None
                    new = search(st.profile, k=k, allowed=allowed)
                    seen = {h.track.id for h in st.hits}
                    st.hits = st.hits + [h for h in new if h.track.id not in seen]
                    st.searches += 1
                elif a.tool == "check_licenses":
                    for h in st.hits:
                        if h.track.id not in st.verdicts:
                            st.verdicts[h.track.id] = check_license(h.track, use)
                elif a.tool == "finish":
                    st.steps.append({"tool": a.tool, "thought": a.thought})
                    break
            st.steps.append({"tool": a.tool, "thought": a.thought, "only_cleared": a.only_cleared})

        # rank: fit first; among near-equal fits prefer cleared tracks
        with span("rank.finalize"):
            hits = sorted(st.hits, key=lambda h: -(h.score + (0.03 if st.verdicts[h.track.id].cleared else 0)))
            items = []
            for h in hits[:k]:
                d = h.to_dict()
                d["why"] = _evidence(h, st.profile)
                d["license"] = st.verdicts[h.track.id].to_dict()
                d["blurb"] = ""
                items.append(d)
            if explain and llm_ok and items:
                _blurbs(items, st.profile)
            warnings = guardrails.apply(items)
            for i, it in enumerate(items):
                it["rank"] = i + 1
        return {"trace_id": tid, "use": use, "profile": st.profile.model_dump() if st.profile else None,
                "analysis": st.analysis.to_dict() if st.analysis else None, "items": items,
                "warnings": warnings, "steps": st.steps}
