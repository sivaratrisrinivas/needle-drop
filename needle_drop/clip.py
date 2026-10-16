"""Phase 3: multimodal clip analysis.

From a video clip (first 30 s) we extract, with ffmpeg + numpy only:
  - shot cuts (ffmpeg scene-change score) -> cuts/min, average shot length
  - motion energy (mean frame difference) and visual palette (brightness, saturation, warmth)
  - audio loudness curve, onset-based tempo estimate, dynamic range
  - subtitles (embedded subtitle stream, or a sidecar .srt)
  - optional keyframe captions from a small local vision model (SmolVLM-500M via llama.cpp)
and turn them into a plain-language scene description for the music brief.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config
from .tracing import span

MAX_SECONDS = 30
W, H = 96, 54
VLM_MODEL = os.getenv("NEEDLE_VLM_GGUF", str(config.MODELS / "smolvlm-500m-q8.gguf"))
VLM_MMPROJ = os.getenv("NEEDLE_VLM_MMPROJ", str(config.MODELS / "smolvlm-500m-mmproj.gguf"))


@dataclass
class ClipAnalysis:
    duration: float
    cuts: list[float]
    cuts_per_min: float
    avg_shot_s: float
    motion: float
    brightness: float
    saturation: float
    warmth: float
    loudness_db: float
    dynamic_range_db: float
    tempo_bpm: float | None
    has_audio: bool
    subtitles: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    keyframes: list[str] = field(default_factory=list)
    pacing: str = ""
    palette: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, check=False, **kw)


def probe(path: str) -> dict:
    r = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(r.stdout or b"{}")


def detect_cuts(f: np.ndarray, fps: int, threshold: float = 0.25, window: int = 3) -> list[float]:
    """Shot boundaries from colour-histogram change between the frames just before and just after t.

    Comparing short windows (not single frames) keeps strobes, siren lights and flicker from
    counting as cuts; a cut must also be the local maximum and at least 0.5 s after the last one.
    """
    if len(f) < 2 * window:
        return []
    H = np.stack([np.concatenate([np.histogram(x[..., c], bins=16, range=(0, 1))[0] for c in range(3)]) for x in f])
    H = H / H.sum(1, keepdims=True)
    d = np.array([0.5 * np.abs(H[max(0, t - window):t].mean(0) - H[t:t + window].mean(0)).sum() for t in range(1, len(H))])
    cuts: list[float] = []
    for t in range(len(d)):
        if d[t] > threshold and d[t] == d[max(0, t - 5):t + 6].max():
            ts = round((t + 1) / fps, 2)
            if not cuts or ts - cuts[-1] > 0.5:
                cuts.append(ts)
    return cuts


def frames(path: str, fps: int = 4) -> np.ndarray:
    r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-t", str(MAX_SECONDS), "-i", path,
              "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    return a[: len(a) // (W * H * 3) * W * H * 3].reshape(-1, H, W, 3).astype(np.float32) / 255


def keyframes(path: str, out_dir: Path, times: list[float]) -> list[str]:
    out = []
    for i, t in enumerate(times):
        p = out_dir / f"key_{i}.jpg"
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", path,
              "-frames:v", "1", "-vf", "scale=384:-2", str(p)])
        if p.exists():
            out.append(str(p))
    return out


def audio(path: str, sr: int = 16000) -> np.ndarray:
    r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-t", str(MAX_SECONDS), "-i", path,
              "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"])
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768


def estimate_tempo(y: np.ndarray, sr: int = 16000) -> float | None:
    """Onset-strength autocorrelation tempo estimate (60-180 BPM)."""
    if len(y) < sr * 4:
        return None
    hop = 256
    n = len(y) // hop
    frames_ = y[: n * hop].reshape(n, hop)
    spec = np.abs(np.fft.rfft(frames_ * np.hanning(hop), axis=1))
    flux = np.maximum(0, np.diff(np.log1p(spec), axis=0)).sum(1)
    flux = flux - np.convolve(flux, np.ones(31) / 31, mode="same")  # remove slow swells, keep onsets
    if flux.std() < 1e-6:
        return None
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    fps = sr / hop
    lags = np.arange(len(ac))
    bpm = 60 * fps / np.maximum(lags, 1)
    ok = (bpm >= 60) & (bpm <= 180)
    if not ok.any():
        return None
    # weight towards ~110 BPM to reduce octave errors
    w = ac * ok * np.exp(-0.5 * (np.log2(bpm / 110) / 0.9) ** 2)
    i = int(np.argmax(w))
    # require a clear periodic peak; noise beds (rain, surf, room tone) have none
    if ac[i] / (ac[0] + 1e-9) < 0.12:
        return None
    return round(float(bpm[i]), 1)


def subtitles(path: str, sidecar: str | None = None) -> list[str]:
    text = ""
    if sidecar and Path(sidecar).exists():
        text = Path(sidecar).read_text(errors="replace")
    else:
        r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:s:0", "-f", "srt", "-"])
        text = r.stdout.decode(errors="replace")
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.strip().isdigit() and "-->" not in l]
    return [re.sub(r"<[^>]+>", "", l) for l in lines][:40]


@lru_cache(maxsize=1)
def _vlm():
    if not (Path(VLM_MODEL).exists() and Path(VLM_MMPROJ).exists()) or os.getenv("NEEDLE_VLM", "1") == "0":
        return None
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MTMDChatHandler

    handler = MTMDChatHandler(clip_model_path=VLM_MMPROJ, verbose=False)
    return Llama(VLM_MODEL, chat_handler=handler, n_ctx=4096, verbose=False, n_threads=os.cpu_count())


def caption(image_path: str) -> str | None:
    import base64

    m = _vlm()
    if m is None:
        return None
    with span("vlm.caption", image=Path(image_path).name) as s:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        r = m.create_chat_completion(messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "In one short sentence, describe this film frame: setting, lighting and mood."}]}],
            max_tokens=40, temperature=0.1)
        text = r["choices"][0]["message"]["content"].strip()
        u = r.get("usage") or {}
        s.usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), "local")
        s.set(caption=text)
        return text


def _describe(a: ClipAnalysis) -> None:
    a.pacing = ("frantic, rapid cutting" if a.cuts_per_min >= 30 else "brisk cutting" if a.cuts_per_min >= 15
                else "measured pacing" if a.cuts_per_min >= 6 else "long, lingering takes")
    light = "dark, low-key" if a.brightness < .3 else "bright, high-key" if a.brightness > .6 else "naturally lit"
    col = "desaturated" if a.saturation < .08 else "vivid, saturated" if a.saturation > .3 else "moderately colorful"
    temp = "warm" if a.warmth > .05 else "cold, blue" if a.warmth < -.05 else "neutral"
    a.palette = f"{light}, {col}, {temp} palette"
    motion = "lots of on-screen motion" if a.motion > .08 else "some movement" if a.motion > .03 else "very still frames"
    parts = [f"A {a.duration:.0f}-second clip with {a.pacing} ({len(a.cuts)} cuts, ~{a.avg_shot_s:.1f}s per shot), "
             f"{motion}, {a.palette}."]
    if a.has_audio:
        loud = "loud" if a.loudness_db > -18 else "quiet" if a.loudness_db < -32 else "moderate"
        parts.append(f"Existing audio is {loud} ({a.loudness_db:.0f} dBFS)"
                     + (f" with a pulse around {a.tempo_bpm:.0f} BPM." if a.tempo_bpm else "."))
    if a.captions:
        parts.append("What is on screen: " + " ".join(a.captions))
    if a.subtitles:
        parts.append("Dialogue: \"" + " / ".join(a.subtitles[:8]) + "\"")
    a.description = " ".join(parts)


def analyze_clip(path: str, sidecar_srt: str | None = None, workdir: str | None = None,
                 use_vlm: bool = True) -> ClipAnalysis:
    workdir = Path(workdir or tempfile.mkdtemp(prefix="clip_", dir=config.CACHE))
    workdir.mkdir(parents=True, exist_ok=True)
    with span("clip.analyze", file=Path(path).name) as s:
        info = probe(path)
        dur = min(MAX_SECONDS, float(info.get("format", {}).get("duration", 0) or 0))
        has_audio = any(st.get("codec_type") == "audio" for st in info.get("streams", []))
        with span("clip.frames"):
            f = frames(path, fps=10)
            if len(f):
                diff = float(np.abs(np.diff(f[::3], axis=0)).mean()) if len(f) > 3 else 0.0
                mx, mn = f.max(-1), f.min(-1)
                sat = float((mx - mn).mean())  # chroma; unlike (max-min)/max it does not blow up in dark frames
                bright = float(f.mean())
                warm = float((f[..., 0] - f[..., 2]).mean())
            else:
                diff = sat = bright = warm = 0.0
        with span("clip.cuts") as cs:
            cuts = [c for c in detect_cuts(f, fps=10) if c > 0.2]
            cs.set(cuts=cuts)
        with span("clip.audio"):
            y = audio(path) if has_audio else np.zeros(0, np.float32)
            if len(y) > 1600:
                blocks = y[: len(y) // 1600 * 1600].reshape(-1, 1600)
                rms_db = 20 * np.log10(np.sqrt((blocks ** 2).mean(1)) + 1e-6)
                loud, dr = float(np.percentile(rms_db, 50)), float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10))
                tempo = estimate_tempo(y)
            else:
                loud, dr, tempo = -90.0, 0.0, None
        shots = len(cuts) + 1
        a = ClipAnalysis(duration=round(dur, 2), cuts=cuts, cuts_per_min=round(len(cuts) / max(dur, 1) * 60, 1),
                         avg_shot_s=round(dur / shots, 2), motion=round(diff, 4), brightness=round(bright, 3),
                         saturation=round(sat, 3), warmth=round(warm, 3), loudness_db=round(loud, 1),
                         dynamic_range_db=round(dr, 1), tempo_bpm=tempo, has_audio=has_audio and loud > -80)
        with span("clip.subtitles"):
            a.subtitles = subtitles(path, sidecar_srt)
        # keyframes: middle of the first, longest and last shots
        bounds = [0.0] + cuts + [dur]
        shots_ = sorted(zip(bounds[:-1], bounds[1:]), key=lambda x: x[1] - x[0], reverse=True)
        times = sorted({round((s0 + s1) / 2, 2) for s0, s1 in [shots_[0], (bounds[0], bounds[1]), (bounds[-2], bounds[-1])]})
        a.keyframes = keyframes(path, workdir, times[:3])
        if use_vlm:  # captions are slow on CPU (~30-60 s each), so only the longest shot by default
            n = int(os.getenv("NEEDLE_VLM_FRAMES", "1"))
            longest = min(range(len(times)), key=lambda j: abs(times[j] - (shots_[0][0] + shots_[0][1]) / 2)) if times else 0
            order = [longest] + [j for j in range(len(a.keyframes)) if j != longest]
            a.captions = [c for c in (caption(a.keyframes[j]) for j in order[:n] if j < len(a.keyframes)) if c]
        _describe(a)
        s.set(**{k: v for k, v in a.to_dict().items() if k not in ("keyframes",)})
        return a
