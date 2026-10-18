"""Synthesize the narration (Kokoro-82M via kokoro-onnx, Apache 2.0, runs on CPU) and mux it with the
recorded video. Each video beat (between two sync flashes) is stretched to fit its line: waiting beats
are sped up by however much fits (up to 12x), short real-time beats hold their last frame.
usage: python scripts/demo/narrate.py TAKE_DIR MODEL.onnx VOICES.bin [voice]"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

take, model, voices = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
voice = sys.argv[4] if len(sys.argv) > 4 else "af_heart"
m = json.loads((take / "marks.json").read_text()); v = m["video"]
lines = json.loads((Path(__file__).parent / "narration.json").read_text())
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v]))

# 1. find the sync flashes (magenta 10x10 square, top-left) in the video
raw = subprocess.check_output(["ffmpeg", "-loglevel", "error", "-i", v, "-vf", "fps=25,crop=10:10:0:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
fr = np.frombuffer(raw, np.uint8).reshape(-1, 10 * 10 * 3).reshape(-1, 100, 3).mean(1)
hit = (fr[:, 0] > 200) & (fr[:, 1] < 70) & (fr[:, 2] > 200)
starts = [i / 25 for i in range(len(hit)) if hit[i] and (i == 0 or not hit[i - 1])]
names = [n for _, n in m["events"]]
assert len(starts) == len(names), f"found {len(starts)} flashes for {len(names)} events"
print("beats:", [(n, round(t, 1)) for n, t in zip(names, starts)])

# 2. narration audio
from kokoro_onnx import Kokoro
k = Kokoro(model, voices)
wav = {}
for key, text in lines.items():
    a, sr = k.create(text, voice=voice, speed=0.97, lang="en-us")
    wav[key] = a.astype(np.float32); print(key, round(len(a) / sr, 1), "s")
SR = sr

# 3. cut each beat to fit its line
beats = [(n, t, (starts[i + 1] if i + 1 < len(starts) else dur) - t) for i, (n, t) in enumerate(zip(names, starts))]
segs, audio = [], []
for i, (n, t0, real) in enumerate(beats):
    L = len(wav[n]) / SR if n in wav else 0.0
    fast = n.startswith("fast") or n.startswith("end_")
    if n.startswith("end_"):
        fast = False
    if n in ("fast1", "fast2"):
        D = max(L + 0.6, min(real / 12, L + 6.0)); speed = real / D; pad = 0.0  # at most 6 s of waiting beyond the narration
    else:
        D = max(real, L + 0.5); speed = 1.0; pad = max(0.0, D - real)
    f = take / f"nseg{i}.mp4"
    vf = f"setpts=PTS/{speed:.4f},fps=25" + (f",tpad=stop_mode=clone:stop_duration={pad:.2f}" if pad > 0.01 else "")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-t", f"{real:.2f}", "-i", v, "-vf", vf, "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "fast", "-t", f"{D:.2f}", str(f)], check=True)
    segs.append(f"file '{f.name}'")
    a = wav.get(n, np.zeros(0, np.float32))
    audio.append(np.concatenate([np.zeros(int(0.15 * SR), np.float32), a, np.zeros(max(0, int(D * SR) - int(0.15 * SR) - len(a)), np.float32)])[: int(D * SR)])
(take / "nlist.txt").write_text("\n".join(segs))
speech = np.concatenate(audio)
speech = speech / max(1e-6, np.abs(speech).max()) * 0.9
# the recording has no sound, so lay the real stand-in music under the beats where the player is on
# (it is quieter while the voice speaks, and comes up once the voice stops)
mp = take / "music.wav"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(Path(__file__).parent / "standin_music.mp3"), "-ac", "1", "-ar", str(SR), str(mp)], check=True)
music, _ = sf.read(mp, dtype="float32")
off, bounds = 0, {}
for i, (n, t0, real) in enumerate(beats):
    bounds[n] = (off, len(audio[i])); off += len(audio[i])
if "play" in bounds:
    a0 = bounds["play"][0] + int(0.9 * SR)
    end = bounds["rate"][0] + bounds["rate"][1] if "rate" in bounds else bounds["play"][0] + bounds["play"][1]
    n_m = min(len(music), end - a0)
    env = np.full(n_m, 0.55, np.float32)
    voice_end = bounds["play"][0] + int((len(wav["play"]) / SR + 0.3) * SR) - a0
    env[: max(0, min(n_m, voice_end))] = 0.16
    ramp = int(0.8 * SR)
    if 0 < voice_end < n_m - ramp:
        env[voice_end: voice_end + ramp] = np.linspace(0.16, 0.55, ramp)
    env[-int(0.6 * SR):] *= np.linspace(1, 0, int(0.6 * SR))
    speech[a0:a0 + n_m] += music[:n_m] * env
speech = np.clip(speech, -1, 1)
sf.write(take / "narration.wav", speech, SR)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-i", str(take / "nlist.txt"), "-i", str(take / "narration.wav"),
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(take / "needle-drop-demo-narrated.mp4")], check=True)
print("done", take / "needle-drop-demo-narrated.mp4")
