"""Render the two procedural test clips (chase, dawn) with Playwright + ffmpeg.

Video: the canvas animation in scene.html recorded by Chromium, encoded as VP9 WebM (plays in every browser; H.264 does not play in open-source Chromium).
Audio: synthesized with ffmpeg (rain + 128 BPM pulse + siren sweep for chase; soft surf for dawn).
Subtitles: a small .srt muxed in as a subtitle stream.
"""
import os
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parent.parent / "samples")
OUT.mkdir(parents=True, exist_ok=True)

SRT = {
    "chase": "1\n00:00:02,000 --> 00:00:04,500\nThey're right behind us.\n\n2\n00:00:09,000 --> 00:00:11,000\nDrive. Just drive!\n\n"
             "3\n00:00:20,000 --> 00:00:22,500\nLeft! Take the left!\n",
    "dawn": "1\n00:00:14,000 --> 00:00:17,000\nI used to come here with my father.\n\n2\n00:00:23,000 --> 00:00:26,000\nIt still feels like home.\n",
}
AUDIO = {
    "chase": ("anoisesrc=color=pink:amplitude=0.25:d=31[r];"
              "sine=f=55:d=31,volume='if(lt(mod(t,0.46875),0.08),1.6,0)':eval=frame[k];"
              "sine=f=700:d=31,vibrato=f=1.5:d=0.9,volume=0.12[s];"
              "[r][k][s]amix=inputs=3:normalize=0[a]"),
    "dawn": ("anoisesrc=color=brown:amplitude=0.18:d=31,volume='0.5+0.5*sin(2*PI*t/7)':eval=frame[w];"
             "sine=f=220:d=31,volume=0.03[p];[w][p]amix=inputs=2:normalize=0[a]"),
}

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.getenv("CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome") if os.path.exists(os.getenv("CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")) else None)
    for name in ("chase", "dawn"):
        ctx = b.new_context(viewport={"width": 960, "height": 540}, record_video_dir=str(OUT / "_raw"),
                            record_video_size={"width": 960, "height": 540})
        page = ctx.new_page()
        page.goto((HERE / "scene.html").as_uri() + f"?scene={name}")
        page.wait_for_timeout(31_000)
        raw = page.video.path()
        ctx.close()
        srt = OUT / f"{name}.srt"
        srt.write_text(SRT[name])
        out = OUT / f"{name}.webm"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.5", "-t", "30", "-i", str(raw),
                        "-f", "lavfi", "-t", "30", "-i", "anullsrc=r=44100:cl=mono",
                        "-filter_complex", AUDIO[name], "-map", "0:v", "-map", "[a]",
                        "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36", "-deadline", "realtime", "-cpu-used", "6", "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "libopus",
                        "-t", "30", str(out)], check=True)
        print("wrote", out)
    b.close()
