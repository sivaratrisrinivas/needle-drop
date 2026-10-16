"""Speed up the marked waiting segments 12x and write needle-drop-demo.mp4 next to marks.json.
Playwright's video starts before our clock does, so the offset is the video length minus the
wall time at the end of the script."""
import json
import subprocess
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "recordings")
m = json.loads((d / "marks.json").read_text()); v = m["video"]
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v]))
off = dur - m["end"]
segs, t = [], 0.0
for a, b in m["marks"]:
    segs += [(t, a + off, 1), (a + off, b + off, 12)]; t = b + off
segs.append((t, dur, 1))
lst = []
for i, (a, b, s) in enumerate(segs):
    f = d / f"seg{i}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", v, "-vf", f"setpts=PTS/{s},fps=25",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "fast", str(f)], check=True)
    lst.append(f"file '{f.name}'")
(d / "list.txt").write_text("\n".join(lst))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-i", str(d / "list.txt"), "-c", "copy", "-movflags", "+faststart",
                str(d / "needle-drop-demo.mp4")], check=True)
print("offset", round(off, 1), "->", d / "needle-drop-demo.mp4")
