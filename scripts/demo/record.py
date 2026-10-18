"""Record the demo video of the live app with Playwright.

Long waits (the local 3B model on CPU takes about a minute per scene and a few minutes per clip) are
marked as "fast" segments and sped up 12x afterwards by ffmpeg; the on-screen caption says so.
usage: python scripts/demo/record.py [BASE_URL] [OUT_DIR]
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "recordings"); OUT.mkdir(parents=True, exist_ok=True)
marks, t0 = [], None


def now():
    return time.time() - t0


events = []


def ev(pg, name):
    """Log a wall-clock event and flash a 10px magenta square in the corner for 250 ms, so the
    assembler can find the exact video frame (Playwright's video clock is not our clock)."""
    events.append([now(), name])
    pg.evaluate("""() => { const d = document.createElement('div');
      d.style.cssText = 'position:fixed;left:0;top:0;width:10px;height:10px;background:#ff00ff;z-index:100000';
      document.body.appendChild(d); setTimeout(() => d.remove(), 250); }""")
    pg.wait_for_timeout(300)


def caption(pg, text, name=None):
    if name:
        ev(pg, name)
    pg.evaluate("""t => { let d = document.getElementById('cap'); if (!d) { d = document.createElement('div'); d.id = 'cap';
      d.style.cssText = 'position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:99999;background:rgba(10,12,18,.92);color:#fff;font:600 18px system-ui;padding:10px 18px;border-radius:10px;border:1px solid #f5b642;max-width:90vw;text-align:center'; document.body.appendChild(d);} d.textContent = t; }""", text)


def fast(pg, label, wait_fn, name):
    start = now(); caption(pg, label + " (sped up 12x)", name); wait_fn(); marks.append([start, now()]); ev(pg, "end_" + name)


with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium", args=["--autoplay-policy=no-user-gesture-required"])
    ctx = b.new_context(viewport={"width": 1280, "height": 800}, record_video_dir=str(OUT), record_video_size={"width": 1280, "height": 800})
    pg = ctx.new_page(); t0 = time.time()
    # The build sandbox cannot reach jamendo.com, so a stand-in tone replaces the stream in this recording (the video says so).
    tone = (Path(__file__).parent / "standin_tone.wav").read_bytes()
    pg.route("**/*.jamendo.com/**", lambda r: r.fulfill(status=200, body=tone, headers={"content-type": "audio/wav", "access-control-allow-origin": "*"}))
    pg.goto(BASE); pg.wait_for_selector("#scene"); pg.wait_for_timeout(1500)
    caption(pg, "Needle Drop: one field. Everything else appears only when you need it", "intro"); pg.wait_for_timeout(3500)

    caption(pg, "1. Describe a scene (Tab uses the suggestion). A local 3B LLM writes the music brief", "describe")
    pg.click("#scene"); pg.keyboard.press("Tab"); pg.wait_for_timeout(2500)
    pg.keyboard.press("Enter")
    fast(pg, "Agent runs: brief, catalog search, license checks", lambda: pg.wait_for_selector("#crate li", timeout=900000), "fast1")
    pg.wait_for_timeout(500)
    caption(pg, "Usable tracks stand out; the rest fade. * means checked against the 2019 license snapshot", "shortlist"); pg.wait_for_timeout(3500)
    pg.locator(".exp").first.scroll_into_view_if_needed(); pg.locator(".exp").first.click()
    caption(pg, "Tap + for why it fits, the license text it cites, and what it costs", "picks"); pg.wait_for_timeout(5000)
    pg.locator(".exp").first.click(); pg.locator("#foot").scroll_into_view_if_needed(); pg.locator("#tracebtn").click(); pg.wait_for_timeout(500)
    caption(pg, "Every run is traced: steps, tokens, and spend (local model, $0)", "trace"); pg.wait_for_timeout(4500)
    pg.evaluate("window.scrollTo(0,0)")
    caption(pg, "The verdicts follow your intended use: monetized video, personal video, or client work", "use"); pg.wait_for_timeout(3500)

    caption(pg, "3. Or drop a clip anywhere on the page (here a 30 s sample with cuts and subtitles)", "clipdrop")
    pg.set_input_files("#file", ["web/samples/chase.webm", "web/samples/chase.srt"]); pg.wait_for_selector("body.has-clip", timeout=30000); pg.wait_for_timeout(2500)
    fast(pg, "Analyzing frames, cuts, audio tempo, subtitles; then the same agent loop",
         lambda: pg.wait_for_function("!document.querySelector('#clipline').hidden && !document.querySelector('#status').classList.contains('busy')", timeout=1500000), "fast2")
    pg.evaluate("window.scrollTo(0,0)"); pg.wait_for_timeout(800)
    pg.locator("#clipline").click()
    caption(pg, "Clip analysis: cuts, tempo, pacing, palette. It feeds the brief", "analysis"); pg.wait_for_timeout(5500)
    pg.locator("#clipline").click()
    caption(pg, "Tap a title to drop the needle: it plays against your clip (stand-in tone here: the sandbox can't reach Jamendo)", "play")
    pg.locator("#crate .ti").first.click(); pg.wait_for_timeout(7000)
    pg.locator("#crate .exp").first.click()
    caption(pg, "Rate the fit: labels calibrate the judge", "rate")
    pg.locator(".rate button").first.click(); pg.wait_for_timeout(3500)

    caption(pg, "4. Labeling screen: Pass / Fail with keyboard shortcuts, export JSON for judge calibration", "review")
    pg.goto(BASE + "/review.html"); pg.wait_for_timeout(2500)
    pg.keyboard.press("1"); pg.wait_for_timeout(1200); pg.keyboard.press("2"); pg.wait_for_timeout(2500)
    caption(pg, "github.com/sivaratrisrinivas/needle-drop", "github"); pg.wait_for_timeout(3500)
    end = now(); vid = pg.video.path(); ctx.close(); b.close()
(OUT / "marks.json").write_text(json.dumps({"video": str(vid), "marks": marks, "end": end, "events": events}))
print("recorded", vid, marks)
