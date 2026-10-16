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


def caption(pg, text):
    pg.evaluate("""t => { let d = document.getElementById('cap'); if (!d) { d = document.createElement('div'); d.id = 'cap';
      d.style.cssText = 'position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:99999;background:rgba(10,12,18,.92);color:#fff;font:600 18px system-ui;padding:10px 18px;border-radius:10px;border:1px solid #f5b642;max-width:90vw;text-align:center'; document.body.appendChild(d);} d.textContent = t; }""", text)


def fast(pg, label, wait_fn):
    start = now(); caption(pg, label + " (sped up 12x)"); wait_fn(); marks.append([start, now()])


with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    ctx = b.new_context(viewport={"width": 1280, "height": 800}, record_video_dir=str(OUT), record_video_size={"width": 1280, "height": 800})
    pg = ctx.new_page(); t0 = time.time()
    pg.goto(BASE); pg.wait_for_timeout(1500)
    caption(pg, "Needle Drop: find the right song for a scene, with a license verdict on every pick"); pg.wait_for_timeout(3500)

    caption(pg, "1. Describe a scene. A local 3B LLM turns it into a structured music brief"); 
    pg.click("#examples button:nth-child(2)"); pg.wait_for_timeout(2000)
    pg.click("#go")
    fast(pg, "Agent runs: brief, catalog search, license checks", lambda: pg.wait_for_selector("#shortlist li", timeout=600000))
    pg.wait_for_timeout(500)
    caption(pg, "Ranked shortlist: every track has a cleared / not cleared flag for the chosen use"); pg.wait_for_timeout(3500)
    pg.locator("#brief").scroll_into_view_if_needed(); pg.wait_for_timeout(2500)
    pg.locator("#shortlist li").first.scroll_into_view_if_needed()
    pg.locator("#shortlist li").first.locator("summary").first.click() if pg.locator("#shortlist li").first.locator("summary").count() else None
    caption(pg, "Each pick explains why, cites the license text, and shows its cost basis"); pg.wait_for_timeout(5000)
    pg.locator("#shortlist li").first.screenshot(path=str(OUT / "first_item.png"))
    pg.locator("#trace summary").click(); pg.wait_for_timeout(500)
    caption(pg, "Every run is traced: steps, tokens, and spend (local model, $0)"); pg.wait_for_timeout(4500)

    caption(pg, "2. Switch the intended use: monetized video, so NC tracks are not cleared")
    pg.evaluate("window.scrollTo(0,0)"); pg.wait_for_timeout(1500)

    pg.click(".tab[data-tab=clip]"); pg.wait_for_timeout(800)
    caption(pg, "3. Drop in a clip (here a 30 s sample with hard cuts and subtitles)")
    pg.click("#sample-chase"); pg.wait_for_selector("#preview:not(.hidden)", timeout=30000); pg.wait_for_timeout(3000)
    pg.click("#go")
    fast(pg, "Analyzing frames, cuts, audio tempo, subtitles; then the same agent loop",
         lambda: pg.wait_for_selector("#analysis:not(.hidden)", timeout=1500000))
    pg.wait_for_selector("#shortlist li", timeout=600000)
    pg.locator("#analysis").scroll_into_view_if_needed()
    caption(pg, "Clip analysis: cuts, pacing, motion, tempo, palette, keyframe captions"); pg.wait_for_timeout(6000)
    pg.locator("#shortlist li").first.scroll_into_view_if_needed()
    caption(pg, "Play any pick against the clip (audio streams from Jamendo in a normal browser)"); pg.wait_for_timeout(1500)
    pg.locator("#shortlist li .play").first.click(); pg.wait_for_timeout(5000)
    caption(pg, "Rate the fit: labels feed the judge calibration")
    btn = pg.locator("#shortlist li").first.get_by_text("Fits")
    if btn.count(): btn.first.click()
    pg.wait_for_timeout(3000)

    caption(pg, "4. Labeling UI: Pass / Fail with keyboard shortcuts, export JSON for judge calibration")
    pg.goto(BASE + "/review.html"); pg.wait_for_timeout(2500)
    pg.keyboard.press("1"); pg.wait_for_timeout(1200); pg.keyboard.press("2"); pg.wait_for_timeout(2500)
    caption(pg, "github.com/sivaratrisrinivas/needle-drop"); pg.wait_for_timeout(3500)
    vid = pg.video.path(); ctx.close(); b.close()
(OUT / "marks.json").write_text(json.dumps({"video": str(vid), "marks": marks}))
print("recorded", vid, marks)
