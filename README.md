# Needle Drop

Describe a scene or drop in a 30-second clip, and Needle Drop returns a ranked shortlist of
Creative Commons tracks that fit it. Every track carries a license verdict for your intended use
(personal, monetized, or client work) with citations, and **nothing is ever marked cleared without a
verified license source**.

Everything runs on free, open components on a CPU: a local Qwen2.5-3B (llama.cpp) for the scene
brief, the agent and the judge, nomic-embed-text for search, SmolVLM-500M for keyframe captions,
ffmpeg plus numpy for clip analysis, SQLite for the queue, cache and traces, and FastAPI with a
plain HTML/JS front end. No paid API is called anywhere.

- **Static demo (GitHub Pages):** runs entirely in the browser (precomputed backend results for the
  examples and sample clips, in-browser tag search and license rules, and the labeling UI).
- **Full app:** `./start.sh` locally, or the `Dockerfile` on any free CPU host with about 8 GB RAM
  (for example a Hugging Face Docker Space).

## What it does, phase by phase

| # | Phase | Where | Status |
|---|---|---|---|
| 1 | Scene text to a structured music brief (moods, genres, instruments, energy, tempo, match vs counterpoint), constrained to the catalog's own tag vocabulary | `needle_drop/scene.py`, `llm.py` | Built and evaluated |
| 2 | Embeddings and vector search over 21,922 MTG-Jamendo tracks, plus a hybrid ranker (semantic, mood, genre, instrument, energy, avoid list, per-artist cap) | `catalog.py`, `embeddings.py`, `search.py` | Built and evaluated |
| 3 | 30 s clip analysis: cuts (windowed colour histograms, robust to strobes), pacing, motion, palette, brightness, audio tempo (spectral flux with a confidence gate), subtitles, SmolVLM keyframe captions | `clip.py` | Built; tested on two synthetic sample clips |
| 4 | RAG over license terms: 9 knowledge-base docs (each CC license, video sync as an adaptation, attribution, the Jamendo source), cited in every verdict, plus cost | `licensing.py`, `data/licenses/` | Built and evaluated |
| 5 | Agent: analyze the clip, write the brief, search, check licenses, explain, rank, with a cleared/not-cleared flag per track. The LLM picks each step as constrained JSON and illegal steps are replaced and traced | `agent.py` | Built; tested through the API |
| 6 | Evals on 50 iconic needle drops plus an LLM-as-judge calibrated against human labels | `evals/` | Built; judge calibrated on provisional labels only (see below) |
| 7 | A small fine-tuned adapter mapping scene text into the track embedding space, compared with off-the-shelf | `finetune/` | Built and evaluated |
| 8 | Guardrails: cleared only with a verified source, clearance claims scrubbed from generated text, uncertain items flagged | `guardrails.py`, `tests/` | Built; 9 unit tests |
| 9 | Durable job queue, content-hash result cache, span tracing, per-request token and spend tracking | `jobs.py`, `tracing.py`, `api.py` | Built; tested through the API |
| 10 | Front end: drag in a clip, play each shortlisted track against it, rate fits, see the trace and spend | `web/` | Built; tested with Playwright on desktop and phone widths |

## Licensing rules (the part that must not be wrong)

Syncing music to video creates an adaptation under Creative Commons, so:

| License | Personal video | Monetized video | Client / ad work |
|---|---|---|---|
| CC BY | Cleared (credit) | Cleared (credit) | Cleared (credit) |
| CC BY-SA | Cleared (video becomes BY-SA) | Cleared (video becomes BY-SA) | Needs review |
| CC BY-NC | Cleared (credit) | Not cleared | Not cleared |
| CC BY-NC-SA | Cleared (video becomes BY-NC-SA) | Not cleared | Not cleared |
| CC BY-ND, BY-NC-ND | Not cleared (no adaptations) | Not cleared | Not cleared |
| No license record | Unverified | Unverified | Unverified |

The license source is the per-track `audio_licenses.txt` from the MTG-Jamendo dataset (a 2019
snapshot), so every cleared track also carries a flag telling you to reconfirm on the Jamendo track
page before publishing. Licenses can change; this tool is a shortlist, not legal advice.

## Evals

All numbers below come from files in `evals/results/` and `finetune/`, produced by the scripts named.
They were built with the [ai-evals-course evals skills](https://github.com/ai-evals-course/evals-skills)
(error analysis, code evals, binary LLM judge, judge validation, synthetic data, RAG evals).

### Retrieval on 50 iconic scenes (`python evals/run_eval.py`)

Each scene (Reservoir Dogs' ear scene, Pulp Fiction's opening, Guardians' "Hooked on a Feeling" and
so on) has the original song plus reference tags in catalog vocabulary. The original songs are
major-label recordings and are not in a CC catalog, so **original@10 is 0 by construction**. The
headline metric is **similar@10**: at least one top-10 track shares a genre and a mood with the
reference and is within 0.25 energy.

| variant | similar@10 (95% CI) | P@10 | MRR | brief mood recall | brief genre recall | counterpoint acc | energy MAE |
|---|---|---|---|---|---|---|---|
| heuristic | 0.22 (0.10-0.34) | 0.03 | 0.06 | 0.12 | 0.03 | 0.80 | 0.20 |
| llm_semantic_v1 | 0.40 (0.26-0.54) | 0.09 | 0.18 | 0.23 | 0.05 | 0.56 | 0.17 |
| llm_hybrid_v1 | 0.42 (0.28-0.56) | 0.15 | 0.22 | 0.23 | 0.05 | 0.56 | 0.17 |
| llm_semantic_v2 | 0.38 (0.26-0.52) | 0.10 | 0.16 | 0.25 | 0.17 | 0.66 | 0.17 |
| llm_hybrid_v2 | 0.40 (0.28-0.54) | 0.18 | 0.24 | 0.25 | 0.17 | 0.66 | 0.17 |
| llm_semantic_v3 | 0.40 (0.28-0.54) | 0.09 | 0.22 | 0.18 | 0.05 | 0.74 | 0.22 |
| llm_hybrid_v3 | 0.46 (0.32-0.60) | 0.14 | 0.18 | 0.17 | 0.06 | 0.72 | 0.20 |
| direct_baseline | 0.26 (0.14-0.38) | 0.04 | 0.08 | – | – | – | – |
| direct_adapter | 0.28 (0.16-0.42) | 0.04 | 0.09 | – | – | – | – |

The heuristic's 0.80 counterpoint accuracy is an artifact: it always answers "match", and 40 of 50 scenes are matches. The adapter row is the phase-7 comparison (raw scene text, no LLM).

What this says: the LLM brief roughly doubles similar@10 over the no-LLM keyword baseline, and the
hybrid ranker improves precision and MRR over pure semantic search. With 50 scenes the confidence
intervals are wide, so small differences between variants are not meaningful.

### Error analysis (`evals/error_analysis.md`)

Every v1 trace was read and open-coded, then grouped into failure modes: genre collapse onto the
prompt example, era ignored, counterpoint over-predicted (30 of 50 vs 10 of 50 true), energy too high
for quiet scenes, mood copying, and a catalog gap (almost no 60s/70s, surf or soul tags in Jamendo).
v2 fixed genre recall (0.05 to 0.17) but the 3B model copied a prompt example verbatim in 20 of 50
briefs; v3 removed the examples. v3 is the shipped prompt: it has the best counterpoint accuracy and no copying, and its retrieval is statistically indistinguishable from v1 and v2 on 50 scenes.

### License RAG (`python evals/license_rag_eval.py`)

16 questions with the expected knowledge-base section: hit@1 0.75, **hit@3 0.94**, MRR@3 0.84.

### Off-the-shelf vs fine-tuned adapter (phase 7)

181 synthetic (scene, track) pairs were generated with the dimension-tuple method (track mood x film
genre x setting, then a separate call writes a scene that never mentions music). A linear adapter
(256x256, identity-initialised, InfoNCE with soft labels, pure numpy) was tuned by 5-fold CV
(`finetune/sweep.py`).

| Model (raw scene text, semantic search only) | Held-out synthetic pairs, similar@10 (5-fold CV) | 50 iconic scenes, similar@10 (95% CI) |
|---|---|---|
| Off-the-shelf nomic-embed | 0.25 | 0.26 |
| Fine-tuned adapter | 0.30 | 0.28 |

The adapter's gain is within noise on the real scenes (one or two scenes out of 50), and both are well
below the LLM-brief pipeline. Honest read: 181 synthetic pairs from a 3B model are not enough signal,
and the track embeddings come from text tags rather than audio. The code accepts any target matrix,
so audio embeddings (for example CLAP over the Jamendo audio) can be swapped in.

### LLM-as-judge (`python evals/judge.py validate LABELS.json`)

A binary Pass/Fail judge ("does this track fit the scene?") with critique before verdict, few-shot
examples from the train split only, and dev/test TPR and TNR, followed by the Rogan-Gladen corrected
pass rate with a bootstrap CI on unlabeled top-10 pairs.

Result: **the local 3B judge does not agree well enough with the labels to be used as an automatic grader yet.**

| judge prompt | dev TPR / TNR (n=45) | test TPR / TNR (n=40) |
|---|---|---|
| v1 | 0.45 / 0.88 | 0.29 / 0.91 |
| v2 (prompt told to fail only clear clashes; did not help) | 0.35 / 0.88 | 0.28 / 0.91 |

It rarely fails good tracks (TNR about 0.9) but passes only about a quarter to a third of the tracks the labels call good. TPR + TNR is only
about 1.2, so the Rogan-Gladen correction is unstable (the corrected pass rate hit 1.0 with a 0-1 confidence interval), and I do not report a
corrected fit rate. Next steps: a larger judge model, or more labeled data.

**These labels are provisional.** They were written by Claude from tags, not by a person listening to
the audio. To calibrate properly, open `web/review.html`, label the 100 pairs (keys 1, 2, D, U), click
Export, and run `python evals/judge.py validate needle-labels.json`.

## Verified vs not verified

Verified in the build environment:
- Catalog built from MTG-Jamendo metadata with per-track licenses; 21,922 tracks embedded.
- The eval scripts above ran and produced the numbers shown.
- 9 guardrail unit tests pass (`pytest tests`): ND never cleared for video, NC never cleared for
  monetized use, a missing source is unverified, LLM text cannot claim clearance, and so on.
- The API was exercised end to end (search job, clip job, cache hit, traces, spend) and the UI was
  driven with Playwright in live and static modes at desktop and phone widths.

Not verified:
- **Audio playback of Jamendo tracks.** The build environment could not reach jamendo.com, so the
  stream URLs were never played here. They follow Jamendo's public format and should work in a browser.
- **Licenses are as of the 2019 snapshot**, not checked live against Jamendo's API.
- **Clip analysis on real film footage.** It was tested on two generated sample clips (a night chase
  with hard cuts and a quiet dawn) and checked for cut count and tempo, not on real movies.
- **The judge's agreement with a human.** The calibration labels are provisional (see above).
- **The Docker image on a hosted Space.** It builds locally from the same steps but was not deployed,
  since that needs your own Hugging Face account.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_models.sh      # Qwen2.5-3B, nomic-embed, SmolVLM GGUFs (about 3 GB)
bash scripts/fetch_data.sh        # MTG-Jamendo metadata and licenses
python scripts/embed_catalog.py   # only if data/catalog_vectors.npy is missing
./start.sh                        # LLM server on :8081, app on :8000
```

Set `NEEDLE_LLM=heuristic` to run without the LLM (keyword brief), or point `NEEDLE_LLM_BASE_URL`
at any OpenAI-compatible server.

## Credits

Track metadata and licenses: [MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset),
metadata licensed CC BY-NC-SA 4.0 by MTG/UPF. Each track is licensed by its artist under the CC
license shown. Models: Qwen2.5-3B-Instruct (Apache 2.0), nomic-embed-text-v1.5 (Apache 2.0),
SmolVLM-500M (Apache 2.0).
