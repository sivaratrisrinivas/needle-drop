# Error analysis (error-discovery skill)

Process: ran the full pipeline (LLM brief → hybrid search) on all 50 iconic scenes, read every
trace (brief + top 10), wrote an open-coded note per failing scene, then grouped the notes into
failure modes (axial coding). Annotator: Claude, reading tags and briefs; nobody listened to audio
in this pass. Counts are from the v1 prompt run (`results/llm_hybrid_v1.json`).

## Failure modes (v1)

| # | Failure mode | Evidence | Count | Fix |
|---|---|---|---|---|
| F1 | **Genre collapse onto the few-shot example.** The brief picks indie/poprock for almost any scene. | indie in 26/50 briefs, poprock in 22/50; reference-genre recall 0.05 | ~35 | v2: two new examples with different genres, explicit "do not default to indie or poprock" |
| F2 | **Era and setting ignored.** "1969 Los Angeles", "1950s restaurant", "space station" do not move the genre. | OUATIH, Pulp Fiction twist, 2001, Shawshank all miss | ~10 | v2: era → genre guidance (60s/70s/80s, classical/orchestral, synthwave) |
| F3 | **Counterpoint over-predicted.** The prompt foregrounded irony, so the model called 30/50 scenes counterpoint (truth: 10/50), and then gave grief scenes "happy, uplifting". | Kill Bill "Bang Bang" → happy/uplifting; Shawshank → dark/powerful | 20 false | v2: default to match; counterpoint only for stylised violence; grief/romance always match |
| F4 | **Energy too high for quiet, awestruck scenes.** | Platoon 0.9 (ref 0.3), Shawshank 0.9 (ref 0.25), 2001 0.7 (ref 0.35) | 8 with error > 0.3 | v2: anchored energy scale |
| F5 | **Mood copying.** "uplifting" appears in 22/50 briefs, copied from the example. | — | ~15 | v2: new examples, no "uplifting" |
| C1 | **Catalog gap, not a pipeline bug.** Decade tags (60s/70s) are rare in Jamendo, and there is almost no surf, soul-era or classic rock. Some references cannot be matched by any query. | 60s: 78 tracks, 70s: 92, surf: ~20 | — | Document; consider adding FMA as a second catalog |

## Round 2: v2 and v3 (same 50 scenes, so optimistic)

| # | Failure mode | Evidence | Outcome |
|---|---|---|---|
| F7 | **Example copying.** With two JSON examples in the prompt, the 3B model copied the counterpoint example's moods and genres verbatim. | 20 of 50 v2 briefs | v3 removed all JSON examples (zero-shot, guidance in prose only) |
| F8 | **Zero-shot genre drift.** Without examples, genres revert to electronic and counterpoint-flavoured picks; brief genre recall falls back to 0.05-0.06 (v2: 0.17). | v3 brief genre recall 0.06 | Open. Mood and genre tags from a 3B model are weak; retrieval leans on the semantic query text |
| F9 | **Energy still high.** | v3 energy MAE 0.20-0.22 vs 0.17 in v1/v2 | Open |

| prompt | relationship acc | similar@10 hybrid (95% CI) | genre recall |
|---|---|---|---|
| v1 | 0.56 | 0.42 (0.28-0.56) | 0.05 |
| v2 | 0.66 | 0.40 (0.28-0.54) | 0.17 |
| v3 | 0.72 | 0.46 (0.32-0.60) | 0.06 |

All three similar@10 intervals overlap almost entirely, so the ranking of prompts by retrieval is not
established. v3 is kept because it has the best counterpoint accuracy, no example copying, and the
highest point estimate. A larger model (for example a 7B) is the obvious next experiment; it would be about 2x slower on this CPU.

## What we did not find
- No invalid JSON reached the search step (schema-constrained decoding plus pydantic validation and one retry).
- No out-of-vocabulary tags after snapping (`brief_valid` = 1.0).
- No licensing claims in generated text in the agent runs reviewed (the guardrail scrubber would remove them; see tests).

## Caveat
The v2 prompt was written after reading all 50 v1 traces, so v2 numbers on these same 50 scenes are
optimistic (the eval set doubled as a dev set). A fresh held-out set of scenes is the next step; the
review UI (`web/review.html`) and judge calibration are set up for it.
