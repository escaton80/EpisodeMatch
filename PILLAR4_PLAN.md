# Pillar 4 Plan — Title-Card & Credits OCR: from beta to production

**Basis:** Ist-Stand v0.11.4 (`extractors/frame_scanner.py`, `extractors/ocr_engine.py`,
`matcher/pipeline.py::_stage_visual`, `matcher/credit_scorer.py`, `core/models.py::TitleCardHit`,
`tests/test_extractors.py`). Arbeitsweise: TDD, kleine Batches, Stop nach Batch 1 für
reale Messungen an 3–5 MKVs.

---

## Bestandsaufnahme

**Was gut funktioniert:**
- Two-window scan design (title + credits), scene detection with lavfi→showinfo
  fallback (C5-hardened), evenly-limited keyframes, RMS dedupe, per-frame OCR with
  graceful failure isolation (`classify_frame` never kills the scan).
- Credit regexes (directed/written/created/guest + "as role"), noise-word trimming,
  height-boosted title heuristic — all covered by pure-function tests.
- Evidence is display-only per hit (timestamps) + a pillar summary; scoring weights
  are configurable (`credit_weight_director/writer/guest/title`).
- OCR backend chain (WinRT → RapidOCR → Tesseract → Null) probes and degrades
  honestly.

**Größte Lücken / Risiken:**
1. **Statische Credits-Rolls produce zero scene cuts** — the credits window is
   skipped entirely when scene detection returns no timestamps. No uniform
   fallback sampling exists. (Biggest structural hole: exactly the frames pillar 4
   wants are missed on slow-rolling credits.)
2. **No frame preprocessing** — WinRT OCR runs on the raw PNG; white text on
   dark/noisy backgrounds relies purely on the backend. No grayscale/contrast/
   invert retry.
3. **Dedupe compares only against the PREVIOUS kept frame** — alternating credit
   cards ("Written by" → "Directed by") can survive as near-duplicates of the
   frame two steps back; and there is no minimum time distance, so fade sequences
   can keep several near-identical frames while two real credit cards 0.4 s apart
   look identical by RMS.
4. **No short-title protection** — `score_episode_credits` accepts any title
   candidate ≥ 80 fuzzy; `"Home"` can hit a generic phrase and inflate a wrong
   episode.
5. **No season-level IDF** — a director credited in 6 of 10 episodes contributes
   full weight to EVERY episode's candidates; only rarity makes credits
   discriminative.
6. **Summary evidence lacks substance** — `credits_summary` shows a counter and
   credited names, not timestamps or the recognized OCR text.
7. **No observability** — frames are deleted after classification; there is no way
   to see what the OCR actually read, what it missed, or how long each phase took.
   This blocks calibration (and the RapidOCR decision).

**Tests today:** pure-function coverage is good (CreditExtraction 7, SplitNames 3,
SceneParsing 5, WindowFilter 4, FrameSimilarity, OcrPlumbing with FakeOcrBackend,
plus a fake-visual pipeline integration test). Missing: scan() end-to-end with a
mocked ffmpeg, dedupe behavior at the threshold, credits fallback sampling, scorer
calibration (short titles, repeated credits), pipeline evidence strings, and the
debug CLI.

---

## Batch 1 — Diagnose-Basis: `episodematch debug-visual <file.mkv>` (TDD) — [x] DONE (522 green + live smoke)

Goal: measure pillar 4 on 3–5 real MKVs before changing any scoring — raw OCR text
per timestamp, classification, potential hits, and phase timings.

- New `FrameScanner.debug_scan(mkv_path, *, duration_seconds, max_frames=None)`
  alongside `scan()` (shared private `_scan(..., keep_unknown=..., keep_frames=...)`):
  keeps **UNKNOWN** frames with their raw OCR text instead of discarding them and
  never deletes PNGs — the diagnostics view of the same pipeline.
- New CLI command (`main.py`): `episodematch debug-visual <file.mkv>
  [--series "Show" --season N] [--save-frames DIR] [--json]`
  - Prints one block per frame: timestamp (MM:SS), kind, OCR confidence, raw
    lines; plus phase timings (scene detection / frame extraction / OCR).
  - With `--series/--season`: fetches the TMDB season guide and additionally lists
    potential credit/title hits per frame (director/writer/guest/title with ratios).
  - `--save-frames DIR` copies the extracted PNGs there (default: extracted frames
    are kept in `data/frames/debug/<stem>/` for this command only).
  - `--json` prints the whole structure machine-readably.
- Tests: `TestDebugScan` (FrameScanner with FakeOcrBackend + mocked
  `detect_scene_changes`/`extract_frame` — UNKNOWN frames visible, raw text kept,
  normal `scan()` unchanged) and a CLI test (monkeypatched scanner, stdout +
  `--json` structure, exit codes).

**Deliverable after Batch 1:** real measurements (hit rate per window, OCR
misses, per-phase seconds) on 3–5 local MKVs → calibration input for Batches 2/3
and the RapidOCR decision.

---

## Batch 2 — Frame-Qualität & Speed (TDD) — [x] DONE (530 green)

- **Dedupe hardening:** compare each frame against **all** previously kept frames
  (not only the previous one) so alternating credit cards survive, and enforce a
  **minimum timestamp distance of ≤ 1.5 s** between kept frames — small enough
  that fast-cutting opening credits ("Written by" → "Directed by" within a second)
  are never collapsed, while fade sequences still collapse to one frame.
- **Credits fallback sampling:** trigger uniform fallback sampling of N frames
  across the credits window already when scene detection yields **< 3 (or < 4)**
  timestamps — not only at exactly zero (single cuts at the scene end or a studio
  logo would otherwise suppress the fallback). A static credits roll has no cuts
  at all; a uniform sample fixes that.
- **Light preprocessing as a retry path:** when OCR returns nothing recognizable,
  retry once on a grayscale + contrast-threshold variant of the frame (white text
  on dark/noisy backgrounds); the original image stays the first attempt.
- Tests: synthetic PNGs (white text on dark and noisy backgrounds), mocked
  subprocess outputs, dedupe/threshold asserts, fallback-sampling asserts.

---

## Batch 3 — Matching & Kalibrierung (TDD) — [x] DONE (539 green)

- **Short-title protection:** for `len(normalized) < 5` titles, `partial_ratio` is
  disabled entirely — only a full-line match (`ratio` / `token_sort_ratio >= 95`)
  may confirm a short title, so `"Home"` cannot hit `"Welcome Home"` at 100 %.
  Longer titles keep the existing two-metric logic.
- **Season-IDF for credits (normalized):** a name credited in k of N season
  episodes gets multiplier **`log(N / k) / log(N)`** — always within `[0.0, 1.0]`
  (`k=1` → 1.0, `k=N` → 0.0), so long seasons cannot inflate weights above 1.0.
  Creator/showrunner credits fade automatically. Requires ≥ 3 episodes; below that
  unweighted + log note (same pattern as the reference-IDF in pillar 2).
- **Transparent evidence strings:** `credits_summary` includes the timestamp
  (MM:SS) and the recognized OCR text of the top hits; per-hit evidence keeps its
  timestamp and gains the raw OCR text in `details`.
- Score capping stays `min(1.0)`; the visual weight is already configurable
  (`matching.weights.visual`).
- Tests: scorer calibration (short titles, a director repeated across 6 of 10
  episodes, normalized IDF bounds), evidence-string asserts.

---

## Optionaler Ausblick — RapidOCR als zweites Visual-Backend

**Decide AFTER Batch 1**, based on the real measurements: WinRT miss rate on
credits frames, OCR seconds per frame, and whether misses are content-related
(stylized fonts) or backend-related. Only if the data shows WinRT gaps that a
second ONNX backend would plausibly close does `rapidocr-onnxruntime` become worth
its install weight (it is already implemented as fallback backend 2 — the question
is whether the *visual* path should try both and merge, rather than just degrade).

---

## Verifikation

- Per batch: TDD (red → green → full suite).
- Batch 1 additionally: run `episodematch debug-visual` on 3–5 real local MKVs and
  record measurements (hit rate, OCR misses, timing).
- After Batch 3: a full `--depth deep` scan against a known disc to calibrate
  confidence vs. the subtitle pillar.

---

## Batch 5 — Gewichts-Normalisierung über aktive Säulen (TDD) — [x] DONE (565 green)

**Nutzer-Fall (Voyager-DVDs außerhalb der DiscDB, keine Untertitel):** Pillar 4 ist
die EINZIGE Säule — ein 100%-Titel-Match trug nur 0.35 (unter der 0.55-Solver-
Schwelle) und wurde als unassigned gemeldet, obwohl der Beweis eindeutig ist.

*Fix:* der Solver normalisiert die Confidence-Schwelle über die Summe der AKTIVEN
Säulen-Gewichte (`effective_min = min_confidence × min(1.0, weight_sum)`):
- nur Pillar 4 aktiv (Gewicht 0.20) → Schwelle 0.11 → der 100%-Titel-Match wird
  zugeordnet, die angezeigte Confidence bleibt die ehrliche absolute Säulen-Score (0.35)
- alle Säulen aktiv (Gewichtssumme 1.40 ≥ 1.0) → Schwelle bleibt 0.55 — das
  Multi-Pillar-Verhalten ist unverändert
- schwache Multi-Säulen-Kombinationen werden konsequent strenger bewertet
  (relativ zur aktiven Gewichtssumme)
- `extraction.ocr_backend` („auto"/„rapidocr") wählt das OCR-Backend — RapidOCR
  liest die stilisierten Voyager-Cards messbar besser (0.92–0.99 vs WinRT-Fehlle-
  sungen), in `data/config.json` gesetzt
- Tests: Threshold-Skalierung (single/two/all pillars), Legacy-Verhalten ohne
  weight_sum, Pipeline-Integration (visual-only DVD-Scan assigned by title match)
*Dateien:* `matcher/solver.py`, `matcher/pipeline.py`, `core/config.py`,
`extractors/ocr_engine.py` (preference passthrough), `main.py` (debug-visual honors
preference), Tests in `tests/test_solver.py`/`tests/test_pipeline.py`/
`tests/test_binaries_downloader.py`/`tests/test_extractors.py`.

*Addendum (v0.12.0, Nutzer-Report „jetzt ist es 0/7"):* Batch 5 zählte die
AKTIVIERTEN Optionen, nicht die Säulen, die wirklich Evidence geliefert haben.
Der GUI-Scan hat DiscDB + Subtitles + Visual an (beide Defaults aktiv), aber
Voyager lieferte weder DiscDB- noch Subtitle-Evidence → weight_sum 0.85 statt
0.20 → Schwelle 0.55 × 0.85 = 0.4675 > 0.35 → ALLES rejected. Fix: die
Gewichtssumme wird über die Säulen gebildet, die tatsächlich Evidence
(> 0.0) beigesteuert haben — „enabled but empty" senkt die Schwelle nicht mehr.
Tests: `TestVisualOnlyDvdScan::test_enabled_but_empty_pillars_do_not_raise_threshold`
(rot reproduziert „No admissible match above the confidence threshold", grün nach Fix).
Zweiter Suite-Fix: der Seriennamen-Filter-Test nutzte den Modul-Helper-Guide
(„Test Show") statt der echten Serie — der Filter konnte dort nie greifen;
der Test trägt jetzt „Rocko's Modern Life". (580 green)

---

## Batch 2.8 — Chapter-Mark-Verdichtung (TDD) — [x] DONE (571 green)

**Nutzer-Beobachtung (Voyager S2):** der Episodentitel steht **kurz vor und nach dem
Ende des ersten Kapitels**, deutlich oben links — und die Messung bestätigte: die
Karte ist **kein Scene-Cut** (statisches Sternenfeld, langsames Fading) und nur
~2-3 s sichtbar.

*Fix:* die MKV-Kapitel-Marker werden gelesen (`mkv.chapters` → `chapter_marks`) und
das Title-Fenster bekommt **dichte 2,5-s-Probes innerhalb ±10 s um jedes Kapitel-Mark**
— zusätzlich zum 10-s-Uniform-Raster und den Scene-Cuts. Der Cap wird **evenly über
das ganze Fenster** angewendet (linspace inklusive Endpunkte — first-N would lose the
late cards), die Pipeline reicht die Chapter-Marks durch
(`_stage_visual → scanner.scan(chapter_marks=…)`).
*Tests:* dense zone um das Kapitel-Mark; Marks außerhalb des Fensters ignorieren;
keine Marks → Uniform-Grid unverändert; Pipeline-Durchreichung.
*Dateien:* `extractors/frame_scanner.py`, `matcher/pipeline.py`,
`core/models.py` (ChapterInfo-Basis), Tests in `tests/test_extractors.py` +
`tests/test_credit_scorer.py`.

---

## Batch 2.9 — Kapitel-verankertes Title-Fenster (TDD) — [x] DONE (576 green)

**Diagnose über alle 7 Voyager-Testdateien:** das Kapitel-1-Ende variiert
(169–346 s) — bei drei Dateien liegt der Opening-Credits-Block (mit den Cards)
**hinter** dem fixen 300-s-Fenster, das Title-Fenster verpasst sie strukturell.

*Fix:* `compute_scan_windows` erweitert das Title-Fenster-Ende auf
`erstes Kapitel-Mark + 120 s`, wenn ein Mark bekannt ist — die Credits-Blöcke
werden komplett abgedeckt, wo auch immer sie beginnen. Die Chapter-Zonen-Probes
(2.5 s ±10 s) bleiben; ohne Marks gilt das fixe Fenster unverändert.
*Tests:* compute_scan_windows mit/ohne Mark; die Probes jenseits 300 s bei
Late-Marks; der Out-of-window-Mark ignoriert.
*Live bestätigt:* die MULGREW-Card bei 04:25 (hinter dem fixen Fenster) erzeugt
jetzt den Serien-Cast-Hinweis.
*Dateien:* `extractors/frame_scanner.py` (`compute_scan_windows`, `_scan`).
