# EpisodeMatch — Post-Review Audit Plan (v0.1.0)

**Created:** 2026-11-10 · **Basis:** senior review (3 parallel audits) + independent
verification of every disputed finding against the actual codebase.
**No code was changed in this pass.** This plan is the worklist for the next pass.

**Verification method:** every finding was checked by code trace; disputed or
high-impact claims were additionally reproduced with isolated offline runs
(`python -B`, no file writes, no network). Verdicts below: **CONFIRMED** (reproduced or
proven by code trace), **PARTIALLY CONFIRMED** (real behavior, overstated impact),
**REJECTED** (claim not supported).

**Summary: 22 confirmed · 2 partially confirmed · 1 rejected · 1 by-design note.**
The existing test suite (400 tests, green) does **not** cover any of the confirmed
findings — each fix below must come with a regression test.

---

## Priority 0 — Wrong high-confidence assignments (breaks the core promise)

- [x] **A1. Season is dropped from DiscDB evidence — CONFIRMED** (repro: S02E01 disc
  evidence becomes S01E01 @ 1.0). Evidence carries only `episode_hint`; the solver
  stamps the requested season.
  *Fix:* store `season_hint` on `Evidence`; in `_stage_disc_db*` skip or downgrade
  evidence whose entry season ≠ requested season; solver asserts season match.
  Regression test: S02E01 disc fixture vs `season=1` scan.
  Files: `core/models.py` (Evidence), `matcher/pipeline.py` (`_apply_disc_matches`),
  `matcher/solver.py`.

- [x] **A2. Contradicting title index outranks an exact MPLS match — CONFIRMED**
  (repro: rip MPLS `00001.mpls` + index 2 ranks the wrong E2 entry at 1.0 above the
  exact-MPLS E1 entry). The index bonus ignores the MPLS contradiction, and equal
  scores fall back to list order.
  *Fix:* in `match_fingerprint`, ignore the index signal when both rip and entry carry
  MPLS names and they differ; give exact MPLS a decisive bonus over index+duration.
  Regression test with a two-entry equal-duration disc.
  File: `providers/thediscdb.py` (`match_fingerprint`).

- [x] **A3. Solver applies the confidence threshold *after* optimization — CONFIRMED**
  (repro: raw-value matrix picks two sub-threshold cells over one feasible 0.90 match
  and returns zero assignments). 
  *Fix:* zero out sub-threshold cells before the assignment; solve; report leftovers.
  Regression test with the raw-value repro.
  File: `matcher/solver.py` (`solve_assignment`).

- [x] **A4. Evidence aggregation double-counts and is uncalibrated — CONFIRMED.**
  (a) Visual pillar adds the same observation twice (per-hit evidence + summary), so a
  single OCR credit reaches 1.0 confidence (squared sum 1² + 0.4²). (b) The new
  unique-duration DiscDB signal (0.70) squares to 0.49 and falls below the 0.55 solver
  threshold despite being labeled near-conclusive.
  *Fix:* aggregate one calibrated score per pillar (no per-hit + summary double add);
  align DiscDB score scale with the solver threshold (report unique-duration at ≥0.90 or
  change threshold semantics); add calibration tests (single credit bounded by visual
  weight; unique-duration assigned).
  Files: `core/models.py` (`add_evidence`), `matcher/pipeline.py` (`_stage_visual`),
  `providers/thediscdb.py` (`match_fingerprint`).

## Priority 1 — Data loss & crashes

- [x] **B1. Partial rename failure has no undo journal — CONFIRMED on Windows** (repro:
  pre-existing temp name for the second item → `FileExistsError` after the first file
  already moved; journal count 0). The escaping `FileExistsError` also bypasses the
  CLI's `EpisodeMatchError` handler.
  *Fix:* write the journal incrementally (open as "pending" before the loop, finalize
  after), validate temp-name availability up-front, and wrap `OSError` in
  `RenameError`. Regression test: pre-existing temp path.
  Files: `postprocess/renamer.py` (`apply`), `main.py` (`_apply_renames`).

- [x] **B2. Second cleanup destroys the previous `.bak` — CONFIRMED.**
  `_swap()` unlinks an existing `episode.mkv.bak` before installing the new one.
  *Fix:* rotate (`bak` → `bak.old`) or refuse/skip with a warning; never unlink a
  backup without consent. Regression test: two consecutive cleanups.
  File: `postprocess/stream_cleaner.py` (`_swap`).

- [x] **B3. AV1 encoding overwrites existing outputs — CONFIRMED.**
  `add_job()` accepts an existing destination, ffmpeg runs with `-y`, completion does
  `replace(destination)`. The GUI's "Add folder" does not check either.
  *Fix:* reject existing destinations in `add_job` (unless an explicit overwrite flag is
  set) and drop `-y`; the GUI's add-folder must skip existing outputs. Regression test.
  Files: `postprocess/av1_encoder.py`, `ui/views/postprocess_view.py`.

- [x] **B4. CLI `--rename --tag` tags stale paths — CONFIRMED by code trace.**
  Renames run first; the tagger then uses `AssignmentEntry.file_path`, which no longer
  exists. Per-file tag failures are printed but the command reports success.
  *Fix:* apply renames first, then update entry paths from the rename map before
  tagging (or tag before renaming). Regression test.
  Files: `main.py` (`_cmd_match`), `postprocess/mkv_tagger.py`.

- [x] **B5. EBML fallback crashes an otherwise successful probe — CONFIRMED**
  (`AttributeError: 'EbmlTrack' object has no attribute 'get'`).
  `_merge_ebml_tracks()` treats the `EbmlTrack` dataclass as a dict.
  *Fix:* use attribute access; add a regression test with a subtitle-only-EBML
  metadata object.
  File: `extractors/ffprobe_reader.py` (`_merge_ebml_tracks`).

- [x] **B6. Malformed HTTP-200 payloads escape the provider/LLM error boundaries —
  CONFIRMED.** `response.json()` is unguarded in SubDL, OpenSubtitles and the LLM
  client; a 200-with-HTML response raises `JSONDecodeError`, which the pipeline's
  `EpisodeMatchError` handlers do not catch and can abort a scan.
  *Fix:* wrap `response.json()` into `ProviderError`/`LlmError` in all three clients;
  the LLM retry loop treats it as retryable. Offline mock tests (200 + HTML body).
  Files: `providers/subdl.py`, `providers/opensubtitles.py`, `matcher/llm_arbiter.py`.

## Priority 2 — Parser correctness (pillar-2 quality)

- [x] **C1. ODS object reuse across display sets drops subtitles — CONFIRMED**
  (repro: epoch-start display set with ODS, later normal-case PCS reusing the same
  object → only one event). The assembled-object store is cleared per display set,
  although PGS objects persist for the whole epoch.
  *Fix:* keep assembled objects until an EPOCH_START clears them; clear only the
  per-display-set placement context. Regression test: reuse stream must yield 2 events.
  File: `extractors/pgs_parser.py` (`_consume_display_set`).

- [x] **C2. Partial PDS update wipes previous palette colors — CONFIRMED** (repro:
  two-color palette, then one-entry update → color 1 alpha drops to 0).
  *Fix:* verify against SubtitleEdit/ffmpeg whether updates merge or replace; if merge
  (likely for real streams), merge entries per palette id. Regression test.
  File: `extractors/pgs_parser.py` (`_consume_display_set` → PDS branch).

- [x] **C3. Terminal PGS event gets zero duration — CONFIRMED** (repro: stream ends
  with a visible display set and no clearing PCS → `end == start`).
  *Fix:* `refine_event_timing` falls back to `default_event_duration` when no later
  boundary exists. Regression test.
  File: `extractors/pgs_parser.py` (`refine_event_timing`).

- [x] **C4. The "max 40 PGS events" cap only limits OCR, not parsing/RLE decode —
  CONFIRMED.** The full `.sup` is read and every event bitmap is decoded before the cap
  applies (also the cause of large `data/subtitles` growth together with B-?). 
  *Fix:* decode lazily (store raw ODS, decode+OCR only the selected events) or stop
  parsing after enough in-window events. Test: full-film sup with tiny cap → bounded
  decode.
  Files: `extractors/pgs_parser.py` (`parse_sup`), `extractors/subtitle_extractor.py`
  (`_extract_pgs`).

- [x] **C5. ffprobe single-column scene CSV is never parsed — CONFIRMED**
  (`-of csv=p=0` with one entry prints bare timestamps; the parser requires a comma, so
  the lavfi path always falls back to the slower ffmpeg path).
  *Fix:* accept single-value rows in `parse_lavfi_scene_lines`. Regression test with
  real ffprobe output.
  File: `extractors/frame_scanner.py`.

## Priority 2 — Threading, shutdown & deadlocks

- [x] **D1. AV1 worker can deadlock on a full stderr pipe — CONFIRMED by code trace**
  (stdout is drained synchronously; stderr is only read after `wait()`, which can never
  be reached if ffmpeg blocks on a full stderr buffer).
  *Fix:* redirect stderr to a temp file or drain it in a background thread; keep the
  stdout progress loop. Test: a fake encoder that floods stderr.
  File: `postprocess/av1_encoder.py` (`_run_job`).

- [x] **D2. App shutdown closes the shared database under live workers — CONFIRMED by
  code trace.** View shutdowns wait only 1.5 s; `FluentWindow.closeEvent` then closes
  the database while network/remux/encode workers can still run. 
  *Fix:* workers get a cancellation + completion handshake; the main window closes the
  database only after all views report idle (with user-visible "wait or cancel" choice).
  Files: `ui/main_window.py`, `ui/views/*.py` (`shutdown`), `ui/workers.py`.

## Priority 2 — Provider & quota behavior

- [x] **E1. SubDL accepts wrong-episode/wrong-language results and mislabels them —
  CONFIRMED** (offline repro: request E1/de → E2/fr content returned as `language=de`).
  The candidate loop only *prefers* the episode match and never validates language or
  episode identity of the downloaded file.
  *Fix:* reject candidates whose `episode` field mismatches (when present) and validate
  the language attribute; label `Transcript.language` from the actual match.
  Regression tests with an E2/fr candidate.
  File: `providers/subdl.py` (`reference_transcript`).

- [x] **E2. OpenSubtitles daily-quota block suppresses cached references — CONFIRMED**
  by code trace. The quota early-return happens before the local cache is consulted,
  so after a 406 even already-downloaded SRTs become unavailable for other episodes.
  *Fix:* consult the local SRT cache before honoring the quota block (move the block
  after the cache lookup). Test: quota-blocked client with cached SRT still returns the
  reference.
  File: `providers/opensubtitles.py` (`reference_transcript`).

## Priority 3 — Config, frozen app, packaging

- [x] **F1. `AppConfig.masked()` leaks the SubDL API key — CONFIRMED** (masked dict
  returns the key verbatim).
  *Fix:* add `subdl_api_key` to the masked fields; test asserts no verbatim key in
  `masked()` output.
  File: `core/config.py` (`masked`).

- [x] **F2. Frozen EXE resolves `bin/`/`data/` against the launch CWD — CONFIRMED by
  code trace** (`_dev_anchor` returns None when frozen). Launching via a shortcut from
  another directory silently forks tools/settings/cache; the build smoke test only
  runs from the project root.
  *Fix:* when frozen, anchor `bin/`/`data/` at the directory of `sys.executable`;
  add a frozen-simulation regression test.
  File: `core/config.py` (`_dev_anchor`/`_resolve_relative`).

- [x] **F3. Release ZIP does not contain the project's own MIT `LICENSE` — CONFIRMED**
  in the existing release asset. The guard also only checks six exact EXE names, not
  arbitrary accidentally-included files.
  *Fix:* include `LICENSE` at the ZIP root (and extend the guard to flag unknown
  top-level executables). Test asserts `LICENSE` present.
  File: `scripts/build_exe.py` (`build_release_zip`).

- [x] **F4. Audio/visual scan-window settings are never passed to the extractors —
  CONFIRMED by code trace** (`ExtractionSettings.audio_window_*` and
  `title_card_window_minutes` exist but the pipeline uses hard-coded defaults).
  *Fix:* plumb the values into `AudioSnipper`/`FrameScanner` construction. Test with a
  non-default config asserting the windows change.
  Files: `matcher/pipeline.py` (`_subtitle_service`/`_audio_service`/`_visual_service`).

- [x] **F5. Season-range releases are only matched for their first season — CONFIRMED**
  (`seasons-6-10-blu-ray` matches a season-6 scan but not season 7–10).
  *Fix:* parse range patterns and treat a season inside the range as a match. Test.
  File: `matcher/pipeline.py` (`_disc_release_folders`).

- [x] **F6. `CacheDatabase(read_only=True)` is not enforced — CONFIRMED.** The
  connection stays writable and `put_json`/`delete`/`purge_expired` mutate.
  *Fix:* open with `file:...?mode=ro` URI and raise on mutation calls in read-only
  mode. Test.
  File: `core/database.py`.

## Rejected claims (no action)

- [x] **REJECTED — "Cropped PCS field order reversed"** (claimed
  `pgs_parser.py:498-514`). Verified against the BDA/ffmpeg field order: the cropping
  rectangle comes *before* the object position, and our parser handles a spec-compliant
  payload correctly (crop=(3,4,5,6), position=(100,200)). The auditor's repro payload
  was built in a non-compliant order. **No fix needed.**

## Housekeeping (non-code)

- [ ] **H1. Uncommitted local change:** `packaging/README-FIRST.txt` (the disclaimer
  addition) is modified in the working tree but not committed. Commit it (and the
  release asset already carries it).
- [ ] **H2.** The repo has no CI workflow; the audit's repros (this plan's regression
  tests) should run automatically on push once added (`build-macos.yml`/CI discussion
  is a separate topic).

---

## Suggested execution order

1. **A1–A4** (wrong high-confidence assignments) — restores trust in the scores.
2. **B1–B4** (data loss) — before letting anyone else run renames/cleanup/encode.
3. **B5, C1–C5** (crashes + parser correctness) — improves pillar-2 yield directly.
4. **E1–E2, F1** (provider correctness + key hygiene).
5. **D1–D2** (threading) and the remaining P3 items.
