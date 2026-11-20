# Review Plan — DiscForge feedback (David), v0.11.3

**Basis:** `_export/Feedback/review-episodematch.md` (Abschnitte 1, 2, 4, 7 und 8/CI).
**Geprüft gegen:** `master` @ 0.11.3 (Commit `d0d46c2` und früher).
**Verfahren:** jeder Punkt wurde gegen den echten Code nachvollzogen; Verdicts:
**CONFIRMED** (trifft zu), **ALREADY FIXED** (abgelöst), **NOT APPLICABLE** (trifft
unsere Architektur nicht), **DEFERRED** (real, aber bewusst zurückgestellt).

Arbeitsweise wie bei `AUDIT_PLAN.md`: TDD — Regressionstest zuerst (rot), Fix,
komplette Suite grün, Punkt abhaken. Nach jedem Batch kurzes Update, nächster
Batch erst auf Freigabe.

---

## Batch 1 — TheDiscDb-Parsing (Korrektheit: falsch verworfene/falsch benannte Titel)

Alle drei Punkte betreffen `providers/thediscdb.py` (`parse_disc_json`,
`DiscTitleEntry`) und wirken direkt auf Zuordnung und Dateinamen — daher Priorität 1.

- [x] **1.1 Episoden-Ranges (`"1-2"`) werden still verworfen — CONFIRMED.**
  `_safe_int("1-2")` → `None` → `DiscTitleEntry.episode is None` → `is_episode`
  = False → ein legitimer Doppel-Episoden-Premiere-Titel ist kein Kandidat und
  landet immer unassigned.
  *Fix:* `DiscTitleEntry` bekommt `episode_end: int | None`;
  `parse_disc_json` parst `"1-2"` / `"1 - 2"` zu `episode=1, episode_end=2`
  (Regex `^\s*(\d+)\s*[-–]\s*(\d+)\s*$`); `is_episode` erfüllt bei Range.
  `_apply_disc_matches`/Solver/Referenz-Fetch nutzen `episode` (Start); die
  Evidence nennt die Spanne. Naming: bei gesetztem `episode_end` erzeugt das
  **Default-Rendering automatisch** `S01E01-E02` (Jellyfin-Multi-Episode-Form) —
  ohne dass der Nutzer ein Custom-Template braucht (`format_episode_name` rendert
  `{episode:02d}` als `01-E02`, wenn `episode_end` übergeben wird).
  **Solver:** eine Range-Zuordnung (E01-E02) verbraucht **beide** Episoden —
  `solve_assignment` markiert die abgedeckten Folge-Episoden (E02) als belegt,
  damit sie nicht an eine andere Datei vergeben werden (`episode_numbers`-Liste
  um die von Ranges abgedeckten Episoden reduzieren bzw. im Assignment-Ergebnis
  als „covered by range" buchen).
  *Test:* `tests/test_thediscdb_url.py` — Range-Entry wird Kandidat, matched auf
  E01, `episode_end=2`; Naming-Test für `S01E01-E02` mit dem Default-Template;
  Solver-Test: zweite Datei bekommt bei Range-Zuordnung nicht E02 (nächste freie
  Episode), Unassigned-Best-Guess bleibt korrekt.
  *Dateien:* `providers/thediscdb.py`, `core/config.py` (`format_episode_name`),
  `matcher/solver.py` (Range-Belegung im globalen Assignment),
  `matcher/pipeline.py` (Evidence-Spanne, Übergabe `episode_end`).

- [x] **1.2 `Title` kann ein suggested filename sein — CONFIRMED.**
  `episode_title=item.get("Title")` läuft ungefiltert ins Template
  (`… - S06E09 - Parks and Recreation.S06.E09.The Cones of Dunshire.mkv.mkv`).
  *Fix:* Normalisierung beim Parsen (nur wenn der String `.mkv`/`.mp4`/`.avi`
  enthält — sonst bleibt ein punktierter echter Titel unangetastet): führendes
  `<anything>.Sxx.Eyy.`-Präfix und Trailing-Extension strippen; Ergebnis zusätzlich
  über `sanitize_filename` jagen.
  *Test:* Range an Grenzfällen — echten Titel mit Punkten (bleibt), Filename-Titel
  (wird sauber), leeren Title (None bleibt None).
  *Dateien:* `providers/thediscdb.py` (+ Helper `_normalize_disc_title`).

- [x] **1.3 Disc-Titel schlägt TMDB-Titel — CONFIRMED (Präzedenz falsch).**
  `AssignmentEntry.episode_title` kommt aus dem TMDB-Guide
  (`episode_titles.get(episode)`); der Disc-Titel erscheint nur im Evidence-String.
  Bei Pilots weicht der „Broadcast"-Titel der Disc vom TMDB-Produktionstitel ab —
  der Nutzer hält die Box in der Hand.
  *Fix:* in `_apply_disc_matches` den Disc-Titel als `episode_title` setzen, wenn
  vorhanden **und nicht generisch**: Platzhalter-Titel wie `"Episode 1"`,
  `"Title 01"`, `"Episode 12"` (Regex `^(episode|title|ep)\s*\d+$`, case-insensitive)
  werden gefiltert — in dem Fall fällt die Präzedenz zurück auf den TMDB-Titel.
  Bei echter Abweichung einen Evidence-Eintrag „Disc title X vs TMDB title Y"
  (display-only) erzeugen statt still zu wählen.
  *Test:* Scan mit abweichenden Titeln → Entry trägt Disc-Titel + Abweichungs-Evidence;
  generischer Disc-Titel („Episode 1") → TMDB-Titel gewinnt.
  *Dateien:* `matcher/pipeline.py` (`_apply_disc_matches`).

---

## Batch 2 — Fingerprint-Scoring (Robustheit der Pillar-1-Scores)

Alle Punkte in `providers/thediscdb.py` (`match_fingerprint`).

- [x] **2.1 Duration-Uniqueness hängt am geladenen Kandidaten-Set — CONFIRMED.**
  `duration_matches` zählt über **alle** Einträge **aller** geladenen Discs
  (line ~220); der Evidence-String behauptet trotzdem „unique on disc". Dieselbe
  Datei kriegt +0.70 (nur Disc 1 geladen) oder +0.20 (ganze Saison geladen).
  *Fix:* Uniqueness **pro Disc** zählen — „unique on its own disc" ist die
  Aussage, die das Evidence-String bereits macht. Der volle `+0.70`-Unique-Bonus
  wird nur vergeben, wenn die Disc **gepinnt** ist (Disc-URL) oder durch das
  2.2-Layout-Gate bestätigt ist; bei mehreren ungepinnten, nicht bestätigten
  Discs wird der Bonus sauber gedämpft (Ladder: unique-on-own-disc mit
  Layout-Bestätigung → 0.70; ohne Bestätigung → 0.35), damit nicht mehrere
  ungepinnte Discs gleichzeitig `+0.70` liefern. Evidence-String korrigieren
  („unique on its own disc" / „unique on its own disc, layout confirmed").
  *Test:* Zwei Discs mit überlappenden Dauern — ungepinnt: gedämpfter Bonus auf
  beiden, kein Doppel-0.70; gepinnt bzw. layout-bestätigt: voller Bonus; Claim-String
  prüft die jeweilige Aussage.
  *Dateien:* `providers/thediscdb.py`, `matcher/pipeline.py` (Pinned-Kontext
  an `match_fingerprint` durchreichen).

- [x] **2.2 Disc-Layout-Gate (name+duration als Einheit) — CONFIRMED als Vorschlag.**
  Heute entscheidet der Single-Title-Score; ein falscher Eintrag mit passender
  Dauer (> 0.55) wird nur durch den Solver abgefangen. Vorschlag von David: vor
  der Titel-Verarbeitung die **Disc-Identität** bewerten — Anteil der Episode-
  Playlists der Disc, die im Rip mit MPLS **und** Dauer vorhanden sind — und
  ein Layout-Match ≥ 0.8 als Gate/Boost nutzen („this is the right disc").
  *Fix:* `match_fingerprint` um einen Disc-Level-Score erweitern
  (`disc_layout_score` pro `ParsedDisc`); bei ≥ 0.8 bekommen die Titel dieser
  Disc einen Bonus und die Evidence nennt „disc layout match N/M"; darunter
  kein Bonus (Gate statt Verwerfen — ein zu aggressives Verwerfen würde
  Contributor-Fehler härter bestrafen als nötig).
  *Test:* TestDiscLayoutGate — korrekte Disc (alle Playlists MPLS+Dauer) → Bonus +
  Layout-Evidence; fremde Disc mit gleicher Nummerierung → kein Bonus.
  *Dateien:* `providers/thediscdb.py`, `matcher/pipeline.py` (Evidence-Übergabe).

- [x] **2.3 Ungenutzte Signale — TEILWEISE / NOT APPLICABLE post-rip.**
  (a) SegmentMap-**String**-Vergleich: die Disc-Seite trägt die Map, aber der
  gerippte MKV enthält keine Blu-ray-Clip-IDs — post-rip steht nur die
  Chapter-Anzahl gegenüber (davon lebt der bestehende +0.05-Count-Check). Ein
  String-Vergleich ist ohne Pre-Rip-Daten (BDMV/makemkvcon) unmöglich → dieser
  Teil ist **NOT APPLICABLE** für unseren post-rip-Ansatz (wie §3). Der
  Count-Vergleich wurde gegen Whitespace-Abweichungen robuster gemacht.
  (b) `ContentHash`/`GlobalDiscId`: ebenfalls **keine Rip-Seiten-Quelle** (MakeMKV
  schreibt weder den Hash noch die Disc-ID in die MKV-Tags) → als Bonus-Gate
  **NOT APPLICABLE** post-rip; die Felder bleiben in `ParsedDisc` für einen
  künftigen Pre-Rip-Pfad erhalten.
  (c) `Type` dominiert Dauer bereits korrekt (`is_episode`) — jetzt durch einen
  Test fixiert: ein Extra mit Episodenlänge ist nie ein Kandidat
  (`TestTypeOverridesDuration`).
  *Umgesetzt:* (c) Test; Count-Vergleich robuster. (a)/(b) als NA dokumentiert.
  *Test:* `tests/test_thediscdb_url.py::TestTypeOverridesDuration`.
  *Dateien:* `providers/thediscdb.py`, `extractors/ffprobe_reader.py`
  (nur falls Fingerprint-Feld ergänzt wird), `core/models.py` (DiscFingerprint).

---

## Batch 3 — Subtitle-IDF & Sync-Check (Präzision von Pillar 2)

- [x] **4.1 IDF über die Kandidaten-Referenzen — CONFIRMED.**
  `text_scorer.score_episode` rechnet `coverage = confirmed / denominator` ohne
  Gewichtung; die 9–20 %-Kollisionslinie kommt aus show-generischen Zeilen, die in
  jeder Referenz stehen. `tfidf_cosine` existiert bereits (Plot-Pfad) — der
  Mechanismus ist da.
  *Fix:* Referenz-Coverage mit IDF über die **N Kandidaten-Referenzen** gewichten:
  Zeilen, die in allen Referenzen vorkommen, → Gewicht ~0; `matched/total`-Falle
  vermeiden (Vokabular-Größe dämpft automatisch über IDF). Bei N < 3 Referenzen
  keine IDF-Anwendung (Log-Hinweis „IDF needs ≥ 3 references"), Coverage unverändert.
  Evidence erweitern: die **passenden Zeilen** (top-3 mit Referenztext) statt nur
  der Zahl. `N >= 3` zählt dabei ausschließlich **erfolgreich geladene,
  nicht-leere Referenz-Transkripte** (leere/fehlgeschlagene Fetches zählen weder
  zur IDF-Grundlage noch zur Log-Meldung).
  *Test:* 3 Referenzen mit geteilten generischen Zeilen — die richtige Episode
  gewinnt auch, wenn eine falsche 3/8 vs. 11/80 „bessere Quote" hat; N=2 → kein
  IDF, Log-Hinweis.
  *Dateien:* `matcher/text_scorer.py`, `matcher/pipeline.py` (Referenz-Set
  an den Scorer reichen).

- [x] **4.2 Sync-/Cut-Check als Evidence — CONFIRMED (fehlt komplett).**
  Ein Referenz-Untertitel ist auf einen Retail-Master getaktet; eine andere Fassung
  verliert legitimerweise Zeilen und sieht im Score aus wie die „falsche Episode".
  *Fix:* Vor dem Scoring first/last Cue der Referenz gegen die eigene Extraktion
  vergleichen; bei einer Abweichung von **mehr als 20 Sekunden** (feste Schwelle,
  am Start- oder End-Cue gemessen) den Kandidaten als
  „reference is a different master (offset ~X s)" markieren — als Evidence-Tag,
  nicht als Score-Strafe.
  *Test:* Verschobene Referenz → Evidence-Tag gesetzt, Ranking unverändert;
  passende Referenz → kein Tag.
  *Dateien:* `matcher/text_scorer.py` oder `matcher/pipeline.py` (Check vor Scoring),
  `core/models.py` (Evidence `kind="sync_check"`).

- [x] **4.3 seconv-„never for delivery" + C4 lazy decode — TEILWEISE ALREADY FIXED.**
  C4 ist gelöst (`pts_windows` begrenzt das RLE-Decode auf das Scan-Fenster —
  TestWindowedDecode); David's Hinweis zielt auf den Zustand vor dem Fix. Der
  „PGS never converted as a delivery decision"-Punkt ist eine **Policy/README-Notiz**,
  kein Code-Change.
  *Fix:* Ein Satz in `README.md` (Pillar-2-Abschnitt): PGS bleibt auf Blu-ray
  unangetastet; Konvertierung betrifft nur die Matching-Extraktion und DVD/VobSub.
  *Test:* keiner (Doku).
  *Dateien:* `README.md`.

- [x] **4.4 SubDL-Tier-Angabe prüfen — UNVERIFIED.**
  README sagt 300 downloads/day, David's Notizen sagen 50. Gegen die API-Doku
  prüfen und README korrigieren; falls 50: Hinweis verstärken, dass Season-Packs
  (1 Download) den Engpass entschärfen (Cache-Verhalten haben wir schon).
  *Test:* keiner (Doku).
  *Dateien:* `README.md`.

---

## Batch 4 — Renamer Cross-FS & GitHub Actions CI

- [x] **7.1 Cross-Filesystem-Rename (EXDEV) — CONFIRMED.**
  `Renamer.apply()` nutzt `item.source.rename(temp)`; liegt die Quelle auf einem
  anderen Laufwerk als das Ziel (typisch: NAS-Share), wirft das `OSError EXDEV`
  → Batch bricht (zwar sauber mit Journal, aber die Aktion scheitert komplett).
  *Fix — Staged-Copy nur als EXDEV-Fallback (der normale Zwei-Step-Rename bleibt):*
  Auf demselben Dateisystem bleibt der sofortige `rename()` aktiv — es werden
  **keine 50-GB-Blu-ray-Dateien kopiert**, nur weil der Pfad es erlaubt. Erst wenn
  `source.rename(temp)` mit `OSError` und `errno.EXDEV` scheitert, greift der
  Fallback:
  1. `temp` liegt bereits **im Zielverzeichnis** (`destination.parent`, Name
     `.{destination-stem}.emtmp{suffix}` — `validate()` prüft ihn vorab);
  2. `shutil.copy2(source, temp)` (funktioniert über Dateisystem-Grenzen und
     erhält Metadaten);
  3. Dateigröße von `temp` gegen `source` verifizieren (Copy-Integrität) —
     Abweichung → `temp` löschen, `RenameError`, Batch bricht mit Journal;
  4. `temp` **atomar** nach `destination` umbenennen (same-FS Rename, immer
     erfolgreich);
  5. `source` erst **ganz zum Schluss** löschen — ein Fehlschlag dieses Löschens
     wird protokolliert, aber bricht den Batch nicht mehr (die Zuordnung ist zu
     dem Zeitpunkt bereits im Journal; die Datei existiert dann doppelt statt gar
     nicht).
  Die Reviewer-Kritik am „original aside before replacement"-Ordering trifft unser
  Schema damit strukturell nicht: die Quelle bleibt bis zum verifizierten Staging
  unangetastet; das Docstring erklärt den Ablauf.
  *Test:* `TestCrossFilesystem` — `Path.rename` monkeypatched zu EXDEV für den
  ersten Schritt → Copy-Pfad läuft, Größe geprüft, Journal korrekt, Undo rundet
  ab; Größen-Abweichung → RenameError + temp aufgeräumt.
  *Dateien:* `postprocess/renamer.py`.

- [x] **7.2 „Skip when target exists"-Datenverlust — NOT APPLICABLE.**
  Wir haben keinen Skip-Guard: `validate()` fail-fast bei existierendem Ziel,
  `_resolve_collision` vergibt " (2)"-Suffixe statt zu überspringen, und ein
  Re-Run nach Mapping-Fix erzeugt neue Namen, während alte falsch benannte Dateien
  unangetastet bleiben. Kein Handlungsbedarf; im Plan dokumentiert, damit es
  nicht als „offen" wieder auftaucht.

- [x] **8.1 GitHub Actions CI — CONFIRMED (H2).**
  Kein Workflow im Repo; die 480 Tests sind nur lokal reproduzierbar.
  *Fix:* `.github/workflows/tests.yml` — `windows-latest`, Python 3.11 + 3.12,
  `pip install -e .[dev,gui]`, `pytest` (Suite ist offline-fähig; Qt-Tests laufen
  offscreen via conftest-Hard-Exit). Ergänzend ein `concurrency`-Guard und die
  Matrix mit **`fail-fast: false`**, damit Fehler auf Python 3.11 und 3.12
  unabhängig voneinander sichtbar bleiben. Optional (separater Schritt): Badge
  ins README.
  *Test:* Workflow selbst — grüner Run auf push; kein Unit-Test (nur
  `.github/workflows/tests.yml` + README-Badge-Zeile).
  *Dateien:* `.github/workflows/tests.yml` (neu), `README.md` (Badge, optional).

---

## Zurückgestellt / ausgeschieden (mit Begründung)

- [ ] **1.4 GitHub Tree API (`/git/trees/main?recursive=1`) — DEFERRED.**
  Real, aber unser Walk ist **cache-gedeckt**: series-/release-/file-Listings
  liegen mit 7-Tage-TTL in SQLite (`thediscdb:series-listing` etc.), ein Cold-Scan
  braucht ~5–10 unauthentifizierte Requests (Limit: 60/h) — weit weg vom Limit.
  Der Tree-Kauf wäre **1 Request à ~11 MB** bei jedem Refresh (Parsing- und
  Transfer-Kosten gegen null Ersparnis im Normalfall). `IncompleteRead` betraf
  `urllib` — unser Client ist `httpx` (streamt sauber). URL-Encoding haben wir
  (`_url_encode_path`).
  *Alternative, falls Rate-Limits doch stören:* `ETag`/`If-None-Match` auf die
  drei Contents-Calls (304 → Cache bleibt gültig) — gleiche Ersparnis ohne
  11-MB-Payload. Erst umsetzen, wenn ein realer 403/Rate-Limit-Fall auftritt.
  *Dateien (später):* `providers/thediscdb.py`.

- [x] **§3 EPL_nn / BDMV/PLAYLIST / TINFO — NOT APPLICABLE (nur geprüft wie gefordert).**
  `TINFO`/`EPL_`-Strings kommen in unserer Pipeline nirgends vor (`grep` im src
  leer): wir arbeiten post-rip auf MKVs und lesen ausschließlich den MKV-`TITLE`-Tag
  (MPLS-Name oder Disc-Label, `_MPLS_PATTERN` in `ffprobe_reader.py`). Das Signal
  existiert nur **pre-rip** (MakeMKV-Konsolen-Output bzw. `BDMV/PLAYLIST/*.mpls`)
  — für unseren post-rip-Ansatz strukturell nicht erreichbar. Interessant erst für
  einen späteren Rip-Integrationspfad (§9-Themenfeld, außerhalb dieses Plans).

- **§5 (Track-Cleanup/AV1), §6 (verify-Befehl), §9 (Linux-Portierung)** — laut
  Aufgabenstellung explizit ausgeschlossen; §6 bleibt als eigenes künftiges Thema
  notiert (die §6-Ideen überschneiden sich teilweise mit Batch 2.2).

- **§8 „smaller" (README-Tonalität, Season-Vorschlag aus Disc-Label, SMB-Kolonnen-
  Hinweis)** — notiert, aber kein Batch: README-Ton und Season-Vorschlag sind
  Produktentscheidungen, der SMB-Hinweis ist eine README-Zeile und kann mit dem
  nächsten README-Durchlauf mitkommen.
