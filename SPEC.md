\# Projekt-Spezifikation: EpisodeMatch



\## 1. Kernziel \& Fokus

\*\*EpisodeMatch\*\* ist eine spezialisierte Windows-Desktop-Applikation in Python (3.11+) mit \*\*PySide6\*\* (LGPL) und einem komplett selbst geschriebenen, modernen Windows-11-Dark/Fluent-QSS-Stylesheet samt eigenen Custom-Widgets (bewusst \*kein\* GPL-lizenziertes PyQt-Fluent-Widgets, da das Projekt unter der MIT-Lizenz veröffentlicht wird).



\*\*Hauptfokus (Priorität 1):\*\*

Beim Rippen von Blu-ray- und DVD-Serien (z. B. mit MakeMKV) entstehen anonyme Dateien (`title\_t00.mkv`, `title\_t01.mkv`...), deren Reihenfolge auf der Disc oft nicht der echten Episoden-Reihenfolge entspricht. \*\*EpisodeMatch\*\* identifiziert jede `.mkv`-Datei mit nahezu 100%iger Sicherheit und ordnet ihr die exakte Staffel und Episode (`SxxExx - Episodentitel`) zu.



\*\*Zusatzfunktionen (Priorität 2 – Optionales Post-Processing):\*\*

Nach der erfolgreichen Identifizierung und Umbenennung kann der Nutzer optional überflüssige Audio-/Untertitelspuren bereinigen (inkl. automatischer Forced-Subtitle-Erkennung) oder die sauber benannten Episoden in eine `ffmpeg` AV1 (`SVT-AV1`) Encoding-Queue schicken.



\---



\## 2. Strikte Coding-Regeln für den Agenten (zcode)

1\. \*\*Keine Platzhalter!\*\* Schreibe niemals `# TODO`, `pass` oder Dummy-Implementierungen. Jede Klasse, jeder Parser und jeder Algorithmus muss vollständig und produktionsreif ausprogrammiert werden.

2\. \*\*Strikte Typisierung:\*\* Nutze durchgehend Python Type Hints und `pydantic` (v2) Datenmodelle.

3\. \*\*Intelligentes Binary-Management (`./bin/`):\*\*

&#x20;  - `src/episodematch/core/binaries.py` sucht benötigte Tools (`ffmpeg.exe`, `ffprobe.exe`, `mkvmerge.exe`, `mkvextract.exe`, `mkvpropedit.exe`) zuerst im lokalen Projektordner `./bin/`.

&#x20;  - Falls dort nicht gefunden, wird `shutil.which()` (System-PATH) und `C:\\Program Files\\MKVToolNix\\` geprüft.

&#x20;  - `bin\_downloader.py` ermöglicht das automatische Herunterladen und Entpacken fehlender Tools direkt nach `./bin/`.



\---



\## 3. Die 5-Säulen Identifizierungs-Engine (Das Herzstück)



Um eine `.mkv`-Datei zweifelsfrei der richtigen Folge zuzuordnen, kombiniert \*\*EpisodeMatch\*\* fünf unabhängige Erkennungsmethoden und berechnet daraus einen gewichteten \*\*Confidence-Score (0–100 %)\*\* inklusive transparenter Begründung für den Nutzer:



1\. \*\*Säule 1: Blu-ray Playlist \& TheDiscDB-Abgleich (`thediscdb\_matcher.py`)\*\*

&#x20;  - Liest via `ffprobe` die von MakeMKV eingebetteten Metadaten aus (insb. `SOURCE\_ID` wie `00800.mpls` oder Segment-Maps, exakte Byte-Größe, Kapitelanzahl und exakte Laufzeit auf die Sekunde).

&#x20;  - Gleicht diese technischen Fingerabdrücke direkt mit den Daten von `thediscdb.com` (GraphQL/GitHub-Datenstruktur) ab.



2\. \*\*Säule 2: Untertitel-Extraktion \& Dialog-Abgleich (`subtitle\_matcher.py`)\*\*

&#x20;  - Extrahiert Text-Untertitel (SRT/ASS) oder Blu-ray Bild-Untertitel (PGS `.sup`) direkt aus der MKV.

&#x20;  - Enthält einen \*\*vollständigen nativen Python-Binärparser für PGS-Untertitel (`pgs\_parser.py`)\*\*, der RLE-komprimierte Untertitel-Bitmaps aus den ersten und letzten 10 Minuten der Episode dekodiert und per lokaler OCR (`winrt-Windows.Media.Ocr` / Windows Media OCR API, alternativ `rapidocr-onnxruntime`, Fallback `pytesseract`) in Text umwandelt.

&#x20;  - Gleicht die extrahierten Dialogzeilen mit Referenz-Untertiteln (OpenSubtitles) sowie mit Episoden-Inhaltsangaben (TMDB / IMDb) ab.



3\. \*\*Säule 3: Audio-Snipping \& Whisper-Transkription (`audio\_matcher.py`)\*\*

&#x20;  - Falls keine Untertitel vorhanden sind: Schneidet per `ffmpeg` zwei kurze Audio-Fenster heraus (z. B. Minute 02:30–04:30 nach dem Intro und 5 Minuten vor Ende) als 16kHz Mono-WAV.

&#x20;  - Transkribiert das Gesprochene lokal via `faster-whisper` und sucht nach Charakternamen, Orten und Schlüsselbegriffen der jeweiligen Episode.



4\. \*\*Säule 4: Visuelle Title-Card \& Credits OCR (`visual\_matcher.py`)\*\*

&#x20;  - Nutzt `ffmpeg` Scene-Change-Detection in den ersten 5 Minuten (typische Position für eingeblendete Episodentitel, "Written by...", "Directed by...") sowie am Beginn des Abspanns (Gastschauspieler).

&#x20;  - Extrahiert relevante Keyframes, liest eingeblendeten Text per OCR aus und gleicht Regisseure, Autoren und Episodentitel direkt mit den IMDb/TMDB-Credits der Staffel ab!



5\. \*\*Säule 5: LLM Deep-Context Arbiter (`llm\_arbiter.py`)\*\*

&#x20;  - Bündelt alle gesammelten Indizien \*aller\* MKV-Dateien eines Ordners (Laufzeiten, MPLS-Namen, OCR-Textfetzen aus Title-Cards, Untertitel-/Whisper-Transkripte) und schickt sie zusammen mit dem kompletten Episodenguide der Staffel (Plots, Autoren, Gaststars von TMDB/IMDb) an eine OpenAI-kompatible LLM-API (GLM Flash).

&#x20;  - Löst selbst schwierigste Fälle über eine globale Zuordnungs-Matrix (Ungarischer Algorithmus / Assignment Problem), sodass jede Episode der Disc eindeutig vergeben wird.



\---



\## 4. Verzeichnisstruktur (Vom Agenten exakt so anzulegen)



```text

EpisodeMatch/

├── bin/                         # Lokaler Ordner für ffmpeg.exe, ffprobe.exe, mkvextract.exe etc.

├── data/                        # SQLite-Cache (TheDiscDB, TMDB, OCR/Whisper-Cache), Temp-Frames

├── pyproject.toml               # Dependencies: PySide6 (Extra gui), pydantic, httpx, faster-whisper (Extra audio), pillow, beautifulsoup4, rapidfuzz, scipy, pytest

├── SPEC.md

├── src/

│   └── episodematch/

│       ├── \_\_init\_\_.py

│       ├── main.py              # App Entrypoint

│       ├── core/

│       │   ├── config.py        # Settings, API-Keys (TMDB, OpenSubtitles, GLM-Flash), Naming-Templates

│       │   ├── binaries.py      # Sucht \& prüft ffmpeg, ffprobe, mkvtoolnix in ./bin/ und System-PATH

│       │   ├── bin\_downloader.py# Automatischer Downloader für fehlende Binaries nach ./bin/

│       │   ├── models.py        # Pydantic-Modelle (MkvFile, StreamInfo, Evidence, EpisodeInfo, MatchCandidate, SeasonAssignment)

│       │   └── database.py      # SQLite-Datenbank zum Cachen von Scans, IMDb/TMDB-Daten \& TheDiscDB

│       ├── extractors/          # <--- DATEN AUS DER MKV ZIEHEN

│       │   ├── ffprobe\_reader.py# Liest Dauer, Kapitel, Streams, MPLS-Tags, Segment-Maps

│       │   ├── pgs\_parser.py    # Nativer Binär-Parser für Blu-ray .sup (PGS) inkl. RLE-Bitmap-Dekodierung!

│       │   ├── ocr\_engine.py    # Wrapper für Windows-OCR (winrt-Windows.Media.Ocr) mit RapidOCR- und Tesseract-Fallback

│       │   ├── subtitle\_extractor.py # Zieht Text-Subs oder PGS-Bitmaps aus der MKV und wandelt sie in Text

│       │   ├── audio\_snipper.py # Extrahiert kurze Audio-Samples via ffmpeg \& transkribiert mit faster-whisper

│       │   └── frame\_scanner.py # Findet Title-Cards (Episodentitel / "Written by") in den ersten 5 Min per FFmpeg + OCR

│       ├── providers/           # <--- EXTERNE WISSENSQUELLEN

│       │   ├── thediscdb.py     # Client \& lokaler Importer für TheDiscDB (MPLS-zu-Episode Mapping)

│       │   ├── tmdb\_client.py   # Lädt Serien, Staffeln, Episoden-Plots, Laufzeiten, Regisseure \& Autoren (Crew)

│       │   ├── imdb\_scraper.py  # Ergänzt detaillierte Plot-Synopsen, Gaststars \& exakte Episoden-Reihenfolgen

│       │   └── opensubtitles.py # Lädt Referenz-SRTs für schnelles lokales Text-Fuzzy-Matching

│       ├── matcher/             # <--- DAS MATCHING-GEHIRN

│       │   ├── pipeline.py      # Steuert den 5-Säulen-Scan von schnell (Metadaten) bis tief (OCR/Audio/LLM)

│       │   ├── text\_scorer.py   # Vergleicht Dialoge/OCR-Texte via RapidFuzz / TF-IDF mit Referenz-Subtitles \& Plots

│       │   ├── credit\_scorer.py # Gleicht per OCR erkannte Regisseure/Autoren/Titel mit TMDB/IMDb-Crew ab

│       │   ├── llm\_arbiter.py   # Schickt alle Indizien eines Ordners + Episodenguide an GLM-Flash für 100% Match

│       │   └── solver.py        # Löst Konflikte (Linear Sum Assignment via scipy), damit keine Folge doppelt vergeben wird

│       ├── postprocess/         # <--- UMBENENNEN \& OPTIONALE ZUSATZFUNKTIONEN

│       │   ├── renamer.py       # Benennt Dateien sicher um (Plex/Jellyfin/Emby-Format) + Undo-Historie!

│       │   ├── mkv\_tagger.py    # Schreibt Episodentitel \& Metadaten verlustfrei per mkvpropedit in den MKV-Header

│       │   ├── stream\_cleaner.py# Optional: Erkennt Forced-Subs (<15% Events) \& entfernt ungewollte Tonspuren

│       │   └── av1\_encoder.py   # Optional: Schlanke Batch-Queue für ffmpeg SVT-AV1 Encoding der fertigen Folgen

│       └── ui/                  # <--- PYSIDE6 WINDOWS 11 OBERFLÄCHE

│           ├── main\_window.py   # FluentWindow (Hauptfokus: Matcher-Workspace)

│           ├── views/

│           │   ├── workspace\_view.py # Hauptansicht: Serien-Suche, MKV-Liste, Live-Confidence-Balken, Beweis-Inspektor

│           │   ├── visual\_verify\_view.py # Synchroner Multi-Episode-Frame-Vergleich (zeigt Frame bei Min X für alle Dateien)

│           │   ├── postprocess\_view.py # Optionales Spuren-Aufräumen \& AV1-Encoding-Queue

│           │   └── settings\_view.py # Tool-Status (./bin/), API-Keys, Naming-Schema-Baukasten

│           └── widgets/

│               ├── evidence\_card.py # Zeigt genau an, WARUM eine Folge erkannt wurde (z.B. "Regisseur im Bild bei 01:42 erkannt")

│               └── multi\_player.py  # Schneller Thumbnail-/Frame-Scrubber für alle Dateien einer Disc gleichzeitig

└── tests/

&#x20;   ├── test\_pgs\_parser.py

&#x20;   ├── test\_solver.py

&#x20;   ├── test\_extractors.py

&#x20;   └── test\_renamer.py

