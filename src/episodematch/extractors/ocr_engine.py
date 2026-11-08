"""OCR engine with a priority chain of backends.

Backends are probed in this order:

1. **WinRtOcrBackend** — native Windows Media OCR via the ``winrt-Windows.Media.Ocr``
   projection (fast, offline, ships with Windows 10/11).
2. **RapidOcrBackend** — ``rapidocr-onnxruntime`` (pure-Python offline ONNX models).
3. **TesseractBackend** — ``pytesseract`` + a ``tesseract.exe`` from ``./bin/`` or PATH.
4. **NullBackend** — reports unavailability honestly; the pipeline then skips
   OCR-dependent evidence instead of fabricating results.

All backends implement :class:`OcrBackend` and return :class:`OcrResult` models with
per-line bounding boxes and confidences.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from episodematch.core.exceptions import OcrUnavailableError
from episodematch.core.models import OcrLine, OcrResult

logger = logging.getLogger(__name__)

#: Minimum line height in pixels below which images are upscaled before OCR.
_MIN_LINE_HEIGHT = 28


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def prepare_for_ocr(image: Image.Image, *, upscale: bool = True) -> Image.Image:
    """Convert to RGB and upscale small bitmaps (PGS crops are often tiny)."""
    converted = image if image.mode == "RGB" else image.convert("RGB")
    if upscale and (converted.height < _MIN_LINE_HEIGHT or converted.width < 64):
        factor = max(2, min(4, _MIN_LINE_HEIGHT // max(1, converted.height) + 1))
        converted = converted.resize(
            (converted.width * factor, converted.height * factor),
            Image.Resampling.LANCZOS,
        )
    return converted


def crop_image(image: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
    """Crop with clamping (PGS positions can theoretically exceed video bounds)."""
    left = max(0, x)
    top = max(0, y)
    right = min(image.width, x + max(1, width))
    bottom = min(image.height, y + max(1, height))
    return image.crop((left, top, right, bottom))


# ---------------------------------------------------------------------------
# Backend protocol + shared plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendAvailability:
    name: str
    available: bool
    reason: str = ""


class OcrBackend(Protocol):
    """Contract every OCR backend fulfills."""

    name: str

    def probe(self) -> BackendAvailability:  # pragma: no cover - interface
        ...

    def recognize(self, image: Image.Image) -> OcrResult:  # pragma: no cover - interface
        ...


def _clamp_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Backend 1: Windows Media OCR (winrt projections)
# ---------------------------------------------------------------------------


class WinRtOcrBackend:
    """Native Windows OCR via ``winrt-Windows.Media.Ocr``."""

    name = "winrt"

    def __init__(self, language: str = "en-US") -> None:
        self._language = language
        self._engine: Any = None

    def probe(self) -> BackendAvailability:
        try:
            from winrt.windows.media.ocr import OcrEngine  # noqa: F401,PLC0415
        except ImportError as exc:
            return BackendAvailability(
                self.name,
                False,
                f"winrt-Windows.Media.Ocr not installed ({exc.name}); pip install -e .[ocr-winrt]",
            )
        try:
            engine = self._create_engine()
        except Exception as exc:  # noqa: BLE001 - WinRT can raise OS-level errors
            return BackendAvailability(self.name, False, f"OcrEngine creation failed: {exc}")
        if engine is None:
            return BackendAvailability(
                self.name,
                False,
                f"No Windows OCR language pack for {self._language!r} installed",
            )
        return BackendAvailability(self.name, True)

    def _create_engine(self) -> Any:
        from winrt.windows.globalization import Language  # noqa: PLC0415
        from winrt.windows.media.ocr import OcrEngine  # noqa: PLC0415

        try:
            engine = OcrEngine.try_create_from_language(Language(self._language))
        except OSError:
            engine = None
        if engine is None:
            engine = OcrEngine.try_create_from_user_profile_languages()
        self._engine = engine
        return engine

    def recognize(self, image: Image.Image) -> OcrResult:
        from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap  # noqa: PLC0415
        from winrt.windows.storage.streams import DataWriter  # noqa: PLC0415

        prepared = prepare_for_ocr(image)
        rgba = prepared.convert("RGBA")
        # Windows OCR expects BGRA8 pixel data.
        bgra = bytearray(rgba.tobytes())
        for offset in range(0, len(bgra), 4):
            bgra[offset], bgra[offset + 2] = bgra[offset + 2], bgra[offset]
        writer = DataWriter()
        writer.write_bytes(bytes(bgra))
        buffer = writer.detach_buffer()
        bitmap = SoftwareBitmap.create_copy_from_buffer(
            buffer,
            BitmapPixelFormat.BGRA8,
            prepared.width,
            prepared.height,
            BitmapAlphaMode.STRAIGHT,
        )
        if self._engine is None and self._create_engine() is None:
            raise OcrUnavailableError("Windows OCR engine unavailable")
        # pywinrt v2: IAsyncOperation.get() blocks until completion (pumps internally);
        # the asyncio wrapper form is not needed for this short, CPU-bound call.
        winrt_result = self._engine.recognize_async(bitmap).get()
        lines: list[OcrLine] = []
        for line in winrt_result.lines:
            words = list(line.words)
            if words:
                left = min(int(w.bounding_rect.x) for w in words)
                top = min(int(w.bounding_rect.y) for w in words)
                right = max(int(w.bounding_rect.x + w.bounding_rect.width) for w in words)
                bottom = max(int(w.bounding_rect.y + w.bounding_rect.height) for w in words)
                box = (left, top, max(0, right - left), max(0, bottom - top))
            else:
                rect = line.bounding_rect
                box = (int(rect.x), int(rect.y), int(rect.width), int(rect.height))
            lines.append(OcrLine(text=line.text, x=box[0], y=box[1], width=box[2], height=box[3], confidence=1.0))
        return OcrResult(lines=lines, backend=self.name, language=self._language)


# ---------------------------------------------------------------------------
# Backend 2: RapidOCR (rapidocr-onnxruntime)
# ---------------------------------------------------------------------------


class RapidOcrBackend:
    """Offline ONNX OCR via ``rapidocr-onnxruntime``."""

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine: Any = None

    def probe(self) -> BackendAvailability:
        try:
            import rapidocr_onnxruntime  # noqa: F401,PLC0415
        except ImportError as exc:
            return BackendAvailability(
                self.name,
                False,
                f"rapidocr-onnxruntime not installed ({exc.name}); pip install -e .[ocr]",
            )
        try:
            self._engine = rapidocr_onnxruntime.RapidOCR()
        except Exception as exc:  # noqa: BLE001 - model load failures
            return BackendAvailability(self.name, False, f"RapidOCR init failed: {exc}")
        return BackendAvailability(self.name, True)

    def recognize(self, image: Image.Image) -> OcrResult:
        import numpy as np  # noqa: PLC0415

        if self._engine is None:
            availability = self.probe()
            if not availability.available:
                raise OcrUnavailableError(availability.reason)
        prepared = prepare_for_ocr(image)
        array = np.array(prepared)
        result, _elapsed = self._engine(array)
        lines: list[OcrLine] = []
        for item in result or []:
            box, text, score = item[0], item[1], float(item[2])
            xs = [int(point[0]) for point in box]
            ys = [int(point[1]) for point in box]
            lines.append(
                OcrLine(
                    text=str(text),
                    x=min(xs),
                    y=min(ys),
                    width=max(xs) - min(xs),
                    height=max(ys) - min(ys),
                    confidence=_clamp_confidence(score),
                )
            )
        # RapidOCR returns bottom-up boxes; sort reading order top-to-bottom.
        lines.sort(key=lambda line: (line.y, line.x))
        return OcrResult(lines=lines, backend=self.name)


# ---------------------------------------------------------------------------
# Backend 3: Tesseract (pytesseract + external binary)
# ---------------------------------------------------------------------------


class TesseractBackend:
    """Tesseract OCR via pytesseract; binary from ./bin/ or PATH."""

    name = "tesseract"

    def __init__(self, *, bin_dir: Path | None = None, language: str = "eng") -> None:
        self._bin_dir = Path(bin_dir) if bin_dir else None
        self._language = language
        self._checked = False

    def _resolve_binary(self) -> Path | None:
        import sys

        candidates: list[Path] = []
        if self._bin_dir is not None:
            exe_name = "tesseract.exe" if sys.platform == "win32" else "tesseract"
            candidates.append(self._bin_dir / exe_name)
        from shutil import which  # noqa: PLC0415

        hit = which("tesseract")
        if hit:
            candidates.append(Path(hit))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def probe(self) -> BackendAvailability:
        try:
            import pytesseract  # noqa: F401,PLC0415
        except ImportError as exc:
            return BackendAvailability(
                self.name,
                False,
                f"pytesseract not installed ({exc.name}); pip install -e .[ocr]",
            )
        binary = self._resolve_binary()
        if binary is None:
            return BackendAvailability(
                self.name,
                False,
                "tesseract binary not found in ./bin/ or PATH",
            )
        import pytesseract  # noqa: PLC0415

        pytesseract.pytesseract.tesseract_cmd = str(binary)
        self._checked = True
        return BackendAvailability(self.name, True)

    def recognize(self, image: Image.Image) -> OcrResult:
        import pytesseract  # noqa: PLC0415

        if not self._checked:
            availability = self.probe()
            if not availability.available:
                raise OcrUnavailableError(availability.reason)
        prepared = prepare_for_ocr(image, upscale=False)
        data = pytesseract.image_to_data(
            prepared,
            lang=self._language,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
        # Group words into lines by (block, paragraph, line).
        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for index in range(len(data["text"])):
            text = (data["text"][index] or "").strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index]) / 100.0
            except (TypeError, ValueError):
                confidence = None
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            grouped.setdefault(key, []).append(
                {
                    "text": text,
                    "left": int(data["left"][index]),
                    "top": int(data["top"][index]),
                    "width": int(data["width"][index]),
                    "height": int(data["height"][index]),
                    "conf": confidence,
                }
            )
        lines: list[OcrLine] = []
        for words in grouped.values():
            words.sort(key=lambda w: w["left"])
            line_text = " ".join(w["text"] for w in words)
            left = min(w["left"] for w in words)
            top = min(w["top"] for w in words)
            right = max(w["left"] + w["width"] for w in words)
            bottom = max(w["top"] + w["height"] for w in words)
            confidences = [w["conf"] for w in words if w["conf"] is not None]
            lines.append(
                OcrLine(
                    text=line_text,
                    x=left,
                    y=top,
                    width=max(0, right - left),
                    height=max(0, bottom - top),
                    confidence=_clamp_confidence(sum(confidences) / len(confidences)) if confidences else None,
                )
            )
        lines.sort(key=lambda line: (line.y, line.x))
        return OcrResult(lines=lines, backend=self.name, language=self._language)


# ---------------------------------------------------------------------------
# Backend 4: Null
# ---------------------------------------------------------------------------


class NullBackend:
    """Always-unavailable backend with an honest reason."""

    def __init__(self, reason: str = "no OCR backend installed") -> None:
        self.name = "null"
        self._reason = reason

    def probe(self) -> BackendAvailability:
        return BackendAvailability(self.name, False, self._reason)

    def recognize(self, image: Image.Image) -> OcrResult:
        raise OcrUnavailableError(self._reason)


# ---------------------------------------------------------------------------
# Engine facade
# ---------------------------------------------------------------------------


#: Factory chain, probed in priority order. Each factory receives the shared kwargs;
#: monkeypatching this tuple lets tests force a "no backend" environment.
_BACKEND_FACTORIES: tuple[Any, ...] = (
    lambda language="en-US", bin_dir=None: WinRtOcrBackend(language=language),
    lambda language="en-US", bin_dir=None: RapidOcrBackend(),
    lambda language="en-US", bin_dir=None: TesseractBackend(bin_dir=bin_dir, language="eng"),
)


class OcrEngine:
    """Facade picking the best available backend (cached)."""

    def __init__(
        self,
        *,
        bin_dir: Path | None = None,
        language: str = "en-US",
        preferred: list[str] | None = None,
    ) -> None:
        self._bin_dir = bin_dir
        self._language = language
        self._preferred = [name.lower() for name in (preferred or [])]
        self._backend: OcrBackend | None = None
        self._probe_lock = threading.Lock()

    @property
    def backend(self) -> OcrBackend:
        with self._probe_lock:
            if self._backend is None:
                self._backend = self._select()
            return self._backend

    @property
    def availability(self) -> BackendAvailability:
        return self.backend.probe()

    def _select(self) -> OcrBackend:
        instances: list[OcrBackend] = [
            factory(language=self._language, bin_dir=self._bin_dir) for factory in _BACKEND_FACTORIES
        ]
        if self._preferred:
            def rank(backend: OcrBackend) -> int:
                try:
                    return self._preferred.index(backend.name)
                except ValueError:
                    return len(self._preferred)

            instances.sort(key=rank)
        for instance in instances:
            try:
                availability = instance.probe()
            except Exception as exc:  # noqa: BLE001 - probing must never crash the app
                logger.debug("OCR backend %s probe crashed: %s", instance.name, exc)
                continue
            if availability.available:
                logger.info("Using OCR backend: %s", instance.name)
                return instance
            logger.debug("OCR backend %s unavailable: %s", instance.name, availability.reason)
        reasons = "; ".join(i.probe().reason for i in instances)
        logger.warning("No OCR backend available: %s", reasons)
        return NullBackend(reason=reasons or "no OCR backend installed")

    @property
    def available(self) -> bool:
        return self.backend.probe().available

    def recognize(self, image: Image.Image) -> OcrResult:
        """OCR one image; raises :class:`OcrUnavailableError` when no backend exists."""
        return self.backend.recognize(image)


def recognize_region(
    image: Image.Image,
    engine: OcrEngine,
    x: int,
    y: int,
    width: int,
    height: int,
) -> OcrResult:
    """OCR a rectangular region of a larger image (frame -> title-card area)."""
    return engine.recognize(crop_image(image, x, y, width, height))
