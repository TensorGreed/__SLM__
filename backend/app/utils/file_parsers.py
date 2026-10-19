"""File parsing utilities for various document types.

Parsers raise :class:`DocumentParseError` when a file can't be turned into
text. They used to return strings like ``"[PDF parse error: …]"``, which the
ingestion step then marked ACCEPTED and cleaning fed to training as if they
were document content.
"""

import hashlib
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from app.utils.tabular_io import load_rows, read_text_file, render_row_text


class DocumentParseError(ValueError):
    """The file couldn't be turned into usable text (message is user-facing)."""


def parse_text(file_path: Path) -> str:
    """Read a plain-text file in whatever encoding it was saved with."""
    return read_text_file(file_path)


def parse_markdown(file_path: Path) -> str:
    """Read Markdown file (treated as plain text)."""
    return parse_text(file_path)


def parse_tabular(file_path: Path) -> str:
    """Readable text for CSV/TSV/JSON(L)/XLSX/Parquet — used for previews and
    the ``.extracted.txt`` sidecar. Training keeps the rows themselves
    (cleaning's row-preserving path), not this rendering."""
    try:
        rows = load_rows(file_path)
    except ValueError as exc:
        raise DocumentParseError(str(exc)) from exc
    if not rows:
        raise DocumentParseError(f"{file_path.name} has no data rows.")
    return "\n\n".join(text for text in (render_row_text(row) for row in rows) if text)


def parse_csv(file_path: Path) -> str:
    return parse_tabular(file_path)


# ── OCR for scanned pages ────────────────────────────────────────────────
#
# A scanned PDF has no text layer. When the system has poppler's pdftoppm
# and tesseract (apt: poppler-utils tesseract-ocr), pages without text are
# rendered and OCR'd; otherwise the parse error says what to install.
# BREWSLM_OCR=0 turns it off; BREWSLM_OCR_LANG (default "eng", tesseract
# language codes joined by "+") and BREWSLM_OCR_DPI (default 300) tune it.

OCR_TIMEOUT_SECONDS = 180


def ocr_available() -> bool:
    import os
    import shutil

    if os.environ.get("BREWSLM_OCR", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(shutil.which("pdftoppm") and shutil.which("tesseract"))


def ocr_pdf_pages(file_path: Path, pages: list[int]) -> dict[int, str]:
    """OCR the given 1-based ``pages`` of a PDF: pdftoppm renders each page
    to PNG, tesseract reads it. Returns ``{page: text}``."""
    import os
    import subprocess
    import tempfile

    dpi = str(int(os.environ.get("BREWSLM_OCR_DPI") or 300))
    lang = os.environ.get("BREWSLM_OCR_LANG") or "eng"
    out: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="brewslm-ocr-") as tmp:
        for page in pages:
            prefix = os.path.join(tmp, f"p{page}")
            subprocess.run(
                ["pdftoppm", "-r", dpi, "-f", str(page), "-l", str(page), "-png", "-singlefile", str(file_path), prefix],
                check=True, capture_output=True, timeout=OCR_TIMEOUT_SECONDS,
            )
            result = subprocess.run(
                ["tesseract", f"{prefix}.png", "stdout", "-l", lang, "--psm", "3"],
                check=True, capture_output=True, timeout=OCR_TIMEOUT_SECONDS,
            )
            out[page] = result.stdout.decode("utf-8", errors="replace").strip()
    return out


PAGE_SEPARATOR = "\n\f\n"
"""How :func:`parse_pdf` joins pages: a form feed on its own line, so
cleaning can find page headers / footers per page
(``services/page_furniture``). Cleaning turns it into a blank line — the
join used before — so the cleaned text reads the same."""


def parse_pdf(file_path: Path) -> str:
    """Extract text from a PDF with PyPDF2; pages without a text layer
    (scanned) are OCR'd when tesseract + pdftoppm are installed. Pages are
    joined with :data:`PAGE_SEPARATOR`."""
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise DocumentParseError("PDF parsing is unavailable — install PyPDF2 on the server.") from exc
    try:
        reader = PdfReader(str(file_path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # noqa: BLE001
        raise DocumentParseError(f"Couldn't read this PDF ({exc}). Is it password-protected or damaged?") from exc
    empty = [index + 1 for index, text in enumerate(pages) if not text]
    if empty and ocr_available():
        try:
            ocr = ocr_pdf_pages(file_path, empty)
        except Exception as exc:  # noqa: BLE001
            if len(empty) == len(pages):
                raise DocumentParseError(f"This PDF looks scanned and OCR failed ({exc}).") from exc
            ocr = {}
        for page, text in ocr.items():
            pages[page - 1] = text
    text = PAGE_SEPARATOR.join(p for p in pages if p)
    if not text.strip():
        raise DocumentParseError(
            "This PDF has no extractable text — it looks like a scanned/image PDF. "
            + (
                "OCR found no text on its pages either."
                if empty and ocr_available()
                else "Install OCR on the server (apt install tesseract-ocr poppler-utils) to read scans, "
                "or export it as text or DOCX and upload that."
            )
        )
    return text


def parse_docx(file_path: Path) -> str:
    """Extract paragraphs and table cells from DOCX using python-docx."""
    try:
        from docx import Document
    except ImportError as exc:
        raise DocumentParseError("DOCX parsing is unavailable — install python-docx on the server.") from exc
    try:
        doc = Document(str(file_path))
    except Exception as exc:  # noqa: BLE001
        raise DocumentParseError(f"Couldn't read this Word document ({exc}).") from exc
    blocks = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    text = "\n\n".join(blocks)
    if not text.strip():
        raise DocumentParseError("This Word document has no text.")
    return text


class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre", "td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):  # noqa: ANN001
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):  # noqa: ANN001
        if not self._skip_depth:
            self.parts.append(data)


def parse_html(file_path: Path) -> str:
    """Visible text of an HTML page (scripts/styles dropped)."""
    extractor = _HTMLText()
    extractor.feed(read_text_file(file_path))
    lines = [" ".join(line.split()) for line in "".join(extractor.parts).splitlines()]
    text = "\n".join(line for line in lines if line)
    if not text.strip():
        raise DocumentParseError("This HTML page has no visible text.")
    return text


# File extension → parser mapping
PARSERS: dict[str, Any] = {
    ".txt": parse_text,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".csv": parse_tabular,
    ".tsv": parse_tabular,
    ".json": parse_tabular,
    ".jsonl": parse_tabular,
    ".xlsx": parse_tabular,
    ".parquet": parse_tabular,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".html": parse_html,
    ".htm": parse_html,
}

SUPPORTED_EXTENSIONS = set(PARSERS.keys())


def parse_file(file_path: Path) -> str:
    """Parse a file based on its extension. Returns extracted text; raises
    :class:`DocumentParseError` when there's no usable text."""
    ext = file_path.suffix.lower()
    parser = PARSERS.get(ext)
    if not parser:
        raise ValueError(f"Unsupported file type: {ext}")
    return parser(file_path)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_type(filename: str) -> str:
    """Return the file extension without dot."""
    return Path(filename).suffix.lower().lstrip(".")
