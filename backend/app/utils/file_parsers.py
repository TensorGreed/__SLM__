"""File parsing utilities for various document types."""

import hashlib
from pathlib import Path
from typing import Any


def parse_text(file_path: Path) -> str:
    """Read plain text file."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return file_path.read_text(encoding="utf-8", errors="replace")


def parse_markdown(file_path: Path) -> str:
    """Read Markdown file (treated as plain text)."""
    return parse_text(file_path)


def parse_csv(file_path: Path) -> str:
    """Read CSV file as text."""
    return parse_text(file_path)


def parse_pdf(file_path: Path) -> str:
    """Extract text from PDF using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(file_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except ImportError:
        return f"[PDF parsing unavailable — install PyPDF2] {file_path.name}"
    except Exception as e:
        return f"[PDF parse error: {e}] {file_path.name}"


def parse_docx(file_path: Path) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        return f"[DOCX parsing unavailable — install python-docx] {file_path.name}"
    except Exception as e:
        return f"[DOCX parse error: {e}] {file_path.name}"


# File extension → parser mapping
PARSERS: dict[str, Any] = {
    ".txt": parse_text,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".csv": parse_csv,
    ".json": parse_text,
    ".jsonl": parse_text,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}

SUPPORTED_EXTENSIONS = set(PARSERS.keys())


def parse_file(file_path: Path) -> str:
    """Parse a file based on its extension. Returns extracted text."""
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
