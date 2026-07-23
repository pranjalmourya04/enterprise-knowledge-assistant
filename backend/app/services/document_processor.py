from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import uuid

import pdfplumber

from app.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


@dataclass
class Chunk:
    chunk_id: str
    text: str
    filename: str
    page_number: int
    chunk_index_on_page: int
    department: str = "general"
    sensitivity: str = "internal"
    metadata: dict = field(default_factory=dict)

def extract_pages(pdf_path: Path) -> List[dict]:
    """
    Extract text from every page of a PDF.
    Returns a list of {"page_number": int, "text": str}, 1-indexed pages.
    Pages with no extractable text (e.g. pure scanned images) are skipped
    but logged, since pdfplumber won't OCR them.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page_number": i, "text": text})
            else:
                print(f"[document_processor] Warning: page {i} of "
                      f"{pdf_path.name} has no extractable text "
                      f"(possibly a scanned image - OCR not implemented).")
    return pages


def _split_text_into_windows(text: str, chunk_size: int, overlap: int) -> List[str]:
    
    words = text.split()
    if not words:
        return []

    windows = []
    start = 0
    step = max(chunk_size - overlap, 1)  # guard against overlap >= chunk_size

    while start < len(words):
        end = start + chunk_size
        window_words = words[start:end]
        windows.append(" ".join(window_words))
        if end >= len(words):
            break
        start += step

    return windows


def chunk_document(pdf_path: Path, department: str = "general",
                    sensitivity: str = "internal") -> List[Chunk]:
    """
    Full pipeline: extract pages, then chunk each page's text with overlap.
    Returns a flat list of Chunk objects ready for embedding.
    """
    pages = extract_pages(pdf_path)
    chunks: List[Chunk] = []

    for page in pages:
        windows = _split_text_into_windows(
            page["text"], CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
        )
        for idx, window_text in enumerate(windows):
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=window_text,
                    filename=pdf_path.name,
                    page_number=page["page_number"],
                    chunk_index_on_page=idx,
                    department=department,
                    sensitivity=sensitivity,
                )
            )

    return chunks
