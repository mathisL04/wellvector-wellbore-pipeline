"""Stage 2: Content extraction — Docling + pdfplumber + OCR for scanned pages."""

from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF — for rendering pages to images and text extraction
import pdfplumber
from PIL import Image
from rich.console import Console

from pipeline.models import TriageResult

console = Console()


# ---------------------------------------------------------------------------
# PDF → Image rendering
# ---------------------------------------------------------------------------

def render_page_to_image(
    pdf_path: Path,
    page_num: int,
    dpi: int = 200,
) -> Image.Image:
    """Render a single PDF page to a PIL Image."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return img


def get_page_count(pdf_path: Path) -> int:
    """Get total number of pages in a PDF."""
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


# ---------------------------------------------------------------------------
# pdfplumber table extraction
# ---------------------------------------------------------------------------

def extract_tables_pdfplumber(
    pdf_path: Path,
    pages: list[int] | None = None,
) -> list[dict]:
    """Extract tables from specific pages using pdfplumber."""
    tables_found: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        target_pages = pages if pages else list(range(len(pdf.pages)))

        for page_idx in target_pages:
            if page_idx >= len(pdf.pages):
                continue

            page = pdf.pages[page_idx]
            page_tables = page.extract_tables()

            for table_idx, table in enumerate(page_tables):
                if not table or len(table) < 2:
                    continue

                headers = [str(cell).strip() if cell else "" for cell in table[0]]
                rows = []
                for row in table[1:]:
                    cleaned = [str(cell).strip() if cell else "" for cell in row]
                    if any(cleaned):
                        rows.append(cleaned)

                if rows:
                    tables_found.append({
                        "page": page_idx,
                        "table_index": table_idx,
                        "headers": headers,
                        "rows": rows,
                    })

    return tables_found


# ---------------------------------------------------------------------------
# PyMuPDF text extraction
# ---------------------------------------------------------------------------

def extract_text_pymupdf(
    pdf_path: Path,
    pages: list[int] | None = None,
) -> dict[int, str]:
    """Extract text from specific pages using PyMuPDF."""
    page_texts: dict[int, str] = {}

    doc = fitz.open(str(pdf_path))
    target_pages = pages if pages else list(range(len(doc)))

    for page_idx in target_pages:
        if page_idx >= len(doc):
            continue
        page_texts[page_idx] = doc[page_idx].get_text("text")

    doc.close()
    return page_texts


# ---------------------------------------------------------------------------
# Docling table extraction
# ---------------------------------------------------------------------------

def extract_with_docling(pdf_path: Path, pages: list[int] | None = None) -> dict:
    """Extract content using Docling with TableFormer.

    Returns dict with tables and text from the document.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    tables: list[dict] = []
    text_parts: list[str] = []

    # Extract tables from Docling result
    doc = result.document
    for table_ix, table in enumerate(doc.tables):
        try:
            df = table.export_to_dataframe()
            headers = [str(c) for c in df.columns.tolist()]
            rows = []
            for _, row in df.iterrows():
                cleaned = [str(v).strip() if v else "" for v in row.values]
                if any(cleaned):
                    rows.append(cleaned)
            if rows:
                tables.append({
                    "page": table_ix,  # Approximate
                    "table_index": table_ix,
                    "headers": headers,
                    "rows": rows,
                })
        except Exception:
            continue

    # Extract text
    text_content = doc.export_to_markdown()
    if text_content.strip():
        text_parts.append(text_content)

    return {
        "tables": tables,
        "text": text_parts,
    }


# ---------------------------------------------------------------------------
# Smart page selection for scanned documents
# ---------------------------------------------------------------------------

def select_pages_for_ocr(pdf_path: Path, doc_name: str = "") -> list[int]:
    total = get_page_count(pdf_path)
    name_upper = doc_name.upper()

    if total <= 3:
        return list(range(total))

    if any(p in name_upper for p in ["WELL_COMPLETION_REPORT", "COMPLETION_REPORT", "INDIVIDUAL_WELL_RECORD"]):
        return list(range(min(10, total)))

    if "WDSS" in name_upper:
        return list(range(min(3, total)))

    if any(p in name_upper for p in [
        "DRILLING_FLUID_SUMMARY",
        "FORMATION_TEST",
        "AAODC_REPORTS",
        "CHANGE_IN_DRILLING_PROGRAM",
    ]):
        return list(range(min(3, total)))

    return list(range(min(3, total)))


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_document(
    triage: TriageResult,
    pdf_path: Path,
    ocr_client=None,
) -> dict:
    """Full extraction for a triaged document.

    Flow:
    - Scanned PDFs → render pages to images → OCR via HF API → return text
    - Text PDFs → try Docling (TableFormer) → fallback to pdfplumber + PyMuPDF

    Returns dict with:
        - tables: list of extracted tables
        - text: dict[page, text] or list[str]
        - ocr_text: dict[page, text] (if OCR was used)
        - source: extraction method used
    """
    pages = triage.relevant_pages or None
    result: dict = {
        "document": triage.document,
        "tables": [],
        "text": {},
        "ocr_text": {},
        "source": "unknown",
    }

    if triage.is_scanned:
        if ocr_client is None:
            console.print(f"    [yellow]Scanned PDF but no OCR client — skipping[/]")
            result["source"] = "skipped_no_ocr"
            return result

        # Scanned PDF → render to images → OCR (with disk cache)
        from pipeline.ocr import ocr_pages

        target_pages = select_pages_for_ocr(pdf_path, doc_name=triage.document.doc_name)
        console.print(f"    [dim]OCR: rendering {len(target_pages)} pages...[/]")

        images = [render_page_to_image(pdf_path, p) for p in target_pages]
        ocr_texts = ocr_pages(
            ocr_client, images,
            page_nums=target_pages,
            doc_name=triage.document.doc_name,
        )
        result["ocr_text"] = dict(zip(target_pages, ocr_texts))
        result["source"] = "qwen_vlm_ocr"

    else:
        # Text-based PDF → try Docling first, fallback to pdfplumber
        try:
            console.print(f"    [dim]Trying Docling TableFormer...[/]")
            docling_result = extract_with_docling(pdf_path, pages)

            if docling_result["tables"]:
                result["tables"] = docling_result["tables"]
                result["text"] = {"docling_md": docling_result["text"][0]} if docling_result["text"] else {}
                result["source"] = "docling"
                return result

            # Docling found no tables — use its text output
            if docling_result["text"]:
                result["text"] = {"docling_md": docling_result["text"][0]}
                result["source"] = "docling"
                return result

        except Exception as e:
            console.print(f"    [yellow]Docling failed: {e}, falling back to pdfplumber[/]")

        # Fallback: pdfplumber + PyMuPDF
        result["tables"] = extract_tables_pdfplumber(pdf_path, pages)
        result["text"] = extract_text_pymupdf(pdf_path, pages)
        result["source"] = "pdfplumber"

        if not result["tables"] and pages:
            result["tables"] = extract_tables_pdfplumber(pdf_path, pages=None)

    return result
