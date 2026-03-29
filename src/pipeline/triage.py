"""Stage 1: Fast triage using PyMuPDF keyword search and document name heuristics."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from pipeline.models import (
    CASING_KEYWORDS,
    HIGH_PRIORITY_DOC_PATTERNS,
    LOW_PRIORITY_DOC_PATTERNS,
    DocumentMeta,
    DocumentRelevance,
    TriageResult,
)

SCANNED_KEEP_PATTERNS = [
    "WELL_COMPLETION_REPORT",
    "COMPLETION_REPORT",
    "COMPLETION_LOG",
    "WDSS",
    "INDIVIDUAL_WELL_RECORD",
    "INTERNATIONAL_DEPARTMENT",
    "DRILLING_FLUID_SUMMARY",
    "FORMATION_TEST",
    "AAODC_REPORTS",
    "CHANGE_IN_DRILLING_PROGRAM",
]

EXTRA_EXCLUDE_PATTERNS = [
    "CORE_STUDY",
    "CORE_ANALYSIS",
    "CORE_DESCRIPTION",
    "CORED_INTERVAL",
    "CORING_ANALYSIS",
    "PARTIAL_ROCK_ANALYSIS",
    "LOG_ANALYSIS",
    "CUTTINGS_AND_CORE",
]


def _matches_any(doc_name: str, patterns: list[str]) -> bool:
    name_upper = doc_name.upper()
    return any(pattern.upper() in name_upper for pattern in patterns)


def _name_priority(doc_name: str) -> tuple[DocumentRelevance, float]:
    """Score document relevance based on its name alone."""
    name_upper = doc_name.upper()

    if _matches_any(name_upper, EXTRA_EXCLUDE_PATTERNS):
        return DocumentRelevance.IRRELEVANT, 0.0

    for pattern in HIGH_PRIORITY_DOC_PATTERNS:
        if pattern.upper() in name_upper:
            return DocumentRelevance.HIGH, 0.8

    for pattern in LOW_PRIORITY_DOC_PATTERNS:
        if pattern.upper() in name_upper:
            return DocumentRelevance.IRRELEVANT, 0.0

    return DocumentRelevance.MEDIUM, 0.4


def _extract_text_and_check(pdf_path: Path) -> tuple[dict[int, str], bool]:
    """Extract text per page and detect if PDF is scanned (image-only).

    Returns (page_texts, is_scanned).
    """
    page_texts: dict[int, str] = {}
    total_chars = 0

    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            page_texts[page_num] = text
            total_chars += len(text.strip())
        doc.close()
    except Exception:
        return {}, True

    is_scanned = total_chars < 50  # Less than 50 chars total → likely scanned
    return page_texts, is_scanned


def _keyword_search(
    page_texts: dict[int, str],
) -> tuple[list[int], list[str], float]:
    """Search all pages for casing-related keywords.

    Returns (relevant_page_indices, matched_keywords, score).
    """
    relevant_pages: list[int] = []
    matched_keywords: set[str] = set()

    for page_num, text in page_texts.items():
        text_lower = text.lower()
        page_hits: list[str] = []

        for keyword in CASING_KEYWORDS:
            if keyword.lower() in text_lower:
                page_hits.append(keyword)
                matched_keywords.add(keyword)

        if page_hits:
            relevant_pages.append(page_num)

    # Score: more keyword variety = higher score
    score = min(len(matched_keywords) / 5.0, 1.0)
    return relevant_pages, sorted(matched_keywords), score


def triage_document(doc: DocumentMeta, pdf_path: Path) -> TriageResult:
    """Run full triage on a single document: name check + keyword search."""
    # Step 1: Name-based priority
    name_relevance, name_score = _name_priority(doc.doc_name)

    # If name says irrelevant and it's not a WDSS or completion report, skip early
    if name_relevance == DocumentRelevance.IRRELEVANT:
        return TriageResult(
            document=doc,
            relevance=DocumentRelevance.IRRELEVANT,
            relevance_score=0.0,
        )

    # Step 2: Extract text and search keywords
    page_texts, is_scanned = _extract_text_and_check(pdf_path)

    if is_scanned:
    # Only strong whitelisted scanned docs should go to OCR
        if _matches_any(doc.doc_name, SCANNED_KEEP_PATTERNS):
            return TriageResult(
                document=doc,
                relevance=DocumentRelevance.HIGH,
                relevance_score=0.8,
                is_scanned=True,
            )

        return TriageResult(
            document=doc,
            relevance=DocumentRelevance.IRRELEVANT,
            relevance_score=0.0,
            is_scanned=True,
        )

    relevant_pages, keyword_hits, keyword_score = _keyword_search(page_texts)

    # Combine name score and keyword score
    combined_score = max(name_score, keyword_score)

    if combined_score >= 0.6 or name_relevance == DocumentRelevance.HIGH:
        relevance = DocumentRelevance.HIGH
    elif combined_score >= 0.3:
        relevance = DocumentRelevance.MEDIUM
    elif keyword_hits:
        relevance = DocumentRelevance.LOW
    else:
        relevance = DocumentRelevance.IRRELEVANT

    return TriageResult(
        document=doc,
        relevance=relevance,
        relevance_score=combined_score,
        relevant_pages=relevant_pages,
        is_scanned=is_scanned,
        keyword_hits=keyword_hits,
    )


def triage_all(
    docs_with_paths: list[tuple[DocumentMeta, Path | None]],
) -> list[TriageResult]:
    """Triage all downloaded documents, returning sorted by relevance score."""
    results: list[TriageResult] = []

    for doc, path in docs_with_paths:
        if path is None:
            results.append(
                TriageResult(
                    document=doc,
                    relevance=DocumentRelevance.IRRELEVANT,
                    relevance_score=0.0,
                    errors=["Download failed"],
                )
            )
            continue

        result = triage_document(doc, path)
        results.append(result)

    # Sort: HIGH first, then by score descending
    priority_order = {
        DocumentRelevance.HIGH: 0,
        DocumentRelevance.MEDIUM: 1,
        DocumentRelevance.LOW: 2,
        DocumentRelevance.IRRELEVANT: 3,
    }
    results.sort(
        key=lambda r: (priority_order[r.relevance], -r.relevance_score)
    )
    return results
