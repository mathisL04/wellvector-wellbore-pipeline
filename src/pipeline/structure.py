"""Stage 3: LLM-based structuring using OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

import json
import os
import re

from openai import OpenAI, APIError

from pipeline.models import (
    NORWEGIAN_GLOSSARY,
    CasingRecord,
    ExtractionResult,
    PipelineStats,
)


# ---------------------------------------------------------------------------
# Default model — can be overridden via OPENROUTER_MODEL env var
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "anthropic/claude-sonnet-4"


# ---------------------------------------------------------------------------
# System prompt (same across all calls)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a petroleum engineering data extraction specialist.

Your task is to extract casing design information from Norwegian Continental Shelf well documents.
The documents may be in English or Norwegian. Use the glossary below for Norwegian terms.

## Norwegian → English Glossary
""" + "\n".join(f"- {k} → {v}" for k, v in NORWEGIAN_GLOSSARY.items()) + """

## Target Schema

Extract ALL casing strings/sections found. Each casing string becomes one row:

{
  "records": [
    {
      "wellbore": "<well name, e.g. 7/11-1>",
      "casing_type": "<Conductor | Surface | Intermediate | Production | Liner | Other>",
      "casing_diameter_in": <diameter in inches, float or null>,
      "casing_depth_m": <setting depth in meters MD, float or null>,
      "hole_diameter_in": <hole/bit size in inches, float or null>,
      "hole_depth_m": <hole section TD in meters MD, float or null>,
      "lot_fit_mud_eqv_gcm3": <LOT/FIT equivalent mud weight in g/cm³, float or null>,
      "formation_test_type": "<LOT | FIT | XLOT | null>"
    }
  ]
}

## Rules
1. Extract ALL casing strings mentioned (conductor, surface, intermediate, production, liner)
2. Convert units if needed: feet → meters (×0.3048), ppg → g/cm³ (×0.1198)
3. Use null for missing/unclear values — NEVER guess or hallucinate
4. If the same casing is mentioned in multiple places with different data, use the most detailed source
5. "Casing depth" = casing shoe depth (setting depth), not top of casing
6. Common casing sizes: 30", 20", 18⅝", 13⅜", 9⅝", 7", 5½", 5"
7. LOT = Leak-Off Test, FIT = Formation Integrity Test, XLOT = Extended Leak-Off Test
8. Return ONLY valid JSON, no markdown fences or extra text
9. Data may be in PROSE TEXT, not just tables — look for mentions like "13⅜ inch casing set at X ft"
10. EMW (equivalent mud weight) from LOT/FIT may be expressed as "leak off at X ppg" — convert ppg to g/cm³
11. Even partial data is valuable — if you find a casing size but no depth, still include it with null for missing fields
"""

EXTRACTION_USER_TEMPLATE = """Extract casing design data from this well document.

**Wellbore: {wellbore}**
**Document: {doc_name}**

## Extracted Content

{content}

Return the JSON with all casing records found. If no casing data is present, return {{"records": []}}."""


def _extract_json(text: str) -> str:
    """Extract JSON object from LLM response that may contain prose and markdown fences."""
    # Try to find JSON in markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find a JSON object directly
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text


def create_openrouter_client() -> tuple[OpenAI, str]:
    """Create an OpenRouter client. Returns (client, model_name)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required.\n"
            "Get one at https://openrouter.ai/keys"
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    return client, model


def _format_extraction_content(extraction: dict) -> str:
    """Format extracted tables and text into a concise string for the LLM."""
    parts: list[str] = []

    # Format tables
    for table in extraction.get("tables", []):
        parts.append(f"### Table (page {table['page']})")
        headers = table["headers"]
        parts.append(" | ".join(headers))
        parts.append(" | ".join(["---"] * len(headers)))
        for row in table["rows"]:
            padded = row + [""] * (len(headers) - len(row))
            parts.append(" | ".join(padded[:len(headers)]))
        parts.append("")

    # Format text — handles both dict[int, str] and dict[str, str] (Docling)
    text_data = extraction.get("text", {})
    if text_data and not extraction.get("tables"):
        for key, text in sorted(text_data.items(), key=lambda x: str(x[0])):
            stripped = text.strip()
            if stripped:
                label = f"Page {key}" if isinstance(key, int) else "Document Content"
                parts.append(f"### {label}")
                if len(stripped) > 2500:
                    stripped = stripped[:2500] + "\n[...truncated...]"
                parts.append(stripped)
                parts.append("")

    # Format OCR text
    for page_num, text in sorted(extraction.get("ocr_text", {}).items()):
        stripped = text.strip()
        if stripped:
            parts.append(f"### OCR Page {page_num}")
            if len(stripped) > 1500:
                stripped = stripped[:1500] + "\n[...truncated...]"
            parts.append(stripped)
            parts.append("")

    return "\n".join(parts) if parts else "(No extractable content found)"


def structure_document(
    extraction: dict,
    client: OpenAI,
    model: str,
    stats: PipelineStats,
) -> ExtractionResult:
    """Send extracted content to LLM via OpenRouter for structured extraction."""
    doc = extraction["document"]
    content = _format_extraction_content(extraction)

    if content == "(No extractable content found)":
        return ExtractionResult(
            document=doc,
            records=[],
            confidence=0.0,
            errors=["No extractable content"],
        )

    user_message = EXTRACTION_USER_TEMPLATE.format(
        wellbore=doc.wellbore,
        doc_name=doc.doc_name,
        content=content,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            extra_headers={
                "HTTP-Referer": "https://github.com/well-extraction-pipeline",
                "X-Title": "Well Extraction Pipeline",
            },
        )

        # Track token usage
        if response.usage:
            stats.llm_input_tokens += response.usage.prompt_tokens or 0
            stats.llm_output_tokens += response.usage.completion_tokens or 0
        stats.llm_calls += 1

        raw_response = (response.choices[0].message.content or "").strip()

        # Extract JSON from response — handle markdown fences, preamble text, etc.
        json_str = _extract_json(raw_response)

        parsed = json.loads(json_str)
        records = []
        for rec in parsed.get("records", []):
            rec["wellbore"] = doc.wellbore  # Override with canonical wellbore name
            records.append(CasingRecord(**rec))

        return ExtractionResult(
            document=doc,
            records=records,
            raw_text_used=content[:500],
            confidence=0.9 if records else 0.1,
        )

    except json.JSONDecodeError as e:
        return ExtractionResult(
            document=doc,
            records=[],
            raw_text_used=content[:500],
            confidence=0.0,
            errors=[f"JSON parse error: {e}"],
        )
    except APIError as e:
        return ExtractionResult(
            document=doc,
            records=[],
            confidence=0.0,
            errors=[f"OpenRouter API error: {e}"],
        )


def deduplicate_records(records: list[CasingRecord]) -> list[CasingRecord]:
    """Deduplicate casing records per wellbore, preferring records with more data."""
    if not records:
        return []

    groups: dict[str, list[CasingRecord]] = {}
    for rec in records:
        key = f"{rec.wellbore}|{rec.casing_type}|{rec.casing_diameter_in}"
        groups.setdefault(key, []).append(rec)

    deduplicated: list[CasingRecord] = []
    for group_records in groups.values():
        best = max(
            group_records,
            key=lambda r: sum(
                1 for v in r.model_dump().values() if v is not None
            ),
        )
        deduplicated.append(best)

    deduplicated.sort(
        key=lambda r: (r.wellbore, r.casing_depth_m or 0)
    )
    return deduplicated
