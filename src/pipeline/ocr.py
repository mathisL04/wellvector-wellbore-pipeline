"""OCR module: Vision model via OpenRouter for scanned PDF pages, with disk caching."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

from openai import OpenAI
from PIL import Image


# Qwen2.5-VL-72B on OpenRouter — excellent document OCR, same family as RolmOCR base
DEFAULT_OCR_MODEL = "qwen/qwen2.5-vl-72b-instruct"

OCR_CACHE_DIR = Path("data/ocr_cache")

OCR_PROMPT = (
    "Extract ALL text from this document page. "
    "Output as clean Markdown. Preserve table structures using Markdown table syntax. "
    "Include ALL numbers, units, measurements, and technical data exactly as shown. "
    "Do not summarize or skip any content."
)


MAX_IMAGE_DIM = 2048  # Max pixels on longest side for API compatibility


def _resize_for_api(img: Image.Image) -> Image.Image:
    """Resize image if any dimension exceeds MAX_IMAGE_DIM."""
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_DIM:
        return img

    scale = MAX_IMAGE_DIM / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string, resizing if needed."""
    img = _resize_for_api(img)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _cache_key(doc_name: str, page_num: int) -> str:
    """Generate a cache key for a specific page of a document."""
    safe = doc_name.replace("/", "_").replace(" ", "_")
    return f"{safe}_page{page_num}"


def _get_cached(doc_name: str, page_num: int) -> str | None:
    """Check if OCR result is cached on disk."""
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = OCR_CACHE_DIR / f"{_cache_key(doc_name, page_num)}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    return None


def _save_cache(doc_name: str, page_num: int, text: str) -> None:
    """Save OCR result to disk cache."""
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = OCR_CACHE_DIR / f"{_cache_key(doc_name, page_num)}.txt"
    cache_file.write_text(text, encoding="utf-8")


def ocr_single_page(
    client: OpenAI,
    image: Image.Image,
    model: str = DEFAULT_OCR_MODEL,
    max_retries: int = 3,
) -> str:
    """OCR a single page image via OpenRouter vision model, with retry."""
    import time as _time

    b64 = _image_to_base64(image)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {
                                "type": "text",
                                "text": OCR_PROMPT,
                            },
                        ],
                    }
                ],
                extra_headers={
                    "HTTP-Referer": "https://github.com/well-extraction-pipeline",
                    "X-Title": "Well Extraction Pipeline",
                },
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt < max_retries - 1 and ("502" in str(e) or "503" in str(e) or "timeout" in str(e).lower()):
                _time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            raise

    return ""


def ocr_pages(
    client: OpenAI,
    images: list[Image.Image],
    page_nums: list[int],
    doc_name: str,
    model: str = DEFAULT_OCR_MODEL,
) -> list[str]:
    """OCR multiple page images with disk caching.

    Cached pages are returned instantly, only uncached pages hit the API.
    """
    results: list[str] = []
    for img, page_num in zip(images, page_nums):
        # Check cache first
        cached = _get_cached(doc_name, page_num)
        if cached is not None:
            results.append(cached)
            continue

        # Not cached — call API
        try:
            text = ocr_single_page(client, img, model)
            _save_cache(doc_name, page_num, text)
            results.append(text)
        except Exception as e:
            results.append(f"[OCR ERROR: {e}]")

    return results
