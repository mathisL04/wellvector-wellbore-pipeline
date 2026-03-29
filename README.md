# Wellvector Wellbore Pipeline

This project builds a pipeline that ingests raw SODIR well documents and outputs structured casing design data for the Cod field dataset.

## Pipeline flow

1. Load CSV metadata for all documents
2. Download and cache PDFs locally
3. Triage documents by relevance using filename heuristics and text checks
4. Extract content
   - scanned PDFs → OCR with Qwen via OpenRouter
   - text PDFs → Docling / PyMuPDF / pdfplumber
5. Structure extracted content into the target schema with Claude via OpenRouter
6. Deduplicate records
7. Export the final CSV output

## Scripts

- `src/pipeline/main.py`  
  Main orchestrator that runs the full pipeline from CSV input to final CSV output.

- `src/pipeline/downloader.py`  
  Downloads all PDFs asynchronously and caches them locally.

- `src/pipeline/triage.py`  
  Filters relevant documents using filename heuristics, scanned-document rules, and keyword-based checks.

- `src/pipeline/extract.py`  
  Extracts content from relevant PDFs using OCR for scanned files and Docling/PyMuPDF/pdfplumber for text-based files.

- `src/pipeline/ocr.py`  
  Handles page-level OCR with Qwen vision through OpenRouter and stores OCR results in a local cache.

- `src/pipeline/structure.py`  
  Sends extracted content to the LLM and converts it into structured casing-design records.

- `src/pipeline/models.py`  
  Defines shared models, enums, keyword lists, and document pattern rules used across the pipeline.

## Output schema

The final output is written to:

`output/casing_design.csv`

Schema:

- `Wellbore`
- `Casing type`
- `Casing diameter [in]`
- `Casing depth [m]`
- `Hole diameter [in]`
- `Hole depth [m]`
- `LOT/FIT mud eqv. [g/cm3]`
- `Formation test type`

## Requirements

Install Python 3.11.

Install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install rich pandas httpx openai pymupdf pdfplumber pillow docling
