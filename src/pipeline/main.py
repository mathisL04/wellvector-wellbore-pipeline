"""Main pipeline orchestrator: OCR → Docling → OpenRouter → CSV output."""

from __future__ import annotations

import asyncio
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from pipeline.downloader import download_all
from pipeline.extract import extract_document
from pipeline.models import (
    CasingRecord,
    DocumentMeta,
    DocumentRelevance,
    PipelineStats,
    TriageResult,
)
from pipeline.structure import create_openrouter_client, deduplicate_records, structure_document
from pipeline.triage import triage_all


console = Console()

CSV_INPUT = Path("wellbore_document_7_11.csv")
OUTPUT_DIR = Path("output")
CACHE_DIR = Path("data/pdfs")


def load_documents(csv_path: Path) -> list[DocumentMeta]:
    """Load document metadata from the input CSV."""
    documents: list[DocumentMeta] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            documents.append(
                DocumentMeta(
                    wellbore=row["wlbName"],
                    doc_type=row["wlbDocumentType"],
                    doc_name=row["wlbDocumentName"],
                    url=row["wlbDocumentUrl"],
                    format=row["wlbDocumentFormat"],
                    size_kb=int(row["wlbDocumentSize"]),
                    npd_id=int(row["wlbNpdidWellbore"]),
                )
            )
    return documents


def print_triage_summary(results: list[TriageResult]) -> None:
    """Print a summary table of triage results."""
    table = Table(title="Stage 1: Triage Results", show_lines=True)
    table.add_column("Relevance", style="bold")
    table.add_column("Wellbore")
    table.add_column("Document")
    table.add_column("Score", justify="right")
    table.add_column("Pages")
    table.add_column("Scanned")
    table.add_column("Keywords")

    colors = {
        DocumentRelevance.HIGH: "green",
        DocumentRelevance.MEDIUM: "yellow",
        DocumentRelevance.LOW: "dim",
        DocumentRelevance.IRRELEVANT: "red",
    }

    for r in results:
        if r.relevance == DocumentRelevance.IRRELEVANT:
            continue
        table.add_row(
            f"[{colors[r.relevance]}]{r.relevance.value}[/]",
            r.document.wellbore,
            r.document.doc_name[:50],
            f"{r.relevance_score:.2f}",
            str(len(r.relevant_pages)) if r.relevant_pages else "-",
            "Yes" if r.is_scanned else "No",
            ", ".join(r.keyword_hits[:3]) if r.keyword_hits else "-",
        )

    console.print(table)

    counts: dict[DocumentRelevance, int] = {}
    for r in results:
        counts[r.relevance] = counts.get(r.relevance, 0) + 1

    console.print(f"\n  [green]HIGH:[/] {counts.get(DocumentRelevance.HIGH, 0)}  "
                  f"[yellow]MEDIUM:[/] {counts.get(DocumentRelevance.MEDIUM, 0)}  "
                  f"[dim]LOW:[/] {counts.get(DocumentRelevance.LOW, 0)}  "
                  f"[red]IRRELEVANT:[/] {counts.get(DocumentRelevance.IRRELEVANT, 0)}")


def records_to_csv(records: list[CasingRecord], output_path: Path) -> None:
    """Write casing records to the final output CSV."""
    if not records:
        console.print("[red]No records to write.[/]")
        return

    rows = []
    for rec in records:
        rows.append({
            "Wellbore": rec.wellbore,
            "Casing type": rec.casing_type or "",
            "Casing diameter [in]": rec.casing_diameter_in if rec.casing_diameter_in else "",
            "Casing depth [m]": rec.casing_depth_m if rec.casing_depth_m else "",
            "Hole diameter [in]": rec.hole_diameter_in if rec.hole_diameter_in else "",
            "Hole depth [m]": rec.hole_depth_m if rec.hole_depth_m else "",
            "LOT/FIT mud eqv. [g/cm3]": rec.lot_fit_mud_eqv_gcm3 if rec.lot_fit_mud_eqv_gcm3 else "",
            "Formation test type": rec.formation_test_type or "",
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"\n[green]Output written to {output_path}[/]")

    output_table = Table(title="Extracted Casing Design Data", show_lines=True)
    for col in df.columns:
        output_table.add_column(col)
    for _, row in df.iterrows():
        output_table.add_row(*[str(v) for v in row.values])
    console.print(output_table)


def print_stats(stats: PipelineStats, elapsed: float, model: str) -> None:
    """Print pipeline performance stats."""
    console.print("\n[bold]Pipeline Statistics[/]")
    console.print(f"  LLM Model:             {model}")
    console.print(f"  OCR Model:             Qwen2.5-VL-7B via HF Hyperbolic")
    console.print(f"  Table Extraction:      Docling TableFormer")
    console.print(f"  Total documents:       {stats.total_documents}")
    console.print(f"  Downloaded:            {stats.documents_downloaded}")
    console.print(f"  Triaged as relevant:   {stats.documents_triaged_relevant}")
    console.print(f"  Documents extracted:   {stats.documents_extracted}")
    console.print(f"  Pages processed:       {stats.pages_processed}")
    console.print(f"  LLM calls:             {stats.llm_calls}")
    console.print(f"  LLM input tokens:      {stats.llm_input_tokens:,}")
    console.print(f"  LLM output tokens:     {stats.llm_output_tokens:,}")
    console.print(f"  OCR pages:             {stats.ocr_pages}")
    console.print(f"  Total time:            {elapsed:.1f}s")


async def run_pipeline(
    csv_path: Path = CSV_INPUT,
    output_path: Path = OUTPUT_DIR / "casing_design.csv",
) -> None:
    """Run the full 3-stage pipeline.

    Flow:
      1. Download all PDFs
      2. Triage by name + keyword search
      3. Extract: scanned → OCR (Qwen2.5-VL via HF), text → Docling TableFormer
      4. Structure: Claude via OpenRouter
      5. Deduplicate & output CSV
    """
    start = time.time()
    stats = PipelineStats()

    console.rule("[bold blue]Well Document Extraction Pipeline[/]")
    console.print(f"Input:  {csv_path}")
    console.print(f"Output: {output_path}")

    # ── Init clients ──────────────────────────────────────────────────
    llm_client, llm_model = create_openrouter_client()
    console.print(f"LLM:    {llm_model} via OpenRouter")

    # OCR uses the same OpenRouter client with a vision model
    from pipeline.ocr import DEFAULT_OCR_MODEL
    ocr_client = llm_client  # Same client, different model
    console.print(f"OCR:    {DEFAULT_OCR_MODEL} via OpenRouter")

    console.print(f"Tables: Docling TableFormer\n")

    # ── Load CSV metadata ─────────────────────────────────────────────
    documents = load_documents(csv_path)
    stats.total_documents = len(documents)
    console.print(f"Loaded {len(documents)} documents across "
                  f"{len(set(d.wellbore for d in documents))} wellbores\n")

    # ── Stage 0: Download PDFs ────────────────────────────────────────
    console.rule("[bold]Stage 0: Downloading PDFs[/]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        dl_task = progress.add_task("Downloading...", total=len(documents))
        downloaded = await download_all(
            documents, cache_dir=CACHE_DIR, progress=progress, task_id=dl_task
        )

    stats.documents_downloaded = sum(1 for _, p in downloaded if p is not None)
    failed = sum(1 for _, p in downloaded if p is None)
    console.print(f"  Downloaded: {stats.documents_downloaded}, Failed: {failed}\n")

    # ── Stage 1: Triage ───────────────────────────────────────────────
    console.rule("[bold]Stage 1: Triage[/]")
    triage_results = triage_all(downloaded)
    print_triage_summary(triage_results)

    # Only process HIGH-priority docs (MEDIUM ones like core studies rarely have casing data)
    relevant = [
        r for r in triage_results
        if r.relevance == DocumentRelevance.HIGH
    ]
    stats.documents_triaged_relevant = len(relevant)
    console.print(f"\n  Processing {len(relevant)} HIGH-priority documents\n")

    # ── Stage 2: Extraction (OCR + Docling) ───────────────────────────
    console.rule("[bold]Stage 2: Extraction (OCR + Docling)[/]")
    extractions: list[dict] = []
    path_lookup = {doc.doc_name: path for doc, path in downloaded if path}

    for triage_result in relevant:
        doc = triage_result.document
        pdf_path = path_lookup.get(doc.doc_name)
        if not pdf_path:
            continue

        console.print(f"  [cyan]{doc.wellbore}[/] | {doc.doc_name[:55]}")
        extraction = extract_document(
            triage_result, pdf_path,
            ocr_client=ocr_client,
        )
        extractions.append(extraction)
        stats.documents_extracted += 1

        n_tables = len(extraction.get("tables", []))
        n_text = len(extraction.get("text", {}))
        n_ocr = len(extraction.get("ocr_text", {}))
        stats.pages_processed += n_text + n_ocr
        stats.ocr_pages += n_ocr

        console.print(f"    Source: {extraction['source']} | "
                      f"Tables: {n_tables} | Text pages: {n_text} | OCR pages: {n_ocr}")

    # ── Stage 3: LLM Structuring (OpenRouter) ─────────────────────────
    console.rule(f"[bold]Stage 3: LLM Structuring ({llm_model})[/]")
    all_records: list[CasingRecord] = []

    for extraction in extractions:
        doc = extraction["document"]

        # Skip extractions with no content
        has_content = (
            extraction.get("tables")
            or any(v.strip() for v in extraction.get("text", {}).values())
            or any(v.strip() for v in extraction.get("ocr_text", {}).values())
        )
        if not has_content:
            continue

        console.print(f"  Structuring: [cyan]{doc.doc_name[:55]}[/]")
        result = structure_document(extraction, llm_client, llm_model, stats)

        if result.records:
            console.print(f"    [green]Found {len(result.records)} casing record(s)[/]")
            all_records.extend(result.records)
        elif result.errors:
            console.print(f"    [red]{result.errors[0]}[/]")
        else:
            console.print(f"    [dim]No casing data[/]")

    # ── Deduplicate & Output ──────────────────────────────────────────
    console.rule("[bold]Results[/]")
    console.print(f"  Raw records: {len(all_records)}")
    final_records = deduplicate_records(all_records)
    console.print(f"  After deduplication: {len(final_records)}")

    records_to_csv(final_records, output_path)
    print_stats(stats, time.time() - start, llm_model)


def main() -> None:
    """CLI entry point."""
    csv_path = CSV_INPUT
    output_path = OUTPUT_DIR / "casing_design.csv"

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--input" and i < len(sys.argv) - 1:
            csv_path = Path(sys.argv[i + 1])
        elif arg == "--output" and i < len(sys.argv) - 1:
            output_path = Path(sys.argv[i + 1])
        elif arg == "--model" and i < len(sys.argv) - 1:
            os.environ["OPENROUTER_MODEL"] = sys.argv[i + 1]

    asyncio.run(run_pipeline(csv_path, output_path))


if __name__ == "__main__":
    main()
