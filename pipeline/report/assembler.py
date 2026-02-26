"""
Collects all extractor outputs into a single report dict and saves
both a master JSON, individual component files, and a PDF report.

Report generation (DOCX, etc.) can be added to pipeline/report/exporter.py.
"""

import json
import logging
import os
from typing import Dict, List

log = logging.getLogger(__name__)

COMPONENT_FILES = [
    ("farmer_questions", "farmer_questions.json"),
    ("challenges",       "farmer_challenges.json"),
    ("participants",     "participant_report.json"),
    ("terminology",      "terminology_output.json"),
]


def assemble(
    summary:      str,
    narration:    Dict,
    terminology:  List[Dict],
    insights:     Dict,
    participants: Dict,
) -> Dict:
    """
    Collect all extractor outputs into a single structured report dict.
    Pure data — no file I/O.
    """
    return {
        "summary":          summary,
        "narration":        narration,
        "terminology":      terminology,
        "farmer_questions": insights.get("farmer_questions", []),
        "challenges":       insights.get("challenges", []),
        "participants":     participants,
    }


def save(report: Dict, output_dir: str, export_pdf: bool = True) -> str:
    """
    Save master report JSON, individual component files, and optionally a PDF.

    Args:
        report:     Assembled report dict from assemble().
        output_dir: Directory to write all outputs into.
        export_pdf: Whether to also generate a PDF via exporter.py (default True).

    Returns:
        Path to the master JSON report file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Master JSON
    master_path = os.path.join(output_dir, "outreach_report.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"Master report saved: {master_path}")

    # Component JSONs
    for key, filename in COMPONENT_FILES:
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({key: report[key]}, f, indent=2, ensure_ascii=False)
        log.info(f"Saved: {path}")

    # PDF
    if export_pdf:
        from pipeline.report.exporter import PDFReportGenerator
        pdf_path = os.path.join(output_dir, "outreach_report.pdf")
        PDFReportGenerator().create_report(report, pdf_path)
        log.info(f"PDF report saved: {pdf_path}")

    return master_path