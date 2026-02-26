"""
Collects all extractor outputs into a single report dict and saves
both a master JSON and individual component files.

Report generation (DOCX, PDF, etc.) will be added to pipeline/report/exporter.py.
"""

import json
import logging
import os
from typing import Dict, List

log = logging.getLogger(__name__)

COMPONENT_FILES = [
    ("metadata",         "meeting_metadata.json"),
    ("conclusion",       "meeting_conclusion.json"),
    ("farmer_questions", "farmer_questions.json"),
    ("challenges",       "farmer_challenges.json"),
    ("participants",     "participant_report.json"),
    ("terminology",      "terminology_output.json"),
]


def assemble(
    # summary:          str,
    conclusion:       str,
    metadata:         Dict,
    narration:        Dict,
    terminology:      List[Dict],
    insights:         Dict,
    participants:     Dict,
) -> Dict:
    """
    Collect all extractor outputs into a single structured report dict.
    Pure data — no file I/O.
    """
    return {
        # "summary":          summary,
        "conclusion":       conclusion,
        "metadata":         metadata,
        "narration":        narration,
        "terminology":      terminology,
        "farmer_questions": insights.get("farmer_questions", []),
        "challenges":       insights.get("challenges", []),
        "participants":     participants,
    }


def save(report: Dict, output_dir: str, export_pdf: bool = True) -> str:
    """
    Save master report JSON and individual component files to output_dir.
    Returns the path to the master report file.
    """
    os.makedirs(output_dir, exist_ok=True)

    master_path = os.path.join(output_dir, "outreach_report.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"Master report saved: {master_path}")

    for key, filename in COMPONENT_FILES:
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({key: report[key]}, f, indent=2, ensure_ascii=False)
        log.info(f"Saved: {path}")

    if export_pdf:
        try:
            from pipeline.report.exporter import PDFReportGenerator
            pdf_path = os.path.join(output_dir, "outreach_report.pdf")
            PDFReportGenerator().create_report(report, pdf_path)
            log.info(f"PDF report saved: {pdf_path}")
        except Exception:
            log.exception("PDF export failed.")

    return master_path