"""
Shared transcript formatting and chunking utilities used by all extraction modules.
"""

from typing import Dict, List


def format_transcript(entries: List[Dict]) -> str:
    """Format entries as 'Speaker <id>: <text>' using translated or raw text."""
    lines = []
    for entry in entries:
        speaker = entry.get("speaker_id", "Unknown")
        text = entry.get("translated_text") or entry.get("text", "")
        if text.strip():
            lines.append(f"Speaker {speaker}: {text.strip()}")
    return "\n".join(lines)


def format_original(entries: List[Dict]) -> str:
    """Format entries using only original_text (for terminology extraction)."""
    return "\n".join(
        e.get("original_text", "").strip()
        for e in entries
        if e.get("original_text", "").strip()
    )


def chunk_entries(
    entries: List[Dict],
    max_chars: int = 4000,
    text_key: str = "original_text",
) -> List[List[Dict]]:
    """
    Split entries into chunks not exceeding max_chars.

    Args:
        entries:   List of transcript entry dicts.
        max_chars: Maximum character count per chunk.
        text_key:  Field to measure for size. Falls back to 'text' if not found.
    """
    chunks, current, length = [], [], 0
    for e in entries:
        text = e.get(text_key) or e.get("text", "")
        if length + len(text) > max_chars and current:
            chunks.append(current)
            current, length = [], 0
        current.append(e)
        length += len(text)
    if current:
        chunks.append(current)
    return chunks