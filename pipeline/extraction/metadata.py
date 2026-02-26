"""
Regex-first metadata extraction with optional LLM backfill (Ministral via BaseLLM).

- First pass: deterministic regex extraction.
- Second pass (optional): LLM fills ONLY missing keys; never overwrites regex hits.
- Uses BaseLLM helpers: _run_inference + _safe_json.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pipeline.extraction.base_llm import BaseLLM

log = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

SCHEMA: Dict[str, Any] = {
    "date": None,
    "day": None,
    "village": None,
    "sarpanch_name": None,
    "panchayat": None,
    "block": None,
    "phone_number": None,
    "event_location": None,
    "district": None,
    "farmers_attended_total": None,
    "coordinator_name": None,
    "reporting_manager_name": None,
    "female_farmers_count": None,
    "male_farmers_count": None,
    "event_start_time": None,
    "event_end_time": None,
}

NUMWORDS = {
    "zero": 0, "nil": 0, "none": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


# =============================================================================
# HELPERS (regex-first)
# =============================================================================

def normalize_text(t: str) -> str:
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\\ u", "\\u")               # fix broken escapes
    t = re.sub(r"\\\s*u0A3C", "", t)           # remove marker seen in your data
    t = re.sub(r"\s+", " ", t).strip()
    return t


def pick_relevant_window(text: str, window: int = 7000) -> str:
    text_n = text.lower()
    keys = [
        "today", "date", "village", "panchayat", "block",
        "coordinator", "reporting manager", "sarpanch",
        "phone", "farmer", "farmers", "event start", "event end", "day",
    ]
    idxs = [text_n.find(k) for k in keys if text_n.find(k) != -1]
    if not idxs:
        return text[:window]
    i = max(min(idxs) - window // 2, 0)
    return text[i:i + window]


def first_match(patterns: List[str], text: str, flags=re.I) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(1).strip()
    return None


def to_int_maybe(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s.isdigit():
        return int(s)
    return NUMWORDS.get(s)


def clean_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    v = re.sub(r"[,\.;:\-]+$", "", v).strip()
    return v if v else None


def extract_meta_regex(text: str) -> Dict[str, Any]:
    text = normalize_text(text)
    out = dict(SCHEMA)

    # Stop at comma OR period OR end (prevents bleeding)
    STOP = r"(?=\s*(?:,|\.|$))"

    out["date"] = clean_value(first_match([
        rf"(?:today'?s\s+date\s*(?:is)?\s*)([^,\.]+){STOP}",
        rf"(?:\bdate\b\s*)([A-Za-z]+\s+\d{{1,2}},\s*\d{{4}}){STOP}",
        rf"(?:\bdate\b\s*)(\d{{1,2}}[\/\-]\d{{1,2}}[\/\-]\d{{2,4}}){STOP}",
    ], text))

    out["day"] = clean_value(first_match([
        rf"(?:day\s*(?:is)?\s*)([A-Za-z]+){STOP}",
    ], text))

    out["village"] = clean_value(first_match([
        rf"(?:village\s+name\s*(?:is)?\s*)([^,\.]+){STOP}",
        rf"(?:\bvillage\b\s*)([^,\.]+){STOP}",
    ], text))

    out["panchayat"] = clean_value(first_match([
        rf"(?:panchayat\s+name\s*(?:is)?\s*)([^,\.]+){STOP}",
        rf"(?:\bpanchayat\b\s*)([^,\.]+){STOP}",
    ], text))

    out["block"] = clean_value(first_match([
        rf"(?:block\s*(?:is)?\s*)([^,\.]+){STOP}",
    ], text))

    out["coordinator_name"] = clean_value(first_match([
        rf"(?:coordinator\s+name\s*(?:is)?\s*)([^,\.]+){STOP}",
        rf"(?:name\s+of\s+the\s+coordinator\s*)([^,\.]+){STOP}",
    ], text))

    out["reporting_manager_name"] = clean_value(first_match([
        rf"(?:reporting\s+manager\s*(?:name)?\s*(?:is)?\s*)([^,\.]+){STOP}",
        rf"(?:name\s+of\s+the\s+reporting\s+manager\s*)([^,\.]+){STOP}",
    ], text))

    out["sarpanch_name"] = clean_value(first_match([
        rf"(?:sarpanch\s+name\s*(?:is)?\s*)([^,\.]+){STOP}",
    ], text))

    out["event_location"] = clean_value(first_match([
        rf"(?:meeting\s+location\s*)([^,\.]+){STOP}",
        rf"(?:event\s+location\s*)([^,\.]+){STOP}",
    ], text))

    out["district"] = clean_value(first_match([
        rf"(?:district\s*)([^,\.]+){STOP}",
    ], text))

    # Phone: digits or spaced digits
    phone_raw = first_match([
        r"(?:phone\s+number\s*(?:is)?\s*)(\+?\d[\d\s]{8,}\d)",
    ], text)
    if phone_raw:
        digits = re.sub(r"\D", "", phone_raw)
        if len(digits) >= 10:
            out["phone_number"] = digits[-10:]

    # Total farmers
    tf = first_match([
        r"number\s+of\s+total\s+farmers\s*\.\s*([A-Za-z]+|\d+)",
        r"number\s+of\s+total\s+farmers\s*[:\-]?\s*([A-Za-z]+|\d+)",
        r"\btotal\s+farmers\b\s*[:\-]?\s*([A-Za-z]+|\d+)",
    ], text)
    tf_int = to_int_maybe(tf)
    if tf_int is not None:
        out["farmers_attended_total"] = str(tf_int)

    # Female/male farmers
    female_raw = first_match([r"female\s+farmers\s*,?\s*(nil|none|\d+|[A-Za-z]+)"], text)
    male_raw   = first_match([r"male\s+farmers\s*,?\s*(nil|none|\d+|[A-Za-z]+)"], text)

    f_int = to_int_maybe(female_raw)
    m_int = to_int_maybe(male_raw)

    # Special-case: "male farmers, nil ... , eight" => choose last numeric token in the clause
    m_clause = re.search(r"male\s+farmers([^\.]{0,60})", text, re.I)
    if m_clause:
        tail = m_clause.group(1).lower()
        tokens = re.findall(
            r"\b(nil|none|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\b",
            tail,
        )
        if tokens:
            last = tokens[-1]
            last_int = to_int_maybe(last)
            if (tokens[0] in ("nil", "none", "zero")) and (last_int is not None) and (last_int != 0):
                m_int = last_int

    if f_int is not None:
        out["female_farmers_count"] = f_int
    if m_int is not None:
        out["male_farmers_count"] = m_int

    # Consistency fix: if total known and female=0, assume all male if male missing/0
    if out["farmers_attended_total"] is not None and out["female_farmers_count"] == 0:
        try:
            tot = int(out["farmers_attended_total"])
            if out["male_farmers_count"] in (None, 0):
                out["male_farmers_count"] = tot
        except Exception:
            pass

    # Start/end time
    ms = re.search(
        r"(?:event\s+start(?:\s+time)?\s*(?:is)?\s*)(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
        text,
        re.I,
    )
    if ms:
        hh, mm, ap = ms.group(1), ms.group(2), ms.group(3).upper()
        out["event_start_time"] = f"{hh}{(':' + mm) if mm else ''}{ap}"

    me = re.search(
        r"(?:event\s+end(?:\s+time)?\s*(?:is)?\s*(?:approximately)?\s*)(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
        text,
        re.I,
    )
    if me:
        hh, mm, ap = me.group(1), me.group(2), me.group(3).upper()
        out["event_end_time"] = f"{hh}{(':' + mm) if mm else ''}{ap}"

    if out["event_start_time"]:
        out["event_start_time"] = out["event_start_time"].replace(" ", "")
    if out["event_end_time"]:
        out["event_end_time"] = out["event_end_time"].replace(" ", "")

    return out


# =============================================================================
# LLM BACKFILL PROMPT
# =============================================================================

def build_fill_prompt(en_text: str, current: Dict[str, Any]) -> str:
    return f"""
Fill ONLY the missing fields for this JSON. Do not change existing values.
Return ONLY one JSON object with the same keys as the schema.
If still not present, keep null.

Schema keys:
{list(SCHEMA.keys())}

Current JSON:
{json.dumps(current, ensure_ascii=False)}

Transcript:
\"\"\"{en_text[:8000]}\"\"\"

Return JSON only.
""".strip()


# =============================================================================
# METADATA EXTRACTOR
# =============================================================================

class MetadataExtractor(BaseLLM):

    def __init__(self, base: Optional[BaseLLM] = None, device: Optional[str] = None):
        if base is not None:
            # share loaded model/tokenizer/device
            self.model = base.model
            self.tokenizer = base.tokenizer
            self.device = base.device
        else:
            super().__init__(device=device)

    def extract(self, english_text: str, use_llm: bool = True, window: int = 7000) -> Dict[str, Any]:
        win = pick_relevant_window(english_text, window=window)
        base = extract_meta_regex(win)

        if use_llm and any(v is None for v in base.values()):
            try:
                base = self._llm_fill_missing(win, base)
            except Exception:
                log.exception("LLM backfill failed; returning regex-only metadata.")
        return base

    def _llm_fill_missing(self, en_text: str, current: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_fill_prompt(en_text, current)
        messages = [{"role": "user", "content": prompt}]

        raw = self._run_inference(messages, max_new_tokens=450)

        # safe parse; if parse fails, keep current
        obj = self._safe_json(raw, fallback={})

        merged = dict(current)
        for k in SCHEMA.keys():
            if merged.get(k) is None and obj.get(k) is not None:
                merged[k] = obj.get(k)
        return merged

# # =========================== Useage =====================================
# from pipeline.extraction.narration import NarrationGenerator
# from pipeline.extraction.metadata import MetadataExtractor

# base = NarrationGenerator()               # loads Ministral once
# meta_extractor = MetadataExtractor(base=base)

# meta = meta_extractor.extract(english_text, use_llm=True)

# #==================Or======================================================
# from pipeline.extraction.metadata import MetadataExtractor
# meta = MetadataExtractor().extract(english_text, use_llm=True)