"""
pipeline/extraction/terminology.py

3-stage extractor:
  Stage 1 — pull local/dialect crop disease and pest names from native-language text.
  Stage 2 — transliterate local names to Roman script (indic-transliteration library).
  Stage 3 — map each transliterated name to Standard English + Scientific names.

Output schema:
  [{"Crop": "", "Local Name": "<transliterated>", "Standard Name": "",
    "Scientific Name": "", "Language": ""}, ...]

Requires:
  pip install indic-transliteration
"""

import asyncio
import json
import logging
import re
from typing import Dict, List

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _transliterate

from pipeline.extraction.base_llm import BaseLLM
from pipeline.transcript.utils import chunk_entries, format_original

log = logging.getLogger(__name__)


# =============================================================================
# TRANSLITERATION
# =============================================================================

# FLORES-200 code → indic-transliteration script constant
FLORES_TO_SCRIPT = {
    "pan_Guru": sanscript.GURMUKHI,
    "hin_Deva": sanscript.DEVANAGARI,
    "tel_Telu": sanscript.TELUGU,
    "tam_Taml": sanscript.TAMIL,
    "kan_Knda": sanscript.KANNADA,
    "mal_Mlym": sanscript.MALAYALAM,
    "ben_Beng": sanscript.BENGALI,
    "guj_Gujr": sanscript.GUJARATI,
    "ory_Orya": sanscript.ORIYA,
    "mar_Deva": sanscript.DEVANAGARI,
}


def transliterate(text: str, flores_lang: str = "pan_Guru") -> str:
    """
    Transliterate Indian language text to Roman/IAST script.
    Returns title-cased result.

    Args:
        text:        Native script string (e.g. "ਪੀਲੀ ਕੁੰਗੀ")
        flores_lang: FLORES-200 language code (e.g. "pan_Guru")
                     — same code used throughout the rest of the pipeline.
    """
    if not text:
        return text

    source_script = FLORES_TO_SCRIPT.get(flores_lang, sanscript.GURMUKHI)
    result        = _transliterate(text, source_script, sanscript.IAST)
    return " ".join(w.capitalize() for w in result.split())


# =============================================================================
# TERMINOLOGY EXTRACTOR
# =============================================================================

class TerminologyExtractor(BaseLLM):

    INVALID_TERMS = {
        "rat", "rats", "dog", "dogs", "cow", "buffalo",
        "animal", "tractor", "water", "rain",
    }

    async def extract(self, entries: List[Dict], flores_lang: str = "pan_Guru") -> List[Dict]:
        """
        Args:
            entries:     Shared transcript schema (uses original_text).
            flores_lang: FLORES-200 source language code — passed in from
                         main.py so transliteration uses the correct script.
        """
        chunks = chunk_entries(entries, max_chars=4000, text_key="original_text")
        log.info(f"TerminologyExtractor: {len(chunks)} chunk(s)...")

        all_local_terms: List[str] = []

        # ── Stage 1: Extract local dialect terms ────────────────────────────
        for i, chunk in enumerate(chunks):
            log.info(f"  Chunk {i+1}/{len(chunks)}")
            transcript = format_original(chunk)
            if len(transcript) < 50:
                continue
            local_terms = await asyncio.get_event_loop().run_in_executor(
                None, self._extract_local_terms, transcript
            )
            all_local_terms.extend(local_terms)

        all_local_terms = self._deduplicate(all_local_terms, threshold=90)
        all_local_terms = self._filter_terms(all_local_terms)

        if not all_local_terms:
            log.info("No valid local terms found.")
            return []

        log.info(f"  {len(all_local_terms)} unique local terms found.")

        # ── Stage 2: Transliterate (indic-transliteration library) ──────────
        transliterated = {
            term: transliterate(term, flores_lang)
            for term in all_local_terms
        }
        log.info(f"  Transliteration: {transliterated}")

        # ── Stage 3: Map to Standard + Scientific names ──────────────────────
        mapped = await asyncio.get_event_loop().run_in_executor(
            None, self._map_to_standard, transliterated
        )
        return mapped

    # ------------------------------------------------------------------
    # STAGE 1 — Extract local terms
    # ------------------------------------------------------------------

    def _extract_local_terms(self, transcript: str) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": "You are an expert in Indian agricultural plant diseases.",
            },
            {
                "role": "user",
                "content": f"""
STRICT RULES:
- Extract ONLY LOCAL/DIALECT crop disease or pest names.
- Must be from ORIGINAL text.
- DO NOT translate.
- DO NOT output English names.
- DO NOT output animals (rat, dog, cow, etc.).
- DO NOT output general farming words.
- If unsure, DO NOT include.

Return ONLY a JSON array of strings.
Example: ["ਪੀਲੀ ਕੁੰਗੀ", "ਟੇਲਾ"]

Transcript:
{transcript}
""",
            },
        ]
        decoded = self._run_inference(messages, max_new_tokens=300)
        return self._safe_parse_list(decoded)

    # ------------------------------------------------------------------
    # STAGE 3 — Map to Standard + Scientific names
    # ------------------------------------------------------------------

    def _map_to_standard(self, transliterated: Dict[str, str]) -> List[Dict]:
        """
        Args:
            transliterated: {"ਪੀਲੀ ਕੁੰਗੀ": "Pīlī Kuṃgī", ...}

        Returns:
            [{"Crop": "", "Local Name": "Pīlī Kuṃgī",
              "Standard Name": "", "Scientific Name": "", "Language": ""}, ...]
        """
        all_mapped: List[Dict] = []
        batch_size = 10
        terms      = list(transliterated.values())

        for i in range(0, len(terms), batch_size):
            batch = terms[i : i + batch_size]
            log.info(f"  Mapping batch {i // batch_size + 1}: {batch}")

            messages = [
                {
                    "role": "system",
                    "content": "You are a plant pathology mapping expert.",
                },
                {
                    "role": "user",
                    "content": f"""
Map the following transliterated Punjabi agricultural disease/pest names
to their Standard English Name and Scientific Name.

CRITICAL:
- The "Local Name" field MUST be copied EXACTLY as given.
- If unknown, leave Standard Name and Scientific Name as empty strings.
- Return ONLY a JSON array, no explanation.

Format:
[
  {{
    "Crop": "",
    "Local Name": "",
    "Standard Name": "",
    "Scientific Name": "",
    "Language": ""
  }}
]

Local Terms:
{json.dumps(batch, ensure_ascii=False)}
""",
                },
            ]

            decoded = self._run_inference(messages, max_new_tokens=800)
            data    = self._safe_parse_array(decoded)
            valid   = [
                item for item in data
                if isinstance(item, dict) and item.get("Local Name", "").strip()
            ]
            all_mapped.extend(valid)

        return all_mapped

    # ------------------------------------------------------------------
    # FILTER
    # ------------------------------------------------------------------

    def _filter_terms(self, items: List[str]) -> List[str]:
        return [
            t for t in items
            if not re.match(r"^[A-Za-z\s]+$", t)
            and t.lower() not in self.INVALID_TERMS
        ]