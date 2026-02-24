"""
2-stage extractor:
  Stage 1 — pull local/dialect crop disease and pest names from native-language text.
  Stage 2 — map each local name to Standard English + Scientific names.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List

from pipeline.extraction.base_llm import BaseLLM
from pipeline.transcript.utils import chunk_entries, format_original

log = logging.getLogger(__name__)


class TerminologyExtractor(BaseLLM):

    INVALID_TERMS = {
        "rat", "rats", "dog", "dogs", "cow", "buffalo",
        "animal", "tractor", "water", "rain",
    }

    async def extract(self, entries: List[Dict]) -> List[Dict]:
        chunks = chunk_entries(entries, max_chars=4000, text_key="original_text")
        log.info(f"TerminologyExtractor: {len(chunks)} chunk(s)...")

        all_local_terms: List[str] = []

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

        mapped = await asyncio.get_event_loop().run_in_executor(
            None, self._map_to_standard, all_local_terms
        )
        return mapped

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

    def _map_to_standard(self, local_terms: List[str]) -> List[Dict]:
        all_mapped: List[Dict] = []
        batch_size = 10

        for i in range(0, len(local_terms), batch_size):
            batch = local_terms[i : i + batch_size]
            log.info(f"  Mapping batch {i // batch_size + 1}: {batch}")

            messages = [
                {"role": "system", "content": "You are a plant pathology mapping expert."},
                {
                    "role": "user",
                    "content": f"""
Map the following LOCAL agricultural disease/pest names
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
            data = self._safe_parse_array(decoded)
            valid = [
                item for item in data
                if isinstance(item, dict) and item.get("Local Name", "").strip()
            ]
            all_mapped.extend(valid)

        return all_mapped

    def _filter_terms(self, items: List[str]) -> List[str]:
        return [
            t for t in items
            if not re.match(r"^[A-Za-z\s]+$", t)
            and t.lower() not in self.INVALID_TERMS
        ]