"""
3-prompt extractor:
  Prompt 1 — extract raw farmer questions and problems per chunk.
  Prompt 2 — clean / filter questions (drop interviewer questions).
  Prompt 3 — categorize problems into thematic groups.
"""

import json
import logging
from typing import Dict, List

from pipeline.extraction.base_llm import BaseLLM
from pipeline.transcript.utils import chunk_entries, format_transcript

log = logging.getLogger(__name__)


class FarmerInsightExtractor(BaseLLM):

    async def extract(self, entries: List[Dict]) -> Dict:
        chunks = chunk_entries(entries, max_chars=8000, text_key="translated_text")
        log.info(f"FarmerInsightExtractor: {len(chunks)} chunk(s)...")

        all_questions: List[str] = []
        all_problems:  List[str] = []

        for i, chunk in enumerate(chunks):
            log.info(f"  Chunk {i+1}/{len(chunks)}")
            transcript = format_transcript(chunk)
            if len(transcript.strip()) < 50:
                continue
            result = self._extract_from_chunk(transcript)
            all_questions.extend(result.get("farmer_questions", []))
            all_problems.extend(result.get("problems", []))

        all_questions = self._deduplicate(all_questions)
        all_problems  = self._deduplicate(all_problems)

        return {
            "farmer_questions": self._clean_questions(all_questions),
            "challenges":       self._categorize_problems(all_problems),
        }

    # ------------------------------------------------------------------

    def _extract_from_chunk(self, transcript: str) -> Dict:
        messages = [
            {
                "role": "user",
                "content": f"""
STRICT RULES:
- Extract ONLY questions and problems from FARMERS about their OWN farming difficulties.
- A farmer question expresses a complaint, grievance, or need for help.
- IGNORE questions asked BY an interviewer TO a farmer.
- IGNORE demographic or background data collection questions.
- IGNORE officer answers, greetings, narration.
- Convert reported questions into direct question form.
- Remove filler words (sir, please, etc.).
- DO NOT invent anything.
- If NO farmer questions or problems exist, return empty lists.

Transcript:
{transcript}

Return STRICT JSON only:
{{
  "farmer_questions": [],
  "problems": []
}}
""",
            }
        ]
        decoded = self._run_inference(messages, max_new_tokens=400)
        return self._safe_json(decoded, {"farmer_questions": [], "problems": []})

    def _clean_questions(self, questions: List[str]) -> List[str]:
        if not questions:
            return []
        messages = [
            {
                "role": "user",
                "content": f"""
Review the following questions from a rural Indian farmer meeting transcript.
Keep ONLY questions where a farmer is expressing a problem, complaint, or need for help.

REMOVE questions that ask the farmer to describe themselves (background/demographic).
KEEP questions like "Why are we getting less than 500 rupees for bhaji?"

Rules: keep meaning exactly the same; improve grammar only if needed; ensure each
ends with '?'; do NOT merge or invent questions.

Questions:
{json.dumps(questions, indent=2)}

Return STRICT JSON:
{{
  "farmer_questions": []
}}
""",
            }
        ]
        decoded = self._run_inference(messages, max_new_tokens=500)
        result  = self._safe_json(decoded, {"farmer_questions": questions})
        return result.get("farmer_questions", questions)

    def _categorize_problems(self, problems: List[str]) -> List[Dict]:
        if not problems:
            return []
        messages = [
            {"role": "system", "content": "You categorize rural Indian farmer problems."},
            {
                "role": "user",
                "content": f"""
Categorize the following farmer problems into logical categories such as:
- Wildlife and Pest Issues
- Input Cost and Availability
- Financial and Loan Issues
- Crop Diseases and Soil Health
- Market and Technology Adoption
- Infrastructure Issues
- Government Support Issues
- Operational Challenges

RULES: every problem must appear exactly once; do NOT invent problems; if unsure, use 'Others'.

Problems:
{json.dumps(problems, indent=2)}

Return STRICT JSON:
{{
  "challenges": [
    {{
      "category": "",
      "challenges": []
    }}
  ]
}}
""",
            },
        ]
        decoded = self._run_inference(messages, max_new_tokens=2500)
        result  = self._safe_json(decoded, {"challenges": []})
        return result.get("challenges", [])