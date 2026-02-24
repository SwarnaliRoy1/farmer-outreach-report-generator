"""
Extracts participant names and roles from the transcript and structures
them by role category.
"""

import logging
import re
from typing import Dict, List, Optional

from pipeline.extraction.base_llm import BaseLLM
from pipeline.transcript.utils import format_transcript

log = logging.getLogger(__name__)


class ParticipantExtractor(BaseLLM):

    async def extract(self, entries: List[Dict]) -> Dict:
        transcript = format_transcript(entries)[:20000]
        result     = self._extract_from_transcript(transcript)
        return self._structure_roles(result.get("participants", []))

    # ------------------------------------------------------------------

    def _extract_from_transcript(self, transcript: str) -> Dict:
        messages = [
            {
                "role": "system",
                "content": "Extract participant names and roles from transcript.",
            },
            {
                "role": "user",
                "content": f"""
Extract all real participants.

Rules:
- Ignore Speaker labels.
- Do not assume farmer.
- If unclear role → Unknown.

Transcript:
{transcript}

Return JSON:
{{
  "participants": [
    {{"name": "", "role": "", "village": "", "phone_number": ""}}
  ]
}}
""",
            },
        ]
        decoded = self._run_inference(messages, max_new_tokens=600)
        return self._safe_json(decoded, {"participants": []})

    def _structure_roles(self, participants: List[Dict]) -> Dict:
        by_role: Dict[str, List] = {
            "coordinators":       [],
            "reporting_managers": [],
            "sarpanch":           [],
            "farmers":            [],
            "other_officials":    [],
        }
        for p in participants:
            role = (p.get("role") or "").lower()
            if "coordinator" in role:
                by_role["coordinators"].append(p)
            elif role in ("reporting manager", "rm"):
                by_role["reporting_managers"].append(p)
            elif "sarpanch" in role:
                by_role["sarpanch"].append(p)
            elif "farmer" in role:
                by_role["farmers"].append(p)
            else:
                by_role["other_officials"].append(p)

        return {
            "total_count":           len(participants),
            "participants_by_role":  by_role,
            "detailed_participants": participants,
        }

    @staticmethod
    def _validate_phone(phone: Optional[str]) -> Optional[str]:
        if not phone:
            return None
        phone = re.sub(r"\D", "", phone)
        return phone if re.match(r"^\d{10}$", phone) else None