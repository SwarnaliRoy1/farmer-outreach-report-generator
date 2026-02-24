"""
Loads Ministral 14B once and exposes shared inference + JSON parsing helpers.
All extractor/generator classes inherit from this.
"""

import json
import logging
import re
from typing import Dict, List, Optional

import torch
from rapidfuzz import fuzz
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512"
log = logging.getLogger(__name__)


class BaseLLM:

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Loading model on {self.device}...")

        self.tokenizer = MistralCommonBackend.from_pretrained(MODEL_ID)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            offload_folder="offload",
            offload_state_dict=True,
        )
        self.model.eval()
        log.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        messages: List[Dict],
        max_new_tokens: int = 800,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Tokenize messages, generate, and return only the new decoded tokens."""
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        prompt_length = tokenized["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
            )

        return self.tokenizer.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        )

    # ------------------------------------------------------------------
    # JSON PARSING
    # ------------------------------------------------------------------

    def _safe_json(self, text: str, fallback: dict) -> dict:
        """Parse a JSON object from raw model output; return fallback on failure."""
        text = re.sub(r"```(?:json)?", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return fallback

    def _safe_parse_list(self, text: str) -> List[str]:
        """Parse a JSON array of strings from raw model output."""
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return []

    def _safe_parse_array(self, text: str) -> List[Dict]:
        """Parse a JSON array of objects from raw model output."""
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        return []

    # ------------------------------------------------------------------
    # DEDUPLICATION
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(items: List[str], threshold: int = 85) -> List[str]:
        """Remove near-duplicate strings using fuzzy ratio."""
        unique = []
        for item in items:
            item = item.strip()
            if item and not any(fuzz.ratio(item, u) > threshold for u in unique):
                unique.append(item)
        return unique