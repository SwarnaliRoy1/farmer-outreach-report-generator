"""
Translates native-language transcript entries to English using
AI4Bharat IndicTrans2 (indic-en-1B).

Operates on the shared List[Dict] transcript schema:
  entry["original_text"]   → source (native language)
  entry["translated_text"] → filled in-place by translate_transcript()

Supported model directions:
  "ai4bharat/indictrans2-indic-en-1B"    — Indic → English  (default)
  "ai4bharat/indictrans2-en-indic-1B"    — English → Indic
  "ai4bharat/indictrans2-indic-indic-1B" — Indic → Indic

Language codes follow FLORES-200 / NLLB convention e.g.:
  "pan_Guru" (Punjabi), "hin_Deva" (Hindi), "tam_Taml" (Tamil)
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

MODEL_ID = "ai4bharat/indictrans2-indic-en-1B"


# =============================================================================
# TRANSLATOR
# =============================================================================

class IndicTrans2Translator:

    def __init__(self, device: str = "cuda", model_id: str = MODEL_ID):
        self.device = device

        print(f"Loading IndicTrans2 ({model_id}) on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

        # Required pre/post processor for IndicTrans2 tokenization artifacts
        self.processor = IndicProcessor(inference=True)
        print("IndicTrans2 loaded.")

    # ------------------------------------------------------------------
    # CORE BATCH TRANSLATION
    # ------------------------------------------------------------------

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str = "eng_Latn",
    ) -> list[str]:
        """
        Translate a list of strings from src_lang to tgt_lang.
        Uses FLORES-200 language codes (e.g. "pan_Guru", "hin_Deva").
        """
        preprocessed = self.processor.preprocess_batch(
            texts, src_lang=src_lang, tgt_lang=tgt_lang
        )

        inputs = self.tokenizer(
            preprocessed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,     # IndicTrans2 recommended max
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=5,
                num_return_sequences=1,
                max_length=256,
            )

        decoded    = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated = self.processor.postprocess_batch(decoded, lang=tgt_lang)
        return translated

    # ------------------------------------------------------------------
    # STRUCTURED TRANSCRIPT TRANSLATION
    # ------------------------------------------------------------------

    def translate_transcript(
        self,
        entries: list[dict],
        src_lang: str,
        tgt_lang: str = "eng_Latn",
        batch_size: int = 32,
    ) -> list[dict]:
        """
        Fill the 'translated_text' field of each entry in-place.
        Reads from entry['original_text'], writes to entry['translated_text'].
        Returns the same list for chaining.
        """
        total_batches = (len(entries) + batch_size - 1) // batch_size

        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            texts = [e["original_text"] for e in batch]

            print(f"  Translating batch {i // batch_size + 1}/{total_batches}...")
            translated_texts = self.translate_batch(texts, src_lang, tgt_lang)

            for entry, translated in zip(batch, translated_texts):
                entry["translated_text"] = translated

        print(f"Translation complete. {len(entries)} segments translated.")
        return entries