"""
pipeline/translation/sarvam_translate.py

Translates native-language transcript entries to English using
sarvamai/sarvam-translate (chat-style causal LM).

Drop-in replacement for indictrans2.py — identical public interface:
  IndicTrans2Translator  →  SarvamTranslator
  translate_transcript() →  same signature, same in-place mutation
  translate_batch()      →  same signature

Operates on the shared List[Dict] transcript schema:
  entry["original_text"]   → source (native language)
  entry["translated_text"] → filled in-place by translate_transcript()

Language names (plain English, not FLORES-200 codes):
  "Punjabi", "Hindi", "Tamil", "Telugu", "Marathi",
  "Kannada", "Gujarati", "Bengali", "Odia", "Malayalam"

FLORES-200 → Sarvam language name map is handled automatically
by translate_transcript() so main.py needs no changes.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "sarvamai/sarvam-translate"

# FLORES-200 short codes → plain English language names for Sarvam prompt
FLORES_TO_LANG = {
    "pan_Guru": "Punjabi",
    "hin_Deva": "Hindi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "mar_Deva": "Marathi",
    "kan_Knda": "Kannada",
    "guj_Gujr": "Gujarati",
    "ben_Beng": "Bengali",
    "ory_Orya": "Odia",
    "mal_Mlym": "Malayalam",
}


# =============================================================================
# TRANSLATOR
# =============================================================================

class SarvamTranslator:
    def __init__(self, device: str = "cuda"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16
        ).to(device)

        self.model.eval()

        if torch.cuda.is_available():
            print(f"Model loaded. Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        else:
            print("Model loaded on CPU.")

    # ------------------------------------------------------------------
    # SINGLE STRING TRANSLATION
    # ------------------------------------------------------------------

    def _translate_one(self, text: str, tgt_lang: str) -> str:
        messages = [
            {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
            {"role": "user", "content": text},
        ]
    
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
    
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,          
                temperature=0.01,        
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
    
        # Slice only the newly generated tokens
        new_tokens = generated_ids[0][input_length:].tolist()
    
        # Guard: if still empty, return raw decode for debugging
        if not new_tokens:
            print(f"[WARN] No tokens generated for input: {text[:80]}")
            return ""
    
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # CORE BATCH TRANSLATION
    # ------------------------------------------------------------------

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str = "eng_Latn",
    ) -> list[str]:
        # Resolve FLORES codes to plain names for both directions
        target_language = FLORES_TO_LANG.get(tgt_lang, tgt_lang)
        if target_language == "eng_Latn":
            target_language = "English"
    
        results = []
        for text in texts:
            if not text or not text.strip():
                results.append("")
                continue
            translated = self._translate_one(text, target_language)
            results.append(translated)
    
        return results

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