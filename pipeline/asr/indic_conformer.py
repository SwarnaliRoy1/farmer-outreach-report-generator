"""
Thin wrapper around AI4Bharat Indic Conformer 600M.
Produces timestamped native-language text segments from audio chunks.
"""

import torch
from transformers import AutoModel


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_asr_model(device: str) -> AutoModel:
    print("Loading ASR model...")
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


# =============================================================================
# INFERENCE
# =============================================================================

def transcribe_chunk(
    model: AutoModel,
    chunk: torch.Tensor,
    language: str,
    device: str,
    decoder: str = "ctc",
) -> str:
    """
    Run ASR on a single audio chunk (already on the correct device).
    Returns the transcribed string, or empty string on failure.
    """
    try:
        with torch.no_grad():
            text = model(chunk, language, decoder)
        return text.strip() if text else ""
    except Exception as e:
        print(f"    ASR failed: {e}")
        return ""