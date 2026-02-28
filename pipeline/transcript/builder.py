"""
Merges diarization speaker turns with ASR output into the shared
List[Dict] transcript schema:

  {
    "speaker_id":    str,
    "start":         float,
    "end":           float,
    "original_text": str       ← native language, filled here
    "translated_text": str     ← filled later by translation stage
  }
"""

import json

import torch
import torchaudio

from pipeline.asr.indic_conformer import transcribe_chunk


# =============================================================================
# AUDIO HELPERS
# =============================================================================

def load_audio(path: str, target_sr: int = 16000, device: str = "cpu") -> torch.Tensor:
    """Load audio file to a (1, samples) tensor on the specified device."""
    wav, sr = torchaudio.load(path)
    wav = torch.mean(wav, dim=0, keepdim=True)  # stereo → mono
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    return wav.to(device)


def extract_chunk(
    wav: torch.Tensor, start: float, end: float, sr: int = 16000
) -> torch.Tensor:
    """Slice a waveform tensor between start and end seconds."""
    return wav[:, int(start * sr) : int(end * sr)]


# =============================================================================
# TRANSCRIPT BUILDER
# =============================================================================

def build_transcript(
    audio_path: str,
    turns: list[tuple[float, float, str]],
    asr_model,
    language: str,
    device: str,
    min_duration: float = 0.5,
) -> list[dict]:
    """
    For each diarization turn, extract the audio chunk, run ASR,
    and return the structured transcript list.
    """
    wav = load_audio(audio_path, device=device)
    transcript = []

    for i, (start, end, speaker) in enumerate(turns):
        duration = end - start
        if duration < min_duration:
            continue

        print(f"  [{i+1}/{len(turns)}] {speaker} | {start:.2f}s → {end:.2f}s ({duration:.2f}s)")

        chunk = extract_chunk(wav, start, end)
        text  = transcribe_chunk(asr_model, chunk, language, device)

        if text:
            transcript.append({
                "speaker_id":      speaker,
                "start":           round(start, 3),
                "end":             round(end, 3),
                "original_text":   text,
                "translated_text": "",       # filled by translation stage
            })

    return transcript


# =============================================================================
# SAVE / LOAD HELPERS
# =============================================================================

def save_transcript(transcript: list[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"transcript": transcript}, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(transcript)} segments → {output_path}")


def load_transcript(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("transcript", data)