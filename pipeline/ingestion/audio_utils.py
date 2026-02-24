"""
Handles discovery, loading, normalization, and combining of raw audio files
into a single 16kHz mono WAV ready for ASR + diarization.
"""

import glob
import os

import torch
import torchaudio


# =============================================================================
# FIND AND SORT FILES
# =============================================================================

def get_sorted_files(directory: str) -> list[str]:
    """
    Discover all audio files in a directory and return them in pipeline order:
    narration file(s) first, then all others sorted alphabetically.
    """
    extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))

    if not files:
        raise FileNotFoundError(f"No audio files found in {directory}")

    narration_files = [f for f in files if "narration" in os.path.basename(f).lower()]
    other_files     = sorted(
        [f for f in files if "narration" not in os.path.basename(f).lower()],
        key=lambda f: os.path.basename(f).lower(),
    )

    ordered = narration_files + other_files

    print("File order:")
    for i, f in enumerate(ordered):
        print(f"  {i+1}. {os.path.basename(f)}")

    return ordered


# =============================================================================
# LOAD + NORMALIZE
# =============================================================================

def load_and_normalize(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load an audio file, convert to mono, and resample to target_sr."""
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # stereo → mono

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)

    return wav  # shape: (1, samples)


# =============================================================================
# COMBINE + SAVE
# =============================================================================

def combine_audio(
    directory: str,
    output_path: str,
    target_sr: int = 16000,
) -> str:
    """
    Load all audio files from directory in order, concatenate them,
    and save as a single 16kHz mono PCM WAV.

    Returns the output_path for chaining.
    """
    files = get_sorted_files(directory)

    chunks = []
    for path in files:
        print(f"Loading: {os.path.basename(path)}")
        wav = load_and_normalize(path, target_sr)
        duration = wav.shape[1] / target_sr
        print(f"  → {duration:.2f}s | shape: {wav.shape}")
        chunks.append(wav)

    combined = torch.cat(chunks, dim=1)
    total_duration = combined.shape[1] / target_sr
    print(f"\nCombined duration: {total_duration:.2f}s")

    torchaudio.save(
        output_path,
        combined,
        sample_rate=target_sr,
        encoding="PCM_S",
        bits_per_sample=16,
    )
    print(f"Saved → {output_path}")
    return output_path