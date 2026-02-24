"""
Thin wrapper around pyannote/speaker-diarization-community-1.
Returns a list of (start, end, speaker_id) turn tuples.
"""

import torch
from pyannote.audio import Pipeline


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_diarization_pipeline(device: str) -> Pipeline:
    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    pipeline.to(torch.device(device))

    # Tuned parameters for better speaker distinction
    pipeline.embedding_exclude_overlap = True
    pipeline._pipelines["clustering"]._instantiated["Fa"] = 0.3

    return pipeline


# =============================================================================
# INFERENCE
# =============================================================================

def diarize(pipeline: Pipeline, audio_path: str) -> list[tuple[float, float, str]]:
    """
    Run full-audio diarization and return a sorted list of
    (start_sec, end_sec, speaker_id) tuples.
    """
    print("Running speaker diarization...")
    diarization = pipeline(audio_path)

    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    print(f"Found {len(turns)} speaker turns.")
    return turns