from typing import List, Optional

from pydantic import BaseModel


class StreamFrameResponse(BaseModel):
    """What POST /practice/{session_id}/stream-frame returns."""
    session_id: str
    buffered_frames: int   # how many frames are currently in the rolling window
    buffer_full: bool      # whether the window has reached its max size (oldest now discarding)
    frame_valid: bool      # whether THIS frame had a usable hand detection
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    stable_prediction: Optional[str] = None
    stable_confidence: Optional[float] = None
    stable_streak: int = 0
    inference_latency: float = 0.0
    processing_fps: float = 0.0
    has_person: bool = False
    hand_count: int = 0
    upper_body_visible: bool = False
    partial_hand_visible: bool = False
    hand_centered: bool = False


class SequenceResponse(BaseModel):
    """
    What GET /practice/{session_id}/sequence returns - the Task 1
    deliverable: exposes the buffered landmark sequence in the exact
    shape a future temporal model (LSTM/GRU/Transformer) would consume:
    a list of frames, each a 63-value normalized landmark vector.
    """
    session_id: str
    frame_count: int
    max_frames: int
    is_full: bool
    sequence: List[List[float]]
