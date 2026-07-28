from typing import List

from pydantic import BaseModel


class StreamFrameResponse(BaseModel):
    """What POST /practice/{session_id}/stream-frame returns."""
    session_id: str
    buffered_frames: int   # how many frames are currently in the rolling window
    buffer_full: bool      # whether the window has reached its max size (oldest now discarding)
    frame_valid: bool      # whether THIS frame had a usable hand detection


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
