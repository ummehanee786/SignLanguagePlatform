"""
frame_buffer.py

Sprint Task 2, Phase 2 prototype: the data preparation stage a future
sequence model (LSTM/GRU/Transformer) will need, built now so the
surrounding infrastructure exists and is tested before that model does.

NO sequence model is implemented here - this is deliberately just the
"capture consecutive landmark vectors -> store the last N frames ->
assemble a fixed-shape sequence tensor" stage:

    Frame 1  -> 63 features
    Frame 2  -> 63 features
    Frame 3  -> 63 features
    ...
    Frame N  -> 63 features
       |
       v
    Sequence Tensor (N x 63)

Everything that turns a raw frame into a validated, normalized 63-value
vector is REUSED from the existing, already-tested pipeline - nothing
about landmark extraction, validation, or normalization is reimplemented
here:

    HandLandmarkDetector.extract_landmarks_with_metadata()  (Sprint 1)
    feature_pipeline.build_feature_vector()                  (Sprint 1)

This is the concrete proof of the design doc's central claim: the
classifier can change later, but the rest of the pipeline is reusable
as-is.
"""

from collections import deque
from typing import Optional

import numpy as np

from app.ai.handtracking.detector import HandLandmarkDetector
from app.ai.ml.inference.feature_pipeline import build_feature_vector, FeatureValidationError

FEATURE_DIM = 63
DEFAULT_SEQUENCE_LENGTH = 20  # e.g. ~0.6-2s of motion at 10-30fps sampling


class FrameBuffer:
    """
    Holds the last `max_frames` normalized feature vectors (one per usable
    frame) in a rolling window - the "Frame Buffer" component from the
    Phase 1 design doc.

    Frames where no hand (or more than one hand) was detected, or where
    landmark validation failed, are skipped rather than inserted as
    garbage/zero vectors - add_image() returns False for these so a
    caller can track how many frames were actually usable, but the buffer
    itself only ever holds real, valid feature vectors.
    """

    def __init__(self, max_frames: int = DEFAULT_SEQUENCE_LENGTH):
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        self.max_frames = max_frames
        self._frames: deque[np.ndarray] = deque(maxlen=max_frames)

    def add_vector(self, feature_vector: np.ndarray) -> None:
        """
        Appends an already-normalized 63-dim feature vector directly.
        Useful for testing, or if landmarks were already extracted
        elsewhere. Most callers should use add_image() instead.
        """
        vector = np.asarray(feature_vector, dtype=np.float32)
        if vector.shape != (FEATURE_DIM,):
            raise ValueError(f"Expected a ({FEATURE_DIM},) vector, got shape {vector.shape}")
        self._frames.append(vector)

    def add_image(self, image, detector: HandLandmarkDetector) -> bool:
        """
        Runs one frame through the full existing per-frame pipeline
        (detect -> validate hand count -> validate + normalize landmarks)
        and appends the result if - and only if - the frame was usable.

        Returns True if a frame was added, False if it was skipped (no
        hand, multiple hands, or failed feature validation) - the caller
        decides what to do with a skipped frame (e.g. just keep reading
        the next one; a brief gap shouldn't reset the whole buffer).
        """
        detection = detector.extract_landmarks_with_metadata(image)
        if detection["hand_count"] != 1:
            return False

        try:
            feature_df = build_feature_vector(detection["landmarks"])
        except FeatureValidationError:
            return False

        vector = feature_df.values[0].astype(np.float32)
        self.add_vector(vector)
        return True

    def is_full(self) -> bool:
        return len(self._frames) == self.max_frames

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def frames(self) -> list:
        """Returns the currently buffered vectors, oldest first."""
        return list(self._frames)


def build_sequence_tensor(
    buffer: FrameBuffer,
    sequence_length: Optional[int] = None,
    pad_mode: str = "zeros",
) -> np.ndarray:
    """
    The "Sequence Builder" component from the Phase 1 design doc: converts
    a FrameBuffer's current (possibly not-yet-full) contents into a fixed
    shape `(sequence_length, 63)` tensor - the shape a future sequence
    model would actually be called with.

    Args:
        buffer: a FrameBuffer instance.
        sequence_length: target sequence length. Defaults to the buffer's
                          own max_frames if not given.
        pad_mode: how to fill missing frames when the buffer isn't full
                   yet:
                     - "zeros"       : pad with zero vectors (default) -
                                       simple and standard for sequence
                                       models, at the cost of the model
                                       needing to learn to ignore padding.
                     - "repeat_first": repeat the earliest available frame
                                       backwards - avoids an artificial
                                       "jump" from zeros to a real pose,
                                       at the cost of implying motion that
                                       didn't happen.
                   Padding is always added at the START of the sequence
                   (oldest end), so the most recent frames stay at the end
                   in stable positions regardless of how full the buffer
                   is - relevant since most sequence architectures weight
                   recent timesteps most heavily.

    Returns:
        np.ndarray of shape (sequence_length, 63), dtype float32.
    """
    target_length = sequence_length or buffer.max_frames
    if pad_mode not in ("zeros", "repeat_first"):
        raise ValueError(f"pad_mode must be 'zeros' or 'repeat_first', got {pad_mode!r}")

    frames = buffer.frames()  # oldest -> newest

    # buffer may hold more than target_length if it was constructed with a
    # larger max_frames than the sequence length being requested here -
    # keep only the most recent `target_length` frames.
    frames = frames[-target_length:]

    missing = target_length - len(frames)
    if missing > 0:
        if not frames:
            # Nothing captured yet at all - zeros is the only sane option
            # regardless of pad_mode.
            pad_frames = [np.zeros(FEATURE_DIM, dtype=np.float32)] * missing
        elif pad_mode == "zeros":
            pad_frames = [np.zeros(FEATURE_DIM, dtype=np.float32)] * missing
        else:  # repeat_first
            pad_frames = [frames[0]] * missing
        frames = pad_frames + frames

    tensor = np.stack(frames, axis=0).astype(np.float32)
    assert tensor.shape == (target_length, FEATURE_DIM), (
        f"internal error: built shape {tensor.shape}, expected ({target_length}, {FEATURE_DIM})"
    )
    return tensor