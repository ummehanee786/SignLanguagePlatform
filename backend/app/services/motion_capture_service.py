"""
motion_capture_service.py

Task 1's infrastructure: continuously ingests webcam frames streamed
during a practice attempt (see POST /practice/{session_id}/stream-frame
in api/practice.py), keeps a rolling per-session window of the last
20-30 frames (oldest automatically discarded), and exposes that window
to two different consumers:

  - the raw sequence tensor (app.ai.ml.sequence.frame_buffer.FrameBuffer,
    the existing Sprint 2 prototype - REUSED here unmodified rather than
    reimplemented) - the "expose the complete sequence to any future
    temporal model" requirement.
  - a per-frame validity/confidence log (motion_metrics.GestureFrameRecord)
    - what Task 2's motion-based metrics are computed from.

One MotionCaptureSession per active practice session_id, mirroring the
same in-memory per-session-dict pattern SessionService already uses
elsewhere in this codebase.
"""

from collections import deque
from typing import Dict, List, Optional

from app.ai.ml.sequence.frame_buffer import FrameBuffer, DEFAULT_SEQUENCE_LENGTH
from app.ai.assessment.motion_metrics import GestureFrameRecord
from app.ai.preprocessing.extract_landmarks import normalize_landmarks


class MotionCaptureSession:
    """
    Bundles the Task 1 landmark-sequence buffer and the Task 2 per-frame
    record log for ONE practice session, updated together from the same
    stream of frames - one add_prediction() call keeps both in lockstep,
    so a caller never has to worry about them drifting out of sync.
    """

    def __init__(self, max_frames: int = DEFAULT_SEQUENCE_LENGTH):
        self.sequence_buffer = FrameBuffer(max_frames=max_frames)
        self._records: "deque[GestureFrameRecord]" = deque(maxlen=max_frames)
        # Parameters for stable gesture detection (configurable)
        # default to 5 consecutive frames for alphabet recognition
        self.consecutive_frames_required = 5
        self.stable_prediction: Optional[str] = None
        self.stable_confidence: Optional[float] = None
        self.stable_streak: int = 0
        self.current_streak_class: Optional[str] = None
        
        # Deque to track timestamps of frames for FPS
        self._timestamps: "deque[float]" = deque(maxlen=20)
        self.latest_latency: float = 0.0

    def add_prediction(self, prediction) -> None:
        """
        `prediction` is a PredictionResponse - the SAME object a single
        classified frame produces (see GestureService.predict) - reused
        here per streamed frame so hand detection only ever runs ONCE
        per frame, regardless of how many downstream consumers need the
        result (the sequence buffer, the motion-metrics log, or both).

        Invalid frames (no hand detected, failed validation) are still
        recorded in the per-frame log (valid=False) - Task 2's "number
        of invalid frames before a valid prediction" needs to see them -
        but are NOT added to the sequence buffer, which only ever holds
        real, usable landmark vectors (matching FrameBuffer's existing
        contract).
        """
        import time
        self._timestamps.append(time.perf_counter())
        self.latest_latency = prediction.processing_time if hasattr(prediction, "processing_time") else (prediction.total_time_ms / 1000.0 if hasattr(prediction, "total_time_ms") and prediction.total_time_ms else 0.0)

        if prediction.success and prediction.landmarks:
            normalized = normalize_landmarks(prediction.landmarks)
            self.sequence_buffer.add_vector(normalized)
            self._records.append(GestureFrameRecord(
                normalized_landmarks=normalized,
                valid=True,
                predicted_class=prediction.predicted_class,
                confidence=prediction.confidence,
            ))

            # Stable prediction logic
            pred_class = prediction.predicted_class
            if pred_class == self.current_streak_class:
                self.stable_streak += 1
            else:
                self.current_streak_class = pred_class
                self.stable_streak = 1
                self.stable_prediction = None
                self.stable_confidence = None

            if self.stable_streak >= self.consecutive_frames_required:
                self.stable_prediction = pred_class
                # Average confidence over the streak frames
                recent = list(self._records)[-self.stable_streak:]
                recent_confs = [r.confidence for r in recent if r.valid and r.confidence is not None]
                if recent_confs:
                    self.stable_confidence = round(sum(recent_confs) / len(recent_confs), 4)
                else:
                    self.stable_confidence = prediction.confidence
        else:
            self._records.append(GestureFrameRecord(
                normalized_landmarks=None,
                valid=False,
            ))

            # Reset streak on invalid frame
            self.current_streak_class = None
            self.stable_streak = 0
            self.stable_prediction = None
            self.stable_confidence = None

    def get_processing_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        duration = self._timestamps[-1] - self._timestamps[0]
        if duration <= 0:
            return 0.0
        return round((len(self._timestamps) - 1) / duration, 2)

    def records(self) -> List[GestureFrameRecord]:
        """The per-frame log, oldest first - what motion_metrics.py consumes."""
        return list(self._records)

    def clear(self) -> None:
        """Resets both buffers - called once an attempt has been graded, so the
        next gesture's capture window starts clean rather than blending
        frames from two different letters."""
        self.sequence_buffer.clear()
        self._records.clear()
        self.stable_prediction = None
        self.stable_confidence = None
        self.stable_streak = 0
        self.current_streak_class = None
        self._timestamps.clear()
        self.latest_latency = 0.0

    def is_full(self) -> bool:
        return self.sequence_buffer.is_full()

    def __len__(self) -> int:
        return len(self._records)


class MotionCaptureManager:
    """Per-session registry - mirrors SessionService's in-memory dict pattern."""

    def __init__(self, max_frames: int = DEFAULT_SEQUENCE_LENGTH):
        self._sessions: Dict[str, MotionCaptureSession] = {}
        self._max_frames = max_frames

    def get_or_create(self, session_id: str) -> MotionCaptureSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = MotionCaptureSession(self._max_frames)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[MotionCaptureSession]:
        return self._sessions.get(session_id)

    def reset(self, session_id: str) -> None:
        """Clears the buffer for the next gesture, without discarding the session's slot."""
        session = self._sessions.get(session_id)
        if session is not None:
            session.clear()

    def remove(self, session_id: str) -> None:
        """Drops the session's capture window entirely, e.g. once practice ends."""
        self._sessions.pop(session_id, None)


# --- Module-level singleton, mirroring get_progress_service() / get_engine() ---

_motion_capture_manager: Optional["MotionCaptureManager"] = None


def get_motion_capture_manager() -> "MotionCaptureManager":
    global _motion_capture_manager
    if _motion_capture_manager is None:
        _motion_capture_manager = MotionCaptureManager()
    return _motion_capture_manager
