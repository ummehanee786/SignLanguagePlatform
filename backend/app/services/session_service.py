"""
session_service.py

Tracks "practice sessions" - one per learning attempt. For now,
sessions live in memory and are also written through to a JSON file
so they survive a server restart. Later this moves to a real
database, but nothing calling into this service will need to change -
that's the whole point of putting it behind a service class instead
of scattering session logic across routers.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Optional


class SessionService:
    def __init__(self, storage_path: Optional[Path] = None):
        self._sessions: dict[str, dict] = {}
        self._storage_path = storage_path or (
            Path(__file__).resolve().parent.parent / "data" / "sessions.json"
        )
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def start_session(self, lesson_id: int, student_id: str, auto_next: bool = True) -> dict:
        student_id = student_id.strip()
        session_id = str(uuid.uuid4())
        now = time.time()
        session = {
            "session_id": session_id,
            "student_id": student_id,
            "lesson_id": lesson_id,          # the lesson practice STARTED on
            "current_lesson_id": lesson_id,  # the lesson currently being shown (advances with auto-next)
            "current_lesson_started_at": now,  # when the CURRENT letter started being shown - used to measure time-taken per attempt
            "auto_next": auto_next,
            "start_time": now,
            "end_time": None,
            "attempts": 0,
            "correct_attempts": 0,
        }
        self._sessions[session_id] = session
        self._save()
        return session

    def record_attempt(self, session_id: str, correct: Optional[bool] = None) -> Optional[dict]:
        """
        Increments the attempt counter. If `correct` is given (i.e. this
        attempt was actually graded - a real prediction was made and
        compared against the expected letter), also updates the
        session's running correct-attempt count, which is what session
        accuracy is computed from.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session["attempts"] += 1
        if correct:
            session["correct_attempts"] += 1
        self._save()
        return session

    def advance_lesson(self, session_id: str, next_lesson_id: Optional[int]) -> Optional[dict]:
        """Updates which lesson is currently being displayed/practiced (auto-next)."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if next_lesson_id is not None:
            session["current_lesson_id"] = next_lesson_id
            session["current_lesson_started_at"] = time.time()
        self._save()
        return session

    @staticmethod
    def time_since_lesson_shown(session: dict) -> float:
        """
        Seconds elapsed since the CURRENT letter started being shown -
        i.e. how long the student has been attempting this specific
        letter, distinct from the AI's own inference_time (how long the
        model took to process one frame).
        """
        started_at = session.get("current_lesson_started_at", session["start_time"])
        return round(time.time() - started_at, 4)

    def reset_lesson_timer(self, session_id: str) -> Optional[dict]:
        """
        Restarts the "time taken" clock for the current letter, without
        changing which letter is being shown. advance_lesson() already
        does this implicitly when moving to a NEW letter (a correct
        answer) - this covers the other case: an incorrect attempt,
        where the student stays on the same letter and retries, and
        each retry should get its own fresh time-taken measurement
        rather than accumulating across attempts.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session["current_lesson_started_at"] = time.time()
        self._save()
        return session

    def end_session(self, session_id: str) -> Optional[dict]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session["end_time"] = time.time()
        self._save()
        return session

    def get_session(self, session_id: str) -> Optional[dict]:
        return self._sessions.get(session_id)

    @staticmethod
    def session_accuracy(session: dict) -> float:
        """Accuracy % for the CURRENT session only (not lifetime/dashboard accuracy)."""
        if session["attempts"] == 0:
            return 0.0
        return round(100 * session["correct_attempts"] / session["attempts"], 2)

    def _save(self):
        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(self._sessions, f, indent=2)

    def _load(self):
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    self._sessions = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._sessions = {}