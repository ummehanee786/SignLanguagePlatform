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

    def start_session(self, lesson_id: int) -> dict:
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "lesson_id": lesson_id,
            "start_time": time.time(),
            "end_time": None,
            "attempts": 0,
        }
        self._sessions[session_id] = session
        self._save()
        return session

    def record_attempt(self, session_id: str) -> Optional[dict]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session["attempts"] += 1
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