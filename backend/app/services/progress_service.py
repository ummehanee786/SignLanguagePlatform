"""
progress_service.py

Every graded practice attempt gets recorded here - student, alphabet
practiced, predicted alphabet, correct/incorrect, confidence, inference
time, time taken, timestamp - and this service turns that raw log into
dashboard metrics (accuracy, streak, strongest/weakest letters, etc) and
feeds the Sign Accuracy Assessment Engine's "gesture accuracy" field
(app/ai/assessment/sign_accuracy_engine.py).

Storage follows the same pattern as SessionService: an in-memory list,
written through to a JSON file so it survives a server restart. This is
deliberately NOT a real database yet, consistent with the rest of this
codebase's current storage - swapping this for a real DB later only
means changing _load()/_save()/record_attempt() internals, since every
other method here only depends on having a list of attempt dicts, not
on how they're stored.

A "student" is just a `student_id` string the caller provides (a name
or client-generated ID) - there's no login/auth system in this project
yet.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# Per-letter accuracy needs at least this many attempts before it's
# included in "strongest"/"weakest" - otherwise one lucky/unlucky
# attempt on a letter practiced once would dominate the ranking.
MIN_ATTEMPTS_FOR_RANKING = 3

RECENT_HISTORY_LIMIT = 10
TOP_N = 5


class ProgressService:
    def __init__(self, storage_path: Optional[Path] = None):
        self._attempts: list[dict] = []
        self._storage_path = storage_path or (
            Path(__file__).resolve().parent.parent / "data" / "attempts.json"
        )
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def record_attempt(
        self,
        student_id: str,
        alphabet_practiced: str,
        predicted_alphabet: str,
        correct: bool,
        confidence: float,
        inference_time: float,
        time_taken_seconds: Optional[float] = None,
        session_id: Optional[str] = None,
        feedback_message: Optional[str] = None,
        feedback_tip: Optional[str] = None,
    ) -> dict:
        """
        Records one graded attempt. Only call this for attempts where a
        gesture was actually predicted - a "no hand detected" response
        isn't a graded attempt and shouldn't be recorded here.

        `inference_time` is how long the MODEL took to process one
        frame (AI processing cost). `time_taken_seconds` is how long
        the STUDENT took to perform the attempt (wall-clock time since
        the letter was shown) - two genuinely different things.
        """
        record = {
            "student_id": student_id.strip(),
            "session_id": session_id,
            "alphabet_practiced": alphabet_practiced.upper(),
            "predicted_alphabet": predicted_alphabet.upper() if predicted_alphabet else None,
            "correct": correct,
            "confidence": confidence,
            "inference_time": inference_time,
            "time_taken_seconds": time_taken_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feedback_message": feedback_message,
            "feedback_tip": feedback_tip,
        }
        self._attempts.append(record)
        self._save()
        return record

    def get_history(self, student_id: str, limit: Optional[int] = None) -> list[dict]:
        """Most recent attempts first."""
        student_id = student_id.strip()
        attempts = [a for a in self._attempts if a["student_id"] == student_id]
        attempts.sort(key=lambda a: a["timestamp"], reverse=True)
        return attempts[:limit] if limit else attempts

    def get_gesture_accuracy(self, student_id: str, alphabet: str) -> float:
        """
        The student's historical accuracy (%) on ONE specific letter,
        across every recorded attempt at it (including the most recent
        one, if it's already been recorded). This is what the
        assessment engine reports as "overall gesture accuracy" -
        distinct from this attempt's binary correct/incorrect and from
        the model's confidence in this one prediction.
        """
        student_id = student_id.strip()
        alphabet = alphabet.upper()
        attempts = [
            a for a in self._attempts
            if a["student_id"] == student_id and a["alphabet_practiced"] == alphabet
        ]
        if not attempts:
            return 0.0
        correct = sum(1 for a in attempts if a["correct"])
        return round(100 * correct / len(attempts), 2)

    def get_dashboard(self, student_id: str) -> dict:
        student_id = student_id.strip()
        attempts = [a for a in self._attempts if a["student_id"] == student_id]

        total_attempts = len(attempts)
        if total_attempts == 0:
            return {
                "student_id": student_id,
                "total_attempts": 0,
                "accuracy_percentage": 0.0,
                "most_mistaken_alphabets": [],
                "strongest_alphabets": [],
                "weakest_alphabets": [],
                "daily_practice_streak": 0,
                "average_confidence": 0.0,
                "recent_practice_history": [],
            }

        correct_count = sum(1 for a in attempts if a["correct"])
        accuracy_percentage = round(100 * correct_count / total_attempts, 2)
        average_confidence = round(sum(a["confidence"] for a in attempts) / total_attempts, 4)

        per_letter = self.per_letter_stats(attempts)

        most_mistaken = sorted(
            (
                {"alphabet": letter, "mistake_count": stats["mistakes"], "attempts": stats["attempts"]}
                for letter, stats in per_letter.items()
                if stats["mistakes"] > 0
            ),
            key=lambda x: x["mistake_count"],
            reverse=True,
        )[:TOP_N]

        ranked_letters = [
            {
                "alphabet": letter,
                "accuracy_percentage": round(100 * stats["correct"] / stats["attempts"], 2),
                "attempts": stats["attempts"],
            }
            for letter, stats in per_letter.items()
            if stats["attempts"] >= MIN_ATTEMPTS_FOR_RANKING
        ]
        # "strongest" excludes 0% - a letter you've never once gotten
        # right shouldn't be labeled a strength just because it happens
        # to be the only (or best of a bad bunch of) candidate(s) that
        # meets the minimum-attempts bar. "weakest" has no such floor -
        # a genuine 0% is exactly what "weakest" should surface.
        strongest = sorted(
            (x for x in ranked_letters if x["accuracy_percentage"] > 0),
            key=lambda x: x["accuracy_percentage"],
            reverse=True,
        )[:TOP_N]
        weakest = sorted(ranked_letters, key=lambda x: x["accuracy_percentage"])[:TOP_N]

        streak = self._compute_daily_streak(attempts)

        recent_history = self.get_history(student_id, limit=RECENT_HISTORY_LIMIT)

        return {
            "student_id": student_id,
            "total_attempts": total_attempts,
            "accuracy_percentage": accuracy_percentage,
            "most_mistaken_alphabets": most_mistaken,
            "strongest_alphabets": strongest,
            "weakest_alphabets": weakest,
            "daily_practice_streak": streak,
            "average_confidence": average_confidence,
            "recent_practice_history": recent_history,
        }

    @staticmethod
    def per_letter_stats(attempts: list[dict]) -> dict:
        """
        Shared aggregation used by both get_dashboard() (Task 2) and
        report_service.py (the follow-up Assessment Report Generator) -
        keyed by the alphabet being PRACTICED (not predicted), so it's
        always "how is the student doing on letter X", not "how often
        does the model confuse letter X" (that's error_analysis.md's
        job, from a different angle). Public because it's explicitly
        meant to be reused outside this class.
        """
        per_letter = defaultdict(lambda: {"attempts": 0, "correct": 0, "mistakes": 0})
        for a in attempts:
            stats = per_letter[a["alphabet_practiced"]]
            stats["attempts"] += 1
            if a["correct"]:
                stats["correct"] += 1
            else:
                stats["mistakes"] += 1
        return per_letter

    @staticmethod
    def _compute_daily_streak(attempts: list[dict]) -> int:
        """
        Counts consecutive calendar days (UTC) of practice, ending at
        the most recent practice day. If the most recent practice day
        is more than 1 day before today, the streak is considered
        broken (0) - practicing "yesterday" still counts as an active
        streak (so a student checking their dashboard first thing in
        the morning, before practicing today, doesn't see their streak
        incorrectly reset to 0).
        """
        practice_dates = {
            datetime.fromisoformat(a["timestamp"]).date() for a in attempts
        }
        if not practice_dates:
            return 0

        today = datetime.now(timezone.utc).date()
        most_recent = max(practice_dates)
        if (today - most_recent) > timedelta(days=1):
            return 0

        streak = 0
        cursor = most_recent
        while cursor in practice_dates:
            streak += 1
            cursor -= timedelta(days=1)
        return streak

    def _save(self):
        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(self._attempts, f, indent=2)

    def _load(self):
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    self._attempts = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._attempts = []

    def get_session_review(self, student_id: str, session_id: str) -> Optional[dict]:
        """
        Builds a rich post-session review for GET /practice/{session_id}/review.

        Returns None when no attempts were recorded for that session_id
        (which usually means the session ID doesn't exist).

        Returned structure matches SessionReviewResponse in
        app/schemas/session_review.py.
        """
        student_id = student_id.strip()
        # Get all attempts for this specific session, in chronological order.
        session_attempts = [
            a for a in self._attempts
            if a.get("session_id") == session_id
            and a["student_id"] == student_id
        ]
        session_attempts.sort(key=lambda a: a["timestamp"])

        if not session_attempts:
            return None

        total_attempts = len(session_attempts)
        correct_count = sum(1 for a in session_attempts if a["correct"])
        incorrect_count = total_attempts - correct_count
        overall_score = round(100 * correct_count / total_attempts, 2)

        # Distinct gesture lists
        correct_gestures = sorted({
            a["alphabet_practiced"]
            for a in session_attempts if a["correct"]
        })
        incorrect_gestures = sorted({
            a["alphabet_practiced"]
            for a in session_attempts if not a["correct"]
        })

        # Confidence trend (one entry per attempt, in order)
        confidence_trend = [
            {
                "attempt_number": i + 1,
                "confidence": round(a["confidence"], 4),
                "correct": a["correct"],
                "expected_gesture": a["alphabet_practiced"],
            }
            for i, a in enumerate(session_attempts)
        ]

        # Most common mistakes: (expected, predicted) pairs when incorrect
        mistake_counts: dict = defaultdict(int)
        for a in session_attempts:
            if not a["correct"] and a.get("predicted_alphabet"):
                key = (a["alphabet_practiced"], a["predicted_alphabet"])
                mistake_counts[key] += 1
        most_common_mistakes = sorted(
            [
                {"expected": exp, "predicted": pred, "count": cnt}
                for (exp, pred), cnt in mistake_counts.items()
            ],
            key=lambda x: x["count"],
            reverse=True,
        )[:5]

        # Gesture-specific feedback: per letter, last feedback message stored
        per_gesture: dict = defaultdict(lambda: {
            "attempts": 0, "correct": 0,
            "last_feedback_message": None, "last_feedback_tip": None,
        })
        for a in session_attempts:  # already in chronological order
            g = per_gesture[a["alphabet_practiced"]]
            g["attempts"] += 1
            if a["correct"]:
                g["correct"] += 1
            if a.get("feedback_message"):
                g["last_feedback_message"] = a["feedback_message"]
            if a.get("feedback_tip"):
                g["last_feedback_tip"] = a["feedback_tip"]

        gesture_feedback = [
            {
                "gesture": letter,
                "attempts": stats["attempts"],
                "correct": stats["correct"],
                "accuracy_percentage": round(
                    100 * stats["correct"] / stats["attempts"], 2
                ),
                "last_feedback_message": stats["last_feedback_message"],
                "last_feedback_tip": stats["last_feedback_tip"],
            }
            for letter, stats in sorted(per_gesture.items())
        ]

        # Recommended gestures: weakest gestures from LIFETIME history that
        # the student didn't fully master in this session (< 100% in session)
        # and have >= MIN_ATTEMPTS_FOR_RANKING lifetime attempts.
        session_gestures = set(per_gesture.keys())
        all_attempts = [a for a in self._attempts if a["student_id"] == student_id]
        lifetime_per_letter = self.per_letter_stats(all_attempts)
        candidates = []
        for letter, stats in lifetime_per_letter.items():
            if stats["attempts"] < MIN_ATTEMPTS_FOR_RANKING:
                continue
            session_stat = per_gesture.get(letter)
            if session_stat and session_stat["correct"] == session_stat["attempts"]:
                continue  # perfect in this session — already mastered
            acc = round(100 * stats["correct"] / stats["attempts"], 2)
            candidates.append((acc, letter))
        candidates.sort()  # lowest accuracy first
        recommended_gestures = [letter for _, letter in candidates[:5]]

        return {
            "session_id": session_id,
            "student_id": student_id,
            "total_attempts": total_attempts,
            "correct_attempts": correct_count,
            "incorrect_attempts": incorrect_count,
            "overall_score": overall_score,
            "correct_gestures": correct_gestures,
            "incorrect_gestures": incorrect_gestures,
            "confidence_trend": confidence_trend,
            "most_common_mistakes": most_common_mistakes,
            "gesture_feedback": gesture_feedback,
            "recommended_gestures": recommended_gestures,
        }



_progress_service: Optional["ProgressService"] = None


def get_progress_service() -> "ProgressService":
    global _progress_service
    if _progress_service is None:
        _progress_service = ProgressService()
    return _progress_service