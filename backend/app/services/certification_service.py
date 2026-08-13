"""
certification_service.py

Task 4: Certification & Skill Evaluation Workflow.

A learner earns a certification level when they demonstrate sustained
accuracy on a required set of gestures. Three levels are defined:

  • Beginner    – ≥60 % accuracy on A–J (10 letters, ≥5 attempts each)
  • Intermediate – ≥75 % accuracy on A–T (20 letters, ≥5 attempts each)
  • Advanced     – ≥85 % accuracy on all 26 letters (≥10 attempts each)

The service is read-only with respect to attempts: it simply reads the
existing ProgressService data and evaluates eligibility. Earned
certifications are stored persistently in a separate JSON file.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json

from app.services.progress_service import ProgressService

# ── Level definitions ────────────────────────────────────────────────────────
LEVELS = {
    "Beginner": {
        "required_letters": list("ABCDEFGHIJ"),
        "min_accuracy": 60.0,
        "min_attempts_per_letter": 5,
    },
    "Intermediate": {
        "required_letters": list("ABCDEFGHIJKLMNOPQRST"),
        "min_accuracy": 75.0,
        "min_attempts_per_letter": 5,
    },
    "Advanced": {
        "required_letters": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "min_accuracy": 85.0,
        "min_attempts_per_letter": 10,
    },
}

CERT_FILE = Path(__file__).resolve().parent.parent / "data" / "certifications.json"


class CertificationService:
    def __init__(self, progress_service: ProgressService):
        self._progress = progress_service
        self._store: dict[str, list[dict]] = {}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, student_id: str) -> dict:
        """
        Evaluate which levels the student currently qualifies for, award any
        new ones, and return the full certification status.
        """
        student_id = student_id.strip()
        attempts = list(reversed(self._progress.get_history(student_id, limit=None)))
        per_letter = self._progress.per_letter_stats(attempts)

        results = {}
        newly_awarded = []

        for level_name, cfg in LEVELS.items():
            status = self._check_level(per_letter, cfg)
            results[level_name] = status

            already_earned = any(
                c["level"] == level_name
                for c in self._store.get(student_id, [])
            )

            if status["eligible"] and not already_earned:
                cert = {
                    "level": level_name,
                    "awarded_at": datetime.now(timezone.utc).isoformat(),
                    "snapshot": status,
                }
                self._store.setdefault(student_id, []).append(cert)
                newly_awarded.append(level_name)

        if newly_awarded:
            self._save()

        return {
            "student_id": student_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "level_status": results,
            "earned_certifications": self._store.get(student_id, []),
            "newly_awarded": newly_awarded,
        }

    def get_certifications(self, student_id: str) -> dict:
        """Return previously earned certifications without re-evaluating."""
        student_id = student_id.strip()
        return {
            "student_id": student_id,
            "earned_certifications": self._store.get(student_id, []),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _check_level(per_letter: dict, cfg: dict) -> dict:
        required = cfg["required_letters"]
        min_acc = cfg["min_accuracy"]
        min_att = cfg["min_attempts_per_letter"]

        letter_results = {}
        for letter in required:
            stats = per_letter.get(letter, {"attempts": 0, "correct": 0})
            att = stats["attempts"]
            acc = round(100 * stats["correct"] / att, 2) if att else 0.0
            letter_results[letter] = {
                "attempts": att,
                "accuracy": acc,
                "meets_criteria": att >= min_att and acc >= min_acc,
            }

        eligible = all(v["meets_criteria"] for v in letter_results.values())
        return {
            "eligible": eligible,
            "required_accuracy": min_acc,
            "required_attempts_per_letter": min_att,
            "letter_breakdown": letter_results,
        }

    def _load(self):
        CERT_FILE.parent.mkdir(parents=True, exist_ok=True)
        if CERT_FILE.exists():
            try:
                with open(CERT_FILE, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._store = {}

    def _save(self):
        with open(CERT_FILE, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2)


# ── Singleton ─────────────────────────────────────────────────────────────────
_cert_service: Optional["CertificationService"] = None


def get_certification_service() -> CertificationService:
    global _cert_service
    if _cert_service is None:
        from app.services.progress_service import get_progress_service
        _cert_service = CertificationService(get_progress_service())
    return _cert_service
