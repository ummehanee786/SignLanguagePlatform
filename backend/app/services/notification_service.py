"""
notification_service.py

Task 6: Notification / Reminder System via API Events.

Generates contextual, rule-based notifications for learners based on their
practice history and learner profile. Notifications are stored persistently
and marked as read/dismissed individually.

Notification Types:
  • STREAK_BROKEN   – no practice in 2+ days
  • STREAK_ACTIVE   – encourage continuing a streak
  • WEAK_ALPHABET   – a letter is falling below 50% accuracy
  • NEW_CERT_READY  – student might qualify for a certification level
  • PRACTICE_DUE    – generic reminder if idle for 1+ day
  • MILESTONE       – every 100th attempt
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import json
import uuid

from app.services.progress_service import ProgressService

NOTIF_FILE = Path(__file__).resolve().parent.parent / "data" / "notifications.json"

# ── Rule thresholds ───────────────────────────────────────────────────────────
IDLE_REMINDER_HOURS = 24        # no practice → PRACTICE_DUE
STREAK_BROKEN_DAYS  = 2         # no practice in N days → STREAK_BROKEN
WEAK_ACCURACY_PCT   = 50.0      # letter below this → WEAK_ALPHABET
MILESTONE_EVERY     = 100       # total attempts multiple → MILESTONE

# ── Certification eligibility hints ──────────────────────────────────────────
_CERT_HINTS = {
    "Beginner":     {"letters": list("ABCDEFGHIJ"),     "min_acc": 60.0, "min_att": 5},
    "Intermediate": {"letters": list("ABCDEFGHIJKLMNOPQRST"), "min_acc": 75.0, "min_att": 5},
    "Advanced":     {"letters": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "min_acc": 85.0, "min_att": 10},
}


class NotificationService:
    def __init__(self, progress_service: ProgressService):
        self._progress = progress_service
        self._store: dict[str, list[dict]] = {}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_notifications(self, student_id: str, unread_only: bool = False) -> list[dict]:
        """Return stored notifications for a student."""
        student_id = student_id.strip()
        notifs = self._store.get(student_id, [])
        if unread_only:
            notifs = [n for n in notifs if not n.get("read")]
        return sorted(notifs, key=lambda n: n["created_at"], reverse=True)

    def generate(self, student_id: str) -> list[dict]:
        """
        Evaluate the student's history and produce any new notifications.
        Deduplicates: won't re-add the same type if an identical unread one
        already exists.
        """
        student_id = student_id.strip()
        attempts   = list(reversed(self._progress.get_history(student_id, limit=None)))
        new_notifs = []

        if not attempts:
            new_notifs.append(self._build(
                "PRACTICE_DUE", "👋 Welcome! Start your first practice session.",
                "Head to Practice to begin learning sign language.", "info"
            ))
        else:
            now   = datetime.now(timezone.utc)
            last  = datetime.fromisoformat(attempts[-1]["timestamp"])
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            idle_hours = (now - last).total_seconds() / 3600

            # ── Streak / idle reminders ───────────────────────────────────────
            if idle_hours >= STREAK_BROKEN_DAYS * 24:
                new_notifs.append(self._build(
                    "STREAK_BROKEN",
                    f"⚠️ Your streak is at risk!",
                    f"You haven't practiced in {int(idle_hours // 24)} days. Jump back in to keep your momentum.",
                    "warning"
                ))
            elif idle_hours >= IDLE_REMINDER_HOURS:
                new_notifs.append(self._build(
                    "PRACTICE_DUE",
                    "⏰ Time to practice!",
                    f"It's been over {int(idle_hours):.0f} hours since your last session. Keep the streak going!",
                    "info"
                ))

            # ── Milestones ────────────────────────────────────────────────────
            total = len(attempts)
            if total > 0 and total % MILESTONE_EVERY == 0:
                new_notifs.append(self._build(
                    "MILESTONE",
                    f"🎉 {total} attempts reached!",
                    "Amazing dedication! You've completed a major milestone on your learning journey.",
                    "success"
                ))

            # ── Weak alphabets ────────────────────────────────────────────────
            per_letter = self._progress.per_letter_stats(attempts)
            for letter, stats in per_letter.items():
                if stats["attempts"] >= 5:
                    acc = 100 * stats["correct"] / stats["attempts"]
                    if acc < WEAK_ACCURACY_PCT:
                        new_notifs.append(self._build(
                            "WEAK_ALPHABET",
                            f"📉 Letter '{letter}' needs work",
                            f"Your accuracy on '{letter}' is {acc:.0f}%. Focus on it in your next session!",
                            "warning",
                            meta={"letter": letter, "accuracy": round(acc, 1)}
                        ))

            # ── Certification hints ───────────────────────────────────────────
            for level_name, cfg in _CERT_HINTS.items():
                qualifying = sum(
                    1 for ltr in cfg["letters"]
                    if (s := per_letter.get(ltr)) and
                    s["attempts"] >= cfg["min_att"] and
                    100 * s["correct"] / s["attempts"] >= cfg["min_acc"]
                )
                total_req = len(cfg["letters"])
                if qualifying == total_req:
                    new_notifs.append(self._build(
                        "NEW_CERT_READY",
                        f"🏆 {level_name} certification within reach!",
                        f"You meet the criteria for '{level_name}'. Go to Certifications to claim it!",
                        "success",
                        meta={"level": level_name}
                    ))

        # Deduplicate types: skip if same type already unread
        existing_unread_types = {
            n["type"] for n in self._store.get(student_id, []) if not n.get("read")
        }
        added = []
        for n in new_notifs:
            if n["type"] not in existing_unread_types:
                self._store.setdefault(student_id, []).append(n)
                added.append(n)
                existing_unread_types.add(n["type"])

        if added:
            self._save()
        return added

    def mark_read(self, student_id: str, notification_id: str) -> bool:
        """Mark a single notification as read. Returns True if found."""
        student_id = student_id.strip()
        for n in self._store.get(student_id, []):
            if n["id"] == notification_id:
                n["read"] = True
                self._save()
                return True
        return False

    def mark_all_read(self, student_id: str) -> int:
        """Mark all notifications as read. Returns count marked."""
        student_id = student_id.strip()
        count = 0
        for n in self._store.get(student_id, []):
            if not n.get("read"):
                n["read"] = True
                count += 1
        if count:
            self._save()
        return count

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build(
        ntype: str, title: str, body: str, severity: str,
        meta: Optional[dict] = None
    ) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "type": ntype,
            "title": title,
            "body": body,
            "severity": severity,   # info | warning | success
            "read": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "meta": meta or {},
        }

    def _load(self):
        NOTIF_FILE.parent.mkdir(parents=True, exist_ok=True)
        if NOTIF_FILE.exists():
            try:
                with open(NOTIF_FILE, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._store = {}

    def _save(self):
        with open(NOTIF_FILE, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2)


# ── Singleton ─────────────────────────────────────────────────────────────────
_notif_service: Optional["NotificationService"] = None


def get_notification_service() -> NotificationService:
    global _notif_service
    if _notif_service is None:
        from app.services.progress_service import get_progress_service
        _notif_service = NotificationService(get_progress_service())
    return _notif_service
