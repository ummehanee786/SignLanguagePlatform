from datetime import datetime, timezone
from typing import List, Optional
from collections import defaultdict

from app.schemas.progress import RecommendationEntry
from app.services.progress_service import ProgressService, get_progress_service
from app.services.error_analysis_service import ErrorAnalysisService, get_error_analysis_service
from app.content.lesson_service import LessonService


class RecommendationService:
    def __init__(
        self,
        progress_service: Optional[ProgressService] = None,
        error_analysis_service: Optional[ErrorAnalysisService] = None,
        lesson_service: Optional[LessonService] = None
    ):
        self._progress_service = progress_service or get_progress_service()
        self._error_analysis_service = error_analysis_service or get_error_analysis_service()
        self._lesson_service = lesson_service or LessonService()

    def get_recommendations(self, student_id: str, limit: int = 5) -> List[RecommendationEntry]:
        student_id = student_id.strip()
        profile = self._progress_service.get_learner_profile(student_id)
        insights = self._error_analysis_service.analyze_student_errors(student_id)

        # Map trends
        trend_dict = {t.alphabet.upper(): t.trend for t in insights.performance_trends}

        # Map top confusion per alphabet
        confusion_by_expected = {}
        for pair in insights.most_confused_pairs:
            exp = pair.expected.upper()
            pred = pair.predicted.upper()
            if exp not in confusion_by_expected:
                confusion_by_expected[exp] = (pred, pair.count)

        # Get list of all available alphabets in the system
        all_lessons = self._lesson_service.get_all_lessons()
        alphabets = [l["sign"].upper() for l in all_lessons]

        # We will calculate candidate recommendations for each alphabet
        # A candidate is { alphabet: score, reason }
        candidates = {}

        # Profile fields
        mastery = profile.get("alphabet_mastery", {})
        consec_correct = profile.get("consecutive_correct", {})
        avg_confidence = profile.get("average_confidence", {})
        last_time_str = profile.get("last_practice_time", {})

        now_dt = datetime.now(timezone.utc)

        for letter in alphabets:
            letter = letter.upper()
            candidates[letter] = []

            # 1. Check: Never practiced
            has_practiced = (letter in mastery) and (profile.get("total_attempts", 0) > 0)
            # Find in attempts list as fallback
            if not has_practiced:
                # Assign baseline starting priority based on curriculum order
                # Earlier alphabets get slightly higher priority to guide the student starting out
                index_val = alphabets.index(letter) if letter in alphabets else 26
                score = 25.0 - (index_val * 0.5)
                candidates[letter].append({
                    "score": score,
                    "reason": "Not practiced recently"
                })
                continue

            # Practiced letter stats
            m_val = mastery.get(letter, 0.0)
            c_val = avg_confidence.get(letter, 0.0)
            consec_corr = consec_correct.get(letter, 0)

            # Hour delta since last practiced
            hours_since = 0.0
            last_t_str = last_time_str.get(letter)
            if last_t_str:
                try:
                    last_dt = datetime.fromisoformat(last_t_str)
                    delta = now_dt - last_dt
                    hours_since = delta.total_seconds() / 3600.0
                except Exception:
                    pass

            # 2. Check: Low mastery level
            if m_val < 0.70:
                score = 20.0 + (1.0 - m_val) * 10
                candidates[letter].append({
                    "score": score,
                    "reason": "Low mastery level"
                })

            # 3. Check: Frequent confusion with another alphabet
            if letter in confusion_by_expected:
                pred, count = confusion_by_expected[letter]
                if count >= 2:
                    score = 19.0 + count
                    candidates[letter].append({
                        "score": score,
                        "reason": f"Frequent confusion with '{pred}'"
                    })

            # 4. Check: Low confidence despite correct predictions
            if consec_corr >= 1 and c_val < 0.80:
                score = 15.0 + (1.0 - c_val) * 10
                candidates[letter].append({
                    "score": score,
                    "reason": "Low confidence despite correct predictions"
                })

            # 5. Check: Not practiced recently
            if hours_since > 24.0:
                score = 12.0 + min(hours_since / 24.0, 10.0)
                candidates[letter].append({
                    "score": score,
                    "reason": "Not practiced recently"
                })

            # 6. Check: High improvement trend (ready to progress)
            # If this alphabet shows improvement trend and is mastered (mastery >= 0.8),
            # we recommend the *next* alphabet in the curriculum with the ready to progress reason.
            if trend_dict.get(letter) == "improving" and m_val >= 0.80:
                # Find the next alphabet in lessons list
                try:
                    idx = alphabets.index(letter)
                    if idx + 1 < len(alphabets):
                        next_letter = alphabets[idx + 1]
                        # Set suggestion for next letter
                        # Only add if the next letter is not already completed
                        next_mastery = mastery.get(next_letter, 0.0)
                        if next_mastery < 0.90:
                            if next_letter not in candidates:
                                candidates[next_letter] = []
                            candidates[next_letter].append({
                                "score": 18.5,
                                "reason": f"High improvement trend (ready to progress) on '{letter}'"
                            })
                except ValueError:
                    pass

            # Fallback spaced repetition if no other rules apply
            if not candidates[letter]:
                candidates[letter].append({
                    "score": 5.0 + m_val,
                    "reason": "Not practiced recently"
                })

        # Process priorities: for each alphabet, get the highest score candidate
        recommendation_list = []
        for letter in alphabets:
            letter_candidates = candidates.get(letter, [])
            if letter_candidates:
                best = max(letter_candidates, key=lambda x: x["score"])
                recommendation_list.append({
                    "alphabet": letter,
                    "score": best["score"],
                    "reason": best["reason"]
                })

        # Sort recommendations: highest score first
        recommendation_list.sort(key=lambda x: (x["score"], -alphabets.index(x["alphabet"])), reverse=True)

        # Convert to schemas
        result = []
        for r in recommendation_list[:limit]:
            result.append(RecommendationEntry(alphabet=r["alphabet"], reason=r["reason"]))

        return result


_recommendation_service: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    return _recommendation_service
