from collections import defaultdict
from typing import List, Optional
from datetime import datetime

from app.services.progress_service import ProgressService, get_progress_service
from app.schemas.error_analysis import (
    ConfusedGesturePair,
    LowConfidenceAlphabet,
    RepeatedMistake,
    AlphabetPerformanceTrend,
    ErrorAnalysisInsight,
)


class ErrorAnalysisService:
    def __init__(self, progress_service: ProgressService):
        self._progress_service = progress_service

    def analyze_student_errors(self, student_id: str) -> ErrorAnalysisInsight:
        student_id = student_id.strip()
        # Retrieve all attempts, reverse them to be in chronological order (oldest first)
        attempts = list(reversed(self._progress_service.get_history(student_id, limit=None)))

        # 1. Most frequently confused gesture pairs (expected -> predicted)
        # Only check incorrect attempts where both expected and predicted are present and expected != predicted
        confusion_counts = defaultdict(int)
        for a in attempts:
            if not a["correct"]:
                expected = a["alphabet_practiced"]
                predicted = a.get("predicted_alphabet")
                if expected and predicted and expected.upper() != predicted.upper():
                    confusion_counts[(expected.upper(), predicted.upper())] += 1

        most_confused_pairs = sorted(
            [
                ConfusedGesturePair(expected=pair[0], predicted=pair[1], count=count)
                for pair, count in confusion_counts.items()
            ],
            key=lambda x: x.count,
            reverse=True,
        )[:5]

        # 2. Alphabets with consistently low confidence (average confidence <= 0.80)
        confidence_by_alphabet = defaultdict(list)
        for a in attempts:
            alphabet = a["alphabet_practiced"].upper()
            confidence_by_alphabet[alphabet].append(a["confidence"])

        low_confidence_list = []
        for alphabet, confidences in confidence_by_alphabet.items():
            if len(confidences) >= 3:
                avg_conf = sum(confidences) / len(confidences)
                if avg_conf < 0.80:
                    low_confidence_list.append(
                        LowConfidenceAlphabet(
                            alphabet=alphabet,
                            average_confidence=round(avg_conf, 4),
                            attempts=len(confidences),
                        )
                    )
        low_confidence_alphabets = sorted(low_confidence_list, key=lambda x: x.average_confidence)

        # 3. Repeated mistakes across multiple sessions
        # For each alphabet, gather the session IDs of incorrect attempts
        incorrect_sessions_by_alphabet = defaultdict(set)
        incorrect_counts_by_alphabet = defaultdict(int)
        for a in attempts:
            if not a["correct"]:
                alphabet = a["alphabet_practiced"].upper()
                session_id = a.get("session_id")
                if session_id:
                    incorrect_sessions_by_alphabet[alphabet].add(session_id)
                    incorrect_counts_by_alphabet[alphabet] += 1

        repeated_mistakes = []
        for alphabet, sessions in incorrect_sessions_by_alphabet.items():
            if len(sessions) >= 2:
                repeated_mistakes.append(
                    RepeatedMistake(
                        alphabet=alphabet,
                        mistake_count=incorrect_counts_by_alphabet[alphabet],
                        sessions=sorted(list(sessions)),
                    )
                )
        repeated_mistakes.sort(key=lambda x: x.mistake_count, reverse=True)

        # 4. Gestures requiring immediate revision
        # Rule: lifetime accuracy < 60% with >= 3 attempts, OR last 2 consecutive attempts incorrect
        revision_required_gestures = []
        # Group attempts by alphabet in chronological order
        attempts_by_alphabet = defaultdict(list)
        for a in attempts:
            alphabet = a["alphabet_practiced"].upper()
            attempts_by_alphabet[alphabet].append(a)

        for alphabet, letter_attempts in attempts_by_alphabet.items():
            # Check 1: Lifetime accuracy < 60% with >= 3 attempts
            correct_count = sum(1 for a in letter_attempts if a["correct"])
            total_attempts = len(letter_attempts)
            accuracy = correct_count / total_attempts if total_attempts > 0 else 1.0

            # Check 2: Last 2 consecutive attempts incorrect
            last_two_incorrect = False
            if len(letter_attempts) >= 2:
                last_two_incorrect = not letter_attempts[-1]["correct"] and not letter_attempts[-2]["correct"]

            if (total_attempts >= 3 and accuracy < 0.60) or last_two_incorrect:
                revision_required_gestures.append(alphabet)

        revision_required_gestures.sort()

        # 5. Improvement or decline in performance for each alphabet (requires >= 4 attempts)
        performance_trends = []
        for alphabet, letter_attempts in attempts_by_alphabet.items():
            if len(letter_attempts) >= 4:
                mid = len(letter_attempts) // 2
                earlier = letter_attempts[:mid]
                later = letter_attempts[mid:]

                earlier_accuracy = sum(1 for a in earlier if a["correct"]) / len(earlier)
                later_accuracy = sum(1 for a in later if a["correct"]) / len(later)

                if later_accuracy > earlier_accuracy:
                    trend = "improving"
                elif later_accuracy < earlier_accuracy:
                    trend = "declining"
                else:
                    trend = "stable"

                performance_trends.append(
                    AlphabetPerformanceTrend(
                        alphabet=alphabet,
                        trend=trend,
                        earlier_accuracy=round(100 * earlier_accuracy, 2),
                        later_accuracy=round(100 * later_accuracy, 2),
                    )
                )
        performance_trends.sort(key=lambda x: x.alphabet)

        return ErrorAnalysisInsight(
            student_id=student_id,
            most_confused_pairs=most_confused_pairs,
            low_confidence_alphabets=low_confidence_alphabets,
            repeated_mistakes=repeated_mistakes,
            revision_required_gestures=revision_required_gestures,
            performance_trends=performance_trends,
        )


_error_analysis_service: Optional[ErrorAnalysisService] = None


def get_error_analysis_service() -> ErrorAnalysisService:
    global _error_analysis_service
    if _error_analysis_service is None:
        _error_analysis_service = ErrorAnalysisService(get_progress_service())
    return _error_analysis_service
