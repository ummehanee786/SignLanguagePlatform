from typing import List, Optional
from pydantic import BaseModel


class ConfusedGesturePair(BaseModel):
    expected: str
    predicted: str
    count: int


class LowConfidenceAlphabet(BaseModel):
    alphabet: str
    average_confidence: float
    attempts: int


class RepeatedMistake(BaseModel):
    alphabet: str
    mistake_count: int
    sessions: List[str]


class AlphabetPerformanceTrend(BaseModel):
    alphabet: str
    trend: str  # "improving" | "declining" | "stable"
    earlier_accuracy: float
    later_accuracy: float


class ErrorAnalysisInsight(BaseModel):
    student_id: str
    most_confused_pairs: List[ConfusedGesturePair]
    low_confidence_alphabets: List[LowConfidenceAlphabet]
    repeated_mistakes: List[RepeatedMistake]
    revision_required_gestures: List[str]
    performance_trends: List[AlphabetPerformanceTrend]
