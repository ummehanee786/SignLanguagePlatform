from typing import Optional

from pydantic import BaseModel


class AttemptRecord(BaseModel):
    student_id: str
    session_id: Optional[str] = None
    alphabet_practiced: str
    predicted_alphabet: Optional[str] = None
    correct: bool
    confidence: float
    inference_time: float
    timestamp: str


class AlphabetMistakeStat(BaseModel):
    alphabet: str
    mistake_count: int
    attempts: int


class AlphabetAccuracyStat(BaseModel):
    alphabet: str
    accuracy_percentage: float
    attempts: int


class LearnerProfile(BaseModel):
    student_id: str
    total_practice_sessions: int
    total_attempts: int
    alphabet_mastery: dict[str, float]  # alphabet name -> mastery (0.0 to 1.0)
    consecutive_correct: dict[str, int]
    consecutive_incorrect: dict[str, int]
    average_confidence: dict[str, float]
    last_practice_time: dict[str, str]

class RecommendationEntry(BaseModel):
    alphabet: str
    reason: str

class DashboardResponse(BaseModel):
    """What GET /progress/{student_id}/dashboard returns."""
    student_id: str
    total_attempts: int
    accuracy_percentage: float
    most_mistaken_alphabets: list[AlphabetMistakeStat]
    strongest_alphabets: list[AlphabetAccuracyStat]
    weakest_alphabets: list[AlphabetAccuracyStat]
    daily_practice_streak: int
    average_confidence: float
    recent_practice_history: list[AttemptRecord]
    learner_profile: Optional[LearnerProfile] = None
    recommendations: list[RecommendationEntry] = []
