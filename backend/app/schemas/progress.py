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