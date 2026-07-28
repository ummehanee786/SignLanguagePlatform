from typing import Optional, List

from pydantic import BaseModel


class GestureWiseStat(BaseModel):
    alphabet: str
    attempts: int
    correct: int
    accuracy_percentage: float


class ImprovementStat(BaseModel):
    earlier_half_accuracy: float
    later_half_accuracy: float
    change_percentage_points: float
    earlier_half_attempts: int
    later_half_attempts: int


class ReportResponse(BaseModel):
    """What GET /reports/{student_id} returns - the Task 2 report."""
    student_id: str
    generated_at: str
    total_assessment_attempts: int
    correct_attempts: int
    incorrect_attempts: int
    overall_assessment_score: float
    gesture_wise_performance: List[GestureWiseStat]
    most_difficult_gestures: List[GestureWiseStat]
    average_confidence: float
    average_response_time_seconds: float
    improvement_across_attempts: Optional[ImprovementStat] = None