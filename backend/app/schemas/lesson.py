from pydantic import BaseModel


class LessonSummary(BaseModel):
    """What GET /lessons returns - one entry per lesson, lightweight."""
    id: int
    sign: str


class LessonDetail(BaseModel):
    """What GET /lessons/{id} returns - full lesson content."""
    id: int
    sign: str
    description: str
    meaning: str
    image: str
    difficulty: str