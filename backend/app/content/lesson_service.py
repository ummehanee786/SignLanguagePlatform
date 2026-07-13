"""
lesson_service.py

Manages the learning content - the actual lesson data (what a sign
looks like, what it means, how hard it is). This has nothing to do
with AI/recognition - it's pure content management, which is why it
lives in its own `content/` folder rather than `services/` or `ai/`.
"""

from typing import Optional


# In-memory lesson data for now. Later this could move to a database
# or a JSON/YAML content file without changing the interface below.
_LESSONS = [
    {
        "id": 1,
        "sign": "A",
        "description": "Closed fist with thumb resting on the side.",
        "meaning": "The letter A",
        "image": "assets/asl/A.jpg",
        "difficulty": "Beginner",
    },
    {
        "id": 2,
        "sign": "B",
        "description": "Flat hand, fingers together pointing up, thumb folded across the palm.",
        "meaning": "The letter B",
        "image": "assets/asl/B.jpg",
        "difficulty": "Beginner",
    },
    {
        "id": 3,
        "sign": "C",
        "description": "Hand curved into a C shape, as if holding a cup.",
        "meaning": "The letter C",
        "image": "assets/asl/C.jpg",
        "difficulty": "Beginner",
    },
    {
        "id": 4,
        "sign": "D",
        "description": "Index finger pointing up, other fingers curled and touching the thumb.",
        "meaning": "The letter D",
        "image": "assets/asl/D.jpg",
        "difficulty": "Beginner",
    },
    {
        "id": 5,
        "sign": "E",
        "description": "Fingers curled down, tips touching the thumb, forming a claw-like shape.",
        "meaning": "The letter E",
        "image": "assets/asl/E.jpg",
        "difficulty": "Beginner",
    },
]


class LessonService:
    def get_all_lessons(self) -> list[dict]:
        """Lightweight summary version (id + sign) - for GET /lessons."""
        return [{"id": lesson["id"], "sign": lesson["sign"]} for lesson in _LESSONS]

    def get_lesson_by_id(self, lesson_id: int) -> Optional[dict]:
        """Full lesson details, or None - for GET /lessons/{id}."""
        for lesson in _LESSONS:
            if lesson["id"] == lesson_id:
                return lesson
        return None