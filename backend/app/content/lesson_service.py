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
#
# Descriptions are for the STATIC alphabet only (matches what the
# current Random Forest model actually supports - see engine.py).
# J and Z are traditionally drawn as motion in ASL; they're included
# here with their static starting handshape since the practice module
# still needs something to show/compare against today, but flagged as
# lower priority for real-time assessment until a sequence model
# exists (see docs/sprint2_live_recognition_design.md) - the static
# snapshot is inherently a partial/ambiguous representation of a
# motion letter, consistent with what error_analysis.md already found
# for J specifically.
_LESSONS = [
    {"id": 1, "sign": "A", "description": "Closed fist with thumb resting on the side.", "meaning": "The letter A", "image": "assets/asl/A.jpg", "difficulty": "Beginner"},
    {"id": 2, "sign": "B", "description": "Flat hand, fingers together pointing up, thumb folded across the palm.", "meaning": "The letter B", "image": "assets/asl/B.jpg", "difficulty": "Beginner"},
    {"id": 3, "sign": "C", "description": "Hand curved into a C shape, as if holding a cup.", "meaning": "The letter C", "image": "assets/asl/C.jpg", "difficulty": "Beginner"},
    {"id": 4, "sign": "D", "description": "Index finger pointing up, other fingers curled and touching the thumb.", "meaning": "The letter D", "image": "assets/asl/D.jpg", "difficulty": "Beginner"},
    {"id": 5, "sign": "E", "description": "Fingers curled down, tips touching the thumb, forming a claw-like shape.", "meaning": "The letter E", "image": "assets/asl/E.jpg", "difficulty": "Beginner"},
    {"id": 6, "sign": "F", "description": "Thumb and index finger touching to form a circle, other three fingers extended.", "meaning": "The letter F", "image": "assets/asl/F.jpg", "difficulty": "Beginner"},
    {"id": 7, "sign": "G", "description": "Index finger and thumb extended horizontally, pointing sideways.", "meaning": "The letter G", "image": "assets/asl/G.jpg", "difficulty": "Intermediate"},
    {"id": 8, "sign": "H", "description": "Index and middle fingers extended together, pointing sideways.", "meaning": "The letter H", "image": "assets/asl/H.jpg", "difficulty": "Intermediate"},
    {"id": 9, "sign": "I", "description": "Fist with the pinky finger extended upward.", "meaning": "The letter I", "image": "assets/asl/I.jpg", "difficulty": "Beginner"},
    {"id": 10, "sign": "J", "description": "Pinky extended, traced in the shape of the letter J (motion-based sign).", "meaning": "The letter J", "image": "assets/asl/J.jpg", "difficulty": "Advanced"},
    {"id": 11, "sign": "K", "description": "Index and middle fingers extended in a V, thumb touching the middle finger.", "meaning": "The letter K", "image": "assets/asl/K.jpg", "difficulty": "Intermediate"},
    {"id": 12, "sign": "L", "description": "Thumb and index finger extended at a right angle, forming an L.", "meaning": "The letter L", "image": "assets/asl/L.jpg", "difficulty": "Beginner"},
    {"id": 13, "sign": "M", "description": "Fist with the thumb tucked under three fingers.", "meaning": "The letter M", "image": "assets/asl/M.jpg", "difficulty": "Intermediate"},
    {"id": 14, "sign": "N", "description": "Fist with the thumb tucked under two fingers.", "meaning": "The letter N", "image": "assets/asl/N.jpg", "difficulty": "Intermediate"},
    {"id": 15, "sign": "O", "description": "Fingers and thumb curved to form an O shape.", "meaning": "The letter O", "image": "assets/asl/O.jpg", "difficulty": "Beginner"},
    {"id": 16, "sign": "P", "description": "Like K, but pointed downward from the wrist.", "meaning": "The letter P", "image": "assets/asl/P.jpg", "difficulty": "Advanced"},
    {"id": 17, "sign": "Q", "description": "Like G, but pointed downward from the wrist.", "meaning": "The letter Q", "image": "assets/asl/Q.jpg", "difficulty": "Advanced"},
    {"id": 18, "sign": "R", "description": "Index and middle fingers crossed.", "meaning": "The letter R", "image": "assets/asl/R.jpg", "difficulty": "Intermediate"},
    {"id": 19, "sign": "S", "description": "Closed fist with the thumb across the front of the fingers.", "meaning": "The letter S", "image": "assets/asl/S.jpg", "difficulty": "Beginner"},
    {"id": 20, "sign": "T", "description": "Fist with the thumb tucked between the index and middle fingers.", "meaning": "The letter T", "image": "assets/asl/T.jpg", "difficulty": "Intermediate"},
    {"id": 21, "sign": "U", "description": "Index and middle fingers extended together, pointing up.", "meaning": "The letter U", "image": "assets/asl/U.jpg", "difficulty": "Beginner"},
    {"id": 22, "sign": "V", "description": "Index and middle fingers extended apart, forming a V.", "meaning": "The letter V", "image": "assets/asl/V.jpg", "difficulty": "Beginner"},
    {"id": 23, "sign": "W", "description": "Index, middle, and ring fingers extended apart.", "meaning": "The letter W", "image": "assets/asl/W.jpg", "difficulty": "Beginner"},
    {"id": 24, "sign": "X", "description": "Index finger bent into a hook shape, other fingers curled.", "meaning": "The letter X", "image": "assets/asl/X.jpg", "difficulty": "Intermediate"},
    {"id": 25, "sign": "Y", "description": "Thumb and pinky extended, other fingers curled.", "meaning": "The letter Y", "image": "assets/asl/Y.jpg", "difficulty": "Beginner"},
    {"id": 26, "sign": "Z", "description": "Index finger traces the shape of the letter Z in the air (motion-based sign).", "meaning": "The letter Z", "image": "assets/asl/Z.jpg", "difficulty": "Advanced"},
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

    def get_lesson_by_sign(self, sign: str) -> Optional[dict]:
        """Full lesson details looked up by letter (e.g. "A") instead of id."""
        sign = sign.upper()
        for lesson in _LESSONS:
            if lesson["sign"] == sign:
                return lesson
        return None

    def get_next_lesson_id(self, current_lesson_id: int) -> Optional[int]:
        """
        Returns the next lesson's id in sequence (A -> B -> C -> ...), or
        None if `current_lesson_id` was the last one. Used by the
        practice module's "auto-next" behavior.
        """
        ids = sorted(lesson["id"] for lesson in _LESSONS)
        try:
            position = ids.index(current_lesson_id)
        except ValueError:
            return None
        if position + 1 >= len(ids):
            return None
        return ids[position + 1]