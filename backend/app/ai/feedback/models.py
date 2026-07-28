"""
models.py  (app/ai/feedback/)

Data contracts for the Gesture Feedback Engine's output.

  LandmarkDeviation  – one detected deviation from the expected hand shape.
  DetailedFeedback   – the full output of GestureFeedbackEngine.evaluate():
                       a list of deviations + pre-rendered correction messages
                       plus the top-level message/tip/severity used by the
                       existing FeedbackObject.

Both are plain dataclasses (not Pydantic) for the same reason as
AssessmentRecord: this is the AI module's internal contract; the API
layer converts these into whatever schema shape it needs.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class LandmarkDeviation:
    """
    One detected deviation from the expected hand shape.

    rule_name   : machine-readable identifier, e.g. "finger_extension"
    description : plain-English explanation of what is wrong,
                  e.g. "Index finger appears bent; try extending it fully."
    severity    : "error" (definitely wrong) | "warning" (likely wrong)
    """
    rule_name: str
    description: str
    severity: str = "warning"   # "error" | "warning"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetailedFeedback:
    """
    Full output of GestureFeedbackEngine.evaluate().

    overall_message     : top-level human-readable message (mirrors
                          FeedbackObject.message)
    severity            : "success" | "info" | "warning" (mirrors
                          FeedbackObject.severity)
    tip                 : optional extra hint (mirrors FeedbackObject.tip)
    deviations          : list of landmark-level deviations found
    correction_messages : pre-rendered plain-English list derived from
                          deviations, for direct use in API responses
                          without the caller needing to iterate deviations.
    """
    overall_message: str
    severity: str = "warning"
    tip: Optional[str] = None
    deviations: list = field(default_factory=list)           # list[LandmarkDeviation]
    correction_messages: list = field(default_factory=list)  # list[str]

    def to_dict(self) -> dict:
        return {
            "overall_message": self.overall_message,
            "severity": self.severity,
            "tip": self.tip,
            "deviations": [d.to_dict() if hasattr(d, "to_dict") else d for d in self.deviations],
            "correction_messages": self.correction_messages,
        }
