"""Transparent practice heuristics; these are not a trained learner model."""

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .audit import MemoryAudit
from .domain import Feedback, FeedbackKind


@dataclass
class PracticeProfile:
    counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    difficulty_preferences: Dict[str, float] = field(default_factory=dict)

    def record_answer(self, skill: str, correct: bool):
        if type(correct) is not bool:
            raise ValueError("correct must be a boolean")
        right, total = self.counts.get(skill, (0, 0))
        self.counts[skill] = (right + int(correct), total + 1)

    def practice_priority(self, skill: str) -> float:
        right, total = self.counts.get(skill, (0, 0))
        return 1.0 - (right + 1) / (total + 2)  #smooth sparse counts toward 50 percent


def apply_feedback(profile: PracticeProfile, feedback: Feedback, audit: MemoryAudit):
    audit.emit(
        "feedback_received", item_id=feedback.item_id,
        item_version=feedback.item_version, skill=feedback.skill,
        feedback_kind=feedback.kind.value,
    )
    if feedback.kind in {FeedbackKind.UNCLEAR, FeedbackKind.INCORRECT_KEY}:
        audit.emit(
            "review_requested", item_id=feedback.item_id,
            item_version=feedback.item_version, issue_code=feedback.kind.value,
        )
        return "review_requested"
    previous = profile.difficulty_preferences.get(feedback.skill, 0.0)
    step = 0.25 if feedback.kind == FeedbackKind.TOO_EASY else -0.25
    profile.difficulty_preferences[feedback.skill] = max(-2.0, min(2.0, previous + step))
    audit.emit(
        "practice_preference_updated", skill=feedback.skill,
        requested_difficulty=profile.difficulty_preferences[feedback.skill],
    )
    return "practice_preference_updated"
