"""Shared records keep generation, delivery, feedback, and scoring separate."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Optional, Tuple


class ItemStatus(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    CALIBRATED = "calibrated"
    RETIRED = "retired"


@dataclass(frozen=True)
class ItemParameters:
    discrimination: float
    difficulty: float
    version: str
    evidence: str  #use "synthetic" for fixtures and "empirical" for fitted parameters

    def __post_init__(self):
        if not isfinite(self.discrimination) or self.discrimination <= 0:
            raise ValueError("discrimination must be finite and positive")
        if not isfinite(self.difficulty) or not self.version:
            raise ValueError("difficulty must be finite and a version is required")
        if self.evidence not in {"synthetic", "empirical"}:
            raise ValueError("unknown calibration evidence")


@dataclass(frozen=True)
class Item:
    item_id: str
    version: str
    skill: str
    stem: str
    options: Tuple[str, ...]
    answer_index: int
    explanation: str
    status: ItemStatus = ItemStatus.DRAFT
    parameters: Optional[ItemParameters] = None
    source: str = "authored-demo"
    family_id: Optional[str] = None  #group shared passages or closely related templates

    def can_score(self, allow_synthetic: bool = False) -> bool:
        return (
            self.status == ItemStatus.CALIBRATED
            and self.parameters is not None
            and (self.parameters.evidence == "empirical" or allow_synthetic)
        )


@dataclass(frozen=True)
class Response:
    item: Item
    correct: bool

    def __post_init__(self):
        if type(self.correct) is not bool:
            raise ValueError("correct must be a boolean")


@dataclass(frozen=True)
class AbilityEstimate:
    mean: float
    posterior_sd: float
    interval_95: Tuple[float, float]
    response_count: int
    edge_mass: float  #flag estimates pressing against the numerical grid


class FeedbackKind(str, Enum):
    TOO_EASY = "too_easy"
    TOO_HARD = "too_hard"
    UNCLEAR = "unclear"
    INCORRECT_KEY = "incorrect_key"


@dataclass(frozen=True)
class Feedback:
    item_id: str
    item_version: str
    skill: str
    kind: FeedbackKind


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    skill: str
    target_difficulty: float
    approved_issue_codes: Tuple[str, ...] = ()
    prompt_version: str = "demo-v1"
