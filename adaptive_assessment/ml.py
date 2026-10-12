"""Future ML contracts: deliberately no fabricated training or fitted models."""

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .domain import ItemParameters


@dataclass(frozen=True)
class TrainingInteraction:
    learner_key: str
    item_id: str
    item_version: str
    skill: str
    timestamp: str
    correct: bool
    mode: str
    used_hint: bool
    policy_version: str
    selection_probability: float


@dataclass(frozen=True)
class ModelArtifact:
    version: str
    dataset_version: str
    code_revision: str
    seed: int
    metrics: Mapping[str, float]
    parameters_uri: str


class ItemCalibrator(Protocol):
    def fit(
        self, interactions: Sequence[TrainingInteraction],
        anchor_parameters: Mapping[str, ItemParameters],
    ) -> ModelArtifact:
        """Fit item parameters on an anchored scale; validate before promotion."""
        ...


class LearnerModel(Protocol):
    def predict_success(
        self, history: Sequence[TrainingInteraction], skill: str, item_id: str,
    ) -> float:
        """Begin with a simple baseline; compare knowledge tracing only with data."""
        ...


class PracticePolicyTrainer(Protocol):
    def fit(self, interactions: Sequence[TrainingInteraction]) -> ModelArtifact:
        """Requires a defined learning outcome and logs with exploration support."""
        ...
