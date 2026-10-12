"""Optional measurement policies with explicit probabilities and family limits."""

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Optional, Sequence, Tuple

from .domain import Item, Response
from .irt import AbilityPosterior, expected_information_gain, information


@dataclass(frozen=True)
class SelectionDecision:
    item: Item
    probability: float
    candidate_probabilities: Tuple[Tuple[str, float], ...]
    candidate_utilities: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class SelectionPolicy:
    criterion: str = "fisher"
    exploration: float = 0.0
    max_per_family: Optional[int] = None

    def __post_init__(self):
        if self.criterion not in {"fisher", "information_gain", "random"}:
            raise ValueError("unknown selection criterion")
        if not isfinite(self.exploration) or not 0 <= self.exploration <= 1:
            raise ValueError("exploration must be between zero and one")
        if self.max_per_family is not None and (
            type(self.max_per_family) is not int or self.max_per_family < 1
        ):
            raise ValueError("family cap must be a positive integer or None")

    @property
    def version(self):
        if self == SelectionPolicy():
            return "max-information-v1"
        return "experimental-selection-v1"  #persist the complete config with this version

    def eligible(self, candidates: Sequence[Item], responses: Sequence[Response]):
        if self.max_per_family is None:
            return list(candidates)
        if any(not item.family_id for item in candidates) or any(not r.item.family_id for r in responses):
            raise ValueError("family control requires family metadata on every eligible item")
        counts = Counter(response.item.family_id for response in responses)
        return [item for item in candidates if counts[item.family_id] < self.max_per_family]

    def choose(self, candidates: Sequence[Item], posterior: AbilityPosterior, rng: Random):
        ordered = sorted(candidates, key=lambda item: item.item_id)
        if not ordered:
            raise ValueError("selection requires at least one eligible item")
        if self.criterion == "random":
            utilities = [0.0] * len(ordered)
            probabilities = [1.0 / len(ordered)] * len(ordered)
        else:
            utilities = [
                expected_information_gain(posterior, item.parameters)
                if self.criterion == "information_gain"
                else information(posterior.estimate.mean, item.parameters)
                for item in ordered
            ]
            best = max(range(len(ordered)), key=utilities.__getitem__)
            probabilities = [self.exploration / len(ordered)] * len(ordered)
            probabilities[best] += 1.0 - self.exploration
        draw, cumulative, selected = rng.random(), 0.0, len(ordered) - 1
        for index, p in enumerate(probabilities):
            cumulative += p
            if draw < cumulative:
                selected = index
                break
        return SelectionDecision(
            ordered[selected], probabilities[selected],
            tuple((item.item_id, p) for item, p in zip(ordered, probabilities)),
            tuple((item.item_id, score) for item, score in zip(ordered, utilities)),
        )
