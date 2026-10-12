"""A single-skill CAT session: choose, answer, update, and check stopping."""

from dataclasses import asdict, dataclass
from math import isfinite
from random import Random
from typing import Optional, Sequence

from .audit import MemoryAudit
from .domain import Item, Response
from .irt import ability_posterior
from .selection import SelectionPolicy


@dataclass(frozen=True)
class StopRule:
    min_items: int = 4
    max_items: int = 8
    target_sd: float = 0.5

    def __post_init__(self):
        if not 1 <= self.min_items <= self.max_items:
            raise ValueError("require 1 <= min_items <= max_items")
        if not isfinite(self.target_sd) or self.target_sd <= 0:
            raise ValueError("target_sd must be finite and positive")


class AssessmentSession:
    POLICY_VERSION = "max-information-v1"

    def __init__(
        self, session_id: str, skill: str, bank: Sequence[Item],
        audit: MemoryAudit, stop_rule: StopRule = StopRule(), *,
        allow_synthetic: bool = False,
        selection: SelectionPolicy = SelectionPolicy(), selection_seed: int = 0,
    ):
        if len({item.item_id for item in bank}) != len(bank):
            raise ValueError("pin exactly one version of each item in a session bank")
        self.session_id = session_id
        self.skill = skill
        self._bank = tuple(bank)
        self.audit = audit
        self.stop_rule = stop_rule
        self.allow_synthetic = allow_synthetic
        self.selection = selection
        self._rng = Random(selection_seed)
        self._responses = []
        self._pending = None
        self._posterior = ability_posterior([], allow_synthetic=allow_synthetic)
        self.audit.emit(
            "session_started", session_id=session_id, skill=skill,
            policy=selection.version, selection_config=asdict(selection),
            selection_seed=selection_seed, stop_rule=asdict(stop_rule),
            allow_synthetic=allow_synthetic,
        )

    @property
    def estimate(self):
        return self._posterior.estimate

    @property
    def responses(self):
        return tuple(self._responses)

    def _base_candidates(self):
        seen = {r.item.item_id for r in self._responses}
        return [item for item in self._bank if (
            item.skill == self.skill and item.item_id not in seen
            and item.can_score(self.allow_synthetic)
        )]

    def _candidates(self):
        return self.selection.eligible(self._base_candidates(), self._responses)

    def stop_reason(self) -> Optional[str]:
        if self._pending is not None:
            return None
        estimate = self.estimate
        if estimate.response_count >= self.stop_rule.max_items:
            return "maximum_length"
        if (estimate.response_count >= self.stop_rule.min_items
                and estimate.posterior_sd <= self.stop_rule.target_sd
                and estimate.edge_mass < 0.01):
            return "precision_target"
        if not self._candidates():
            return "family_limit_reached" if self._base_candidates() else "bank_exhausted"
        return None

    def next_item(self) -> Item:
        if self._pending is not None:
            return self._pending  #a retry must not issue another item
        reason = self.stop_reason()
        if reason:
            raise StopIteration(reason)
        candidates = self._candidates()
        before = self.estimate
        decision = self.selection.choose(candidates, self._posterior, self._rng)
        chosen = decision.item
        self.audit.emit(
            "item_selected", session_id=self.session_id,
            item_id=chosen.item_id, item_version=chosen.version,
            calibration=asdict(chosen.parameters), estimate=asdict(before),
            candidates=[item.item_id for item in candidates],
            policy=self.selection.version, selection_config=asdict(self.selection),
            reason=self.selection.criterion, family_id=chosen.family_id,
            chosen_probability=decision.probability,
            candidate_probabilities=dict(decision.candidate_probabilities),
            candidate_utilities=dict(decision.candidate_utilities),
        )
        self._pending = chosen
        return chosen

    def submit(self, item_id: str, answer_index: int):
        item = self._pending
        if item is None or item.item_id != item_id:
            raise ValueError("answer must match the pending item")
        if type(answer_index) is not int or not 0 <= answer_index < len(item.options):
            raise ValueError("answer_index is outside the available options")
        correct = answer_index == item.answer_index  #grade against the stored key
        response = Response(item, correct)
        posterior = ability_posterior(
            self._responses + [response], allow_synthetic=self.allow_synthetic,
        )
        after = posterior.estimate
        self.audit.emit(
            "answer_scored", session_id=self.session_id, item_id=item.item_id,
            item_version=item.version, answer_index=answer_index,
            correct=correct, estimate=asdict(after),
        )
        self._responses.append(response)
        self._posterior = posterior
        self._pending = None
        return after
