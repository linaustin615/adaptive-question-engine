"""Generation and validation boundaries, with an offline template substitute."""

from typing import Protocol, Tuple

from .domain import GenerationRequest, Item


class QuestionGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> Item:
        ...


class LLMQuestionGenerator:
    def generate(self, request: GenerationRequest) -> Item:
        """Retrieve approved sources, call a model, and parse a strict item schema."""
        raise NotImplementedError(
            "Connect a provider here; persist prompt/model/source versions and cost. "
            "Return a draft with no calibrated IRT parameters."
        )


class TemplateGenerator:
    def generate(self, request: GenerationRequest) -> Item:
        if request.skill != "algebra":
            raise ValueError("the offline template supports algebra only")
        return Item(
            item_id=request.request_id, version="1", skill=request.skill,
            stem="If x + 3 = 8, what is x?", options=("3", "4", "5", "6"),
            answer_index=2, explanation="Subtract 3 from both sides: x = 5.",
            source="offline-template-v1",
        )


def structural_issues(item: Item) -> Tuple[str, ...]:
    """A format gate cannot establish correctness, fairness, or calibration."""
    issues = []
    if not all(value.strip() for value in (item.item_id, item.version, item.skill, item.stem)):
        issues.append("missing_required_text")
    options = [option.strip().casefold() for option in item.options]
    if len(options) != 4 or any(not option for option in options):
        issues.append("invalid_options")
    if len(set(options)) != len(options):
        issues.append("duplicate_options")
    if type(item.answer_index) is not int or not 0 <= item.answer_index < len(options):
        issues.append("invalid_key")
    if not item.explanation.strip():
        issues.append("missing_explanation")
    return tuple(issues)
