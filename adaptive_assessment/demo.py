"""Run with python3 -m adaptive_assessment.demo from this project folder."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from random import Random

from .adaptive import AssessmentSession
from .audit import MemoryAudit
from .domain import Feedback, FeedbackKind, GenerationRequest, Item, ItemParameters, ItemStatus
from .generation import TemplateGenerator, structural_issues
from .irt import probability
from .personalization import PracticeProfile, apply_feedback


def synthetic_bank():
    """Invented parameters do not describe measured difficulty of these stems."""
    return tuple(
        Item(
            item_id=f"synthetic-{i:02d}", version="1", skill="algebra",
            stem=f"If x + {i + 1} = {2 * i + 5}, what is x?",
            options=tuple(str(i + offset) for offset in (2, 3, 4, 5)),
            answer_index=2, explanation=f"Subtract {i + 1}: x = {i + 4}.",
            status=ItemStatus.CALIBRATED,
            parameters=ItemParameters(1.2, -2.2 + i * 0.4, "synthetic-v1", "synthetic"),
        )
        for i in range(12)
    )


def run(seed=7, audit_path=None):
    rng = Random(seed)
    audit = MemoryAudit()
    profile = PracticeProfile()
    session = AssessmentSession(
        "synthetic-session", "algebra", synthetic_bank(), audit, allow_synthetic=True,
    )
    print("SYNTHETIC DEMO: invented item parameters; no trained model or SAT score")
    while session.stop_reason() is None:
        item = session.next_item()
        simulated_correct = rng.random() < probability(0.6, item.parameters)
        answer = item.answer_index if simulated_correct else (item.answer_index + 1) % 4
        estimate = session.submit(item.item_id, answer)
        profile.record_answer(item.skill, answer == item.answer_index)
        print(f"{item.item_id}: correct={simulated_correct}, "
              f"theta={estimate.mean:.2f}, posterior SD={estimate.posterior_sd:.2f}")

    audit.emit("session_stopped", session_id=session.session_id, reason=session.stop_reason())
    score_before_feedback = session.estimate
    apply_feedback(profile, Feedback(item.item_id, item.version, "algebra", FeedbackKind.TOO_HARD), audit)
    apply_feedback(profile, Feedback(item.item_id, item.version, "algebra", FeedbackKind.UNCLEAR), audit)
    request = GenerationRequest("draft-demo", "algebra", profile.difficulty_preferences["algebra"])
    draft = TemplateGenerator().generate(request)
    audit.emit(
        "draft_generated", request=asdict(request), item_id=draft.item_id,
        source=draft.source, structural_issues=structural_issues(draft),
    )
    report = {
        "seed": seed,
        "stop_reason": session.stop_reason(),
        "estimate": asdict(session.estimate),
        "feedback_changed_score": session.estimate != score_before_feedback,
        "next_practice_difficulty_request": request.target_difficulty,
        "template_uses_requested_difficulty": False,
        "new_draft_can_score": draft.can_score(),
        "audit_events": len(audit.events),
    }
    print(json.dumps(report, indent=2))
    if audit_path is not None:
        audit.export(Path(audit_path))
        print(f"Audit written to {audit_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--audit", type=Path, help="optional local JSONL output")
    args = parser.parse_args()
    run(args.seed, args.audit)
