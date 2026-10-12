from dataclasses import replace
from math import isfinite
import unittest

from adaptive_assessment.adaptive import AssessmentSession, StopRule
from adaptive_assessment.audit import MemoryAudit
from adaptive_assessment.demo import synthetic_bank
from adaptive_assessment.domain import Feedback, FeedbackKind, ItemParameters, ItemStatus, Response
from adaptive_assessment.generation import structural_issues
from adaptive_assessment.irt import estimate_ability, information, probability
from adaptive_assessment.personalization import PracticeProfile, apply_feedback


class ScoringTests(unittest.TestCase):
    def test_probability_and_information_reference_values(self):
        parameters = ItemParameters(2.0, 0.7, "test", "synthetic")
        self.assertEqual(probability(0.7, parameters), 0.5)
        self.assertEqual(information(0.7, parameters), 1.0)
        self.assertAlmostEqual(probability(1.7, parameters), 0.8807970779778823)
        self.assertEqual(probability(-1000, parameters), 0.0)
        self.assertEqual(probability(1000, parameters), 1.0)

    def test_prior_and_symmetric_single_answer(self):
        prior = estimate_ability([])
        self.assertAlmostEqual(prior.mean, 0.0)
        self.assertAlmostEqual(prior.posterior_sd, 1.0, places=3)
        item = replace(synthetic_bank()[0], parameters=ItemParameters(1, 0, "test", "synthetic"))
        right = estimate_ability([Response(item, True)], allow_synthetic=True)
        wrong = estimate_ability([Response(item, False)], allow_synthetic=True)
        self.assertGreater(right.mean, prior.mean)
        self.assertAlmostEqual(right.mean, -wrong.mean)
        self.assertAlmostEqual(right.posterior_sd, wrong.posterior_sd)

    def test_uniform_answer_patterns_remain_finite(self):
        for correct in (True, False):
            result = estimate_ability([Response(item, correct) for item in synthetic_bank()], allow_synthetic=True)
            self.assertTrue(isfinite(result.mean) and isfinite(result.posterior_sd))
            self.assertLess(result.interval_95[0], result.interval_95[1])

    def test_duplicate_and_mixed_skill_evidence_is_rejected(self):
        bank = synthetic_bank()
        with self.assertRaises(ValueError):
            estimate_ability([Response(bank[0], True)] * 2, allow_synthetic=True)
        with self.assertRaises(ValueError):
            estimate_ability([Response(bank[0], True), Response(replace(bank[1], skill="reading"), False)], allow_synthetic=True)

    def test_drafts_and_unapproved_synthetic_parameters_are_rejected(self):
        item = synthetic_bank()[0]
        with self.assertRaises(ValueError):
            estimate_ability([Response(item, True)])
        with self.assertRaises(ValueError):
            estimate_ability([Response(replace(item, status=ItemStatus.DRAFT), True)], allow_synthetic=True)

    def test_invalid_parameters_and_non_boolean_response_are_rejected(self):
        for a, b in ((0, 0), (-1, 0), (float("nan"), 0), (1, float("inf"))):
            with self.assertRaises(ValueError):
                ItemParameters(a, b, "test", "synthetic")
        with self.assertRaises(ValueError):
            Response(synthetic_bank()[0], "false")


class SessionTests(unittest.TestCase):
    def make_session(self, bank=None, stop=StopRule()):
        return AssessmentSession("test", "algebra", synthetic_bank() if bank is None else bank, MemoryAudit(), stop, allow_synthetic=True)

    def test_pending_retry_invalid_answer_and_duplicate_submission(self):
        session = self.make_session()
        item = session.next_item()
        self.assertEqual(session.next_item(), item)
        self.assertEqual(len(session.audit.events), 2)
        for bad in (-1, 4, True):
            with self.assertRaises(ValueError):
                session.submit(item.item_id, bad)
        self.assertEqual(len(session.responses), 0)
        session.submit(item.item_id, item.answer_index)
        with self.assertRaises(ValueError):
            session.submit(item.item_id, item.answer_index)

    def test_no_repeats_and_maximum_length_stop(self):
        session = self.make_session(stop=StopRule(2, 3, 0.01))
        seen = set()
        while session.stop_reason() is None:
            item = session.next_item()
            self.assertNotIn(item.item_id, seen)
            seen.add(item.item_id)
            session.submit(item.item_id, item.answer_index)
        self.assertEqual(len(seen), 3)
        self.assertEqual(session.stop_reason(), "maximum_length")

    def test_bank_exhaustion_and_eligibility(self):
        bank = synthetic_bank()
        session = self.make_session([
            bank[0], replace(bank[1], status=ItemStatus.RETIRED),
            replace(bank[2], status=ItemStatus.DRAFT), replace(bank[3], skill="reading"),
        ])
        item = session.next_item()
        self.assertEqual(item, bank[0])
        session.submit(item.item_id, item.answer_index)
        self.assertEqual(session.stop_reason(), "bank_exhausted")
        self.assertEqual(self.make_session([]).stop_reason(), "bank_exhausted")

    def test_information_selector_and_precision_stop(self):
        session = self.make_session(stop=StopRule(1, 8, 2.0))
        item = session.next_item()
        peak = max(information(0, candidate.parameters) for candidate in synthetic_bank())
        self.assertAlmostEqual(information(0, item.parameters), peak)
        session.submit(item.item_id, item.answer_index)
        self.assertEqual(session.stop_reason(), "precision_target")

    def test_sessions_do_not_share_answer_state(self):
        first, second = self.make_session(), self.make_session()
        item = first.next_item()
        first.submit(item.item_id, item.answer_index)
        self.assertEqual(second.estimate.response_count, 0)

    def test_feedback_does_not_change_score_or_calibration(self):
        session = self.make_session()
        item = session.next_item()
        session.submit(item.item_id, item.answer_index)
        before = session.estimate
        profile = PracticeProfile()
        for kind in FeedbackKind:
            apply_feedback(profile, Feedback(item.item_id, item.version, item.skill, kind), session.audit)
        self.assertEqual(session.estimate, before)
        self.assertEqual(item.parameters, synthetic_bank()[int(item.item_id.split("-")[1])].parameters)
        self.assertEqual(profile.counts, {})
        self.assertEqual(sum(event["kind"] == "review_requested" for event in session.audit.events), 2)

    def test_audit_readers_cannot_mutate_stored_events(self):
        audit = MemoryAudit()
        payload = {"nested": [1]}
        audit.emit("test", value=payload)
        payload["nested"].append(2)
        audit.events[0]["payload"]["value"]["nested"].append(3)
        self.assertEqual(audit.events[0]["payload"]["value"], {"nested": [1]})

    def test_structural_gate_catches_duplicate_choices_and_bad_key(self):
        item = replace(synthetic_bank()[0], options=(" A ", "a", "b", "c"), answer_index=4)
        self.assertEqual(set(structural_issues(item)), {"duplicate_options", "invalid_key"})


if __name__ == "__main__":
    unittest.main()
