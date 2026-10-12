from dataclasses import replace
from math import log
from random import Random
import unittest

from adaptive_assessment.adaptive import AssessmentSession, StopRule
from adaptive_assessment.audit import MemoryAudit
from adaptive_assessment.benchmark import measurement_bank, potential_responses
from adaptive_assessment.demo import synthetic_bank
from adaptive_assessment.domain import Response
from adaptive_assessment.irt import ability_posterior, expected_information_gain, probability
from adaptive_assessment.selection import SelectionPolicy


class SelectionTests(unittest.TestCase):
    def test_gain_matches_explicit_two_outcome_posterior_entropy(self):
        bank = synthetic_bank()
        history = [Response(bank[0], True), Response(bank[-1], False)]
        prior = ability_posterior(history, allow_synthetic=True)
        item = bank[5]
        p = sum(w * probability(t, item.parameters) for t, w in zip(prior.grid, prior.weights))

        def entropy(posterior):
            return -sum(w * log(w) for w in posterior.weights if w > 0)

        right = ability_posterior(history + [Response(item, True)], allow_synthetic=True)
        wrong = ability_posterior(history + [Response(item, False)], allow_synthetic=True)
        expected = entropy(prior) - p * entropy(right) - (1 - p) * entropy(wrong)
        self.assertAlmostEqual(expected_information_gain(prior, item.parameters), expected, places=12)

    def test_probabilities_and_seed_replay(self):
        bank = synthetic_bank()
        policy = SelectionPolicy("information_gain", exploration=0.2)
        posterior = ability_posterior([])
        first = policy.choose(bank, posterior, Random(2))
        second = policy.choose(tuple(reversed(bank)), posterior, Random(2))
        self.assertEqual(first, second)
        probabilities = dict(first.candidate_probabilities)
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertTrue(all(p > 0 for p in probabilities.values()))
        self.assertEqual(first.probability, probabilities[first.item.item_id])
        self.assertAlmostEqual(max(probabilities.values()), 0.8 + 0.2 / len(bank))

    def test_sampling_matches_logged_distribution(self):
        bank, posterior = synthetic_bank()[:3], ability_posterior([])
        policy, rng = SelectionPolicy("fisher", exploration=0.3), Random(42)
        counts = {item.item_id: 0 for item in bank}
        for _ in range(4000):
            decision = policy.choose(bank, posterior, rng)
            counts[decision.item.item_id] += 1
        for item_id, p in decision.candidate_probabilities:
            self.assertLess(abs(counts[item_id] / 4000 - p), 0.025)

    def test_family_exclusion_and_honest_stop_reason(self):
        bank = [replace(item, family_id="shared") for item in synthetic_bank()]
        session = AssessmentSession(
            "test", "algebra", bank, MemoryAudit(), StopRule(4, 8),
            allow_synthetic=True, selection=SelectionPolicy("information_gain", max_per_family=1),
        )
        item = session.next_item()
        session.submit(item.item_id, item.answer_index)
        self.assertEqual(session.stop_reason(), "family_limit_reached")
        self.assertEqual(session.estimate.response_count, 1)

    def test_family_mode_refuses_missing_metadata(self):
        with self.assertRaises(ValueError):
            SelectionPolicy(max_per_family=1).eligible(synthetic_bank(), [])

    def test_posterior_cache_matches_recomputation_without_mutation(self):
        session = AssessmentSession("test", "algebra", synthetic_bank(), MemoryAudit(), allow_synthetic=True)
        before = session.estimate
        item = session.next_item()
        self.assertIs(before, session.estimate)
        session.submit(item.item_id, item.answer_index)
        expected = ability_posterior(session.responses, allow_synthetic=True).estimate
        self.assertEqual(session.estimate, expected)
        self.assertEqual(before.response_count, 0)

    def test_different_versions_of_same_item_cannot_count_twice(self):
        item = synthetic_bank()[0]
        with self.assertRaises(ValueError):
            ability_posterior([Response(item, True), Response(replace(item, version="2"), True)], allow_synthetic=True)

    def test_benchmark_outcomes_replay_and_stress_changes_responses(self):
        bank = measurement_bank()
        base = potential_responses(bank, 0, 0, "matched_model")
        self.assertEqual(base, potential_responses(bank, 0, 0, "matched_model"))
        changed = potential_responses(bank, 0, 0, "family_dependence")
        self.assertEqual(base[0], changed[0])
        self.assertNotEqual(base[1], changed[1])

    def test_audit_contains_full_selection_distribution(self):
        audit = MemoryAudit()
        session = AssessmentSession(
            "test", "algebra", synthetic_bank(), audit, allow_synthetic=True,
            selection=SelectionPolicy("information_gain", exploration=0.1),
        )
        chosen = session.next_item()
        event = audit.events[-1]["payload"]
        self.assertEqual(event["chosen_probability"], event["candidate_probabilities"][chosen.item_id])
        self.assertAlmostEqual(sum(event["candidate_probabilities"].values()), 1)
        self.assertEqual(len(event["candidate_utilities"]), len(synthetic_bank()))


if __name__ == "__main__":
    unittest.main()
