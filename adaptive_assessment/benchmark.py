"""Paired synthetic measurement experiments; no training or human learning claim."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from math import sin, sqrt
from pathlib import Path
from random import Random
from statistics import mean, stdev
from time import perf_counter

from .adaptive import AssessmentSession, StopRule
from .audit import MemoryAudit
from .domain import Item, ItemParameters, ItemStatus
from .irt import probability, response_curve
from .selection import SelectionPolicy


POLICIES = {
    "random": SelectionPolicy("random"),
    "fisher": SelectionPolicy(),
    "fisher_family": SelectionPolicy("fisher", max_per_family=1),
    "gain": SelectionPolicy("information_gain"),
    "gain_explore": SelectionPolicy("information_gain", exploration=0.1),
    "gain_family": SelectionPolicy("information_gain", max_per_family=1),
    "gain_family_explore": SelectionPolicy("information_gain", exploration=0.1, max_per_family=1),
}
SCENARIOS = ("matched_model", "family_dependence", "calibration_shift")


def measurement_bank():
    """A response-model fixture, not a collection of usable assessment questions."""
    return tuple(
        Item(
            item_id=f"family-{family:02d}-sibling-{sibling}", version="1", skill="synthetic-trait",
            stem="Synthetic measurement item; do not administer to people.",
            options=("correct", "incorrect"), answer_index=0,
            explanation="The simulation supplies a potential binary response.",
            status=ItemStatus.CALIBRATED, family_id=f"family-{family:02d}",
            parameters=ItemParameters(
                (0.8, 1.2, 1.6, 2.0)[family % 4],
                -2.1 + family * 0.3 + (sibling - 1.5) * 0.08,
                "measurement-fixture-v1", "synthetic",
            ),
        )
        for family in range(15) for sibling in range(4)
    )


def potential_responses(bank, seed, learner, scenario):
    """Pre-sample every item outcome so all policies see the same simulated person."""
    if scenario not in SCENARIOS:
        raise ValueError("unknown response scenario")
    rng = Random(f"response-v1:{seed}:{learner}")
    theta = rng.gauss(0, 1)
    effects = {family: rng.gauss(0, 0.8) for family in sorted({i.family_id for i in bank})}
    responses = {}
    for index, item in enumerate(bank):
        response_theta = theta
        if scenario == "family_dependence":
            response_theta += effects[item.family_id]
        if scenario == "calibration_shift":
            response_theta -= 0.8 * sin(index // 4 + 1)  #shift true difficulty within each family
        responses[item.item_id] = rng.random() < probability(response_theta, item.parameters)
    return theta, responses


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def run_benchmark(seeds=5, learners=40, length=8, scenarios=SCENARIOS):
    if seeds < 2 or learners < 1 or not 1 <= length <= 15:
        raise ValueError("require >=2 seeds, >=1 learner, and 1..15 answers")
    if not scenarios or any(scenario not in SCENARIOS for scenario in scenarios):
        raise ValueError("select at least one known scenario")
    bank = measurement_bank()
    for item in bank:
        response_curve(item.parameters)  #time steady-state decisions with the common cache warm
    rows = []
    for scenario in scenarios:
        for policy_name, policy in POLICIES.items():
            per_seed = []
            for seed in range(seeds):
                errors, coverages, sds, repeats, latencies, counts = [], [], [], [], [], Counter()
                stops = Counter()
                for learner in range(learners):
                    theta, outcomes = potential_responses(bank, seed, learner, scenario)
                    session = AssessmentSession(
                        f"{seed}:{learner}", "synthetic-trait", bank, MemoryAudit(),
                        StopRule(length, length, 0.01), allow_synthetic=True,
                        selection=policy, selection_seed=seed * learners + learner,
                    )
                    families = set()
                    while session.stop_reason() is None:
                        start = perf_counter()
                        item = session.next_item()
                        result = session.submit(item.item_id, 0 if outcomes[item.item_id] else 1)
                        latencies.append((perf_counter() - start) * 1000)
                        counts[item.item_id] += 1
                        families.add(item.family_id)
                    stops[session.stop_reason()] += 1
                    errors.append(result.mean - theta)
                    coverages.append(result.interval_95[0] <= theta <= result.interval_95[1])
                    sds.append(result.posterior_sd)
                    repeats.append(result.response_count - len(families))
                per_seed.append({
                    "seed": seed, "rmse": sqrt(mean(e * e for e in errors)),
                    "bias": mean(errors), "coverage_95": mean(coverages),
                    "mean_posterior_sd": mean(sds), "repeated_families": mean(repeats),
                    "bank_exposure_fraction": len(counts) / len(bank),
                    "max_item_exposure_rate": max(counts.values()) / learners,
                    "p95_decision_and_update_ms": percentile(latencies, 0.95),
                    "stop_counts": dict(stops),
                })
            metrics = {
                key: {"mean": mean(row[key] for row in per_seed),
                      "sd_across_seeds": stdev(row[key] for row in per_seed)}
                for key in per_seed[0] if key not in {"seed", "stop_counts"}
            }
            rows.append({"scenario": scenario, "policy": policy_name,
                         "config": asdict(policy), "metrics": metrics, "per_seed": per_seed})
            print(f"{scenario:19s} {policy_name:20s} "
                  f"RMSE={metrics['rmse']['mean']:.3f} "
                  f"coverage={metrics['coverage_95']['mean']:.3f} "
                  f"repeat families={metrics['repeated_families']['mean']:.2f}", flush=True)
    contrasts = []
    for scenario in scenarios:
        relevant = {row["policy"]: row for row in rows if row["scenario"] == scenario}
        for candidate, baseline in (
            ("gain", "fisher"), ("gain_family", "gain"),
            ("gain_explore", "gain"), ("gain_family_explore", "gain_family"),
            ("gain_family_explore", "fisher"),
        ):
            differences = [a["rmse"] - b["rmse"] for a, b in zip(
                relevant[candidate]["per_seed"], relevant[baseline]["per_seed"],
            )]
            contrasts.append({
                "scenario": scenario, "candidate": candidate, "baseline": baseline,
                "rmse_delta_mean": mean(differences),
                "rmse_delta_sd_across_seeds": stdev(differences),
                "per_seed_rmse_delta": differences,
            })
    return {
        "protocol": "measurement-simulation-v1",
        "seeds": list(range(seeds)), "learners_per_seed": learners, "answers_per_learner": length,
        "bank_items": len(bank), "bank_families": 15, "family_effect_sd": 0.8,
        "difficulty_shift_amplitude": 0.8, "ability_distribution": "Normal(0, 1)",
        "cache": "prewarmed item curves; per-session posterior cached between answers",
        "limitation": "Synthetic fixed-parameter inference; no trained models or human learning evidence",
        "rows": rows, "paired_contrasts": contrasts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--learners", type=int, default=40)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/measurement-benchmark.json"))
    args = parser.parse_args()
    report = run_benchmark(args.seeds, args.learners, args.length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Full results: {args.output}")
