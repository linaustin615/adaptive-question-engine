"""A unidimensional 2PL model with a fixed standard-normal prior.

This estimates learner ability given item parameters. It does not fit parameters.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import exp, isfinite, log, log1p, sqrt
from typing import Sequence, Tuple

from .domain import AbilityEstimate, ItemParameters, Response


ABILITY_GRID = tuple(-4.0 + i * 0.05 for i in range(161))


@dataclass(frozen=True)
class AbilityPosterior:
    grid: Tuple[float, ...]
    weights: Tuple[float, ...]
    estimate: AbilityEstimate


def binary_entropy(p: float) -> float:
    """Uncertainty of a binary outcome, measured in natural-log units."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log(p) - (1.0 - p) * log1p(-p)


@lru_cache(maxsize=2048)
def response_curve(parameters: ItemParameters):
    """Cache immutable curves, never learner-specific state."""
    values = tuple(probability(theta, parameters) for theta in ABILITY_GRID)
    return values, tuple(binary_entropy(p) for p in values)


def expected_information_gain(posterior: AbilityPosterior, parameters: ItemParameters) -> float:
    """I(theta; next answer) = H(E[p]) - E[H(p)], conditional on fixed parameters."""
    if posterior.grid != ABILITY_GRID:
        raise ValueError("information gain requires the standard ability grid")
    probabilities, entropies = response_curve(parameters)
    predictive_p = sum(w * p for w, p in zip(posterior.weights, probabilities))
    conditional_entropy = sum(w * h for w, h in zip(posterior.weights, entropies))
    return max(0.0, binary_entropy(predictive_p) - conditional_entropy)


def probability(theta: float, parameters: ItemParameters) -> float:
    if not isfinite(theta):
        raise ValueError("theta must be finite")
    z = parameters.discrimination * (theta - parameters.difficulty)
    if z >= 0:
        return 1.0 / (1.0 + exp(-z))
    e = exp(z)
    return e / (1.0 + e)


def information(theta: float, parameters: ItemParameters) -> float:
    p = probability(theta, parameters)
    return parameters.discrimination ** 2 * p * (1.0 - p)


def _softplus(value: float) -> float:
    return max(value, 0.0) + log1p(exp(-abs(value)))


def ability_posterior(
    responses: Sequence[Response], *, allow_synthetic: bool = False
) -> AbilityPosterior:
    """Recompute the posterior from the original prior and all unique answers."""
    keys = [r.item.item_id for r in responses]
    if len(keys) != len(set(keys)):
        raise ValueError("repeated responses would count evidence twice")
    if len({r.item.skill for r in responses}) > 1:
        raise ValueError("this example estimates one skill at a time")
    for response in responses:
        if not response.item.can_score(allow_synthetic):
            raise ValueError("scoring requires eligible calibrated items")

    grid = ABILITY_GRID
    log_weights = []
    for theta in grid:
        value = -0.5 * theta ** 2  #start from the same prior on every call
        for response in responses:
            parameters = response.item.parameters
            z = parameters.discrimination * (theta - parameters.difficulty)
            value -= _softplus(-z if response.correct else z)
        log_weights.append(value)

    peak = max(log_weights)
    weights = [exp(value - peak) for value in log_weights]
    total = sum(weights)
    weights = [weight / total for weight in weights]
    mean = sum(theta * weight for theta, weight in zip(grid, weights))
    variance = sum(weight * (theta - mean) ** 2 for theta, weight in zip(grid, weights))

    def quantile(target):
        cumulative = 0.0
        for theta, weight in zip(grid, weights):
            cumulative += weight
            if cumulative >= target:
                return theta
        return grid[-1]

    estimate = AbilityEstimate(
        mean, sqrt(variance), (quantile(0.025), quantile(0.975)),
        len(responses), sum(weights[:3]) + sum(weights[-3:]),
    )
    return AbilityPosterior(grid, tuple(weights), estimate)


def estimate_ability(
    responses: Sequence[Response], *, allow_synthetic: bool = False
) -> AbilityEstimate:
    return ability_posterior(responses, allow_synthetic=allow_synthetic).estimate
