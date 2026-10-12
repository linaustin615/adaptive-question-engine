# Evaluation plan

The included tests check software mechanics. They do not establish human learning, measurement validity, fairness, or accurate difficulty calibration. A small synthetic selection comparison has now been run; see [measurement experiments](measurement_experiments.md) for its exact budget and results. The larger studies proposed below remain future work.

## Before real data

Run a reproducible simulation with known generating parameters. A suggested initial budget is 20 seeds (`0` through `19`), 250 simulated learners per seed, and 20 answers per learner. Use a larger synthetic bank than the twelve-item walkthrough. Record the exact bank generator, ability distribution, seed, prior, policy, and stopping settings. These are proposed budgets, not statistical guarantees.

Compare uniform random selection, nearest-difficulty selection, and maximum-information selection using the same bank and answer budget. Generate one potential response per learner/item under a documented seed scheme so policies receive comparable simulated evidence. Report ability RMSE, bias across ability ranges, 95% interval coverage, mean test length, exposure concentration, and bank-exhaustion frequency. Give variability across seeds rather than only an average.

Then stress the assumptions: perturb item parameters, add guessing and response dependence, shift the ability population, and remove coverage constraints. An estimator tested only on data from its own assumed model can look unrealistically good. Simulation can show mechanics and failure modes, not prove effects on actual learners.

## With real response data

| Question | Evaluation | Essential comparison |
| --- | --- | --- |
| Does a learner model predict? | Held-out log loss, Brier score, calibration plots | Global success rate and smoothed per-skill accuracy |
| Are item parameters useful? | Response prediction, parameter uncertainty, item fit and drift | Rasch/2PL baselines and simple difficulty labels |
| Does CAT measure efficiently? | Precision, external validity, content coverage, exposure | Fixed forms and random selection at equal budgets |
| Are generated items valid? | Blind expert review of keys, ambiguity, skill match, accessibility | Authored templates and generation without feedback |
| Does personalization help learning? | Randomized study with independent delayed assessment | Same content/time budget under a simple practice policy |
| Does feedback improve generation? | Defects per reviewed item and per exposure | Remove issue summaries while keeping model and prompt budget fixed |

Split by learner to prevent personal history leakage. Hold out later time periods for drift and separate near-duplicate item families and shared passages across splits. Report new-learner and new-item settings separately. Fit calibration parameters and feature transforms on training data only; keep anchor/linking choices explicit. Select thresholds using validation data and reserve the final test set.

For learned models, begin with five initialization seeds and report all runs. Record learners, items, responses, missingness, exclusions, training budget, and compute. Do not interpret a small improvement without uncertainty. Bootstrap at the learner level; item clustering may require additional treatment. For a human study, choose sample size from a power analysis and prespecified effect size rather than copying the simulation budget.

## Selection bias and audit gates

Adaptive policies choose which evidence becomes visible. The selector logs candidate sets, policy configuration, utilities, and selection probabilities. The default deterministic policy has probability one for its chosen item and zero support for alternatives. The optional exploration policy assigns positive probability to each eligible candidate, but family-excluded candidates still have zero probability. This does not provide support for arbitrary alternative policies. Curriculum constraints, a justified exploration budget, and sufficient data remain prerequisites for useful off-policy evaluation; no such evaluator is implemented here.

Check differential item functioning and calibration errors across relevant groups with adequate sample sizes and appropriate data governance. Raw accuracy differences alone do not establish item bias. Include accommodations and device/accessibility effects in the study design. The scaffold has no demographic dataset and cannot pass these checks.

Release gates should be chosen in advance: expert content approval; credible parameter precision and fit; held-out performance against baselines; acceptable coverage and exposure; reviewed subgroup results; replayable artifacts; and rollback readiness. The maximum test length is an operational cap, not a scientific validation gate.

## Honest claims

Acceptable now: “The Python demo implements a synthetic Bayesian scoring and adaptive-selection loop, with separate feedback routing.”

Requires additional evidence: “The model learns from user feedback,” “personalization improves learning,” “AI-generated difficulty is calibrated,” “scores predict SAT results,” or “the system is fair.”
