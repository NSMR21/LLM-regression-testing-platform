# Success criteria: measured results

Generated 2026-11-04 22:27 UTC by `experiments/success_criteria.py` (RegressionLens 0.1.0, Python 3.13.18, full mode, 277 s).

All numbers below come from the **simulated provider**, whose true per-case accuracy is known, so detection power and false-alarm rates are measured against ground truth. The same harness runs on a real model with `--model`; see the last section.

| # | Criterion | Result | Status |
|---|---|---|---|
| 1 | A deliberately introduced prompt regression is detected | detected in 88% (35/40; 95% CI 74–95%) of paired runs | **met** |
| 2 | Identical runs don't raise false alarms beyond 2.5% | 0.7% (2/300; 95% CI 0.2–2.4%) | **met** |
| 3 | Model graders are checked against human labels, agreement reported | κ reported per judge; poor judges flagged (95% judge → calibrated, 85% judge → calibrated, 70% judge → weak) | **met** |
| 4 | Any past result can be reproduced from its stored config | re-score 100% (1800/1800; 95% CI 100–100%) identical; replay 100% (1800/1800; 95% CI 100–100%) identical outputs | **met** |

## 1. Regression detection

**Prompt regression.** The example suite (`examples/support-qa`, 30 cases) compares `prompts/v1.yaml` with `prompts/v2-shorter.yaml`, a plausible cleanup that drops the output-format instructions and the final-sale policy line.

Detected = the paired test on `pass_rate` reports a significant drop (absolute gates are not counted).

| Model | Pairs | Repeats | Detected | Verdicts | Mean Δ pass rate | Median p |
|---|---|---|---|---|---|---|
| simulated | 40 | 5 | 88% (35/40; 95% CI 74–95%) | {'fail': 39, 'inconclusive': 1} | -12.8 pts | 0.0026 |

**Power curve.** Synthetic suite of 60 cases × 5 repeats; the candidate's true per-case accuracy is shifted by the given amount; 40 independent comparisons per row.

| True drop | Mean observed Δ | Detected by the paired test |
|---|---|---|
| +0 pts | -0.8 pts | 5% (2/40; 95% CI 1–17%) |
| -2 pts | -2.9 pts | 10% (4/40; 95% CI 4–23%) |
| -5 pts | -5.9 pts | 40% (16/40; 95% CI 26–55%) |
| -8 pts | -8.7 pts | 70% (28/40; 95% CI 55–82%) |
| -10 pts | -10.7 pts | 82% (33/40; 95% CI 68–91%) |
| -15 pts | -15.8 pts | 100% (40/40; 95% CI 91–100%) |
| -20 pts | -20.8 pts | 100% (40/40; 95% CI 91–100%) |

Smallest tested drop detected at least 80% of the time: **10 pts**.

Each run reports its own minimum detectable effect (MDE, 80% power): 8.6 pts on average here. The measured curve crosses 80% at about 9.6 pts, so the reported MDE is slightly optimistic compared with the measured power (by 1.0 pts). Treat it as a guide to what a suite can catch, not a guarantee.

All rows share the same 40 baseline runs, which is why every row's observed Δ is offset from the true drop by about the same amount.

## 2. False alarms on identical runs

**Full pipeline A/A.** The same configuration run twice with different seeds, then compared. Threshold: 2.5% (α/2 for the two-sided test at α = 0.05).

| Model | Pairs | Repeats | Regression alarms | Verdicts |
|---|---|---|---|---|
| simulated | 300 | 5 | 0.7% (2/300; 95% CI 0.2–2.4%) | {'fail': 14, 'inconclusive': 237, 'pass': 49} |

Most A/A comparisons end **inconclusive**, not pass: with 30 cases the CI is wider than the 3-point margin, so the platform refuses to certify equivalence it cannot see. That is the intended behaviour; more cases or repeats turn those into passes.

**Why per-case statistics matter.** 1000 simulated A/A comparisons per row (40 cases × 5 repeats). ρ is the probability that a case's repeats within one run share a single draw (deterministic decoding, batching or server state make repeats correlated). The naive method pools all trials as independent samples and runs a two-proportion z-test.

| ρ | RegressionLens false alarms | Naive pooled z-test false alarms |
|---|---|---|
| 0.0 | 1.9% (19/1000; 95% CI 1.2–2.9%) | 1.7% (17/1000; 95% CI 1.1–2.7%) |
| 0.5 | 2.4% (24/1000; 95% CI 1.6–3.5%) | 11.8% (118/1000; 95% CI 9.9–13.9%) |
| 1.0 | 0.5% (5/1000; 95% CI 0.2–1.2%) | 16.8% (168/1000; 95% CI 14.6–19.2%) |

RegressionLens stays at or below the threshold whatever the correlation; the naive test's false-alarm rate grows with ρ because it counts correlated repeats as independent evidence.

## 3. Grader agreement with human labels

A simulated judge with a given accuracy grades 60 stratified outputs; simulated human labels follow the truth with 3% noise. The dashboard and API compute the same report from real labels.

| Judge accuracy | Labels | Agreement | Cohen's κ (95% CI) | False pass | False fail | Trust |
|---|---|---|---|---|---|---|
| 95% | 60 | 95% | 0.90 (0.77–1.00) | 7% | 3% | calibrated |
| 85% | 60 | 92% | 0.83 (0.67–0.97) | 3% | 13% | calibrated |
| 70% | 60 | 70% | 0.40 (0.16–0.62) | 20% | 40% | weak |

A grader is `calibrated` with κ ≥ 0.6 on at least 30 labels. Verdicts that rely on a grader that is not calibrated carry a note saying so.

## 4. Reproducibility

20 past runs of the example suite.

| Check | Result |
|---|---|
| Re-score stored outputs with freshly built graders: identical scores | 100% (1800/1800; 95% CI 100–100%) |
| Re-run stored config with the stored seed: identical outputs | 100% (1800/1800; 95% CI 100–100%) |
| Re-run: same config fingerprint | 100% (20/20; 95% CI 84–100%) |
| Re-run vs original: verdicts | {'pass': 20} |
| Re-run with a fresh seed (non-replayable sampling): verdicts | {'fail': 2, 'inconclusive': 14, 'pass': 4} |
| Re-run with a fresh seed: regression alarms (paired test only; gates not counted) | 5% (1/20; 95% CI 1–24%) |

On a real GPU backend exact replay is not guaranteed (kernels and batching are non-deterministic), so the server's `POST /runs/{id}/rerun` treats reproduction as statistical equivalence: it compares the rerun with the original and refuses to run at all when the model digest no longer matches (tested in `backend/tests/test_server.py`).

## Real model

Not measured in this report: no model server was available where it was generated. To measure on a local model:

```bash
ollama pull llama3.2:3b
python experiments/success_criteria.py --model examples/support-qa/models/llama3.2-3b.yaml --real-pairs 5
```
