# Results (auto-generated)

## HoosierHelp — workflow module ablation (text controller, no agent-to-agent asks)

| workflow | 4 behaviors (192) | all 5 (240) | self-contradictory (48) | questions to person / episode |
|---|---|---|---|---|
| official single-agent loop (same Qwen3-8B) | 0.000 (n=192) | 0.000 (n=240) | 0.000 (n=48) | 0.00 |
| v1 plain role chain (start) | 0.354 (n=192) | 0.354 (n=240) | 0.354 (n=48) | 4.01 |
| v2 + counterfactual relaxation diagnosis | 0.396 (n=192) | 0.371 (n=240) | 0.271 (n=48) | 4.13 |
| v3 + deterministic planner, grounding | 0.500 (n=192) | 0.463 (n=240) | 0.312 (n=48) | 4.10 |
| v4 + direct-to-person questions | 0.609 (n=192) | 0.583 (n=240) | 0.479 (n=48) | 3.92 |
| v5 without fallback clarification (ablation) | 0.391 (n=192) | 0.383 (n=240) | 0.354 (n=48) | 3.36 |
| **v5 full system** | 0.594 (n=192) | 0.571 (n=240) | 0.479 (n=48) | 3.94 |

## HoosierHelp — related-work style baselines (same Qwen3-8B, same simulated user, same turn cap; errors count as failures)

| system | 4 behaviors (192) | all 5 (240) | self-contradictory (48) | questions to person (avg) |
|---|---|---|---|---|
| single agent, official prompt, direct | 0.000 (n=192) | 0.000 (n=240) | 0.000 (n=48) | 0.13 |
| single agent, official prompt, step-by-step (thinking) | 0.073 (n=192) | 0.067 (n=240) | 0.042 (n=48) | 0.47 |
| plain role chain (one greedy handoff) | 0.359 (n=192) | 0.333 (n=240) | 0.229 (n=48) | 3.55 |
| 9-way majority-vote handoff | 0.344 (n=192) | 0.338 (n=240) | 0.312 (n=48) | 3.53 |
| confidence handoff (uncertainty propagation) | 0.323 (n=192) | 0.317 (n=240) | 0.292 (n=48) | 3.55 |
| **ours: v5 text controller** | 0.594 (n=192) | 0.571 (n=240) | 0.479 (n=48) | 3.94 |
| **ours: v5 + learned rich-text controller** | 0.615 (n=192) | 0.583 (n=240) | 0.458 (n=48) | 3.99 |

## HoosierHelp — clarification controllers on the v5 workflow (paired vs. v5 text controller)

| controller | 4 behaviors | Δ vs no-ask [95% CI] | all 5 | Δ all | controller asks (total) | questions to person / episode |
|---|---|---|---|---|---|---|
| no agent-to-agent asks (v5 text) | 0.594 (n=192) | +0.000 [+0.000, +0.000] | 0.571 (n=240) | +0.000 [+0.000, +0.000] | 0 | 3.94 |
| ask whenever candidates disagree | 0.599 (n=192) | +0.005 [-0.031, +0.042] | 0.558 (n=240) | -0.013 [-0.046, +0.021] | 83 | 4.22 |
| max-uncertainty | 0.620 (n=192) | +0.026 [-0.016, +0.068] | 0.579 (n=240) | +0.008 [-0.029, +0.046] | 80 | 4.20 |
| fixed field order | 0.609 (n=192) | +0.016 [-0.031, +0.062] | 0.571 (n=240) | +0.000 [-0.042, +0.042] | 86 | 4.22 |
| decision-relevance VOI (deterministic) | 0.604 (n=192) | +0.010 [-0.031, +0.052] | 0.583 (n=240) | +0.013 [-0.025, +0.050] | 32 | 4.04 |
| learned rich-text controller (v5 labels) | 0.615 (n=192) | +0.021 [-0.016, +0.062] | 0.583 (n=240) | +0.013 [-0.021, +0.046] | 17 | 3.99 |
|   ablation: no dependency features | 0.625 (n=192) | +0.031 [+0.000, +0.062] | 0.588 (n=240) | +0.017 [-0.013, +0.046] | 13 | 3.99 |
|   ablation: no receiver state | 0.615 (n=192) | +0.021 [-0.010, +0.052] | 0.588 (n=240) | +0.017 [-0.013, +0.050] | 8 | 3.94 |

## HoosierHelp v5 — episodes with interpretation disagreement (n=84) vs. without (n=156); fixed resolver code

| controller | success, disagreement subset | Δ vs no-ask [95% CI] | success, no-disagreement subset | controller questions to person / episode (subset) |
|---|---|---|---|---|
| no agent-to-agent asks (v5 text) | 0.440 (n=84) | +0.000 [+0.000, +0.000] | 0.686 (n=156) | 0.00 |
| ask whenever candidates disagree | 0.357 (n=84) | -0.083 [-0.179, +0.000] | 0.699 (n=156) | 0.61 |
| max-uncertainty | 0.393 (n=84) | -0.048 [-0.131, +0.036] | 0.699 (n=156) | 0.65 |
| fixed field order | 0.381 (n=84) | -0.060 [-0.143, +0.012] | 0.692 (n=156) | 0.63 |
| decision-relevance VOI | 0.381 (n=84) | -0.060 [-0.131, +0.012] | 0.686 (n=156) | 0.35 |
| **learned rich-text controller** | 0.452 (n=84) | +0.012 [-0.071, +0.095] | 0.673 (n=156) | 0.11 |

## HoosierHelp v6 — Table A: free interaction budget (10-turn cap). success / questions to the person per episode

| controller | success / questions |
|---|---|
| ask on every open question (disagreement-first order) | (pending) |
| fixed field order, ask everything | 0.613 / 4.36 |
| budget-aware value of information (train priors, no learning) | 0.621 / 4.08 |
| ours: learned rich-text controller, free question choice (λ=0.2) | (pending) |
| ours: learned stop rule, fixed question order (λ=?) | (pending) |
| **ours: learned + VOI hybrid (β=?, λ=0.2)** | (pending) |

## HoosierHelp v6 — Table B: fixed interaction budgets. success (questions per episode)

| controller | 8 turns (native) | 7 turns | 6 turns |
|---|---|---|---|
| ask on every open question (disagreement-first order) | 0.588 / 4.20 | 0.500 / 3.91 | 0.388 / 3.53 |
| fixed field order, ask everything | 0.596 / 4.16 | 0.496 / 3.95 | 0.392 / 3.54 |
| budget-aware value of information (train priors, no learning) | 0.579 / 3.95 | 0.533 / 3.55 | 0.454 / 2.69 |
| ours: learned rich-text controller, free question choice (λ=0.2) | 0.442 / 2.90 | (pending) | 0.383 / 2.66 |
| ours: learned stop rule, fixed question order (λ=?) | (pending) | (pending) | (pending) |
| **ours: learned + VOI hybrid (β=?, λ=0.2)** | (pending) | (pending) | (pending) |

Supplementary v6 rows (8 turns): max-uncertainty 0.575 / 4.17; never ask 0.000 / 0.00; learned at default λ=0.02 (pending); learned without dependency features (pending); learned without receiver state (pending).

## HoosierHelp v7 — ATOMIC questions (one field per question). Table A: free budget (10-turn cap): success / questions per episode

| controller | success / questions |
|---|---|
| ask on every open question (disagreement-first order) | 0.458 / 5.83 |
| fixed field order, ask everything | 0.446 / 5.88 |
| myopic value of information (literature rule: no budget model) | 0.471 / 5.76 |
| **ours: learned + budget-aware VOI (β=0.25, λ=0.05)** | 0.379 / 3.58 |
|   ours without learning: budget-aware VOI only | 0.463 / 5.51 |
|   ours without VOI: learned only, free question choice (λ=0.05) | 0.388 / 4.12 |
|   ours, learned stop rule with fixed question order (λ=0.05) | 0.321 / 3.06 |

## HoosierHelp v7 — ATOMIC questions. Table B: turn-cap sweep: success / questions per episode

| controller | 5 turns | 6 turns | 7 turns | 8 turns (native) | 10 turns |
|---|---|---|---|---|---|
| ask on every open question (disagreement-first order) | 0.263 / 3.00 | 0.275 / 3.99 | 0.296 / 4.85 | 0.367 / 5.26 | 0.458 / 5.83 |
| fixed field order, ask everything | 0.275 / 3.00 | 0.263 / 3.99 | 0.304 / 4.85 | 0.350 / 5.28 | 0.446 / 5.88 |
| myopic value of information (literature rule: no budget model) | 0.267 / 3.00 | 0.275 / 3.99 | 0.292 / 4.84 | 0.333 / 5.14 | 0.471 / 5.76 |
| **ours: learned + budget-aware VOI (β=0.25, λ=0.05)** | (pending) | 0.325 / 2.69 | (pending) | 0.371 / 3.30 | 0.379 / 3.58 |
|   ours without learning: budget-aware VOI only | 0.279 / 2.16 | 0.342 / 2.54 | 0.367 / 2.74 | 0.408 / 3.76 | 0.463 / 5.51 |
|   ours without VOI: learned only, free question choice (λ=0.05) | (pending) | 0.229 / 3.18 | (pending) | 0.317 / 3.83 | 0.388 / 4.12 |
|   ours, learned stop rule with fixed question order (λ=0.05) | (pending) | 0.308 / 2.62 | (pending) | 0.342 / 2.95 | 0.321 / 3.06 |

Supplementary v7 rows (8 turns): never ask 0.000 / 0.00; hybrid without dependency features (pending); hybrid without receiver state (pending).

## HoosierHelp — sampling-seed robustness (all 5 behaviors, 240 episodes per run; mean ± std over seeds)

| system | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| v5 text controller (no asks) | 0.571 | 0.583 | 0.567 | 0.574 ± 0.009 (n=3) |
| v5 without fallback clarification | 0.383 | 0.383 | 0.396 | 0.388 ± 0.007 (n=3) |
| v5 + decision-relevance VOI | 0.583 | 0.579 | 0.562 | 0.575 ± 0.011 (n=3) |
| v5 + learned rich-text controller | 0.583 | 0.588 | 0.567 | 0.579 ± 0.011 (n=3) |

## HoosierHelp v5 — by behavior and by case (text controller)

| behavior | success | | case | success |
|---|---|---|---|---|
| self_contradictory | 0.479 (n=48) | | composite|direct_match | 0.733 (n=60) |
| unsupported_request | 0.396 (n=48) | | composite|fallback_required | 0.283 (n=60) |
| rambling | 0.562 (n=48) | | single|direct_match | 0.817 (n=60) |
| normal | 0.771 (n=48) | | single|fallback_required | 0.450 (n=60) |
| impatience | 0.646 (n=48) | |  |  |

## NGQA (native knowledge-QA, 300 test items, Qwen3-8B)

| system | binary acc | sparse / standard / complex | tag F1 | explain ROUGE-L |
|---|---|---|---|---|
| single agent, direct answer | 0.623 | 0.66 / 0.54 / 0.58 | 0.535 | 0.503 |
| single agent, step-by-step | 0.647 | 0.66 / 0.59 / 0.66 | 0.555 | 0.516 |
| plain role chain (one greedy handoff) | 0.623 | 0.66 / 0.53 / 0.61 | 0.451 | 0.483 |
| 5-way majority vote | 0.620 | 0.65 / 0.53 / 0.61 | 0.454 | 0.478 |
| confidence handoff (uncertainty propagation) | 0.577 | 0.61 / 0.47 / 0.58 | 0.459 | 0.464 |
| **ours (candidate-indexed handoff + guideline analysis)** | 1.000 | 1.00 / 1.00 / 1.00 | 0.976 | 0.745 |

Original NGQA paper, best reported configuration (GPT-4o-mini + ToG): standard 0.86, complex 0.53.

## Cost accounting (HoosierHelp)

| item | wall time | generation calls / completion tokens | simulated-user spend |
|---|---|---|---|
| v5 label collection (200 roots, 1 seed, ~2 prefixes/root) | 8.8 min | 3600 / 1.32M | $0.22 |
| v5 calibration labels (30 roots) | 1.5 min | 468 / 0.18M | $0.03 |
| v6 label collection (200 roots) | 32.7 min | 7903 / 3.52M | $0.70 |
| v6 calibration labels (30 roots) | (pending) | | |
| v5 controller training (A + C phases) | 40 s (GPU, tiny model) | — | — |

| controller (v5 / v6 eval, 240 episodes) | wall per episode | completion tokens per episode | user spend per episode | questions to person |
|---|---|---|---|---|
| v5 text (no asks) | 1.8 s | 6232 | $0.0001 | 3.93 |
| v5 VOI (online search signatures) | 1.9 s | 6457 | $0.0002 | 4.04 |
| v5 learned rich-text | 2.3 s | 6349 | $0.0001 | 3.99 |
| v6 fixed order | 2.0 s | 6686 | $0.0001 | 4.16 |
| v6 VOI (priors) | 1.9 s | 6220 | $0.0001 | 3.95 |
| v6 learned rich-text | (pending) | | | |

## Spend

Simulated-user API spend to date: $100.17 over 61797 calls (gpt-4.1 until 2026-10-19 05:30, gpt-4.1-mini after; replies cached by transcript prefix).
