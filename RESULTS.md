# Results (auto-generated)

## HoosierHelp — workflow module ablation (text controller, no agent-to-agent asks)

| workflow | 4 behaviors (192) | all 5 (240) | self-contradictory (48) |
|---|---|---|---|
| official single-agent loop (same Qwen3-8B) | 0.000 (n=192) | 0.000 (n=240) | 0.000 (n=48) |
| v1 plain role chain (start) | 0.354 (n=192) | 0.354 (n=240) | 0.354 (n=48) |
| v2 + counterfactual relaxation diagnosis | 0.396 (n=192) | 0.371 (n=240) | 0.271 (n=48) |
| v3 + deterministic planner, grounding | 0.500 (n=192) | 0.463 (n=240) | 0.312 (n=48) |
| v4 + direct-to-person questions | 0.609 (n=192) | 0.583 (n=240) | 0.479 (n=48) |
| v5 without fallback clarification (ablation) | 0.391 (n=192) | 0.383 (n=240) | 0.354 (n=48) |
| **v5 full system** | 0.594 (n=192) | 0.571 (n=240) | 0.479 (n=48) |

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

| controller | 4 behaviors | Δ vs no-ask [95% CI] | all 5 | Δ all | asks to person |
|---|---|---|---|---|---|
| no agent-to-agent asks (v5 text) | 0.594 (n=192) | +0.000 [+0.000, +0.000] | 0.571 (n=240) | +0.000 [+0.000, +0.000] | 0 |
| ask whenever candidates disagree | 0.599 (n=192) | +0.005 [-0.031, +0.042] | 0.558 (n=240) | -0.013 [-0.046, +0.021] | 83 |
| max-uncertainty | 0.620 (n=192) | +0.026 [-0.016, +0.068] | 0.579 (n=240) | +0.008 [-0.029, +0.046] | 80 |
| fixed field order | 0.609 (n=192) | +0.016 [-0.031, +0.062] | 0.571 (n=240) | +0.000 [-0.042, +0.042] | 86 |
| decision-relevance VOI (deterministic) | 0.604 (n=192) | +0.010 [-0.031, +0.052] | 0.583 (n=240) | +0.013 [-0.025, +0.050] | 32 |
| learned rich-text controller (v5 labels) | 0.615 (n=192) | +0.021 [-0.016, +0.062] | 0.583 (n=240) | +0.013 [-0.021, +0.046] | 17 |
|   ablation: no dependency features | 0.625 (n=192) | +0.031 [+0.000, +0.062] | 0.588 (n=240) | +0.017 [-0.013, +0.046] | 13 |
|   ablation: no receiver state | 0.615 (n=192) | +0.021 [-0.010, +0.052] | 0.588 (n=240) | +0.017 [-0.013, +0.050] | 8 |

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
| v6 label collection (200 roots) | (pending) | | |
| v6 calibration labels (30 roots) | (pending) | | |
| v5 controller training (A + C phases) | 40 s (GPU, tiny model) | — | — |

| controller (v5 / v6 eval, 240 episodes) | wall per episode | completion tokens per episode | user spend per episode | questions to person |
|---|---|---|---|---|
| v5 text (no asks) | 1.8 s | 6232 | $0.0001 | 3.93 |
| v5 VOI (online search signatures) | 1.9 s | 6457 | $0.0002 | 4.04 |
| v5 learned rich-text | 2.3 s | 6349 | $0.0001 | 3.99 |
| v6 fixed order | 2.9 s | 6710 | $0.0010 | 4.19 |
| v6 VOI (priors) | (pending) | | | |
| v6 learned rich-text | (pending) | | | |

## Spend

Simulated-user API spend to date: $91.92 over 46902 calls (gpt-4.1 until 2026-10-19 05:30, gpt-4.1-mini after; replies cached by transcript prefix).
