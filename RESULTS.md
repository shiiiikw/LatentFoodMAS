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

## HoosierHelp — clarification controllers on the v5 workflow (paired vs. v5 text controller)

| controller | 4 behaviors | Δ vs no-ask [95% CI] | all 5 | Δ all | asks to person |
|---|---|---|---|---|---|
| no agent-to-agent asks (v5 text) | 0.594 (n=192) | +0.000 [+0.000, +0.000] | 0.571 (n=240) | +0.000 [+0.000, +0.000] | 0 |
| ask whenever candidates disagree | 0.599 (n=192) | +0.005 [-0.031, +0.042] | 0.558 (n=240) | -0.013 [-0.046, +0.021] | 83 |
| max-uncertainty | 0.620 (n=192) | +0.026 [-0.016, +0.068] | 0.579 (n=240) | +0.008 [-0.029, +0.046] | 80 |
| fixed field order | 0.609 (n=192) | +0.016 [-0.031, +0.062] | 0.571 (n=240) | +0.000 [-0.042, +0.042] | 86 |
| decision-relevance VOI (deterministic) | 0.604 (n=192) | +0.010 [-0.031, +0.052] | 0.583 (n=240) | +0.013 [-0.025, +0.050] | 32 |
| learned rich-text controller (v5 labels) | 0.615 (n=192) | +0.021 [-0.016, +0.062] | 0.583 (n=240) | +0.013 [-0.021, +0.046] | 17 |

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

## Spend

Simulated-user API spend to date: $90.75 over 45062 calls (gpt-4.1 until 2026-10-19 05:30, gpt-4.1-mini after; replies cached by transcript prefix).
