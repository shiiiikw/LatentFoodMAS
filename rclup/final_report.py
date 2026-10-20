"""One-shot results report for the paper draft: HoosierHelp (module ablation, controller comparison, behaviors, cases),
NGQA (ours vs baselines), official single-agent baseline, spend. Reads whatever run directories exist.

  python -m rclup.final_report --out eval/final/REPORT.md
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from .common import RUNS, read_json, read_jsonl
from .report import bootstrap_ci

BEH = ("normal", "rambling", "impatience", "unsupported_request", "self_contradictory")
B4 = BEH[:4]


def load(path: str) -> dict | None:
    d = RUNS / path
    if not (d / "records.jsonl").exists():
        return None
    recs = read_jsonl(d / "records.jsonl")   # errored episodes count as failures
    def key(r):
        rid = r.get("root_id") or f"hoosier:eval:{r.get('user_spec_id', '')}"
        return rid + "|" + (r.get("behavior") or r.get("user_behavior") or "")
    return {key(r): r for r in recs}


def em(r) -> float:
    return float(bool((r.get("score") or {}).get("end_to_end_match")))


def rate(run: dict, keys) -> str:
    ks = [k for k in keys if k in run]
    return f"{statistics.mean(em(run[k]) for k in ks):.3f} (n={len(ks)})" if ks else "—"


def asks(run: dict, keys=None) -> str:
    ks = [k for k in (keys if keys is not None else run) if k in run]
    return f"{statistics.mean((run[k].get('score') or {}).get('user_questions', 0) for k in ks):.2f}" if ks else "—"


def paired(run: dict, ref: dict, keys) -> str:
    ks = [k for k in keys if k in run and k in ref]
    if not ks:
        return "—"
    d = [em(run[k]) - em(ref[k]) for k in ks]
    lo, hi = bootstrap_ci(d)
    return f"{statistics.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eval/final/REPORT.md")
    a = p.parse_args()
    out = []
    # ---------------- HoosierHelp
    v5 = load("eval/final/hoosier_text_mas_b5")
    allk = sorted(v5) if v5 else []
    k4 = [k for k in allk if not k.endswith("self_contradictory")]
    ksc = [k for k in allk if k.endswith("self_contradictory")]
    out.append("# Results (auto-generated)\n\n## HoosierHelp — workflow module ablation (text controller, no agent-to-agent asks)\n")
    out.append("| workflow | 4 behaviors (192) | all 5 (240) | self-contradictory (48) | questions to person / episode |\n|---|---|---|---|---|")
    for label, path in (("official single-agent loop (same Qwen3-8B)", "baselines/native_single_agent_b5"),
                        ("v1 plain role chain (start)", "eval/full/hoosier_text_mas_b5"),
                        ("v2 + counterfactual relaxation diagnosis", "eval/full2/hoosier_text_mas_b5"),
                        ("v3 + deterministic planner, grounding", "eval/full3/hoosier_text_mas_b5"),
                        ("v4 + direct-to-person questions", "eval/final/hoosier_text_mas_v4_b5"),
                        ("v5 without fallback clarification (ablation)", "eval/final/hoosier_text_mas_nofb_b5"),
                        ("**v5 full system**", "eval/final/hoosier_text_mas_b5")):
        run = load(path)
        if path.startswith("baselines/native"):
            if run is None:
                continue
            recs = list(run.values())
            def r_(bs):
                xs = [em(r) for r in recs if r["user_behavior"] in bs]
                return f"{statistics.mean(xs):.3f} (n={len(xs)})" if xs else "—"
            uq = statistics.mean(len(r.get("user_reply_indices", r.get("messages", [])) or []) if "score" not in r else (r["score"].get("user_questions", 0)) for r in recs)
            out.append(f"| {label} | {r_(B4)} | {r_(BEH)} | {r_(('self_contradictory',))} | {uq:.2f} |")
            continue
        if run is None:
            out.append(f"| {label} | (pending) | | | |")
            continue
        out.append(f"| {label} | {rate(run, k4)} | {rate(run, allk)} | {rate(run, ksc)} | {asks(run)} |")
    out.append("\n## HoosierHelp — related-work style baselines (same Qwen3-8B, same simulated user, same turn cap; errors count as failures)\n")
    out.append("| system | 4 behaviors (192) | all 5 (240) | self-contradictory (48) | questions to person (avg) |\n|---|---|---|---|---|")
    for label, path in (("single agent, official prompt, direct", "eval/hoosier_baselines/baseline_single_direct"),
                        ("single agent, official prompt, step-by-step (thinking)", "eval/hoosier_baselines/baseline_single_cot"),
                        ("plain role chain (one greedy handoff)", "eval/hoosier_baselines/baseline_plain_mas"),
                        ("9-way majority-vote handoff", "eval/hoosier_baselines/baseline_vote_mas"),
                        ("confidence handoff (uncertainty propagation)", "eval/hoosier_baselines/baseline_conf_mas"),
                        ("**ours: v5 text controller**", "eval/final/hoosier_text_mas_b5"),
                        ("**ours: v5 + learned rich-text controller**", "eval/final/hoosier_richtext_mc_b5")):
        run = load(path)
        if run is None:
            out.append(f"| {label} | (pending) | | | |")
            continue
        ks = sorted(run)
        k4_ = [k for k in ks if not k.endswith("self_contradictory")]
        ksc_ = [k for k in ks if k.endswith("self_contradictory")]
        uq = statistics.mean((r.get("score") or {}).get("user_questions", 0) for r in run.values())
        out.append(f"| {label} | {rate(run, k4_)} | {rate(run, ks)} | {rate(run, ksc_)} | {uq:.2f} |")
    out.append("\n## HoosierHelp — clarification controllers on the v5 workflow (paired vs. v5 text controller)\n")
    out.append("| controller | 4 behaviors | Δ vs no-ask [95% CI] | all 5 | Δ all | controller asks (total) | questions to person / episode |\n|---|---|---|---|---|---|---|")
    for label, path in (("no agent-to-agent asks (v5 text)", "eval/final/hoosier_text_mas_b5"),
                        ("ask whenever candidates disagree", "eval/final/hoosier_disagreement_ask_b5"),
                        ("max-uncertainty", "eval/final/hoosier_max_uncertainty_b5"),
                        ("fixed field order", "eval/final/hoosier_fixed_order_b5"),
                        ("decision-relevance VOI (deterministic)", "eval/final/hoosier_voi_det_b5"),
                        ("learned rich-text controller (v5 labels)", "eval/final/hoosier_richtext_mc_b5"),
                        ("  ablation: no dependency features", "eval/final/hoosier_richtext_mc_no_dependency_b5"),
                        ("  ablation: no receiver state", "eval/final/hoosier_richtext_mc_no_receiver_state_b5")):
        run = load(path)
        if run is None or v5 is None:
            out.append(f"| {label} | (pending) | | | | | |")
            continue
        n_asks = sum(1 for r in run.values() for e in (r.get("trace") or []) if e.get("kind") == "clarification" and "user_reply" in e)
        out.append(f"| {label} | {rate(run, k4)} | {paired(run, v5, k4)} | {rate(run, allk)} | {paired(run, v5, allk)} | {n_asks} | {asks(run)} |")
    # ---------------- v6: budget-aware question menu. Table A: free budget (cap 10) -- success and questions asked.
    # Table B: fixed budgets (caps 8/7/6) -- success. Three baselines + ours (learned, lam chosen on offline validation).
    def cell(path, with_asks=False):
        run = load(path)
        if run is None:
            return "(pending)"
        ks = sorted(run)
        acc = statistics.mean(em(run[k]) for k in ks)
        if not with_asks:
            return f"{acc:.3f}"
        uq = statistics.mean((run[k].get("score") or {}).get("user_questions", 0) for k in ks)
        return f"{acc:.3f} / {uq:.2f}"
    lam_file = RUNS / "eval/analysis_lam/chosen_lam.txt"
    lam_txt = lam_file.read_text().strip() if lam_file.exists() else "?"
    lam_ord = (RUNS / "eval/analysis_lam_ord/chosen_lam.txt").read_text().strip() if (RUNS / "eval/analysis_lam_ord/chosen_lam.txt").exists() else "?"
    beta_txt = (RUNS / "eval/analysis_beta/chosen_beta.txt").read_text().strip() if (RUNS / "eval/analysis_beta/chosen_beta.txt").exists() else "?"
    ROWS = (("ask on every open question (disagreement-first order)", "hoosier_disagreement_ask_b5"),
            ("fixed field order, ask everything", "hoosier_fixed_order_b5"),
            ("budget-aware value of information (train priors, no learning)", "hoosier_voi_det_b5"),
            (f"ours: learned rich-text controller, free question choice (λ={lam_txt})", "LEARNED"),
            (f"ours: learned stop rule, fixed question order (λ={lam_ord})", "LEARNED_ORD"),
            (f"**ours: learned + VOI hybrid (β={beta_txt}, λ={lam_txt})**", "LEARNED_VOI"))
    LDIR = {"LEARNED": "lam", "LEARNED_ORD": "ord", "LEARNED_VOI": "voi"}
    out.append("\n## HoosierHelp v6 — Table A: free interaction budget (10-turn cap). success / questions to the person per episode\n")
    out.append("| controller | success / questions |\n|---|---|")
    for label, name in ROWS:
        path = f"eval/v6/{LDIR[name]}/turns10/hoosier_richtext_mc_b5" if name in LDIR else f"eval/v6/turns10/{name}"
        out.append(f"| {label} | {cell(path, True)} |")
    out.append("\n## HoosierHelp v6 — Table B: fixed interaction budgets. success (questions per episode)\n")
    out.append("| controller | 8 turns (native) | 7 turns | 6 turns |\n|---|---|---|---|")
    for label, name in ROWS:
        cells = []
        for T in (8, 7, 6):
            if name in LDIR:
                path = f"eval/v6/{LDIR[name]}/turns{T}/hoosier_richtext_mc_b5"
            else:
                path = f"eval/v6/{name}" if T == 8 else f"eval/v6/turns{T}/{name}"
            cells.append(cell(path, True))
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("\nSupplementary v6 rows (8 turns): max-uncertainty " + cell("eval/v6/hoosier_max_uncertainty_b5", True) +
               "; never ask " + cell("eval/v6/hoosier_text_mas_b5", True) +
               "; learned at default λ=0.02 " + cell("eval/v6/hoosier_richtext_mc_b5", True) +
               "; learned without dependency features " + cell("eval/v6/hoosier_richtext_no_dependency_mc_b5", True) +
               "; learned without receiver state " + cell("eval/v6/hoosier_richtext_no_receiver_state_mc_b5", True) + ".")
    # ---------------- multi-seed robustness (same 240 episodes, sampling seeds 0/1/2)
    out.append("\n## HoosierHelp — sampling-seed robustness (all 5 behaviors, 240 episodes per run; mean ± std over seeds)\n")
    out.append("| system | seed 0 | seed 1 | seed 2 | mean ± std |\n|---|---|---|---|---|")
    for label, name in (("v5 text controller (no asks)", "hoosier_text_mas_b5"), ("v5 without fallback clarification", "hoosier_text_mas_nofb_b5"),
                        ("v5 + decision-relevance VOI", "hoosier_voi_det_b5"), ("v5 + learned rich-text controller", "hoosier_richtext_mc_b5")):
        vals = []
        for sd in (0, 1, 2):
            run = load(f"eval/final/{name}" if sd == 0 else f"eval/final/seed{sd}/{name}")
            vals.append(statistics.mean(em(r) for r in run.values()) if run else None)
        got = [v for v in vals if v is not None]
        cells = " | ".join(f"{v:.3f}" if v is not None else "(pending)" for v in vals)
        ms = f"{statistics.mean(got):.3f} ± {(statistics.stdev(got) if len(got) > 1 else 0.0):.3f} (n={len(got)})" if got else "—"
        out.append(f"| {label} | {cells} | {ms} |")
    if v5:
        out.append("\n## HoosierHelp v5 — by behavior and by case (text controller)\n")
        byb = collections.defaultdict(list); byc = collections.defaultdict(list)
        for k, r in v5.items():
            byb[r["behavior"]].append(em(r)); byc[f"{r['case_type']}|{r['constraint_profile']}"].append(em(r))
        out.append("| behavior | success | | case | success |\n|---|---|---|---|---|")
        rows_b = [(b, statistics.mean(v), len(v)) for b, v in byb.items()]
        rows_c = [(c, statistics.mean(v), len(v)) for c, v in sorted(byc.items())]
        for i in range(max(len(rows_b), len(rows_c))):
            b = rows_b[i] if i < len(rows_b) else ("", "", "")
            c = rows_c[i] if i < len(rows_c) else ("", "", "")
            fb = f"{b[1]:.3f} (n={b[2]})" if b[0] else ""
            fc = f"{c[1]:.3f} (n={c[2]})" if c[0] else ""
            out.append(f"| {b[0]} | {fb} | | {c[0]} | {fc} |")
    # ---------------- NGQA
    out.append("\n## NGQA (native knowledge-QA, 300 test items, Qwen3-8B)\n")
    out.append("| system | binary acc | sparse / standard / complex | tag F1 | explain ROUGE-L |\n|---|---|---|---|---|")
    for label, path in (("single agent, direct answer", "eval/ngqa_full/baseline_single_direct"), ("single agent, step-by-step", "eval/ngqa_full/baseline_single_cot"),
                        ("plain role chain (one greedy handoff)", "eval/ngqa_full/baseline_plain_mas"), ("5-way majority vote", "eval/ngqa_full/baseline_vote_mas"),
                        ("confidence handoff (uncertainty propagation)", "eval/ngqa_full/baseline_conf_mas"), ("**ours (candidate-indexed handoff + guideline analysis)**", "eval/ngqa_full/text_mas")):
        sp = RUNS / path / "summary.json"
        if not sp.exists():
            out.append(f"| {label} | (pending) | | | |")
            continue
        s = read_json(sp)
        recs = read_jsonl(RUNS / path / "records.jsonl")
        rl = statistics.mean(r["score"].get("explain_rouge_l", 0.0) for r in recs)
        bd = s.get("by_difficulty", {})
        out.append(f"| {label} | {s['end_to_end_match']:.3f} | {bd.get('easy', 0):.2f} / {bd.get('medium', 0):.2f} / {bd.get('hard', 0):.2f} | {s.get('tag_f1', 0):.3f} | {rl:.3f} |")
    out.append("\nOriginal NGQA paper, best reported configuration (GPT-4o-mini + ToG): standard 0.86, complex 0.53.")
    # ---------------- cost accounting (reviewer chain 3): offline labels vs online VOI vs no-ask, per-episode inference
    out.append("\n## Cost accounting (HoosierHelp)\n")
    out.append("| item | wall time | generation calls / completion tokens | simulated-user spend |\n|---|---|---|---|")
    for label, path in (("v5 label collection (200 roots, 1 seed, ~2 prefixes/root)", "collect/hoosier_v5_repr"),
                        ("v5 calibration labels (30 roots)", "collect/hoosier_v5_cal"),
                        ("v6 label collection (200 roots)", "collect/hoosier_v6_repr"), ("v6 calibration labels (30 roots)", "collect/hoosier_v6_cal")):
        sp = RUNS / path / "summary.json"
        if not sp.exists():
            out.append(f"| {label} | (pending) | | |")
            continue
        c = read_json(sp)
        g = c.get("gen_usage", {})
        out.append(f"| {label} | {c.get('wall_seconds', 0) / 60:.1f} min | {g.get('calls', 0)} / {g.get('completion_tokens', 0) / 1e6:.2f}M | ${c.get('openai_usd', 0):.2f} |")
    for label, path in (("v5 controller training (A + C phases)", "models/hoosier5_A_richtext"), ("v6 controller training", "models/hoosier6_A_richtext")):
        hp = RUNS / path / "history.json"
        if hp.exists():
            h = read_json(hp)
            secs = sum(e.get("seconds", 0.0) for e in h.get("history", []))
            out.append(f"| {label} | {secs:.0f} s (GPU, tiny model) | — | — |")
    out.append("\n| controller (v5 / v6 eval, 240 episodes) | wall per episode | completion tokens per episode | user spend per episode | questions to person |\n|---|---|---|---|---|")
    for label, path in (("v5 text (no asks)", "eval/final/hoosier_text_mas_b5"), ("v5 VOI (online search signatures)", "eval/final/hoosier_voi_det_b5"),
                        ("v5 learned rich-text", "eval/final/hoosier_richtext_mc_b5"), ("v6 fixed order", "eval/v6/hoosier_fixed_order_b5"),
                        ("v6 VOI (priors)", "eval/v6/hoosier_voi_det_b5"), ("v6 learned rich-text", "eval/v6/hoosier_richtext_mc_b5")):
        sp = RUNS / path / "summary.json"
        if not sp.exists():
            out.append(f"| {label} | (pending) | | | |")
            continue
        c = read_json(sp)
        n = max(1, c.get("n", 1)); g = c.get("gen_usage", {})
        out.append(f"| {label} | {c.get('wall_seconds', 0) / n:.1f} s | {g.get('completion_tokens', 0) / n:.0f} | ${c.get('openai_usd_this_run', 0) / n:.4f} | {c.get('user_questions_avg', 0):.2f} |")
    # ---------------- spend
    led = RUNS / "ledger_validation.json"
    if led.exists():
        L = read_json(led)
        out.append(f"\n## Spend\n\nSimulated-user API spend to date: ${L['usd']:.2f} over {L['calls']} calls (gpt-4.1 until 2026-10-19 05:30, gpt-4.1-mini after; replies cached by transcript prefix).")
    text = "\n".join(out) + "\n"
    o = RUNS / a.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
