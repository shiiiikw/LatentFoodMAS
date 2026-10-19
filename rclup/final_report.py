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
    recs = [r for r in read_jsonl(d / "records.jsonl") if "error" not in r]
    def key(r):
        rid = r.get("root_id") or f"hoosier:eval:{r.get('user_spec_id', '')}"
        return rid + "|" + (r.get("behavior") or r.get("user_behavior") or "")
    return {key(r): r for r in recs}


def em(r) -> float:
    return float(bool(r["score"]["end_to_end_match"]))


def rate(run: dict, keys) -> str:
    ks = [k for k in keys if k in run]
    return f"{statistics.mean(em(run[k]) for k in ks):.3f} (n={len(ks)})" if ks else "—"


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
    out.append("| workflow | 4 behaviors (192) | all 5 (240) | self-contradictory (48) |\n|---|---|---|---|")
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
                xs = [float(bool(r["score"].get("end_to_end_match"))) for r in recs if r["user_behavior"] in bs]
                return f"{statistics.mean(xs):.3f} (n={len(xs)})" if xs else "—"
            out.append(f"| {label} | {r_(B4)} | {r_(BEH)} | {r_(('self_contradictory',))} |")
            continue
        if run is None:
            out.append(f"| {label} | (pending) | | |")
            continue
        out.append(f"| {label} | {rate(run, k4)} | {rate(run, allk)} | {rate(run, ksc)} |")
    out.append("\n## HoosierHelp — clarification controllers on the v5 workflow (paired vs. v5 text controller)\n")
    out.append("| controller | 4 behaviors | Δ vs no-ask [95% CI] | all 5 | Δ all | asks to person |\n|---|---|---|---|---|---|")
    for label, path in (("no agent-to-agent asks (v5 text)", "eval/final/hoosier_text_mas_b5"),
                        ("ask whenever candidates disagree", "eval/final/hoosier_disagreement_ask_b5"),
                        ("max-uncertainty", "eval/final/hoosier_max_uncertainty_b5"),
                        ("fixed field order", "eval/final/hoosier_fixed_order_b5"),
                        ("decision-relevance VOI (deterministic)", "eval/final/hoosier_voi_det_b5"),
                        ("learned rich-text controller (v5 labels)", "eval/final/hoosier_richtext_mc_b5")):
        run = load(path)
        if run is None or v5 is None:
            out.append(f"| {label} | (pending) | | | | |")
            continue
        asks = sum(1 for r in run.values() for e in r.get("trace", []) if e.get("kind") == "clarification" and "user_reply" in e)
        out.append(f"| {label} | {rate(run, k4)} | {paired(run, v5, k4)} | {rate(run, allk)} | {paired(run, v5, allk)} | {asks} |")
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
