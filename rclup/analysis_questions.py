"""Offline question-selection analysis on collected branch labels (no new API calls).

For every collected prefix we know the mean final-error of committing each receiver candidate and of asking each
menu question (then continuing with pi_0). This script asks: which questions are actually worth asking, and which
selection rule picks them? Rules: never ask, disagreement-first, max-uncertainty, fixed field order, deterministic
decision-relevance VOI, and the learned heads. Reports oracle regret, ask precision/recall, and a breakdown by
question category (disagreement that changes the outcome vs. one that does not).

  python -m rclup.analysis_questions --collect collect/hoosier_repr_full --dataset hoosier \
      --carriers richtext=models/hoosierF_A_richtext/carrier.pt,latent=models/hoosierF_A/carrier.pt --out eval/analysis_questions_hoosier
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import torch

from .common import RUNS, write_json
from .cost import CostLedger


def oracle_stats(rec: dict) -> dict:
    yc, ya = rec["y_commit"], rec["y_ask"]
    best_commit = min(yc.values())
    gains = {q: best_commit - e for q, e in ya.items()}
    return {"best_commit": best_commit, "greedy": yc.get(rec["data"]["receiver"]["greedy_index"], best_commit),
            "gains": gains, "oracle": min([best_commit] + list(ya.values()))}


def rule_pick(rule: str, rec: dict, aux: dict, scores: dict | None) -> tuple[str, str | None]:
    """Return ('commit', None) or ('ask', qid) for the rule at this prefix (budget: items_left of the context)."""
    data = rec["data"]
    menu = data["menu"]
    if data["receiver"]["items_left"] <= 0 or not menu:
        return ("commit", None)
    pk = data["packets"][0]
    weights = {c["key"]: c["weight"] for c in pk["candidates"]}
    fv, sig = aux.get("field_value", {}), aux.get("signature", {})
    greedy_key = next((c["key"] for c in pk["candidates"] if c["greedy"]), pk["candidates"][0]["key"])

    def masses(q):
        wa = weights.get(q["a"], 0.0)
        if q["b"] == "__residual__":
            wb = pk["residual_weight"] + sum(w for k, w in weights.items() if k != q["a"] and fv.get((k, q["qid"])) == "unknown")
        else:
            wb = weights.get(q["b"], 0.0)
        return wa, wb

    if rule == "never":
        return ("commit", None)
    if rule == "disagreement_first":
        return ("ask", menu[0]["qid"])
    if rule == "max_uncertainty":
        return ("ask", max(menu, key=lambda q: min(masses(q)))["qid"])
    if rule == "fixed_order":
        order = aux.get("field_order", {})
        return ("ask", min(menu, key=lambda q: order.get(q["qid"], 10 ** 6))["qid"])
    if rule == "voi_det":
        vals = []
        for q in menu:
            g_sig, g_val = sig.get(greedy_key), fv.get((greedy_key, q["qid"]))
            v = sum(w for k, w in weights.items() if k != greedy_key and fv.get((k, q["qid"])) != g_val and sig.get(k) != g_sig)
            if g_val == "unknown":
                v += pk["residual_weight"]
            vals.append(v)
        if max(vals) > 0:
            return ("ask", menu[max(range(len(vals)), key=lambda i: vals[i])]["qid"])
        return ("commit", None)
    if rule.startswith("learned:"):
        # Eq. (8)/(10) with raw sigmoid risks: ask the max positive gain question, else commit b*
        lam = 0.02
        r_commit = torch.sigmoid(scores["commit"])
        b_star = int(torch.argmin(scores["commit"]))
        best_q, best_gain = None, 0.0
        if scores["ask"] is not None:
            for q, r in zip(menu, torch.sigmoid(scores["ask"]).tolist()):
                g = float(r_commit[b_star]) - r - lam
                if g > best_gain:
                    best_q, best_gain = q["qid"], g
        return ("ask", best_q) if best_q else ("commit", None)
    raise ValueError(rule)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collect", required=True)
    p.add_argument("--dataset", choices=["hoosier", "ngqa"], required=True)
    p.add_argument("--carriers", default="", help="name=path[,name=path] learned carriers to score (richtext uses --no-latent)")
    p.add_argument("--ablate", default="", help="name:flag[,..] ablation flags per carrier name, e.g. nodep:dependency")
    p.add_argument("--val-only", action="store_true", help="restrict to the training-time validation split (seed 13, 20%)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from . import features as F
    from .train import load_prefixes, forward_prefix
    from .carrier import Carrier
    d = Path(a.collect) if Path(a.collect).is_absolute() else RUNS / a.collect
    paths = sorted(str(x) for x in (d / "prefixes").glob("*.pt"))
    recs = load_prefixes(paths)
    if a.val_only:  # same root-level split as rclup.train (seed 13, 20% of roots)
        import random
        rng = random.Random(13)
        roots = sorted({r["root_id"] for r in recs})
        rng.shuffle(roots)
        val_roots = set(roots[: max(1, int(len(roots) * 0.2))])
        recs = [r for r in recs if r["root_id"] in val_roots]
    ledger = CostLedger(RUNS / "ledger_analysis.json", cap_usd=1.0)
    # deterministic side information per prefix (same code path as the online controllers)
    aux_all = []
    if a.dataset == "hoosier":
        from .envs.hoosier_env import HoosierEnv
        from .workflow import HoosierWorkflow
        for rec in recs:
            env = HoosierEnv.from_state(rec["root"], rec["root"].get("behavior", "normal"), ledger, rec["env_state"])
            inv, menu = rec["snapshot"]["inv"], rec["snapshot"]["menu"]
            aux_all.append(HoosierWorkflow.planner_aux(None, env, inv.packet, menu))
    else:
        from .envs.ngqa_env import NGQAEnv
        from .workflow import NGQAWorkflow
        for rec in recs:
            env = NGQAEnv(rec["root"], ledger, disclosure=rec.get("disclosure", "half"))
            inv, menu = rec["snapshot"]["inv"], rec["snapshot"]["menu"]
            aux_all.append(NGQAWorkflow.reasoner_aux(None, env, inv.packet, menu))
    # learned scores
    learned = {}
    abl = dict(x.split(":") for x in a.ablate.split(",") if x)
    for spec in [x for x in a.carriers.split(",") if x]:
        name, path = spec.split("=")
        F.ABLATION.clear()
        F.ABLATION.update(abl.get(name, "").split("+") if abl.get(name) else [])
        carrier = Carrier(4096)
        carrier.load_state_dict(torch.load(path if Path(path).is_absolute() else RUNS / path, map_location="cpu"))
        carrier.eval()
        out = []
        with torch.no_grad():
            for rec in recs:
                c, q, _ = forward_prefix(carrier, None, rec, no_latent=("latent" not in name), head="q_dep")
                out.append({"commit": c.detach(), "ask": None if q is None else q.detach()})
        learned[name] = out
    F.ABLATION.clear()
    rules = ["never", "disagreement_first", "max_uncertainty", "fixed_order", "voi_det"] + [f"learned:{n}" for n in learned]
    # question categories from labels: for each menu question, does asking beat the best commit?
    cat = collections.Counter()
    per_rule = {r: collections.defaultdict(list) for r in rules}
    for i, rec in enumerate(recs):
        st = oracle_stats(rec)
        menu = rec["data"]["menu"]
        useful = {q: g > 0 for q, g in st["gains"].items()}
        any_useful = any(useful.values())
        for q in menu:
            cat[("useful" if useful.get(q["qid"]) else "useless", "disagreement" if q["b"] != "__residual__" else "unknown/confirm")] += 1
        for r in rules:
            kind, qid = rule_pick(r, rec, aux_all[i], learned[r.split(":", 1)[1]][i] if r.startswith("learned:") else None)
            y = rec["y_ask"].get(qid, st["greedy"]) if kind == "ask" else st["greedy"]
            s = per_rule[r]
            s["regret"].append(y - st["oracle"])
            s["error"].append(y)
            s["asked"].append(kind == "ask")
            if kind == "ask":
                s["ask_useful"].append(bool(useful.get(qid)))
            if any_useful:
                s["recall"].append(kind == "ask" and bool(useful.get(qid)))
    table = []
    for r in rules:
        s = per_rule[r]
        table.append({"rule": r, "n": len(s["error"]), "mean_error": statistics.mean(s["error"]), "regret": statistics.mean(s["regret"]),
                      "ask_rate": statistics.mean(s["asked"]), "ask_precision": statistics.mean(s["ask_useful"]) if s["ask_useful"] else None,
                      "useful_recall": statistics.mean(s["recall"]) if s["recall"] else None})
    out = Path(a.out) if Path(a.out).is_absolute() else RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "table.json", {"rules": table, "question_categories": {f"{k[0]}|{k[1]}": v for k, v in cat.items()},
                                    "n_prefixes": len(recs), "oracle_error": statistics.mean(oracle_stats(r)["oracle"] for r in recs),
                                    "never_error": statistics.mean(oracle_stats(r)["greedy"] for r in recs)})
    cols = ["rule", "n", "mean_error", "regret", "ask_rate", "ask_precision", "useful_recall"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in table:
        md.append("| " + " | ".join("" if row[c] is None else (f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c])) for c in cols) + " |")
    md.append(f"\nquestion categories: {dict(cat)}; oracle error {statistics.mean(oracle_stats(r)['oracle'] for r in recs):.3f}")
    (out / "table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
