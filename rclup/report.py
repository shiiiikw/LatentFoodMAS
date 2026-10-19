"""Aggregate closed-loop runs and training diagnostics into paper-style tables.

Usage:
  python -m rclup.report --runs baselines/v3/ngqa_test_text_mas_half,eval/v1/ngqa_rclup_half,... --reference baselines/v3/ngqa_test_text_mas_half --out eval/v1/report_ngqa_half
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

from .common import RUNS, read_json, read_jsonl, write_json


def load_run(path: str) -> dict:
    d = Path(path) if Path(path).is_absolute() else RUNS / path
    recs = read_jsonl(d / "records.jsonl")
    summary = read_json(d / "summary.json") if (d / "summary.json").exists() else {}
    return {"name": d.name, "path": str(d), "records": {r["root_id"] + "|" + r.get("behavior", ""): r for r in recs}, "summary": summary}


def bootstrap_ci(values: list[float], n: int = 10000, seed: int = 13) -> tuple[float, float]:
    import random
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n)])


def table_rows(runs: list[dict], reference: dict | None, dataset: str) -> list[dict]:
    rows = []
    for run in runs:
        recs = list(run["records"].values())
        ok = [r for r in recs if "error" not in r]
        em = [float(bool(r["score"]["end_to_end_match"])) for r in recs]
        row = {"method": run["name"], "n": len(recs), "errors": len(recs) - len(ok),
               "EM": statistics.mean(em) if em else float("nan"), "EM_ci": bootstrap_ci(em),
               "Q_items": statistics.mean(r.get("items_used", 0) for r in recs) if recs else 0.0,
               "user_questions": statistics.mean(r["score"].get("user_questions", 0) for r in recs) if recs else 0.0}
        if dataset == "hoosier":
            row["ToolEM"] = statistics.mean(float(bool(r["score"].get("all_match"))) for r in recs)
            row["ResourceEM"] = statistics.mean(float(bool(r["score"].get("resource_exact_match"))) for r in recs)
            row["agent_turns"] = statistics.mean(r["score"].get("agent_turns", 0) for r in recs)
            by = collections.defaultdict(list)
            for r in recs:
                by[f"{r.get('case_type')}|{r.get('constraint_profile')}"].append(float(bool(r["score"]["end_to_end_match"])))
            row["by_case"] = {k: statistics.mean(v) for k, v in sorted(by.items())}
        else:
            row["tag_f1"] = statistics.mean(r["score"].get("tag_f1", 0.0) for r in recs)
            by = collections.defaultdict(list)
            for r in recs:
                by[r.get("difficulty")].append(float(bool(r["score"]["end_to_end_match"])))
            row["by_difficulty"] = {k: statistics.mean(v) for k, v in sorted(by.items())}
        if reference is not None and run is not reference:
            rec_ref = reference["records"]
            paired = [(float(bool(r["score"]["end_to_end_match"])), float(bool(rec_ref[k]["score"]["end_to_end_match"])))
                      for k, r in run["records"].items() if k in rec_ref]
            if paired:
                row["recovery"] = sum(1 for a, b in paired if a == 1 and b == 0) / len(paired)
                row["regression"] = sum(1 for a, b in paired if a == 0 and b == 1) / len(paired)
                diffs = [a - b for a, b in paired]
                row["delta_EM"] = statistics.mean(diffs)
                row["delta_ci"] = bootstrap_ci(diffs)
                row["paired_n"] = len(paired)
        asks = [e for r in recs for e in r.get("trace", []) if e.get("kind") == "clarification"]
        row["clarifications"] = len(asks)
        row["clarifications_resolved_from_evidence"] = sum(1 for e in asks if e.get("resolved") and "user_reply" not in e)
        row["clarifications_asked_user"] = sum(1 for e in asks if "user_reply" in e)
        rows.append(row)
    return rows


def markdown(rows: list[dict], dataset: str) -> str:
    cols = ["method", "n", "EM", "EM_ci", "delta_EM", "delta_ci", "recovery", "regression", "Q_items", "user_questions",
            "clarifications", "clarifications_asked_user"] + (["ToolEM", "ResourceEM", "agent_turns"] if dataset == "hoosier" else ["tag_f1"])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            elif isinstance(v, tuple):
                v = f"[{v[0]:.3f}, {v[1]:.3f}]"
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    for r in rows:
        extra = r.get("by_case") or r.get("by_difficulty")
        if extra:
            out.append(f"\n{r['method']}: " + ", ".join(f"{k}={v:.2f}" for k, v in extra.items()))
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--reference", default=None)
    p.add_argument("--dataset", choices=["hoosier", "ngqa"], required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    runs = [load_run(r) for r in a.runs.split(",")]
    ref = next((r for r in runs if r["path"].endswith(a.reference.rstrip("/").split("/")[-1])), None) if a.reference else None
    rows = table_rows(runs, ref, a.dataset)
    out = Path(a.out) if Path(a.out).is_absolute() else RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "table.json", rows)
    md = markdown(rows, a.dataset)
    (out / "table.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Reporting-only calibration of the executed policy (Sec. 4.6, Phase D)
# ---------------------------------------------------------------------------
def final_risk(rec: dict) -> float | None:
    """Pre-action risk attached to the final committed artifact by the deployed controller (from the trace)."""
    last = None
    for e in rec.get("trace", []):
        if e.get("kind") == "decision" and e.get("decision") == "commit" and e.get("scores"):
            s = e["scores"]
            b = e.get("candidate")
            if b is not None and "commit_risk" in s and b < len(s["commit_risk"]):
                last = float(s["commit_risk"][b])
    return last


def fit_reporting_head(run: dict) -> dict:
    """Platt map r_final = sigmoid(a * logit(r) + b) fitted on completed trajectories of the frozen policy."""
    import torch
    xs, ys = [], []
    for r in run["records"].values():
        p = final_risk(r)
        if p is None:
            continue
        p = min(max(p, 1e-4), 1 - 1e-4)
        xs.append(math.log(p / (1 - p)))
        ys.append(0.0 if r["score"]["end_to_end_match"] else 1.0)
    if len(xs) < 5:
        return {"a": 1.0, "b": 0.0, "n": len(xs)}
    x, y = torch.tensor(xs), torch.tensor(ys)
    a, b = torch.ones(()), torch.zeros(())
    a.requires_grad_(True); b.requires_grad_(True)
    opt = torch.optim.LBFGS([a, b], max_iter=200)

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(a * x + b).clamp(1e-6, 1 - 1e-6)
        loss = -(y * p.log() + (1 - y) * (1 - p).log()).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return {"a": float(a), "b": float(b), "n": len(xs)}


def apply_reporting(run: dict, head: dict) -> dict:
    briers, nlls, raw = [], [], []
    for r in run["records"].values():
        p = final_risk(r)
        if p is None:
            continue
        y = 0.0 if r["score"]["end_to_end_match"] else 1.0
        p = min(max(p, 1e-4), 1 - 1e-4)
        q = 1 / (1 + math.exp(-(head["a"] * math.log(p / (1 - p)) + head["b"])))
        q = min(max(q, 1e-6), 1 - 1e-6)
        briers.append((q - y) ** 2); raw.append((p - y) ** 2)
        nlls.append(-(y * math.log(q) + (1 - y) * math.log(1 - q)))
    return {"n": len(briers), "brier_raw": statistics.mean(raw) if raw else None, "brier_cal": statistics.mean(briers) if briers else None,
            "nll_cal": statistics.mean(nlls) if nlls else None}
