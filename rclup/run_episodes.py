"""Run closed-loop episodes for a controller on a slice of roots and save records.

Examples:
  python -m rclup.run_episodes --dataset hoosier --split test --controller text_mas --limit 8 --out runs/smoke_hoosier
  python -m rclup.run_episodes --dataset ngqa --split test --controller pi0 --limit 8 --out runs/smoke_ngqa
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import time
import traceback
from pathlib import Path

from .backbone import VLLMClient
from .common import DATA, RUNS, append_jsonl, read_jsonl, write_json
from .cost import BudgetExceeded, CostLedger
from .workflow import (DisagreementController, FixedOrderController, GreedyController, HoosierWorkflow, MaxUncertaintyController,
                       NGQAWorkflow, Pi0Controller, VOIController)

CONTROLLERS = {
    "text_mas": lambda seed: GreedyController(),
    "pi0": lambda seed: Pi0Controller(seed),
    "disagreement_ask": lambda seed: DisagreementController(),
    "max_uncertainty": lambda seed: MaxUncertaintyController(),
    "fixed_order": lambda seed: FixedOrderController(),
    "voi_det": lambda seed: VOIController(),
}
LEARNED = ("rclup", "rich_text_mc", "generic_latent_mc", "rclup_q_ref", "scalar_mc")


def build_learned(name: str, carrier_path: str, calibrator_path: str | None, lam: float, device: str = "cuda"):
    """Learned controllers share the carrier architecture; variants differ in which latent content they consume."""
    import torch
    from .backbone import HiddenEncoder
    from .carrier import Carrier, Calibrator
    from .features import CarrierRuntime, RCLUPController
    from .train import D_BACKBONE
    from .carrier import ScalarControl
    carrier = ScalarControl() if name == "scalar_mc" else Carrier(D_BACKBONE)
    carrier.load_state_dict(torch.load(carrier_path, map_location="cpu"))
    encoder = HiddenEncoder()
    rt = CarrierRuntime(carrier, encoder, device=device, generic_latent=(name == "generic_latent_mc"))
    rt.no_latent = name == "rich_text_mc"
    cal = None
    if calibrator_path:
        obj = torch.load(calibrator_path, map_location="cpu")
        cal = Calibrator(obj["groups"], obj.get("l2", 1.0))
        cal.load_state_dict(obj["state_dict"])
        cal.to(device)
    head = "q_ref" if name == "rclup_q_ref" else "q_dep"
    return RCLUPController(rt, cal, lam=lam, head=head, name=name), rt, encoder


def load_roots(dataset: str, split: str, limit: int | None, seed: int = 0, difficulty: str | None = None,
               behaviors: tuple[str, ...] = ("normal",), min_statuses: int = 0) -> list[dict]:
    if dataset == "hoosier":
        rows = read_jsonl(DATA / "hoosier" / ("eval_specs.jsonl" if split == "test" else "train_specs.jsonl"))
        rows = [r for r in rows if r["split"] == split]
        rows = [dict(r, behavior=b) for r in rows for b in behaviors]
    else:
        rows = [r for r in read_jsonl(DATA / "ngqa" / "roots.jsonl") if r["split"] == split]
        if difficulty:
            rows = [r for r in rows if r["difficulty"] == difficulty]
        if min_statuses:
            rows = [r for r in rows if len(r["private_profile"]["statuses"]) >= min_statuses]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def make_env(dataset: str, root: dict, ledger: CostLedger, disclosure: str = "half", max_turns: int | None = None):
    if dataset == "hoosier":
        from .envs.hoosier_env import HoosierEnv, MAX_AGENT_TURNS
        return HoosierEnv(root, root.get("behavior", "normal"), ledger, max_agent_turns=max_turns or MAX_AGENT_TURNS)
    from .envs.ngqa_env import NGQAEnv
    return NGQAEnv(root, ledger, disclosure=disclosure)


def summarize(records: list[dict], dataset: str) -> dict:
    ok = [r for r in records if "error" not in r]
    out = {"n": len(records), "errors": len(records) - len(ok)}
    if not ok:
        return out
    out["end_to_end_match"] = statistics.mean(r["score"]["end_to_end_match"] for r in ok)
    out["user_questions_avg"] = statistics.mean(r["score"]["user_questions"] for r in ok)
    out["items_used_avg"] = statistics.mean(r.get("items_used", 0) for r in ok)
    if dataset == "hoosier":
        out["tool_all_match"] = statistics.mean(bool(r["score"]["all_match"]) for r in ok)
        out["resource_exact_match"] = statistics.mean(bool(r["score"]["resource_exact_match"]) for r in ok)
        out["explicit_final"] = statistics.mean(bool(r["score"]["explicit_final_action"]) for r in ok)
        out["agent_turns_avg"] = statistics.mean(r["score"]["agent_turns"] for r in ok)
        by = collections.defaultdict(list)
        for r in ok:
            by[f"{r['case_type']}|{r['constraint_profile']}"].append(r["score"]["end_to_end_match"])
        out["by_case"] = {k: statistics.mean(v) for k, v in by.items()}
    else:
        out["binary_valid"] = statistics.mean(r["score"]["binary_valid"] for r in ok)
        out["tag_f1"] = statistics.mean(r["score"]["tag_f1"] for r in ok)
        by = collections.defaultdict(list)
        for r in ok:
            by[r["difficulty"]].append(r["score"]["end_to_end_match"])
        out["by_difficulty"] = {k: statistics.mean(v) for k, v in by.items()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hoosier", "ngqa"], required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--controller", choices=sorted(CONTROLLERS) + list(LEARNED), default="text_mas")
    p.add_argument("--carrier", default=None)
    p.add_argument("--calibrator", default=None)
    p.add_argument("--lam", type=float, default=0.02)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--items-budget", type=int, default=2)
    p.add_argument("--difficulty", default=None)
    p.add_argument("--behaviors", default="normal")
    p.add_argument("--disclosure", default="half", choices=["half", "none", "full"])
    p.add_argument("--min-statuses", type=int, default=0)
    p.add_argument("--variant", default="v4", choices=["v1", "v2", "v3", "v4", "v5", "v5nofb", "v6"], help="HoosierHelp module ablation preset")
    p.add_argument("--ablate", default="", help="comma list of controller-input ablations (must match training): dependency, receiver_state")
    p.add_argument("--max-turns", type=int, default=None, help="HoosierHelp agent-turn cap (native 8); budget sweep")
    p.add_argument("--intake-policy", default="three", choices=["three", "two", "skip_docs", "atomic"],
                   help="ordinary intake questions: three low-burden groups (native), two messages (location; rest), or three groups minus documents/eligibility")
    p.add_argument("--out", required=True)
    p.add_argument("--ledger", default=str(RUNS / "ledger_validation.json"))
    p.add_argument("--cap-usd", type=float, default=40.0)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    out = Path(a.out) if Path(a.out).is_absolute() else RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    ledger = CostLedger(Path(a.ledger), a.cap_usd, label="validation")
    bb = VLLMClient()
    learned, runtime, encoder = None, None, None
    if a.ablate:
        from . import features as _F
        _F.ABLATION.update(x for x in a.ablate.split(",") if x)
    if a.controller in LEARNED:
        learned, runtime, encoder = build_learned(a.controller, a.carrier, a.calibrator, a.lam)
        if a.ablate:
            learned.name = f"{a.controller}_no_{a.ablate.replace(',', '_')}"
    roots = load_roots(a.dataset, a.split, a.limit, a.seed, a.difficulty, tuple(a.behaviors.split(",")), a.min_statuses)
    WF = HoosierWorkflow if a.dataset == "hoosier" else NGQAWorkflow
    records_path = out / "records.jsonl"
    done = {r["root_id"] + "|" + r.get("behavior", "") for r in read_jsonl(records_path)} if records_path.exists() else set()
    start = time.monotonic()
    usd0 = ledger.usd

    def one(root):
        key = root["root_id"] + "|" + root.get("behavior", "")
        if key in done:
            return None
        env = make_env(a.dataset, root, ledger, a.disclosure, a.max_turns)
        if learned is not None:
            wf = WF(bb, learned, seed=a.seed, items_budget=a.items_budget, encoder=encoder, carrier=runtime)
        else:
            wf = WF(bb, CONTROLLERS[a.controller](a.seed), seed=a.seed, items_budget=a.items_budget)
        if hasattr(wf, "set_variant"):
            wf.set_variant(a.variant)
            if a.intake_policy == "two":
                wf.intake_groups = (("location",), ("schedule", "intake_methods", "available_documents", "eligibility"))
            elif a.intake_policy == "atomic":   # one field per question: questions are comparable across controllers
                wf.intake_groups = tuple((f,) for f in ("location", "schedule", "intake_methods", "available_documents", "eligibility"))
            elif a.intake_policy == "skip_docs":
                wf.skip_fields = ("available_documents", "eligibility")
        try:
            rec = wf.run(env)
        except BudgetExceeded:
            raise
        except Exception as e:  # keep failures as records (unresolved = failure)
            rec = env.record() if env.messages else {"root_id": root["root_id"], "score": {"end_to_end_match": False, "user_questions": 0}}
            rec.update({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:], "controller": a.controller,
                        "trace": wf.trace.events})
        rec["behavior"] = root.get("behavior", "")
        rec["gen_usage"] = bb.usage_dict()
        return rec

    from concurrent.futures import ThreadPoolExecutor, as_completed
    records = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(one, r) for r in roots]
        for f in as_completed(futures):
            try:
                rec = f.result()
            except BudgetExceeded as e:
                print("STOP:", e)
                break
            if rec is None:
                continue
            records.append(rec)
            append_jsonl(records_path, rec)
            s = rec["score"]
            print(f"[{len(records)}/{len(roots)}] {rec['root_id']} ok={s.get('end_to_end_match')} uq={s.get('user_questions')} "
                  f"items={rec.get('items_used')} err={'error' in rec} ${ledger.usd - usd0:.3f}", flush=True)
    all_records = read_jsonl(records_path)
    summary = summarize(all_records, a.dataset)
    summary.update({"controller": a.controller, "dataset": a.dataset, "split": a.split, "items_budget": a.items_budget, "lam": a.lam, "disclosure": a.disclosure, "variant": a.variant,
                    "max_turns": a.max_turns, "intake_policy": a.intake_policy,
                    "carrier": a.carrier, "calibrator": a.calibrator,
                    "wall_seconds": time.monotonic() - start, "openai_usd_this_run": ledger.usd - usd0,
                    "openai_usd_ledger_total": ledger.usd, "gen_usage": bb.usage_dict()})
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
