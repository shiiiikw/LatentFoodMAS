"""Phase A label collection (App. B.1-B.2): saved prefixes + branch outcomes.

For each fitting root, a prefix policy (pi_0 or a round-robin menu policy,
alternating by root) runs the workflow; at up to `prefixes_per_root` receiver
decision points the state is forked and saved. Every admissible action at a
prefix is then executed with `seeds` branch seeds under the fixed reference
executor pi_0: commit(b) for each receiver candidate, ask(q) for each menu
question (reply obtained, state rebuilt, pi_0 continuation). Outcomes are the
native final-error event E (1 = final answer wrong/missing).

Must run where the HF encoder has a GPU (attach to the vLLM worker job with
`srun --jobid=<job> --overlap`).
"""
from __future__ import annotations

import argparse
import copy
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from .backbone import HiddenEncoder, VLLMClient
from .common import RUNS, append_jsonl, read_jsonl, write_json
from .cost import BudgetExceeded, CostLedger
from .features import TextCache, encode_context
from .run_episodes import load_roots, make_env
from .workflow import Decision, ForcedController, HoosierWorkflow, NGQAWorkflow, Pi0Controller, RoundRobinMenuController, Question


def outcome_error(rec: dict) -> int:
    return int(not bool(rec.get("score", {}).get("end_to_end_match", False)))


class Collector:
    def __init__(self, dataset: str, bb: VLLMClient, encoder: HiddenEncoder, ledger: CostLedger, out: Path,
                 seeds: tuple[int, ...], max_commits: int, max_questions: int, prefixes_per_root: int, items_budget: int, disclosure: str = "half"):
        self.dataset, self.bb, self.encoder, self.ledger, self.out = dataset, bb, encoder, ledger, out
        self.variant = "v5"
        self.disclosure = disclosure
        self.seeds, self.max_commits, self.max_questions = seeds, max_commits, max_questions
        self.prefixes_per_root, self.items_budget = prefixes_per_root, items_budget
        self.text = TextCache(encoder)
        self.WF = HoosierWorkflow if dataset == "hoosier" else NGQAWorkflow
        self.lock = threading.Lock()

    def _branch(self, env_prefix, wf_prefix, snapshot: dict, forced: Decision, seed: int) -> dict:
        env = env_prefix.fork()
        wf = wf_prefix.fork(ForcedController(forced, seed), seed=seed)
        resume = {"st": copy.deepcopy(snapshot["st"]), "menu": snapshot["menu"], "forced": forced, "inv": snapshot["inv"]}
        if self.dataset == "hoosier":
            resume["pinv"] = snapshot["pinv"]
        else:
            resume["rinv"] = snapshot["rinv"]
        try:
            rec = wf.run(env, resume=resume)
            return {"error": 0 if rec["score"].get("end_to_end_match") else 1, "score": rec["score"], "ok": True,
                    "user_questions": rec["score"].get("user_questions"), "items_used": rec.get("items_used"),
                    "n_events": len(rec.get("trace", []))}
        except BudgetExceeded:
            raise
        except Exception as e:
            return {"error": 1, "ok": False, "exception": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}

    def collect_root(self, root: dict, index: int) -> dict:
        seed = 1000 + index
        policy = Pi0Controller(seed) if index % 2 == 0 else RoundRobinMenuController(seed)
        prefixes: list[dict] = []
        env = make_env(self.dataset, root, self.ledger, self.disclosure, getattr(self, "max_turns", None))
        wf = self.WF(self.bb, policy, seed=seed, encoder=self.encoder, items_budget=self.items_budget)
        if hasattr(wf, "set_variant"):
            wf.set_variant(self.variant)

        def hook(wf_live, ctx, snapshot):
            if len(prefixes) >= self.prefixes_per_root:
                return
            data = encode_context(ctx, self.text, external_budget_frac=ctx.external_left / 8.0)
            prefixes.append({"env": env.fork(), "wf": wf_live.fork(Pi0Controller(seed)), "snapshot": snapshot, "data": data,
                             "ctx_candidates": [c for c in ctx.candidates], "menu": list(ctx.menu), "decision_index": len(prefixes)})

        wf.prefix_hook = hook
        prefix_rec = wf.run(env)
        results = []
        for p in prefixes:
            labels = {"commit": {}, "ask": {}}
            cands = p["ctx_candidates"][: self.max_commits]
            for b, c in enumerate(cands):
                labels["commit"][str(b)] = [self._branch(p["env"], p["wf"], p["snapshot"], Decision("commit", b), s) for s in self.seeds]
            for q in p["menu"][: self.max_questions]:
                labels["ask"][q.qid] = [self._branch(p["env"], p["wf"], p["snapshot"], Decision("ask", question=q), s) for s in self.seeds]
            snap = p["snapshot"]
            wf_state = {"versions": dict(p["wf"].versions), "items_used": p["wf"].items_used, "asked": set(p["wf"].asked),
                        "fallback_overrides": dict(getattr(p["wf"], "fallback_overrides", {}) or {}),
                        "last_fallback_request": getattr(p["wf"], "last_fallback_request", None), "seed": p["wf"].seed,
                        "decision_points": p["wf"].decision_points, "invocations": list(p["wf"].invocations)}
            rec = {"root_id": root["root_id"], "dataset": self.dataset, "policy": policy.name, "decision_index": p["decision_index"],
                   "data": p["data"], "labels": labels, "root": root, "env_state": p["env"].state(),
                   "snapshot": {k: v for k, v in snap.items() if k != "sender_invs"}, "wf_state": wf_state,
                   "receiver_candidates": [{"key": c.key, "text": c.text, "is_unresolved": c.is_unresolved, "greedy": c.greedy} for c in cands],
                   "menu": [q.__dict__ for q in p["menu"][: self.max_questions]], "prefix_score": prefix_rec["score"]}
            path = self.out / "prefixes" / f"{root['root_id'].replace(':', '_')}__{p['decision_index']}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(rec, path)
            summary = {"root_id": root["root_id"], "decision_index": p["decision_index"], "policy": policy.name, "path": str(path),
                       "n_commit": len(labels["commit"]), "n_ask": len(labels["ask"]),
                       "commit_err": {b: [x["error"] for x in v] for b, v in labels["commit"].items()},
                       "ask_err": {q: [x["error"] for x in v] for q, v in labels["ask"].items()},
                       "branch_failures": sum(not x["ok"] for v in list(labels["commit"].values()) + list(labels["ask"].values()) for x in v),
                       "prefix_e2e": prefix_rec["score"].get("end_to_end_match")}
            results.append(summary)
        return {"root_id": root["root_id"], "prefixes": results, "prefix_e2e": prefix_rec["score"].get("end_to_end_match"),
                "prefix_error": prefix_rec.get("error")}


class LearnedCommitController:
    """Phase B executor: commit via the frozen learned commit head (Eq. 8); never asks."""
    name = "learned_commit"

    def __init__(self, runtime):
        self.rt = runtime

    def decide(self, ctx):
        sc = self.rt.score(ctx.features)
        return Decision("commit", int(torch.argmin(sc["commit_logits"])), scores={"commit_logits": sc["commit_logits"].tolist()})


class ForcedThenLearned:
    name = "forced_then_learned_then_pi0"

    def __init__(self, forced: Decision, learned: LearnedCommitController, seed: int):
        self.forced, self.learned, self.pi0, self.n = forced, learned, Pi0Controller(seed), 0
        self.used = False

    def decide(self, ctx):
        if not self.used:
            self.used = True
            return self.forced
        self.n += 1
        return self.learned.decide(ctx) if self.n == 1 else self.pi0.decide(ctx)


def retarget(a):
    """Phase B (App. B.3): re-execute every saved fitting question; after the reply the frozen learned
    commitment rule selects the artifact, then pi_0 continues. Stores labels_star in each prefix file."""
    import glob
    from .carrier import Carrier
    from .features import CarrierRuntime
    from .train import D_BACKBONE
    out = Path(a.out) if Path(a.out).is_absolute() else RUNS / a.out
    ledger = CostLedger(Path(a.ledger), a.cap_usd, label="validation")
    bb = VLLMClient()
    encoder = HiddenEncoder()
    carrier = Carrier(D_BACKBONE)
    carrier.load_state_dict(torch.load(a.carrier, map_location="cpu"))
    rt = CarrierRuntime(carrier, encoder, generic_latent=a.generic_latent)
    rt.no_latent = a.no_latent
    learned = LearnedCommitController(rt)
    seeds = tuple(int(x) for x in a.seeds.split(","))
    paths = sorted(glob.glob(str(out / "prefixes" / "*.pt")))
    WF = HoosierWorkflow if a.dataset == "hoosier" else NGQAWorkflow

    def one(path):
        rec = torch.load(path, map_location="cpu", weights_only=False)
        if "labels_star" in rec and not a.force:
            return path, "skip"
        root = rec["root"]
        if a.dataset == "hoosier":
            from .envs.hoosier_env import HoosierEnv
            env0 = HoosierEnv.from_state(root, root.get("behavior", "normal"), ledger, rec["env_state"])
        else:
            from .envs.ngqa_env import NGQAEnv
            env0 = NGQAEnv.from_state(root, ledger, rec["env_state"])
        ws = rec["wf_state"]
        labels_star = {"ask": {}}
        for q in rec["menu"][: a.max_questions]:
            qobj = Question(**q)
            runs = []
            for seed in seeds:
                env = env0.fork()
                wf = WF(bb, ForcedThenLearned(Decision("ask", question=qobj), learned, seed), seed=seed, encoder=encoder, carrier=rt,
                        items_budget=a.items_budget)
                if hasattr(wf, "set_variant"):
                    wf.set_variant(a.variant)
                wf.versions, wf.items_used, wf.asked = dict(ws["versions"]), ws["items_used"], set(ws["asked"])
                wf.decision_points, wf.invocations = ws["decision_points"], list(ws["invocations"])
                if hasattr(wf, "fallback_overrides"):
                    wf.fallback_overrides, wf.last_fallback_request = dict(ws["fallback_overrides"]), ws["last_fallback_request"]
                resume = {"st": copy.deepcopy(rec["snapshot"]["st"]), "menu": [Question(**m) for m in rec["menu"]],
                          "forced": Decision("ask", question=qobj), "inv": rec["snapshot"]["inv"]}
                resume["pinv" if a.dataset == "hoosier" else "rinv"] = rec["snapshot"]["pinv" if a.dataset == "hoosier" else "rinv"]
                try:
                    r = wf.run(env, resume=resume)
                    runs.append({"error": 0 if r["score"].get("end_to_end_match") else 1, "ok": True, "score": r["score"]})
                except BudgetExceeded:
                    raise
                except Exception as e:
                    runs.append({"error": 1, "ok": False, "exception": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]})
            labels_star["ask"][qobj.qid] = runs
        rec["labels_star"] = labels_star
        rec["retarget_carrier"] = a.carrier
        torch.save(rec, path)
        return path, {q: [x["error"] for x in v] for q, v in labels_star["ask"].items()}

    start, usd0 = time.monotonic(), ledger.usd
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(one, pth) for pth in paths]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                path, res = f.result()
            except BudgetExceeded as e:
                print("STOP:", e)
                break
            except Exception as e:
                path, res = "?", f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(paths)}] {Path(str(path)).name} {res} ${ledger.usd - usd0:.2f} {time.monotonic() - start:.0f}s", flush=True)
    write_json(out / "retarget_summary.json", {"prefixes": len(paths), "openai_usd": ledger.usd - usd0, "wall_seconds": time.monotonic() - start,
                                               "carrier": a.carrier, "gen_usage": bb.usage_dict()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hoosier", "ngqa"], required=True)
    p.add_argument("--retarget", action="store_true", help="Phase B: executor-matched question outcomes with a trained carrier")
    p.add_argument("--carrier", default=None)
    p.add_argument("--generic-latent", action="store_true")
    p.add_argument("--no-latent", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--partition", default="representation")
    p.add_argument("--roots", type=int, default=24)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seeds", default="13,29,47")
    p.add_argument("--max-commits", type=int, default=5)
    p.add_argument("--max-questions", type=int, default=6)
    p.add_argument("--prefixes-per-root", type=int, default=2)
    p.add_argument("--items-budget", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--difficulty", default=None)
    p.add_argument("--disclosure", default="half", choices=["half", "none", "full"])
    p.add_argument("--min-statuses", type=int, default=0)
    p.add_argument("--variant", default="v5", choices=["v1", "v2", "v3", "v4", "v5", "v5nofb", "v6"], help="HoosierHelp workflow preset used while collecting")
    p.add_argument("--max-turns", type=int, default=None, help="HoosierHelp agent-turn cap while collecting (budget-pressure labels)")
    p.add_argument("--out", required=True)
    p.add_argument("--ledger", default=str(RUNS / "ledger_validation.json"))
    p.add_argument("--cap-usd", type=float, default=40.0)
    a = p.parse_args()
    if a.retarget:
        return retarget(a)
    out = Path(a.out) if Path(a.out).is_absolute() else RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    ledger = CostLedger(Path(a.ledger), a.cap_usd, label="validation")
    bb = VLLMClient()
    encoder = HiddenEncoder()
    roots = load_roots(a.dataset, a.partition, None, seed=0, difficulty=a.difficulty, min_statuses=a.min_statuses)[a.offset: a.offset + a.roots]
    done = {r["root_id"] for r in read_jsonl(out / "roots.jsonl")} if (out / "roots.jsonl").exists() else set()
    roots = [r for r in roots if r["root_id"] not in done]
    col = Collector(a.dataset, bb, encoder, ledger, out, tuple(int(s) for s in a.seeds.split(",")), a.max_commits, a.max_questions,
                    a.prefixes_per_root, a.items_budget, a.disclosure)
    col.variant = a.variant
    col.max_turns = a.max_turns
    start = time.monotonic()
    usd0 = ledger.usd
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = {ex.submit(col.collect_root, r, i): r for i, r in enumerate(roots)}
        n = 0
        for f in as_completed(futures):
            root = futures[f]
            try:
                res = f.result()
            except BudgetExceeded as e:
                print("STOP:", e)
                break
            except Exception as e:
                res = {"root_id": root["root_id"], "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
            n += 1
            append_jsonl(out / "roots.jsonl", res)
            print(f"[{n}/{len(roots)}] {root['root_id']} prefixes={len(res.get('prefixes', []))} err={res.get('error')} "
                  f"${ledger.usd - usd0:.2f} {time.monotonic() - start:.0f}s", flush=True)
    write_json(out / "summary.json", {"roots": n, "wall_seconds": time.monotonic() - start, "openai_usd": ledger.usd - usd0,
                                      "gen_usage": bb.usage_dict(), "encoder": encoder.usage_dict(), "args": vars(a)})


if __name__ == "__main__":
    main()
