"""Conservative NGQA baselines in the style of related single-agent / multi-agent work (native knowledge-QA mode, Qwen only).

  single_direct : one agent sees record + food + guideline, answers directly (prompting baseline, as in the NGQA paper)
  single_cot    : same, with free-text step-by-step reasoning before the answer
  plain_mas     : role chain profile -> reasoner passing ONE greedy artifact as text (AutoGen / MetaGPT-style message passing)
  vote_mas      : plain_mas with 5 sampled reasoners and majority vote (debate / self-consistency style)
  conf_mas      : profile agent passes its greedy artifact with per-status agreement confidence from 8 samples
                  (uncertainty-propagation style handoff, e.g. PropUQ-MAS), reasoner answers once

  python -m rclup.ngqa_baselines --limit 300 --workers 10 --out eval/ngqa_full
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import roles as R
from .backbone import VLLMClient
from .common import RUNS, append_jsonl, write_json
from .cost import CostLedger
from .envs.ngqa_env import NGQAEnv, guideline_text
from .run_episodes import load_roots

BASELINES = ("single_direct", "single_cot", "plain_mas", "vote_mas", "conf_mas")


def food_text(food: dict) -> str:
    return json.dumps({k: food[k] for k in ("name", "category", "ingredients", "nutrition_tags") if k in food}, ensure_ascii=False)


def run_one(name: str, bb: VLLMClient, root: dict, ledger: CostLedger, seed: int) -> dict:
    env = NGQAEnv(root, ledger, disclosure="full")
    record = env.reset()
    gl = guideline_text()
    tools = R.REASONER_TOOLS
    t0 = time.monotonic()
    if name in ("single_direct", "single_cot"):
        sys_p = ("You are a nutrition assistant. Decide whether the food is healthy for this person from their record, the food record and the "
                 "guideline. " + ("Reason step by step in plain text first, then " if name == "single_cot" else "") +
                 "call report_verdict exactly once (binary yes/no, the decisive nutrition tags, one-sentence explanation). "
                 "Rule: only the person's listed conditions and special diets count; the food is unhealthy only if a listed status contradicts one of "
                 "the food's tags, unless matching tags strictly outnumber contradicting ones.")
        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": f"Person record:\n{record}\n\nFood record: {food_text(env.public_food)}\n\n{gl}"}]
        out = bb.chat(msgs, n=1, temperature=0.0, max_tokens=1536 if name == "single_cot" else 512, tools=tools, purpose=f"ngqa:{name}")[0].text
        art = R.parse_reasoner(out)
    else:
        pmsgs = [{"role": "system", "content": R.NGQA_PROFILE_TASK}, {"role": "user", "content": "Conversation so far:\n[person 1] " + record}]
        greedy = bb.chat(pmsgs, n=1, temperature=0.0, max_tokens=512, tools=R.PROFILE_TOOLS, purpose=f"ngqa:{name}:profile")[0].text
        prof = R.parse_profile(greedy) or {"statuses": {s: R.UNKNOWN for s in R.status_order()}, "habits": [], "sources": []}
        prof = R.canon_profile(prof)[1]
        handoff = R.render_profile(prof)
        if name == "conf_mas":
            samples = bb.chat(pmsgs, n=8, temperature=0.7, top_p=0.9, max_tokens=512, seed=seed, tools=R.PROFILE_TOOLS, purpose=f"ngqa:{name}:profile_samples")
            arts = [R.parse_profile(s.text) for s in samples]
            arts = [R.canon_profile(a)[1] for a in arts if a]
            conf = {}
            for s in R.status_order():
                v = prof["statuses"].get(s, R.UNKNOWN)
                conf[s] = (sum(1 for a in arts if a["statuses"].get(s) == v) + 1) / (len(arts) + 1)
            handoff += "\nConfidence that each status value above is right (agreement across 8 independent readings): " + \
                       ", ".join(f"{s}={conf[s]:.2f}" for s in R.status_order())
        rmsgs = [{"role": "system", "content": R.NGQA_REASONER_TASK},
                 {"role": "user", "content": f"Food record: {food_text(env.public_food)}\n\n{gl}\n\nHandoff from the profile agent:\n{handoff}"}]
        if name == "vote_mas":
            outs = bb.chat(rmsgs, n=5, temperature=0.7, top_p=0.9, max_tokens=512, seed=seed, tools=tools, purpose=f"ngqa:{name}:vote")
            arts = [R.parse_reasoner(o.text) for o in outs]
            arts = [a for a in arts if a]
            votes = collections.Counter((a.get("binary") or "").strip().lower() for a in arts)
            win = votes.most_common(1)[0][0] if votes else ""
            art = next((a for a in arts if (a.get("binary") or "").strip().lower() == win), None)
        else:
            out = bb.chat(rmsgs, n=1, temperature=0.0, max_tokens=512, tools=tools, purpose=f"ngqa:{name}")[0].text
            art = R.parse_reasoner(out)
    if art is None:
        art = {"binary": "", "tags": [], "explain": ""}
    score = env.finish(art)
    rec = env.record()
    rec.update({"controller": name, "wall_seconds": time.monotonic() - t0})
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baselines", default=",".join(BASELINES))
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="eval/ngqa_full")
    a = p.parse_args()
    ledger = CostLedger(RUNS / "ledger_validation.json", 220.0, label="validation")
    bb = VLLMClient()
    roots = load_roots("ngqa", "test", a.limit, a.seed)
    for name in a.baselines.split(","):
        out = RUNS / a.out / f"baseline_{name}"
        out.mkdir(parents=True, exist_ok=True)
        rec_path = out / "records.jsonl"
        done = set()
        if rec_path.exists():
            done = {json.loads(l)["root_id"] for l in open(rec_path) if l.strip()}
        todo = [r for r in roots if r["root_id"] not in done]
        print(f"[{name}] {len(todo)} to run ({len(done)} done)", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            for rec in ex.map(lambda r: run_one(name, bb, r, ledger, a.seed), todo):
                append_jsonl(rec_path, rec)
        recs = [json.loads(l) for l in open(rec_path) if l.strip()]
        by = collections.defaultdict(list)
        for r in recs:
            by[r.get("difficulty")].append(float(bool(r["score"]["end_to_end_match"])))
        summary = {"controller": name, "n": len(recs), "end_to_end_match": statistics.mean(float(bool(r["score"]["end_to_end_match"])) for r in recs),
                   "by_difficulty": {k: statistics.mean(v) for k, v in sorted(by.items())},
                   "tag_f1": statistics.mean(r["score"].get("tag_f1", 0.0) for r in recs),
                   "explain_rouge_l": statistics.mean(r["score"].get("explain_rouge_l", 0.0) for r in recs),
                   "binary_valid": statistics.mean(1.0 if r["final"]["binary"] in ("yes", "no") else 0.0 for r in recs),
                   "wall_seconds_avg": statistics.mean(r.get("wall_seconds", 0.0) for r in recs)}
        write_json(out / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
