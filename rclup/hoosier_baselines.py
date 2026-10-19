"""Conservative HoosierHelp baselines in the style of related single-agent / multi-agent work (same Qwen3-8B backbone,
same simulated user, same native turn cap and scorer as our system). Mirrors rclup.ngqa_baselines.

  single_direct : one agent with the official HoosierHelp agent prompt and tools, no reasoning (prompting baseline)
  single_cot    : same agent with Qwen3 thinking enabled before every action (step-by-step reasoning)
  plain_mas     : role chain intake -> planner -> retriever passing ONE greedy artifact as text (AutoGen / MetaGPT-style
                  message passing); the LLM planner writes the program; ordinary intake questions and a plain fallback
                  question to the person; none of our modules (no candidate sets, no grounding, no relaxation diagnosis,
                  no deterministic repair, no contradiction normaliser, no clarification controller)
  vote_mas      : plain_mas where every stage samples 8 draws + greedy and passes the majority artifact (self-consistency
                  / debate-style voting)
  conf_mas      : plain_mas where the greedy artifact is passed together with per-field agreement confidence from the
                  8 draws (uncertainty-propagation style handoff, e.g. PropUQ-MAS)

  python -m rclup.hoosier_baselines --baselines single_direct,single_cot --workers 10 --out eval/hoosier_baselines
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import roles as R
from .backbone import VLLMClient
from .common import RUNS, append_jsonl, read_jsonl, write_json
from .cost import BudgetExceeded, CostLedger
from .envs.hoosier_env import HoosierEnv, native_agent_prompt, native_tool_schemas
from .packets import Packet
from .run_episodes import load_roots, summarize
from .workflow import Controller, Decision, GreedyController, HoosierWorkflow, S_DRAWS

BASELINES = ("single_direct", "single_cot", "plain_mas", "vote_mas", "conf_mas")
SEARCH, FINAL = "search_resources", "final_recommendation"


# --------------------------------------------------------------------------
# single agent (official prompt + tools, our loop around the native environment)
# --------------------------------------------------------------------------
def _tool_parsing():
    from .common import ensure_vendor_on_path
    ensure_vendor_on_path()
    from eval import tool_parsing
    return tool_parsing


def run_single(name: str, bb: VLLMClient, env: HoosierEnv, schemas: list[dict], trace: list[dict]) -> None:
    tp = _tool_parsing()
    sys_p = native_agent_prompt()
    thinking = name == "single_cot"
    env.reset()
    while not env.done:
        msgs = [{"role": "system", "content": sys_p}] + env.messages
        out = bb.chat(msgs, n=1, temperature=0.0, max_tokens=6144 if thinking else 1024, tools=schemas,
                      purpose=f"hoosier:{name}", thinking=thinking)[0]
        text = out.text
        if thinking and "</think>" in text:
            text = text.rsplit("</think>", 1)[1].strip()
        calls = tp.parse_qwen_xml_tool_calls(text)
        if thinking and not calls and (out.finish_reason == "length" or not tp.clean_tool_call_text(text)):
            # reasoning ran out of budget or produced no action: act without thinking for this turn
            text = bb.chat(msgs, n=1, temperature=0.0, max_tokens=1024, tools=schemas, purpose=f"hoosier:{name}:nothink")[0].text
            calls = tp.parse_qwen_xml_tool_calls(text)
        trace.append({"kind": "agent_output", "turn": env.agent_turns + 1, "text": out.text[:4000], "finish_reason": out.finish_reason})
        searches = [c.arguments for c in calls if c.name == SEARCH]
        finals = [c.arguments for c in calls if c.name == FINAL]
        if searches:
            env.search(searches)
            if env.done:
                break
        if finals:
            a = finals[0]
            env.finish([str(x) for x in (a.get("resource_ids") or [])], str(a.get("message") or ""))
            break
        if not searches:
            msg = tp.clean_tool_call_text(text) or "Could you tell me a bit more about what you need?"
            env.ask_user(msg)


# --------------------------------------------------------------------------
# multi-agent baselines on the shared role chain
# --------------------------------------------------------------------------
class VoteController(Controller):
    name = "vote_mas"

    def decide(self, ctx):
        return Decision("commit", ctx.greedy_index())   # the vote winner carries the greedy flag (see BaselineWorkflow.invoke)


def flatten(obj, prefix="") -> dict:
    """artifact -> {path: json value}; lists of dicts are indexed by need_id when present."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        for i, x in enumerate(obj):
            tag = x.get("need_id", str(i)) if isinstance(x.get("need_id"), str) else str(i)
            out.update(flatten({k: v for k, v in x.items() if k != "need_id"}, f"{prefix}{tag}."))
    else:
        out[prefix.rstrip(".")] = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return out


class BaselineWorkflow(HoosierWorkflow):
    """Plain message-passing role chain; see module docstring for the three handoff modes."""

    def __init__(self, bb: VLLMClient, mode: str, seed: int = 0):
        super().__init__(bb, VoteController() if mode == "vote_mas" else GreedyController(), seed=seed, items_budget=0)
        self.set_variant("v1")
        self.use_relaxation, self.llm_planner, self.confirmation_questions, self.user_direct, self.prune_on_answer = False, True, False, False, False
        self.contradiction_clarification = False
        self.combined_fallback = False          # no counterfactual relaxation round / deterministic repair
        self.contradiction_normalizer = False   # agents read the person's words as written
        self.mode = mode
        self.controller.name = mode
        self.s_draws = 0 if mode == "plain_mas" else S_DRAWS
        self.roles = dict(self.roles)
        pl = self.roles["planner"]
        self.roles["planner"] = R.RoleSpec(pl.name, pl.task.replace("alternative interpretations of the person's needs with empirical support",
                                                                   "the structured evidence state of the person's needs"),
                                           pl.parse, pl.canon, pl.render, pl.tools)

    # --- handoff rendering: one artifact (+ optional per-field confidence) ---
    def handoff(self, packet: Packet) -> str:
        g = packet.greedy
        text = f"Handoff from the {packet.sender_role} agent (message version {packet.version}):\n{g.text}"
        if self.mode == "conf_mas":
            cands = [c for c in packet.candidates if not c.is_unresolved]
            gf = flatten(g.artifact)
            total = packet.n_stochastic + 1
            lines = []
            for path, val in gf.items():
                agree = 1 + sum(c.count for c in cands if c is not g and flatten(c.artifact).get(path) == val)
                lines.append(f"{path}={agree / total:.2f}")
            text += (f"\nConfidence that each item above is right (agreement across {total} independent readings): " + ", ".join(lines))
        return text

    def planner_messages(self, opening: str, packet: Packet, replies: list[dict]) -> list[dict]:
        return [{"role": "system", "content": self.roles["planner"].task},
                {"role": "user", "content": f"Public opening from the person: {opening}\n\n{self.handoff(packet)}"}]

    def retriever_messages(self, opening: str, planner_packet: Packet, selected, results: list[dict], diagnosis=None) -> list[dict]:
        user = (f"Public opening from the person: {opening}\n\n{self.handoff(planner_packet)}\n\nExecuted program: {selected.text}\n\n"
                "Search results (JSON):\n" + json.dumps(results, ensure_ascii=False))
        return [{"role": "system", "content": self.roles["retriever"].task}, {"role": "user", "content": user}]

    # --- majority vote: the most frequent artifact becomes the handed-off one ---
    def invoke(self, *a, **kw):
        inv = super().invoke(*a, **kw)
        if self.mode == "vote_mas":
            cands = [c for c in inv.packet.candidates if not c.is_unresolved]
            if cands:
                win = max(cands, key=lambda c: (c.count + (1 if c.greedy else 0), c.greedy))
                for c in inv.packet.candidates:
                    c.greedy = c is win
                inv.packet.candidates.sort(key=lambda c: (c.is_unresolved, not c.greedy, -c.count))
                self.trace.add("vote", inv_id=inv.inv_id, winner=win.key, votes=win.count + 1, total=inv.packet.n_stochastic + 1)
        return inv

    # --- none of our modules: no grounding, no literal checks, no clarification menus, no resolver ---
    def check_intake_packet(self, env, packet: Packet) -> None:
        spec = self.roles["intake"]
        merged = {}
        for c in packet.candidates:
            if c.is_unresolved:
                merged[c.key] = c
                continue
            art = copy.deepcopy(c.artifact)
            changed = False
            for n in art["needs"]:
                for f in R.FIELDS:
                    if n[f] == R.EMPTY:
                        was_asked = (n["need_id"], f) in self.asked or ("*", f) in self.asked
                        n[f] = R.NONE if was_asked else R.UNKNOWN
                        changed = True
            if changed:
                key, norm = spec.canon(art)
                c.key, c.artifact, c.text = key, norm, spec.render(norm)
            if c.key in merged:
                m = merged[c.key]
                m.count += c.count
                m.weight += c.weight
                m.greedy = m.greedy or c.greedy
            else:
                merged[c.key] = c
        packet.candidates = sorted(merged.values(), key=lambda c: (c.is_unresolved, not c.greedy, -c.count))

    def check_planner_packet(self, intake_packet: Packet, planner_packet: Packet) -> None:
        return None

    def planner_menu(self, packet: Packet, asked: set, env=None) -> list:
        return []

    def planner_aux(self, env, packet: Packet, menu: list) -> dict:
        return {}

    def ask_user_fallback(self, env, request: str, need_name: str | None = None) -> None:
        what = f"for {need_name}" if need_name else "matching your request"
        text = (f"I could not find any resources {what} with your current {request}. "
                f"Could your {request} {'for ' + need_name + ' ' if need_name else ''}be changed, and if so, to what?")
        reply = env.ask_user(text)
        self.last_fallback_request = request   # the intake agent re-reads the conversation with the person's answer
        self.fallback_overrides = {}
        for f in ("location", "schedule", "intake_methods"):
            if f.split("_")[0] in request:
                self.asked.add(("*", f))
        self.trace.add("user_question", kind_detail="fallback_repair", request=request, text=text, reply=reply)


# --------------------------------------------------------------------------
def run_one(name: str, bb: VLLMClient, root: dict, ledger: CostLedger, seed: int, schemas: list[dict]) -> dict:
    env = HoosierEnv(root, root.get("behavior", "normal"), ledger)
    t0 = time.monotonic()
    trace: list[dict] = []
    wf = None
    try:
        if name.startswith("single"):
            run_single(name, bb, env, schemas, trace)
            rec = env.record()
            rec.update({"controller": name, "terminated": "final" if env.final_action else "unresolved", "trace": trace})
        else:
            wf = BaselineWorkflow(bb, name, seed=seed)
            rec = wf.run(env)
    except BudgetExceeded:
        raise
    except Exception as e:
        rec = env.record() if env.messages else {"root_id": root["root_id"], "score": {"end_to_end_match": False, "user_questions": 0}}
        rec.update({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:], "controller": name,
                    "trace": wf.trace.events if wf is not None else trace})
    rec.update({"behavior": root.get("behavior", ""), "wall_seconds": time.monotonic() - t0, "gen_usage": bb.usage_dict()})
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baselines", default=",".join(BASELINES))
    p.add_argument("--behaviors", default="normal,rambling,impatience,unsupported_request,self_contradictory")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="eval/hoosier_baselines")
    p.add_argument("--cap-usd", type=float, default=250.0)
    a = p.parse_args()
    ledger = CostLedger(RUNS / "ledger_validation.json", a.cap_usd, label="validation")
    bb = VLLMClient()
    from .common import ensure_vendor_on_path
    ensure_vendor_on_path()
    from tools.tool_protocol import qwen_tool_schemas
    schemas = qwen_tool_schemas(native_tool_schemas())
    roots = load_roots("hoosier", "test", a.limit, a.seed, behaviors=tuple(a.behaviors.split(",")))
    for name in a.baselines.split(","):
        out = RUNS / a.out / f"baseline_{name}"
        out.mkdir(parents=True, exist_ok=True)
        rec_path = out / "records.jsonl"
        done = {r["root_id"] + "|" + r.get("behavior", "") for r in read_jsonl(rec_path)} if rec_path.exists() else set()
        todo = [r for r in roots if r["root_id"] + "|" + r.get("behavior", "") not in done]
        print(f"[{name}] {len(todo)} to run ({len(done)} done)", flush=True)
        usd0, t0 = ledger.usd, time.monotonic()
        k = 0
        with ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(run_one, name, bb, r, ledger, a.seed, schemas) for r in todo]
            for f in as_completed(futs):
                try:
                    rec = f.result()
                except BudgetExceeded as e:
                    print("STOP:", e)
                    break
                append_jsonl(rec_path, rec)
                k += 1
                s = rec["score"]
                print(f"[{name} {k}/{len(todo)}] {rec['root_id']}|{rec['behavior']} ok={s.get('end_to_end_match')} uq={s.get('user_questions')} "
                      f"err={'error' in rec} ${ledger.usd - usd0:.3f}", flush=True)
        recs = read_jsonl(rec_path)
        summary = summarize(recs, "hoosier")
        byb = collections.defaultdict(list)
        for r in recs:
            byb[r.get("behavior")].append(float(bool(r["score"].get("end_to_end_match"))))
        summary.update({"controller": name, "by_behavior": {b: sum(v) / len(v) for b, v in byb.items()},
                        "end_to_end_match_incl_errors": sum(float(bool(r["score"].get("end_to_end_match"))) for r in recs) / max(1, len(recs)),
                        "wall_seconds": time.monotonic() - t0, "openai_usd_this_run": ledger.usd - usd0, "gen_usage": bb.usage_dict()})
        write_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
