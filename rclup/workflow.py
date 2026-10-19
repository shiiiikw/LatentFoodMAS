"""Episode driver: role invocations, candidate packets, controllers, branching.

The workflow is the *public planner* shared by every method (App. C.1):
ordinary intake questioning, search, final recommendation and fallback repair
are fixed rules. Methods differ only in the receiver-side controller that
chooses between committing a candidate and asking the sender one atomic
clarification (Sec. 4.5). Episodes can be forked at a receiver decision point
and resumed with a forced first decision, which is how saved prefixes are
branched for label collection (App. B.2).
"""
from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .backbone import VLLMClient
from .packets import Packet, Candidate, build_packet, UNRESOLVED_KEY
from . import roles as R

S_DRAWS = 8
PROPOSAL_MAX_TOKENS = 512
Q_MAX = 6


@dataclass
class Question:
    qid: str
    target_role: str
    scope: dict            # {"need_id","field"} or {"status"}
    a: str                 # candidate key (or "__residual__")
    b: str
    text: str
    items: int = 1


@dataclass
class Decision:
    kind: str              # "commit" | "ask"
    candidate: int | None = None
    question: Question | None = None
    scores: dict | None = None


class Controller:
    """Receiver-side action rule. Baselines never ask."""
    name = "base"

    def decide(self, ctx: "ReceiverContext") -> Decision:
        raise NotImplementedError


class GreedyController(Controller):
    name = "text_mas"

    def decide(self, ctx):
        return Decision("commit", ctx.greedy_index())


class Pi0Controller(Controller):
    """Reference continuation pi_0: stochastic backbone commitment, no learned asks."""
    name = "pi0"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def decide(self, ctx):
        cands = [c for c in ctx.candidates if not c.is_unresolved]
        weights = [max(c.count, 0) for c in cands]
        if sum(weights) == 0:
            return Decision("commit", ctx.greedy_index())
        pick = self.rng.choices(range(len(cands)), weights=weights, k=1)[0]
        return Decision("commit", ctx.candidates.index(cands[pick]))


class DisagreementController(Controller):
    """Heuristic control: ask the first menu question while budget remains."""
    name = "disagreement_ask"

    def decide(self, ctx):
        if ctx.menu and ctx.items_left > 0:
            return Decision("ask", question=ctx.menu[0])
        return Decision("commit", ctx.greedy_index())


class RoundRobinMenuController(Controller):
    """Prefix-generation policy (App. B.1): round-robin over the menu up to the budget, then pi_0 commit."""
    name = "round_robin_menu"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.k = 0

    def decide(self, ctx):
        if ctx.menu and ctx.items_left > 0:
            q = ctx.menu[self.k % len(ctx.menu)]
            self.k += 1
            return Decision("ask", question=q)
        return Pi0Controller.decide(self, ctx)


class ForcedController(Controller):
    """Branch executor: forced first decision, then pi_0 for the rest of the trajectory."""
    name = "forced_then_pi0"

    def __init__(self, forced: Decision, seed: int):
        self.forced = forced
        self.inner = Pi0Controller(seed)
        self.used = False

    def decide(self, ctx):
        if not self.used:
            self.used = True
            return self.forced
        return self.inner.decide(ctx)


@dataclass
class ReceiverContext:
    role: str
    incoming: list[Packet]
    candidates: list[Candidate]
    menu: list[Question]
    items_left: int
    external_left: int
    task_state_text: str
    features: Any = None
    state_vec: Any = None

    def greedy_index(self) -> int:
        for i, c in enumerate(self.candidates):
            if c.greedy:
                return i
        return 0


@dataclass
class Invocation:
    inv_id: str
    role: str
    version: int
    messages: list[dict]
    tools: list[dict] | None
    packet: Packet
    selected: int
    purpose: str


@dataclass
class Trace:
    events: list[dict] = field(default_factory=list)

    def add(self, kind: str, **kw):
        self.events.append({"kind": kind, **kw})


class Workflow:
    """Base workflow with sender/receiver invocations; task subclasses define roles and rules."""
    dataset = "base"
    roles: dict[str, R.RoleSpec]

    def __init__(self, backbone: VLLMClient, controller: Controller, *, seed: int = 0, encoder=None,
                 carrier=None, items_budget: int = 2, prefix_hook: Callable | None = None):
        self.bb = backbone
        self.controller = controller
        self.seed = seed
        self.encoder = encoder
        self.carrier = carrier
        self.items_budget = items_budget
        self.prefix_hook = prefix_hook
        self.versions: dict[str, int] = {}
        self.invocations: list[Invocation] = []
        self.trace = Trace()
        self.items_used = 0
        self.rng = random.Random(seed)
        self.asked: set = set()
        self.decision_points = 0

    def fork(self, controller: Controller, seed: int | None = None) -> "Workflow":
        """Copy mutable state for a branch; backbone/encoder/carrier and packets are shared."""
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone.controller = controller
        clone.seed = self.seed if seed is None else seed
        clone.versions = dict(self.versions)
        clone.invocations = list(self.invocations)
        clone.trace = Trace()
        clone.items_used = self.items_used
        clone.rng = random.Random(clone.seed * 7919 + self.decision_points)
        clone.asked = set(self.asked)
        clone.prefix_hook = None
        if hasattr(self, "fallback_overrides"):
            clone.fallback_overrides = dict(self.fallback_overrides)
        return clone

    # ----- helpers -----
    def next_version(self, role: str) -> int:
        self.versions[role] = self.versions.get(role, 0) + 1
        return self.versions[role]

    def invoke(self, role: str, messages: list[dict], *, tools=None, ancestors=(), sources=(), unresolved_text=None,
               purpose: str = "") -> Invocation:
        spec = self.roles[role]
        version = self.next_version(role)
        inv_id = f"{role}#{version}"
        samples = self.bb.propose(messages, s=S_DRAWS, max_tokens=PROPOSAL_MAX_TOKENS, seed=self.seed * 1000 + version,
                                 tools=tools, purpose=f"{self.dataset}:{role}")
        packet = build_packet(inv_id, version, role, samples, spec.parse, spec.canon, spec.render,
                              ancestors=ancestors, sources=sources, unresolved_text=unresolved_text)
        if not any(not c.is_unresolved for c in packet.candidates):
            # All draws invalid: one retry with a different seed (logged); otherwise fail the invocation.
            samples = self.bb.propose(messages, s=S_DRAWS, max_tokens=PROPOSAL_MAX_TOKENS, seed=self.seed * 1000 + version + 500,
                                     tools=tools, purpose=f"{self.dataset}:{role}:retry")
            packet = build_packet(inv_id, version, role, samples, spec.parse, spec.canon, spec.render,
                                  ancestors=ancestors, sources=sources, unresolved_text=unresolved_text)
            packet.extra["retried"] = True
            if not any(not c.is_unresolved for c in packet.candidates):
                self.trace.add("invocation_failed", inv_id=inv_id, role=role, examples=[x.text[:1500] for x in samples[:3]])
                raise RuntimeError(f"No valid proposals for {inv_id}")
        invalid_examples = [x.text[:900] for x in samples[1:] if spec.parse(x.text) is None][:2]
        inv = Invocation(inv_id, role, version, messages, tools, packet, 0, purpose)
        self.invocations.append(inv)
        self.trace.add("invocation", inv_id=inv_id, role=role, version=version, purpose=purpose,
                       packet=packet.public(), n_samples=len(samples), invalid=packet.invalid_count,
                       invalid_examples=invalid_examples, finish_reasons=[x.finish_reason for x in samples])
        return inv

    def attach_states(self, inv: Invocation) -> None:
        """Teacher-forced response states for each candidate's representative generation (Sec. 4.1)."""
        if self.encoder is None or inv.packet.extra.get("states_attached"):
            return
        prompt_mean = None
        for c in inv.packet.candidates:
            if c.is_unresolved or c.states is not None:
                continue
            states, pm = self.encoder.encode(inv.messages, c.rep_text, tools=inv.tools, pool_prompt=prompt_mean is None)
            c.states = states
            if pm is not None:
                prompt_mean = pm
        inv.packet.extra["prompt_mean"] = prompt_mean
        inv.packet.extra["states_attached"] = True
        if self.carrier is not None:
            self.carrier.pool_packet(inv.packet)

    def decide(self, ctx: ReceiverContext, inv: Invocation | None = None, snapshot: dict | None = None) -> Decision:
        self.decision_points += 1
        if self.encoder is not None:
            for p_inv in (snapshot.get("sender_invs", []) if snapshot else []):
                self.attach_states(p_inv)
            if inv is not None:
                self.attach_states(inv)
                ctx.state_vec = inv.packet.extra.get("prompt_mean")
        if self.carrier is not None:
            ctx.features = self.carrier.featurize(ctx)
        if self.prefix_hook is not None and snapshot is not None:
            self.prefix_hook(self, ctx, snapshot)
        d = self.controller.decide(ctx)
        self.trace.add("decision", role=ctx.role, decision=d.kind, candidate=d.candidate,
                       question=None if d.question is None else d.question.__dict__, scores=d.scores,
                       menu=[q.__dict__ for q in ctx.menu], items_left=ctx.items_left)
        return d

    def greedy_text(self, messages, max_tokens=256, purpose="aux") -> str:
        return self.bb.chat(messages, n=1, temperature=0.0, max_tokens=max_tokens, purpose=f"{self.dataset}:{purpose}")[0].text.strip()

    def record_extra(self, rec: dict, outcome: dict, cycles: int) -> dict:
        rec.update(outcome, controller=self.controller.name, items_used=self.items_used, cycles=cycles,
                   trace=self.trace.events, invocations=[{"inv_id": i.inv_id, "role": i.role, "selected": i.selected} for i in self.invocations])
        return rec


# ==========================================================================
# HoosierHelp
# ==========================================================================
class HoosierWorkflow(Workflow):
    dataset = "hoosier"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from .envs.hoosier_env import native_tool_schemas, native_agent_prompt, service_categories
        from .common import ensure_vendor_on_path
        ensure_vendor_on_path()
        from tools.tool_protocol import qwen_tool_schema
        self.schemas = native_tool_schemas()
        self.search_schema = [s for s in self.schemas if s["name"] == "search_resources"]
        final_schema = [s for s in self.schemas if s["name"] == "final_recommendation"][0]
        self.roles = R.hoosier_roles(qwen_tool_schema(self.search_schema[0]), qwen_tool_schema(final_schema))
        self.native_prompt = native_agent_prompt()
        self.categories = service_categories()
        self.last_fallback_request = None
        self.fallback_overrides = {}
        from .envs.hoosier_env import _Native
        index = _Native.load()["index"]
        self.known_counties = {c.upper() for r in index.resources for c in (r.counties or ())}
        self.known_cities = {str(r.city).upper() for r in index.resources if getattr(r, "city", None)}

    # ----- prompts -----
    def _conversation_text(self, env) -> str:
        lines, k = [], 0
        for m in env.messages:
            if m["role"] == "user" and not m["content"].startswith("Tool result for search_resources call"):
                k += 1
                lines.append(f"[person {k}] {m['content']}")
            elif m["role"] == "assistant" and "<tool_call>" not in m["content"]:
                lines.append(f"[agent] {m['content']}")
            elif m["role"] == "user":
                lines.append("[system] " + m["content"].split("\n")[0] + (" (no resources matched)" if "No resources matched" in m["content"] else " (resources returned)"))
        return "\n".join(lines)

    def _person_text(self, env) -> str:
        return " ".join(m["content"] for m in env.messages if m["role"] == "user"
                        and not m["content"].startswith("Tool result for search_resources call")).lower()

    def intake_messages(self, env) -> list[dict]:
        system = self.roles["intake"].task + "\n\nAllowed service_categories values: " + ", ".join(self.categories)
        user = "Conversation so far:\n" + self._conversation_text(env)
        note = getattr(self, "last_fallback_request", None)
        if note:
            user += (f"\n\nNote: the last search returned no resources and the person was asked whether their {note} could change. "
                     "Apply their latest answer to the affected need(s): a new value replaces the old one; 'I can change it', 'if needed', "
                     "'flexible' or 'no longer needed' means set that field to \"none\"; 'cannot change' keeps the old value.")
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def planner_messages(self, opening: str, packet: Packet, replies: list[dict]) -> list[dict]:
        system = self.roles["planner"].task
        user = f"Public opening from the person: {opening}\n\n{packet.readable()}"
        if replies:
            user += "\n\nClarification replies from the intake agent:\n" + "\n".join(
                f"Q: {r['question']}\nA: {r['answer']} (sources: person turns {r.get('sources')}; resolved={r.get('resolved')})" for r in replies)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def retriever_messages(self, opening: str, planner_packet: Packet, selected: Candidate, results: list[dict]) -> list[dict]:
        user = (f"Public opening from the person: {opening}\n\n{planner_packet.readable()}\n\nExecuted program: {selected.text}\n\n"
                "Search results (JSON):\n" + json.dumps(results, ensure_ascii=False))
        return [{"role": "system", "content": self.roles["retriever"].task}, {"role": "user", "content": user}]

    def recover_location(self, replies_text: str) -> dict | None:
        """Source-grounded literal recovery: county/ZIP/city mentioned by the person, validated against the index."""
        import re as _re
        found = {}
        counties = [m.group(1).strip().upper() for m in _re.finditer(r"([A-Za-z][A-Za-z.' ]{1,30}?)\s+county\b", replies_text, flags=_re.I)]
        counties = [c.split()[-1] if c.split()[-1] in self.known_counties else c for c in counties]
        counties = sorted({c for c in counties if c in self.known_counties})
        zips = sorted(set(_re.findall(r"\b(\d{5})\b", replies_text)))
        if counties:
            found["counties"] = counties
        if zips:
            found["zipcodes"] = zips
        if not found:
            cities = sorted({c for c in self.known_cities if len(c) > 3 and _re.search(r"\b" + _re.escape(c.lower()) + r"\b", replies_text)})
            if len(cities) == 1:
                found["cities"] = cities
        return found or None

    # ----- task-specific module: provenance checks on intake candidates -----
    def check_intake_packet(self, env, packet: Packet) -> None:
        """(a) an empty container is 'none' only if the field was asked, else 'unknown'; a 'none' claim for an
        unasked field becomes 'unknown'; (b) location literals must appear in the person's own words."""
        text = self._person_text(env)
        replies_text = " ".join(m["content"] for m in env.messages[1:] if m["role"] == "user"
                                and not m["content"].startswith("Tool result for search_resources call")).lower()
        spec = self.roles["intake"]
        merged: dict[str, Candidate] = {}
        for c in packet.candidates:
            if c.is_unresolved:
                merged[c.key] = c
                continue
            art = copy.deepcopy(c.artifact)
            changed = False
            for n in art["needs"]:
                for f in R.FIELDS:
                    was_asked = (n["need_id"], f) in self.asked or ("*", f) in self.asked
                    if n[f] == R.EMPTY:
                        n[f] = R.NONE if was_asked else R.UNKNOWN
                        changed = True
                    elif n[f] == R.NONE and not was_asked:
                        n[f] = R.UNKNOWN
                        changed = True
                loc = n["location"]
                if isinstance(loc, dict):
                    kept = {}
                    for k, vals in loc.items():
                        ok = [v for v in vals if re.search(r"\b" + re.escape(v.lower()) + r"\b", text)]
                        if ok:
                            kept[k] = ok
                    if kept != loc:
                        n["location"] = kept if kept else (self.recover_location(replies_text) or R.UNKNOWN)
                        changed = True
                elif loc == R.UNKNOWN and ((n["need_id"], "location") in self.asked or ("*", "location") in self.asked):
                    rec = self.recover_location(replies_text)
                    if rec:
                        n["location"] = rec
                        changed = True
                # documents / eligibility literals must be traceable to the person's replies (not the need statement)
                for f in ("available_documents", "eligibility"):
                    if isinstance(n[f], list):
                        ok = [v for v in n[f] if any(k in replies_text for k in R.FIELD_VALUE_LEXICON.get(v, [v.replace("_", " ")]))]
                        if ok != n[f]:
                            was_asked = (n["need_id"], f) in self.asked or ("*", f) in self.asked
                            n[f] = ok if ok else (R.NONE if was_asked else R.UNKNOWN)
                            changed = True
                sch = n["schedule"]
                if isinstance(sch, dict):
                    cleaned = {k: v for k, v in sch.items() if k not in ("time", "start_time", "end_time") or R.time_supported(v, text)}
                    if cleaned != sch:
                        n["schedule"] = cleaned if cleaned else R.UNKNOWN
                        changed = True
                # fallback resolution overrides (ordinary repair, applied to every candidate)
                for (need_id, f), action in (self.fallback_overrides or {}).items():
                    if need_id in (n["need_id"], "*") and action is not None:
                        if n[f] != action:
                            n[f] = action
                            changed = True
            if changed:
                key, norm = spec.canon(art)
                c.key, c.artifact, c.text = key, norm, spec.render(norm)
            if c.key in merged:
                m = merged[c.key]
                m.count += c.count
                m.weight += c.weight
                m.greedy = m.greedy or c.greedy
                m.sources = sorted(set(m.sources) | set(c.sources))
                if c.rep_logprob > m.rep_logprob:
                    m.rep_text, m.rep_logprob, m.states, m.latent = c.rep_text, c.rep_logprob, c.states, c.latent
            else:
                merged[c.key] = c
        packet.candidates = sorted(merged.values(), key=lambda c: (c.is_unresolved, not c.greedy, -c.count))

    # ----- task-specific module: atomic question menu over (need, field) distinctions -----
    def planner_menu(self, packet: Packet, asked: set[str]) -> list[Question]:
        """Atomic (need, field) questions: candidate-vs-candidate where interpretations disagree, candidate-vs-unknown
        where a field is unresolved, and candidate-vs-residual confirmations of agreed concrete values. Ordered by
        tier (disagreement, unknown, confirmation) then field order; capped at Q_MAX."""
        cands = [c for c in packet.candidates if not c.is_unresolved]
        if not cands:
            return []
        tiers = {0: [], 1: [], 2: []}
        need_ids = sorted({n["need_id"] for c in cands for n in c.artifact["needs"]})
        for f in ("service_categories",) + R.FIELDS:
            for need_id in need_ids:
                qid = f"{need_id}:{f}"
                if qid in asked:
                    continue
                values = {}
                for c in cands:
                    need = next((n for n in c.artifact["needs"] if n["need_id"] == need_id), None)
                    v = json.dumps(need[f], sort_keys=True) if need is not None else "missing"
                    values.setdefault(v, []).append(c)
                has_unknown = any(json.loads(v) == R.UNKNOWN for v in values if v != "missing")
                ordered = sorted(values.items(), key=lambda kv: -sum(c.weight for c in kv[1]))
                a_val, a_c = ordered[0]
                label = R.FIELD_LABEL.get(f, f)
                names = {n["need_id"]: (n["service_categories"][0] if isinstance(n["service_categories"], list) and n["service_categories"] else n["need_id"])
                         for c in cands for n in c.artifact["needs"]}
                who = names.get(need_id, need_id)
                if len(values) >= 2:
                    b_val, b_c = ordered[1]
                    text = (f"For the {who} need, which {label} applies: {a_val} or {b_val}? "
                            "Answer from what the person actually said; if they never said, ask them.")
                    tiers[0].append(Question(qid, "intake", {"need_id": need_id, "field": f}, a_c[0].key, b_c[0].key, text))
                elif has_unknown:
                    text = (f"For the {who} need, what is the person's {label}? It is currently unknown. "
                            "Answer from what the person said; if they never said, ask them.")
                    tiers[1].append(Question(qid, "intake", {"need_id": need_id, "field": f}, a_c[0].key, "__residual__", text))
                elif f != "service_categories":
                    text = (f"For the {who} need, is {label} = {a_val} exactly what the person said (including whether they have no constraint)? "
                            "Quote the person's words; if the value is not supported, correct it or ask them.")
                    tiers[2].append(Question(qid, "intake", {"need_id": need_id, "field": f}, a_c[0].key, "__residual__", text))
        menu = tiers[0] + tiers[1] + tiers[2]
        return menu[:Q_MAX]

    # ----- ordinary intake rule -----
    def ask_user_for_missing(self, env, artifact: dict) -> bool:
        missing = R.intake_missing(artifact)
        if not missing:
            return False
        first_field = missing[0][1]
        group = next((g for g in R.INTAKE_GROUPS if first_field in g), (first_field,))
        targets = [(n, f) for (n, f) in missing if f in group]
        fields_text = ", ".join(sorted({R.FIELD_LABEL.get(f, f) for _, f in targets}))
        names = {}
        for n in artifact["needs"]:
            cats = n["service_categories"]
            names[n["need_id"]] = (cats[0] if isinstance(cats, list) and cats else "the service need")
        needs_text = " and ".join(sorted({names.get(n, n) for n, _ in targets}))
        prompt = [{"role": "system", "content": self.native_prompt},
                  {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) +
                   f"\n\nWrite the next assistant message only (no tool call): ask the person, in one short low-burden message, "
                   f"about their {fields_text} for {needs_text}. Ask only these facts; do not repeat already answered facts. "
                   "Do not list example options or suggest values; ask whether they have a requirement or no preference. "
                   "If there are two needs, ask them to answer for each need separately."}]
        text = self.greedy_text(prompt, purpose="intake_question")
        wanted = {R.FIELD_LABEL.get(f, f) for _, f in targets}
        keyword = {"location": ("where", "location", "city", "county", "zip", "area"), "schedule": ("schedule", "day", "time", "when"),
                   "intake method": ("intake", "apply", "contact", "reach", "method"), "available documents": ("document",),
                   "eligibility": ("eligib", "qualif", "requirement")}
        for label in sorted(wanted):
            if not any(k in text.lower() for k in keyword.get(label, (label,))):
                text = text.rstrip() + f" Also, do you have any {label} requirements, or no preference?"
        reply = env.ask_user(text)
        for n, f in targets:
            self.asked.add((n, f))
        self.trace.add("user_question", kind_detail="ordinary_intake", fields=fields_text, text=text, reply=reply)
        return True

    def ask_user_fallback(self, env, request: str) -> None:
        prompt = [{"role": "system", "content": self.native_prompt},
                  {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) +
                   f"\n\nThe last search returned no resources. Write the next assistant message only (no tool call): "
                   f"ask whether the person's {request} can be adjusted, in one short message."}]
        text = self.greedy_text(prompt, purpose="fallback_question")
        reply = env.ask_user(text)
        self.last_fallback_request = request
        self.fallback_overrides = self.resolve_fallback_reply(env, text, reply)
        self.trace.add("fallback_resolution", request=request, overrides={f"{k[0]}|{k[1]}": v for k, v in self.fallback_overrides.items()},
                       adjustable=list(getattr(self, "_adjustable", [])))
        # "I can change it" without a value: ask which value works (ordinary repair), while turns allow search + final.
        adjustable = [a for a in getattr(self, "_adjustable", []) if a not in self.fallback_overrides]
        if adjustable and env.turns_left > 2:
            labels = ", ".join(sorted({R.FIELD_LABEL.get(f, f) for _, f in adjustable}))
            prompt2 = [{"role": "system", "content": self.native_prompt},
                       {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) +
                        f"\n\nThe person said they can adjust their {labels}. Write the next assistant message only (no tool call): ask which specific {labels} "
                        "would work for them instead (or whether any value is fine), in one short message; answer per need if there are two needs."}]
            text2 = self.greedy_text(prompt2, purpose="fallback_followup")
            reply2 = env.ask_user(text2)
            more = self.resolve_fallback_reply(env, text2, reply2)
            for key in adjustable:
                if key not in more and key in getattr(self, "_adjustable", []):
                    more[key] = R.NONE  # still vague after a direct follow-up: treat as no constraint
            self.fallback_overrides.update(more)
            self.trace.add("user_question", kind_detail="fallback_followup", fields=labels, text=text2, reply=reply2)
            self.trace.add("fallback_resolution", request=labels, overrides={f"{k[0]}|{k[1]}": v for k, v in self.fallback_overrides.items()})
        for f in ("location", "schedule", "intake_methods"):
            if f.split("_")[0] in request:
                self.asked.add(("*", f))
        self.trace.add("user_question", kind_detail="fallback_repair", request=request, text=text, reply=reply)

    def resolve_fallback_reply(self, env, question_text: str, reply: str) -> dict:
        """Ordinary repair: interpret the person's answer to a fallback question as per-need field actions."""
        needs = [n for inv in reversed(self.invocations) if inv.role == "intake" for n in inv.packet.candidates[inv.selected].artifact["needs"]][:2]
        need_desc = "; ".join(f"{n['need_id']} = {n['service_categories']}" for n in needs) or "need-1"
        prompt = [{"role": "system", "content": "You interpret a person's reply to a resource-search agent that asked whether a constraint could be relaxed after an empty search. "
                   "For each need and each of the fields location, schedule, intake_methods decide: \"keep\" (not mentioned or cannot change), \"none\" (the person explicitly says any value is fine / they have no preference / no longer need that constraint), "
                   "\"adjustable\" (the person says they can change or adjust it but does not name a new specific value), or a new concrete value in the search format "
                   "(location {\"counties\":[...]}/{\"cities\":[...]}/{\"zipcodes\":[...]}; schedule {\"day\":\"mon\"..\"sun\",\"time\"/\"start_time\"/\"end_time\":\"HH:MM\"} or {\"requires_24_hours\":true}; "
                   "intake_methods array from [appointment, call, email, mail, online, text, walk_in]). A statement about one need does not apply to the other need. "
                   "Output ONLY compact JSON: {\"need-1\": {\"location\": ..., \"schedule\": ..., \"intake_methods\": ...}, \"need-2\": {...}}"},
                  {"role": "user", "content": f"Needs: {need_desc}\nAgent asked: {question_text}\nPerson replied: {reply}"}]
        obj = R.parse_json_object(self.greedy_text(prompt, purpose="fallback_resolve")) or {}
        overrides, adjustable = {}, []
        for n in needs:
            entry = obj.get(n["need_id"]) if isinstance(obj.get(n["need_id"]), dict) else {}
            for f in ("location", "schedule", "intake_methods"):
                action = entry.get(f, "keep")
                if isinstance(action, str) and action.strip().lower() == "keep":
                    continue
                if isinstance(action, str) and action.strip().lower() in ("adjustable", "can change", "flexible"):
                    adjustable.append((n["need_id"], f))
                    continue
                norm = R._norm_field(f, action)
                if norm in (R.UNKNOWN, R.EMPTY):
                    continue
                overrides[(n["need_id"], f)] = norm
        self._adjustable = adjustable
        return overrides

    # ----- sender answers a clarification (source-backed) -----
    def intake_answer(self, env, question: Question) -> dict:
        prompt = [{"role": "system", "content": "You are the intake agent. Answer the planner's clarification strictly from what the person said. "
                   "Output ONLY compact JSON: {\"answer\": \"...\", \"resolved\": true|false, \"sources\": [person turn indices]}. "
                   "Set resolved=false if the conversation does not settle it."},
                  {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) + "\n\nQuestion: " + question.text}]
        obj = R.parse_json_object(self.greedy_text(prompt, purpose="intake_answer")) or {}
        return {"question": question.text, "qid": question.qid, "answer": str(obj.get("answer") or ""),
                "resolved": bool(obj.get("resolved")), "sources": obj.get("sources") or []}

    def intake_ask_user_for(self, env, question: Question) -> str:
        prompt = [{"role": "system", "content": self.native_prompt},
                  {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) +
                   f"\n\nWrite the next assistant message only (no tool call): ask the person one short, specific question to settle this: {question.text} "
                   "Do not suggest values; ask whether they have a requirement or no preference."}]
        text = self.greedy_text(prompt, purpose="clarify_user")
        reply = env.ask_user(text)
        self.asked.add((question.scope.get("need_id", "*"), question.scope.get("field", "")))
        self.trace.add("user_question", kind_detail="clarification", qid=question.qid, text=text, reply=reply)
        return reply

    def _pi0_pick(self, packet: Packet) -> Candidate:
        cands = [c for c in packet.candidates if not c.is_unresolved]
        w = [c.count for c in cands]
        return self.rng.choices(cands, weights=w, k=1)[0] if sum(w) else packet.greedy

    # ----- episode -----
    def run(self, env, *, max_cycles: int = 12, resume: dict | None = None) -> dict:
        if resume is None:
            opening = env.reset()
            st = {"replies": [], "asked": set(), "executed_programs": [], "fallback_fields_asked": []}
        else:
            opening = env.messages[0]["content"]
            st = resume["st"]
        cycles = 0
        outcome = {"terminated": "unresolved"}
        while cycles < max_cycles and not env.done:
            cycles += 1
            if resume is not None and cycles == 1:
                inv, pinv, menu = resume["inv"], resume["pinv"], resume["menu"]
                selected_intake = inv.packet.candidates[inv.selected]
            else:
                # --- sender: intake ---
                inv = self.invoke("intake", self.intake_messages(env), tools=self.roles["intake"].tools,
                                  sources=list(range(1, len(env.user_replies) + 2)), purpose="intake")
                self.check_intake_packet(env, inv.packet)
                self.trace.add("intake_checked", inv_id=inv.inv_id, packet=inv.packet.public())
                self.last_fallback_request = None
                self.fallback_overrides = {}
                selected_intake = self._pi0_pick(inv.packet) if isinstance(self.controller, Pi0Controller) else inv.packet.greedy
                inv.selected = inv.packet.candidates.index(selected_intake)
                # ordinary rule: collect missing facts before searching (reserve 2 turns for search + final)
                if R.intake_missing(selected_intake.artifact) and env.turns_left > 2:
                    self.ask_user_for_missing(env, selected_intake.artifact)
                    continue
                if env.turns_left < 2:
                    break
                # --- receiver: planner ---
                pinv = self.invoke("planner", self.planner_messages(opening, inv.packet, st["replies"]), tools=self.roles["planner"].tools,
                                   ancestors=[inv.inv_id], unresolved_text="UNRESOLVED: the evidence does not support a search program yet.", purpose="planner")
                menu = self.planner_menu(inv.packet, st["asked"]) if env.turns_left > 2 else []
            ctx = ReceiverContext("planner", [inv.packet], pinv.packet.candidates, menu,
                                  self.items_budget - self.items_used, env.turns_left, opening)
            snapshot = {"inv": inv, "pinv": pinv, "menu": menu, "st": copy.deepcopy(st), "sender_invs": [inv]}
            if resume is not None and cycles == 1:
                d = resume["forced"]
                self.trace.add("decision", role="planner", decision=d.kind, candidate=d.candidate, forced=True,
                               question=None if d.question is None else d.question.__dict__, menu=[q.__dict__ for q in menu])
                self.controller = ForcedController(d, self.seed)
                self.controller.used = True
            else:
                d = self.decide(ctx, pinv, snapshot)
            if d.kind == "ask" and d.question is not None:
                self.items_used += d.question.items
                st["asked"].add(d.question.qid)
                ans = self.intake_answer(env, d.question)
                if not ans["resolved"] and env.turns_left > 2:
                    ans["user_reply"] = self.intake_ask_user_for(env, d.question)
                    ans["resolved"] = True
                st["replies"].append(ans)
                self.trace.add("clarification", **ans)
                continue  # intake re-proposes with the reply / new evidence
            chosen = pinv.packet.candidates[d.candidate]
            pinv.selected = d.candidate
            if chosen.is_unresolved:
                if R.intake_missing(selected_intake.artifact) and env.turns_left > 2:
                    self.ask_user_for_missing(env, selected_intake.artifact)
                    continue
                break
            programs = [p["arguments"] for p in chosen.artifact["programs"]]
            prog_key = json.dumps(programs, sort_keys=True)
            if prog_key in st["executed_programs"]:
                remaining = [f for f in ("location", "schedule", "intake") if f not in st["fallback_fields_asked"]]
                if remaining and env.turns_left > 2:
                    st["fallback_fields_asked"].append(remaining[0])
                    self.ask_user_fallback(env, remaining[0])
                    st["replies"], st["asked"] = [], set()
                    continue
                break
            st["executed_programs"].append(prog_key)
            results = env.search(programs)
            if env.done:
                break
            # --- receiver: retriever ---
            rinv = self.invoke("retriever", self.retriever_messages(opening, pinv.packet, chosen, results), tools=self.roles["retriever"].tools,
                               ancestors=[pinv.inv_id], purpose="retriever")
            rctx = ReceiverContext("retriever", [pinv.packet], rinv.packet.candidates, [], 0, env.turns_left, opening)
            rd = self.controller.decide(rctx) if self.carrier is None else self.decide(rctx, rinv, None)
            self.trace.add("decision", role="retriever", decision=rd.kind, candidate=rd.candidate, menu=[])
            rchosen = rinv.packet.candidates[rd.candidate]
            rinv.selected = rd.candidate
            if rchosen.artifact.get("empty") or all(not r.get("resources") for r in results):
                if env.turns_left > 2:
                    req = rchosen.artifact.get("fallback_request", "location, schedule or intake")
                    for f in ("location", "schedule", "intake"):
                        if f in req:
                            st["fallback_fields_asked"].append(f)
                    self.ask_user_fallback(env, req)
                    st["replies"], st["asked"] = [], set()
                    continue
                break
            env.finish(rchosen.artifact["resource_ids"], rchosen.artifact.get("message", ""))
            outcome = {"terminated": "final"}
        return self.record_extra(env.record(), outcome, cycles)


# ==========================================================================
# NGQA-Interactive
# ==========================================================================
class NGQAWorkflow(Workflow):
    dataset = "ngqa"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.roles = R.ngqa_roles()
        from .envs.ngqa_env import guideline_text, food_record_text
        self.guideline = guideline_text()
        self.food_text = food_record_text

    def _conversation_text(self, env) -> str:
        lines, k = [], 0
        for m in env.messages:
            if m["role"] == "user":
                k += 1
                lines.append(f"[person {k}] {m['content']}")
            else:
                lines.append(f"[agent] {m['content']}")
        return "\n".join(lines)

    def profile_messages(self, env) -> list[dict]:
        return [{"role": "system", "content": self.roles["profile"].task},
                {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env)}]

    def check_profile_packet(self, env, packet: Packet) -> None:
        """Lexicon grounding: a 'yes' status must be supported by the person's own words, else it becomes 'unknown'."""
        text = " ".join(m["content"] for m in env.messages if m["role"] == "user").lower()
        spec = self.roles["profile"]
        merged: dict[str, Candidate] = {}
        for c in packet.candidates:
            if c.is_unresolved:
                merged[c.key] = c
                continue
            art = copy.deepcopy(c.artifact)
            changed = False
            for status, val in art["statuses"].items():
                if val == R.YES and not any(k in text for k in R.STATUS_LEXICON.get(status, [])):
                    art["statuses"][status] = R.UNKNOWN
                    changed = True
            if changed:
                key, norm = spec.canon(art)
                c.key, c.artifact, c.text = key, norm, spec.render(norm)
            if c.key in merged:
                m = merged[c.key]
                m.count += c.count
                m.weight += c.weight
                m.greedy = m.greedy or c.greedy
                if c.rep_logprob > m.rep_logprob:
                    m.rep_text, m.rep_logprob, m.states, m.latent = c.rep_text, c.rep_logprob, c.states, c.latent
            else:
                merged[c.key] = c
        packet.candidates = sorted(merged.values(), key=lambda c: (c.is_unresolved, not c.greedy, -c.count))

    def reasoner_messages(self, env, packet: Packet, replies: list[dict]) -> list[dict]:
        analyses = []
        for i, c in enumerate(packet.candidates, 1):
            if c.is_unresolved:
                continue
            yes = [s for s, v in c.artifact["statuses"].items() if v == R.YES]
            ga = R.guideline_analysis(yes, env.public_food["nutrition_tags"])
            analyses.append(f"Candidate {i}: confirmed statuses {yes or 'none'}; contradicted food tags {ga['contradicted'] or 'none'}; "
                            f"matched food tags {ga['matched'] or 'none'}; rule verdict: {ga['verdict']}")
        user = (f"Food record: {self.food_text(env.public_food)}\n\n{self.guideline}\n\n{packet.readable()}\n\nGuideline analysis per candidate:\n"
                + "\n".join(analyses))
        if replies:
            user += "\n\nClarification replies from the profile agent:\n" + "\n".join(
                f"Q: {r['question']}\nA: {r['answer']} (resolved={r.get('resolved')})" for r in replies)
        return [{"role": "system", "content": self.roles["reasoner"].task}, {"role": "user", "content": user}]

    # task-specific module: atomic status questions, ordered by (disagreement, food relevance, support)
    def reasoner_menu(self, env, packet: Packet, asked: set[str]) -> list[Question]:
        from .envs.ngqa_env import guideline
        cands = [c for c in packet.candidates if not c.is_unresolved]
        if not cands:
            return []
        tags = set(env.public_food["nutrition_tags"])
        rel = guideline()["relations"]
        menu = []
        for status in R.status_order():
            if status in asked:
                continue
            values = {}
            for c in cands:
                values.setdefault(c.artifact["statuses"][status], []).append(c)
            relevant = bool(tags & set(rel.get(status, {}).keys()))
            disagree = len(values) >= 2
            if not disagree and R.UNKNOWN not in values and not relevant:
                continue
            ordered = sorted(values.items(), key=lambda kv: -sum(c.weight for c in kv[1]))
            a_val, a_c = ordered[0]
            b_val, b_c = ordered[1] if len(ordered) > 1 else ("the opposite", [])
            tier = 0 if (disagree and relevant) else 1 if (R.UNKNOWN in values and relevant) else 2 if relevant else 3
            menu.append((tier, -sum(c.weight for c in b_c),
                         Question(status, "profile", {"status": status}, a_c[0].key, b_c[0].key if b_c else "__residual__",
                                  f"Does the person have '{status}'? Current interpretations: {a_val} vs {b_val}. "
                                  "Answer from what the person said; if unknown, ask them.")))
        menu.sort(key=lambda t: (t[0], t[1]))
        return [q for _, _, q in menu[:Q_MAX]]

    def profile_answer(self, env, question: Question) -> dict:
        prompt = [{"role": "system", "content": "You are the profile agent. Answer the reasoner's clarification strictly from what the person said. "
                   "Output ONLY compact JSON: {\"answer\": \"...\", \"resolved\": true|false, \"sources\": [person turn indices]}."},
                  {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) + "\n\nQuestion: " + question.text}]
        obj = R.parse_json_object(self.greedy_text(prompt, purpose="profile_answer")) or {}
        return {"question": question.text, "qid": question.qid, "answer": str(obj.get("answer") or ""),
                "resolved": bool(obj.get("resolved")), "sources": obj.get("sources") or []}

    def profile_ask_user(self, env, question: Question) -> str:
        status = question.scope["status"]
        prompt = [{"role": "system", "content": "You are a friendly nutrition assistant talking with a person."},
                  {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) +
                   f"\n\nWrite the next assistant message only: ask the person one short, specific yes/no question about whether they have '{status}', in everyday words."}]
        text = self.greedy_text(prompt, purpose="clarify_user")
        reply = env.ask_user(text)
        self.asked.add(status)
        self.trace.add("user_question", kind_detail="clarification", qid=question.qid, text=text, reply=reply)
        return reply

    def ordinary_open_question(self, env) -> None:
        prompt = [{"role": "system", "content": "You are a friendly nutrition assistant talking with a person."},
                  {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) +
                   "\n\nWrite the next assistant message only: ask the person, in one short open question, whether they have any other health conditions or special diets you should know about."}]
        text = self.greedy_text(prompt, purpose="open_question")
        reply = env.ask_user(text)
        self.trace.add("user_question", kind_detail="ordinary_open", text=text, reply=reply)

    def run(self, env, *, max_cycles: int = 8, resume: dict | None = None) -> dict:
        if resume is None:
            env.reset()
            st = {"replies": [], "asked": set(), "did_open": False}
        else:
            st = resume["st"]
        cycles = 0
        outcome = {"terminated": "unresolved"}
        while cycles < max_cycles and not env.done:
            cycles += 1
            if resume is not None and cycles == 1:
                inv, rinv, menu = resume["inv"], resume["rinv"], resume["menu"]
            else:
                if not st["did_open"] and env.questions_left > 0:
                    self.ordinary_open_question(env)
                    st["did_open"] = True
                inv = self.invoke("profile", self.profile_messages(env), tools=self.roles["profile"].tools,
                                  sources=list(range(1, len(env.user_replies) + 2)), purpose="profile")
                self.check_profile_packet(env, inv.packet)
                self.trace.add("profile_checked", inv_id=inv.inv_id, packet=inv.packet.public())
                inv.selected = inv.packet.candidates.index(inv.packet.greedy)
                rinv = self.invoke("reasoner", self.reasoner_messages(env, inv.packet, st["replies"]), tools=self.roles["reasoner"].tools,
                                   ancestors=[inv.inv_id], purpose="reasoner")
                menu = self.reasoner_menu(env, inv.packet, st["asked"])
            ctx = ReceiverContext("reasoner", [inv.packet], rinv.packet.candidates, menu, self.items_budget - self.items_used,
                                  env.questions_left, self.food_text(env.public_food))
            snapshot = {"inv": inv, "rinv": rinv, "menu": menu, "st": copy.deepcopy(st), "sender_invs": [inv]}
            if resume is not None and cycles == 1:
                d = resume["forced"]
                self.trace.add("decision", role="reasoner", decision=d.kind, candidate=d.candidate, forced=True,
                               question=None if d.question is None else d.question.__dict__, menu=[q.__dict__ for q in menu])
                self.controller = ForcedController(d, self.seed)
                self.controller.used = True
            else:
                d = self.decide(ctx, rinv, snapshot)
            if d.kind == "ask" and d.question is not None:
                self.items_used += d.question.items
                st["asked"].add(d.question.qid)
                ans = self.profile_answer(env, d.question)
                if not ans["resolved"] and env.questions_left > 0:
                    ans["user_reply"] = self.profile_ask_user(env, d.question)
                    ans["resolved"] = True
                st["replies"].append(ans)
                self.trace.add("clarification", **ans)
                continue
            chosen = rinv.packet.candidates[d.candidate]
            rinv.selected = d.candidate
            env.finish(chosen.artifact)
            outcome = {"terminated": "final"}
        return self.record_extra(env.record(), outcome, cycles)
