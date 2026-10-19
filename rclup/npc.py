"""Simulated users (environment-side, GPT-4.1) with cost accounting.

HoosierHelp reuses the benchmark's own LLMSimulatedUser prompt logic verbatim;
only the transport is wrapped so token usage flows into the cost ledger.
NGQA-Interactive defines a nutrition-assistant user who holds a private
health profile and discloses it incrementally.
"""
from __future__ import annotations

import json
import random
import threading
from typing import Any

from .common import ensure_vendor_on_path, load_openai_key
from .cost import CostLedger

NPC_MODEL = "gpt-4.1"
_client_lock = threading.Lock()
_client = None


def openai_client():
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(api_key=load_openai_key(), timeout=90.0, max_retries=4)
        return _client


def responses_call(ledger: CostLedger, *, instructions: str, input_messages: list[dict], max_output_tokens: int,
                   purpose: str, temperature: float = 0.0, model: str = NPC_MODEL) -> str:
    ledger.check()
    response = openai_client().responses.create(model=model, instructions=instructions, input=input_messages,
                                                temperature=temperature, max_output_tokens=max_output_tokens, store=False)
    ledger.record(getattr(response, "model", model) or model, response.usage, purpose)
    return (getattr(response, "output_text", "") or "").strip()


class HoosierUser:
    """Native HoosierHelp simulated user with metered GPT-4.1 transport."""

    def __init__(self, spec: dict, behavior: str, ledger: CostLedger, seed: int = 7, max_output_tokens: int = 512):
        ensure_vendor_on_path()
        from eval import llm_user as native
        self.native_module = native
        self.ledger = ledger
        outer = self

        class Metered(native.LLMSimulatedUser):
            def _generate_responses_api(self, messages, context):
                return responses_call(outer.ledger, instructions=native.USER_SYSTEM_PROMPT,
                                      input_messages=[*native.llm_role_mapped_messages(messages), {"role": "user", "content": context}],
                                      max_output_tokens=self.max_output_tokens, purpose="hoosier_npc", temperature=self.temperature)

        self.user = Metered(spec=spec, user_behavior=behavior, provider="openai", model=NPC_MODEL, seed=seed,
                            temperature=0.0, max_output_tokens=max_output_tokens, enable_thinking=False,
                            thinking_budget_tokens=None, client=object())

    def opening(self) -> str:
        return self.user.opening()

    def respond(self, messages: list[dict], agent_message: str) -> str:
        return self.user.respond(messages, agent_message)

    @property
    def state(self) -> dict:
        return {"contradiction_area": self.user.contradiction_area, "fallback_revealed": self.user.fallback_revealed}


NGQA_USER_SYSTEM = """You are role-playing a real person chatting with a nutrition assistant about whether a particular food is a good choice for you. Stay in character as the person; never reveal that you are simulated or that you have a profile.

Your private health profile is given below. It is the complete truth about you. Answer the assistant's questions truthfully and naturally, in one to three plain sentences, in everyday language (for example, say "my blood pressure is high" or "I'm on a low-salt diet" rather than quoting labels).

Rules for disclosure:
- Your conditions and diets are EXACTLY the items listed under conditions_and_diets; the list "not_your_conditions_or_diets" tells you what you do NOT have. Dietary habits (how you usually eat) are not diets or conditions: never describe a habit as a diet (for example, using little salt does not mean you follow a low-salt diet unless that diet is listed).
- Only answer what the assistant actually asked. Do not volunteer other conditions, diets, or habits.
- If the assistant asks a specific yes/no question about a named condition, diet, or habit, answer it directly and accurately. If it asks about several named items at once, answer each of them.
- If the assistant asks an open-ended question (for example "do you have any health conditions?"), mention at most ONE additional condition or diet from your profile that you have not mentioned yet, the one you consider most important; you may also say you are not sure what else is relevant.
- If something is not in your profile, say you do not have it or that it does not apply to you. Never invent conditions.
- Do not use lists, JSON, labels, or quotation marks. Write exactly one reply."""


ALL_STATUSES = ["obesity", "hypertension", "diabetes", "opioid_misuse", "Low fat/Low cholesterol diet", "Low carbohydrate diet",
                "Sugar free/Low sugar diet", "High protein diet", "Weight loss/Low calorie diet", "Low salt/Low sodium diet",
                "Diabetic diet", "Weight gain/Muscle building diet", "Renal/Kidney diet"]


def _lay(status: str) -> str:
    return {
        "obesity": "I'm quite overweight; my doctor says I'm obese",
        "hypertension": "I have high blood pressure",
        "diabetes": "I have diabetes",
        "opioid_misuse": "I've been struggling with misusing opioid painkillers",
        "Low fat/Low cholesterol diet": "I'm on a low-fat, low-cholesterol diet",
        "Low carbohydrate diet": "I follow a low-carb diet",
        "Sugar free/Low sugar diet": "I'm on a sugar-free, low-sugar diet",
        "High protein diet": "I'm on a high-protein diet",
        "Weight loss/Low calorie diet": "I'm on a low-calorie diet to lose weight",
        "Low salt/Low sodium diet": "I'm on a low-salt diet",
        "Diabetic diet": "I follow a diabetic diet",
        "Weight gain/Muscle building diet": "I'm on a diet to gain weight and build muscle",
        "Renal/Kidney diet": "I'm on a kidney diet",
    }.get(status, status)


class NGQAUser:
    """Simulated user for NGQA-Interactive. The opening discloses a sampled subset."""

    def __init__(self, root: dict, ledger: CostLedger, max_output_tokens: int = 200, disclosure: str = "half"):
        self.root = root
        self.ledger = ledger
        self.max_output_tokens = max_output_tokens
        self.turn = 0
        self.disclosure = disclosure

    def disclosed(self) -> dict:
        d = self.root["opening_disclosure"]
        if self.disclosure == "none":
            return {"statuses": [], "habits": d["habits"]}
        if self.disclosure == "full":
            return {"statuses": list(self.root["private_profile"]["statuses"]), "habits": list(self.root["private_profile"]["habits"])}
        return d

    def opening(self) -> str:
        self.turn += 1
        d = self.disclosed()
        if self.disclosure == "full":
            # native knowledge-QA mode: the whole user record is given as text (no simulated user, no questions)
            return (f"Question: is \"{self.root['food']['name']}\" healthy for this user?\n"
                    f"User record: conditions and special diets: {d['statuses'] or 'none'}; dietary habits: {d['habits'] or 'none'}.")
        parts = [f"Hi, I'm wondering whether \"{self.root['food']['name']}\" is a healthy choice for me."]
        if d["statuses"]:
            parts.append("About me: " + "; ".join(_lay(s) for s in d["statuses"]) + ".")
        if d["habits"]:
            parts.append("Also, " + ", ".join(h[0].lower() + h[1:] for h in d["habits"]) + ".")
        return " ".join(parts)

    def respond(self, messages: list[dict], agent_message: str) -> str:
        self.turn += 1
        profile = self.root["private_profile"]
        context = "\n\n".join([
            "Private profile (the full truth about you):\n" + json.dumps({
                "conditions_and_diets": [_lay(s) for s in profile["statuses"]],
                "not_your_conditions_or_diets": [_lay(s) for s in ALL_STATUSES if s not in profile["statuses"]],
                "dietary_habits": profile["habits"],
                "already_mentioned_in_opening": [_lay(s) for s in self.disclosed()["statuses"]],
            }, ensure_ascii=False, indent=1),
            "The assistant's latest message is the last user turn above. Reply as the person, following the disclosure rules.",
        ])
        mapped = []
        for m in messages[-10:]:
            if m["role"] == "assistant":
                mapped.append({"role": "user", "content": m["content"]})
            elif m["role"] == "user":
                mapped.append({"role": "assistant", "content": m["content"]})
        return responses_call(self.ledger, instructions=NGQA_USER_SYSTEM,
                              input_messages=[*mapped, {"role": "user", "content": context}],
                              max_output_tokens=self.max_output_tokens, purpose="ngqa_npc")
