"""NGQA-Interactive environment: public food record, private user profile.

Scoring follows the native NGQA tasks: binary accuracy (primary), tag-set F1
for the multi-label task, ROUGE-L F1 for the explanation task. The final
error event E=1 is a wrong (or missing) binary answer.
"""
from __future__ import annotations

import copy
import json
import re

from ..common import DATA, read_json
from ..cost import CostLedger
from ..npc import NGQAUser

MAX_USER_QUESTIONS = 4
_GUIDELINE = None


def guideline() -> dict:
    global _GUIDELINE
    if _GUIDELINE is None:
        _GUIDELINE = read_json(DATA / "ngqa" / "guideline.json")
    return _GUIDELINE


def guideline_text() -> str:
    g = guideline()["relations"]
    lines = ["Nutrition guideline (condition or diet -> food nutrition tag relation):"]
    for cond in guideline()["status_order"]:
        rel = g.get(cond, {})
        matches = sorted(t for t, r in rel.items() if r == "match")
        contras = sorted(t for t, r in rel.items() if r == "contradict")
        lines.append(f"- {cond}: contradicts {', '.join(contras) or 'none'}; matches {', '.join(matches) or 'none'}")
    lines.append("A food is unhealthy for the user when any of the user's conditions/diets contradicts one of the food's tags "
                 "(sparse/standard profiles); when both matching and contradicting tags exist, weigh them (complex profiles).")
    return "\n".join(lines)


def food_record_text(food: dict) -> str:
    return json.dumps({"food": food["name"], "category": food["category"], "ingredients": food["ingredients"],
                       "nutrition_tags": food["nutrition_tags"]}, ensure_ascii=False)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def rouge_l_f1(pred: str, ref: str) -> float:
    a, b = _tokens(pred), _tokens(ref)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


class NGQAEnv:
    def __init__(self, root: dict, ledger: CostLedger, max_user_questions: int = MAX_USER_QUESTIONS, disclosure: str = "half"):
        self.root = root
        self.max_user_questions = 0 if disclosure == "full" else max_user_questions
        self.disclosure = disclosure
        self.user = NGQAUser(root, ledger, disclosure=disclosure)
        self.messages: list[dict] = []
        self.user_replies: list[int] = []
        self.final: dict | None = None
        self.done = False

    def reset(self) -> str:
        opening = self.user.opening()
        self.messages = [{"role": "user", "content": opening}]
        return opening

    @property
    def public_food(self) -> dict:
        return copy.deepcopy(self.root["food"])

    @property
    def questions_left(self) -> int:
        return self.max_user_questions - len(self.user_replies)

    def ask_user(self, text: str) -> str:
        if self.done:
            raise RuntimeError("Episode finished")
        if self.questions_left <= 0:
            raise RuntimeError("User-question budget exhausted")
        self.messages.append({"role": "assistant", "content": text})
        reply = self.user.respond(self.messages, text)
        self.messages.append({"role": "user", "content": reply})
        self.user_replies.append(len(self.messages) - 1)
        return reply

    def fork(self) -> "NGQAEnv":
        clone = NGQAEnv.__new__(NGQAEnv)
        clone.__dict__.update({k: v for k, v in self.__dict__.items() if k != "user"})
        clone.messages = copy.deepcopy(self.messages)
        clone.user_replies = list(self.user_replies)
        clone.final = copy.deepcopy(self.final)
        clone.user = NGQAUser(self.root, self.user.ledger, self.user.max_output_tokens, disclosure=self.disclosure)
        clone.user.turn = self.user.turn
        return clone

    def state(self) -> dict:
        return {"messages": copy.deepcopy(self.messages), "user_replies": list(self.user_replies), "final": copy.deepcopy(self.final),
                "done": self.done, "user_turn": self.user.turn, "disclosure": self.disclosure}

    @classmethod
    def from_state(cls, root: dict, ledger: CostLedger, state: dict) -> "NGQAEnv":
        env = cls(root, ledger, disclosure=state.get("disclosure", "half"))
        env.messages = copy.deepcopy(state["messages"])
        env.user_replies = list(state["user_replies"])
        env.final = copy.deepcopy(state["final"])
        env.done = state["done"]
        env.user.turn = state["user_turn"]
        return env

    def finish(self, answer: dict) -> dict:
        if self.done:
            raise RuntimeError("Episode finished")
        self.final = {"binary": (answer.get("binary") or "").strip().lower(), "tags": [t.strip().lower() for t in (answer.get("tags") or [])],
                      "explain": answer.get("explain") or ""}
        self.messages.append({"role": "assistant", "content": json.dumps(self.final, ensure_ascii=False)})
        self.done = True
        return self.score()

    def score(self) -> dict:
        labels = self.root["labels"]
        f = self.final or {"binary": "", "tags": [], "explain": ""}
        gold_bin = labels["binary"].lower()
        valid = f["binary"] in {"yes", "no"}
        pred_tags, gold_tags = set(f["tags"]), {t.lower() for t in labels["tags"]}
        inter = len(pred_tags & gold_tags)
        tag_f1 = (2 * inter / (len(pred_tags) + len(gold_tags))) if (pred_tags or gold_tags) else 1.0
        return {"binary_correct": bool(valid and f["binary"] == gold_bin), "binary_valid": valid,
                "tag_f1": tag_f1, "explain_rouge_l": rouge_l_f1(f["explain"], labels["explain"]),
                "user_questions": len(self.user_replies), "answered": self.final is not None,
                "end_to_end_match": bool(valid and f["binary"] == gold_bin)}

    def record(self) -> dict:
        return {"root_id": self.root["root_id"], "difficulty": self.root["difficulty"], "disclosure": self.disclosure, "messages": copy.deepcopy(self.messages),
                "user_reply_indices": list(self.user_replies), "final": copy.deepcopy(self.final), "score": self.score()}
