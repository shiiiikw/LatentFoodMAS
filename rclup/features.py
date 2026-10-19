"""Glue between workflow objects and the learned carrier.

`encode_context` turns a receiver decision context into plain data (texts,
weights, provenance, frozen-backbone states) that can be saved for offline
training; `build_features` turns that data into tensors for the carrier
(ReceiverBatch + QuestionFeatures). Dependency features (Eq. 11) per slot pair:
  [same origin, origin_u in ancestors_v, origin_v in ancestors_u,
   Jaccard(ancestors), Jaccard(source spans)].
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from .carrier import Carrier, Calibrator, M_SLOTS, QuestionFeatures, ReceiverBatch

ROLE_IDS = {"intake": 0, "planner": 1, "retriever": 2, "profile": 0, "reasoner": 1, "unknown": 3}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class TextCache:
    def __init__(self, encoder, max_items: int = 20000):
        self.encoder = encoder
        self.cache: dict[str, torch.Tensor] = {}
        self.max_items = max_items

    def __call__(self, text: str) -> torch.Tensor:
        key = hashlib.sha256(text.encode()).hexdigest()
        v = self.cache.get(key)
        if v is None:
            v = self.encoder.encode_text(text)
            if len(self.cache) < self.max_items:
                self.cache[key] = v
        return v


def encode_context(ctx, text_cache: TextCache, external_budget_frac: float = 0.0) -> dict:
    """Plain, saveable description of a receiver decision point."""
    packets = []
    for p in ctx.incoming:
        cands = []
        for c in p.candidates:
            if c.is_unresolved:
                continue
            cands.append({"key": c.key, "text": c.text, "weight": float(c.weight), "count": int(c.count), "greedy": bool(c.greedy),
                          "sources": list(c.sources), "states": c.states, "text_vec": text_cache(c.text)})
        packets.append({"origin": p.origin, "version": p.version, "sender_role": p.sender_role, "ancestors": list(p.ancestors),
                        "sources": list(p.sources), "residual_weight": float(p.residual_weight), "candidates": cands})
    receiver = {"role": ctx.role, "task_state_text": ctx.task_state_text,
                "state_vec": ctx.state_vec if ctx.state_vec is not None else text_cache(ctx.task_state_text),
                "candidates": [{"key": c.key, "text": c.text, "is_unresolved": bool(c.is_unresolved), "greedy": bool(c.greedy),
                                "weight": float(c.weight), "text_vec": text_cache(c.text)} for c in ctx.candidates],
                "greedy_index": ctx.greedy_index(), "items_left": ctx.items_left, "external_left": ctx.external_left}
    menu = [{"qid": q.qid, "target_role": q.target_role, "scope": q.scope, "a": q.a, "b": q.b, "text": q.text, "items": q.items,
             "text_vec": text_cache(q.text)} for q in ctx.menu]
    return {"packets": packets, "receiver": receiver, "menu": menu, "external_budget_frac": external_budget_frac}


@dataclass
class Featurized:
    batch: ReceiverBatch
    qfeat: QuestionFeatures
    slot_candidates: list[tuple[str, str]]
    cand_meta: list[tuple[float, float, float]] | None = None  # (weight, greedy, unresolved) per receiver candidate


def build_features(data: dict, pooled: list[torch.Tensor] | None = None, null_slot: torch.Tensor | None = None,
                   m: int = M_SLOTS) -> Featurized:
    """Tensors for the carrier. `pooled[i]` are the [m, d] slots per sender candidate row (None -> null slots)."""
    latents, texts, roles, missing, weights, pids, slot_cands = [], [], [], [], [], [], []
    origins, ancestors, sources = [], [], []
    row = 0
    for p_i, packet in enumerate(data["packets"]):
        for c in packet["candidates"]:
            lat = pooled[row] if pooled is not None and pooled[row] is not None else null_slot
            latents.append(lat.float())
            texts.append(c["text_vec"].float())
            roles.append(ROLE_IDS.get(packet["sender_role"], 3))
            missing.append(int(pooled is None or pooled[row] is None))
            weights.append(float(c["weight"]))
            pids.append(p_i)
            slot_cands.append((packet["origin"], c["key"]))
            origins.append(packet["origin"])
            ancestors.append(set(packet["ancestors"]) | {packet["origin"]})
            sources.append(set(c["sources"]))
            row += 1
    n_c = len(latents)
    if n_c == 0:
        raise ValueError("Receiver context without incoming candidates")
    n = n_c * m
    dep = torch.zeros(n, n, 5)
    for u in range(n_c):
        for v in range(n_c):
            f = torch.tensor([float(origins[u] == origins[v]), float(origins[u] in ancestors[v]), float(origins[v] in ancestors[u]),
                              _jaccard(ancestors[u], ancestors[v]), _jaccard(sources[u], sources[v])])
            dep[u * m:(u + 1) * m, v * m:(v + 1) * m] = f
    packets = data["packets"]
    n_p = len(packets)
    summ = torch.zeros(n_p, 4)
    overlap = torch.ones(n_p)
    for i, p in enumerate(packets):
        ov = [max(_jaccard(set(p["ancestors"]) | {p["origin"]}, set(q["ancestors"]) | {q["origin"]}),
                  _jaccard(set(p["sources"]), set(q["sources"]))) for j, q in enumerate(packets) if j != i]
        summ[i] = torch.tensor([sum(ov) / len(ov) if ov else 0.0, max(ov) if ov else 0.0, float(len(p["ancestors"]) + 1),
                                float(sum(pooled is None or pooled[k] is None for k in range(len(p["candidates"]))))])
        overlap[i] = 1.0 + sum(ov)
    cand_weights = torch.tensor(weights)
    ent = float(-(cand_weights * cand_weights.clamp_min(1e-6).log()).sum())
    graph = torch.tensor([float(n_p), float(len(set(origins))), float(max(len(a) for a in ancestors)),
                          float(summ[:, 1].max()) if n_p else 0.0, float(sum(r == 0 for r in roles)), float(sum(r == 1 for r in roles)),
                          ent, float(data.get("external_budget_frac", 0.0))])
    rec = data["receiver"]
    cand_text = torch.stack([c["text_vec"].float() for c in rec["candidates"]])
    batch = ReceiverBatch(latent=torch.stack(latents), text=torch.stack(texts), roles=torch.tensor(roles), missing=torch.tensor(missing),
                          weight=cand_weights, packet_id=torch.tensor(pids), dep=dep, packet_summary=summ, packet_overlap=overlap,
                          graph=graph, state_vec=rec["state_vec"].float(), receiver_role=ROLE_IDS.get(rec["role"], 3), cand_text=cand_text)
    key_to_row = {}
    for i, sc in enumerate(slot_cands):
        key_to_row.setdefault(sc[1], i)
    items = []
    best_b = rec["greedy_index"]
    for q in data["menu"]:
        ra, rb = key_to_row.get(q["a"]), key_to_row.get(q["b"])
        items.append({"a": ra, "b": rb, "wa": weights[ra] if ra is not None else 0.0, "wb": weights[rb] if rb is not None else 0.0,
                      "text_vec": q["text_vec"].float(), "best_b": best_b, "qid": q["qid"]})
    meta = [(float(c["weight"]), float(c["greedy"]), float(c["is_unresolved"])) for c in rec["candidates"]]
    return Featurized(batch, QuestionFeatures(items), slot_cands, meta)


class CarrierRuntime:
    """Holds the carrier + encoder cache; produces features online for a receiver context."""

    def __init__(self, carrier: Carrier, encoder, device: str = "cuda", generic_latent: bool = False):
        self.carrier = carrier.to(device)
        self.encoder = encoder
        self.device = device
        self.text = TextCache(encoder)
        self.generic_latent = generic_latent

    @torch.no_grad()
    def pool_packet(self, packet) -> None:
        self.carrier.eval()
        if not hasattr(self.carrier, "pool_candidate"):  # ScalarControl: no latent slots
            for c in packet.candidates:
                c.latent = None
            return
        if self.generic_latent:
            states = [c.states for c in packet.candidates if c.states is not None]
            joint = torch.cat(states, 0) if states else None
            pooled = self.carrier.pool_candidate(joint).detach().cpu()
            for c in packet.candidates:
                c.latent = pooled.clone()
            return
        for c in packet.candidates:
            c.latent = self.carrier.pool_candidate(c.states).detach().cpu() if c.states is not None else None

    no_latent = False

    @torch.no_grad()
    def pooled_rows(self, data: dict) -> list[torch.Tensor | None]:
        self.carrier.eval()
        rows = []
        if self.no_latent:
            return [None for p in data["packets"] for _ in p["candidates"]]
        if self.generic_latent:
            for p in data["packets"]:
                states = [c["states"] for c in p["candidates"] if c.get("states") is not None]
                joint = self.carrier.pool_candidate(torch.cat(states, 0)).detach().cpu() if states else None
                rows += [joint] * len(p["candidates"])
            return rows
        for p in data["packets"]:
            for c in p["candidates"]:
                rows.append(self.carrier.pool_candidate(c["states"]).detach().cpu() if c.get("states") is not None else None)
        return rows

    def featurize(self, ctx) -> Featurized:
        data = encode_context(ctx, self.text)
        if not hasattr(self.carrier, "pool_candidate"):
            return build_features(data, None, torch.zeros(M_SLOTS, 256))
        return build_features(data, self.pooled_rows(data), self.carrier.pooler.null.detach().cpu())

    @torch.no_grad()
    def score(self, feat: Featurized, head: str = "q_dep") -> dict:
        self.carrier.eval()
        if hasattr(self.carrier, "commit_logits"):  # ScalarControl
            cands = feat.cand_meta
            commit = self.carrier.commit_logits(feat.batch, torch.tensor([c[0] for c in cands]), torch.tensor([c[1] for c in cands]),
                                                torch.tensor([c[2] for c in cands])).detach().cpu()
            qlog = self.carrier.question_logits(feat.batch, feat.qfeat, head).detach().cpu() if feat.qfeat.items else torch.zeros(0)
            return {"commit_logits": commit, "question_logits": qlog}
        read = self.carrier.read(feat.batch)
        commit = read["commit_logit"].detach().cpu()
        if feat.qfeat.items:
            qx = self.carrier.question_features(read, feat.batch, feat.qfeat)
            qlog = getattr(self.carrier, head)(qx).squeeze(-1).detach().cpu()
        else:
            qlog = torch.zeros(0)
        return {"commit_logits": commit, "question_logits": qlog}


class RCLUPController:
    """Eq. (8) commit rule + Eq. (10) budgeted clarification with calibrated risks."""
    name = "rclup"

    def __init__(self, runtime: CarrierRuntime, calibrator: Calibrator | None, lam: float = 0.02, head: str = "q_dep",
                 min_external_reserve: int = 2, name: str | None = None):
        self.rt = runtime
        self.cal = calibrator
        self.lam = lam
        self.head = head
        self.reserve = min_external_reserve
        if name:
            self.name = name

    def _risk(self, logits: torch.Tensor, groups: list[str]) -> torch.Tensor:
        if self.cal is None:
            return torch.sigmoid(logits)
        return self.cal(logits.to(next(self.cal.parameters()).device), groups).detach().cpu()

    def decide(self, ctx):
        from .workflow import Decision
        feat = ctx.features
        sc = self.rt.score(feat, self.head)
        commit_logits = sc["commit_logits"]
        b_star = int(torch.argmin(commit_logits))
        r_commit = self._risk(commit_logits, [f"{ctx.role}|commit"] * len(commit_logits))
        scores = {"commit_logits": commit_logits.tolist(), "commit_risk": r_commit.tolist(), "b_star": b_star}
        best_q, best_gain = None, 0.0
        if ctx.menu and sc["question_logits"].numel():
            r_ask = self._risk(sc["question_logits"], [f"{ctx.role}|ask"] * len(ctx.menu))
            gains = []
            for q, r in zip(ctx.menu, r_ask.tolist()):
                g = float(r_commit[b_star]) - r - self.lam * q.items
                gains.append(g)
                if q.items <= ctx.items_left and ctx.external_left > self.reserve and g > best_gain:
                    best_q, best_gain = q, g
            scores.update({"ask_risk": r_ask.tolist(), "gains": gains})
        if best_q is not None:
            return Decision("ask", question=best_q, scores=scores)
        return Decision("commit", b_star, scores=scores)
