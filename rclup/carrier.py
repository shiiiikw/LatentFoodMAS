"""Learned modules of RCLUP (Sec. 4.1-4.4, App. A.2).

  * CandidatePooler   : Eq. (3) attention pooling of response states -> m slots.
  * DependencyReader  : Eq. (4) two joint attention blocks with dependency
                        features in the logits over all incoming slots.
  * PacketNormalizer  : Eq. (5) per-packet candidate attention and packet gate.
  * Heads             : commitment logit l_j(b) (Eq. 1 target), reference
                        question head g_k0 / deployment head g_k, recursive
                        forwarding gate (Eq. 6), reporting head h_psi.
  * Calibrator        : Eq. (9) partially pooled positive-slope logistic map.

Features are built from frozen-backbone states (fp16 on CPU) and projected to
d=256. The generic-latent control shares the same pooling budget but pools all
candidate traces jointly without candidate-to-slot association (Sec. 5.2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 256
M_SLOTS = 4
N_HEADS = 4
DROPOUT = 0.1


def _mlp(inp: int, hidden=(256, 128), out: int = 1) -> nn.Sequential:
    layers, prev = [], inp
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(DROPOUT)]
        prev = h
    layers.append(nn.Linear(prev, out))
    return nn.Sequential(*layers)


class CandidatePooler(nn.Module):
    """Z = softmax(Q (H W_K)^T / sqrt d) H W_V  (Eq. 3). Residual gets learned null slots."""

    def __init__(self, d_backbone: int, d: int = D, m: int = M_SLOTS):
        super().__init__()
        self.q = nn.Parameter(torch.randn(m, d) * 0.02)
        self.wk = nn.Linear(d_backbone, d, bias=False)
        self.wv = nn.Linear(d_backbone, d, bias=False)
        self.null = nn.Parameter(torch.zeros(m, d))
        self.d = d

    def forward(self, states: torch.Tensor) -> torch.Tensor:  # [L, d_B] -> [m, d]
        k = self.wk(states)
        v = self.wv(states)
        att = torch.softmax(self.q @ k.T / math.sqrt(self.d), dim=-1)
        return att @ v


class SlotEncoder(nn.Module):
    """x_u = latent slot + candidate-text enc + role + slot position + missingness (Sec. 4.2)."""

    def __init__(self, d_backbone: int, n_roles: int, d: int = D, m: int = M_SLOTS):
        super().__init__()
        self.text = nn.Linear(d_backbone, d)
        self.role = nn.Embedding(n_roles, d)
        self.slot = nn.Embedding(m, d)
        self.missing = nn.Embedding(2, d)
        self.support = nn.Linear(1, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, latent, text_vec, role_id, slot_ids, missing, weight):
        # latent [n, m, d]; text_vec [n, d_B]; role_id [n]; slot_ids [m]; missing [n]; weight [n]
        n, m, d = latent.shape
        x = (latent + self.text(text_vec)[:, None, :] + self.role(role_id)[:, None, :] + self.slot(slot_ids)[None, :, :]
             + self.missing(missing)[:, None, :] + self.support(weight[:, None])[:, None, :])
        return self.ln(x).reshape(n * m, d)


class DependencyAttention(nn.Module):
    """Multi-head attention with additive dependency features on the logits (Eq. 4)."""

    def __init__(self, d: int = D, heads: int = N_HEADS, n_dep: int = 5):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.eta = nn.Parameter(torch.zeros(heads, n_dep))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, dep):  # x [n, d]; dep [n, n, n_dep]
        n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(n, self.h, self.dk).transpose(0, 1)
        k = k.view(n, self.h, self.dk).transpose(0, 1)
        v = v.view(n, self.h, self.dk).transpose(0, 1)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dk) + torch.einsum("hf,uvf->huv", self.eta, dep)
        att = self.drop(torch.softmax(logits, dim=-1))
        y = (att @ v).transpose(0, 1).reshape(n, d)
        x = x + self.drop(self.o(y))
        return x + self.drop(self.ff(self.ln2(x)))


class Carrier(nn.Module):
    """All trainable modules; shared across roles within a backbone (App. A.2)."""

    def __init__(self, d_backbone: int, n_roles: int = 4, d: int = D, m: int = M_SLOTS, n_graph: int = 8, n_packet_summary: int = 4):
        super().__init__()
        self.d, self.m = d, m
        self.pooler = CandidatePooler(d_backbone, d, m)
        self.slots = SlotEncoder(d_backbone, n_roles, d, m)
        self.reader = nn.ModuleList([DependencyAttention(d), DependencyAttention(d)])
        self.state = nn.Linear(d_backbone, d)                  # s_j from mean-pooled task/evidence states
        self.state_role = nn.Embedding(n_roles, d)
        self.enc_text = nn.Linear(d_backbone, d)               # enc(b), enc(q)
        self.wt = nn.Linear(2 * d, d)                           # t_jb (Eq. 5)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.f_eta = _mlp(3 * d + n_packet_summary, (256,), 1)  # rho logits (Eq. 5)
        self.graph = nn.Linear(n_graph, d)                      # v_j
        self.commit = _mlp(4 * d, (256, 128), 1)                # l_j(b) = f_theta(xi_jb)
        q_in = 9 * d + 2                                        # s, enc(q), hbar_k, hbar_k', diff, |diff|, weights(2), attention summaries(2d), v
        self.q_ref = _mlp(q_in, (256, 128), 1)                  # g_k0
        self.q_dep = _mlp(q_in, (256, 128), 1)                  # g_k (initialized from g_k0 in Phase B)
        self.gate = nn.Linear(2 * d, d)                         # gamma (Eq. 6)
        self.wo = nn.Linear(d, d, bias=False)
        self.out_ln = nn.LayerNorm(d)
        self.report = _mlp(4 * d + 2, (256, 128), 1)            # h_psi (Sec. 4.6)

    # ---------- packet-level ----------
    def pool_candidate(self, states: torch.Tensor | None) -> torch.Tensor:
        if states is None:
            return self.pooler.null
        return self.pooler(states.to(self.pooler.wk.weight.dtype).to(self.pooler.wk.weight.device))

    # ---------- receiver-level ----------
    def read(self, batch: "ReceiverBatch") -> dict:
        """Jointly contextualize all incoming slots and build per-candidate contexts.

        batch.latent [n_c, m, d] (already pooled, possibly through the forwarding gate),
        batch.text [n_c, d_B], roles [n_c], missing [n_c], weight [n_c], packet_id [n_c],
        dep [n_c*m, n_c*m, 5], packet_summary [n_p, 4], graph [n_graph], state_vec [d_B],
        receiver_role (int), cand_text [n_b, d_B], question feats (optional).
        """
        dev = self.state.weight.device
        n_c = batch.latent.shape[0]
        slot_ids = torch.arange(self.m, device=dev)
        x = self.slots(batch.latent.to(dev), batch.text.to(dev).float(), batch.roles.to(dev), slot_ids,
                       batch.missing.to(dev), batch.weight.to(dev).float())
        for block in self.reader:
            x = block(x, batch.dep.to(dev).float())
        h = x.view(n_c, self.m, self.d)                                   # contextual slots h_u
        s = self.state(batch.state_vec.to(dev).float()) + self.state_role(torch.tensor(batch.receiver_role, device=dev))
        v = self.graph(batch.graph.to(dev).float())
        enc_b = self.enc_text(batch.cand_text.to(dev).float())           # [n_b, d]
        # Eq. (5): per-packet candidate attention, then packet attention.
        t = self.wt(torch.cat([s[None, :].expand(enc_b.shape[0], -1), enc_b], -1))   # [n_b, d]
        kh = self.wk(h)                                                  # [n_c, m, d]
        logw = torch.log(batch.weight.to(dev).float().clamp_min(1e-4))[:, None].expand(n_c, self.m) - math.log(self.m)
        contexts, rho_all = [], []
        packet_ids = batch.packet_id.tolist()
        uniq = sorted(set(packet_ids))
        for b in range(enc_b.shape[0]):
            c_pack = []
            for p in uniq:
                idx = [i for i, pid in enumerate(packet_ids) if pid == p]
                logits = torch.einsum("d,cmd->cm", t[b], kh[idx]) / math.sqrt(self.d) + logw[idx]
                beta = torch.softmax(logits.reshape(-1), 0).view(len(idx), self.m)
                c_pack.append(torch.einsum("cm,cmd->d", beta, self.wv(h[idx])))
            c_pack = torch.stack(c_pack)                                 # [n_p, d]
            summ = batch.packet_summary.to(dev).float()                  # [n_p, 4]
            logit_rho = self.f_eta(torch.cat([s[None].expand(len(uniq), -1), enc_b[b][None].expand(len(uniq), -1), c_pack, summ], -1)).squeeze(-1)
            logit_rho = logit_rho - torch.log(batch.packet_overlap.to(dev).float().clamp_min(1.0))
            rho = torch.softmax(logit_rho, 0)
            contexts.append(rho @ c_pack)
            rho_all.append(rho)
        c = torch.stack(contexts)                                        # [n_b, d]
        xi = torch.cat([s[None].expand(c.shape[0], -1), enc_b, c, v[None].expand(c.shape[0], -1)], -1)
        return {"h": h, "s": s, "v": v, "enc_b": enc_b, "c": c, "xi": xi, "rho": rho_all,
                "commit_logit": self.commit(xi).squeeze(-1)}

    def question_features(self, read: dict, batch: "ReceiverBatch", qfeat: "QuestionFeatures") -> torch.Tensor:
        """xi_jq = [s; enc(q); hbar_k; hbar_k'; diff; |diff|; weights; receiver-attention summaries; v]."""
        dev = self.state.weight.device
        h = read["h"]
        hbar = h.mean(1)                                                 # [n_c, d]
        rows = []
        for q in qfeat.items:
            a = hbar[q["a"]] if q["a"] is not None else self.pooler.null.mean(0)
            b = hbar[q["b"]] if q["b"] is not None else self.pooler.null.mean(0)
            enc_q = self.enc_text(q["text_vec"].to(dev).float())
            w = torch.tensor([q["wa"], q["wb"]], device=dev)
            # attention summaries: mean context of the receiver's best candidate, and s
            att = torch.cat([read["c"][q["best_b"]], read["s"]])
            rows.append(torch.cat([read["s"], enc_q, a, b, a - b, (a - b).abs(), w, att, read["v"]]))
        return torch.stack(rows) if rows else torch.zeros(0, 1, device=dev)

    def forward_gate(self, local_latent: torch.Tensor, s: torch.Tensor, enc_b: torch.Tensor, c_b: torch.Tensor) -> torch.Tensor:
        """Eq. (6): Z_out = LN(Z_loc + gamma * W_o c)."""
        gamma = torch.sigmoid(self.gate(torch.cat([s, enc_b])))
        return self.out_ln(local_latent + gamma[None, :] * self.wo(c_b)[None, :])


@dataclass
class ReceiverBatch:
    latent: torch.Tensor          # [n_c, m, d]
    text: torch.Tensor            # [n_c, d_B]
    roles: torch.Tensor           # [n_c]
    missing: torch.Tensor         # [n_c]
    weight: torch.Tensor          # [n_c]
    packet_id: torch.Tensor       # [n_c]
    dep: torch.Tensor             # [n_c*m, n_c*m, 5]
    packet_summary: torch.Tensor  # [n_p, 4]
    packet_overlap: torch.Tensor  # [n_p]  v_i = 1 + sum overlaps
    graph: torch.Tensor           # [n_graph]
    state_vec: torch.Tensor       # [d_B]
    receiver_role: int
    cand_text: torch.Tensor       # [n_b, d_B]


@dataclass
class QuestionFeatures:
    items: list[dict]             # each: a, b (candidate row index or None), wa, wb, text_vec [d_B], best_b


class Calibrator(nn.Module):
    """Eq. (9): r = sigmoid(exp(a0 + a_g) * l + b0 + b_g), groups = (role, family)."""

    def __init__(self, groups: list[str], l2: float = 1.0):
        super().__init__()
        self.groups = {g: i for i, g in enumerate(groups)}
        self.a0 = nn.Parameter(torch.zeros(()))
        self.b0 = nn.Parameter(torch.zeros(()))
        self.ag = nn.Parameter(torch.zeros(len(groups)))
        self.bg = nn.Parameter(torch.zeros(len(groups)))
        self.l2 = l2

    def forward(self, logit: torch.Tensor, group: list[str]) -> torch.Tensor:
        idx = torch.tensor([self.groups.get(g, -1) for g in group], device=logit.device)
        ag = torch.where(idx >= 0, self.ag[idx.clamp_min(0)], torch.zeros_like(logit))
        bg = torch.where(idx >= 0, self.bg[idx.clamp_min(0)], torch.zeros_like(logit))
        return torch.sigmoid(torch.exp(self.a0 + ag) * logit + self.b0 + bg)

    def penalty(self) -> torch.Tensor:
        return self.l2 * (self.ag.pow(2).sum() + self.bg.pow(2).sum())

    def fit(self, logits: torch.Tensor, groups: list[str], labels: torch.Tensor, weights: torch.Tensor | None = None, steps: int = 500):
        opt = torch.optim.LBFGS(self.parameters(), lr=0.5, max_iter=steps)
        w = weights if weights is not None else torch.ones_like(labels)

        def closure():
            opt.zero_grad()
            p = self(logits, groups).clamp(1e-6, 1 - 1e-6)
            loss = (w * F.binary_cross_entropy(p, labels, reduction="none")).sum() / w.sum() + self.penalty()
            loss.backward()
            return loss
        opt.step(closure)
        return self


class ScalarControl(nn.Module):
    """Action-trained scalar controller: risk heads over scalar features only (support, counts, budgets, role)."""

    def __init__(self, n_roles: int = 4, n_graph: int = 8):
        super().__init__()
        self.role = nn.Embedding(n_roles, 8)
        self.commit = _mlp(8 + n_graph + 6, (64, 32), 1)
        self.q_ref = _mlp(8 + n_graph + 7, (64, 32), 1)
        self.q_dep = _mlp(8 + n_graph + 7, (64, 32), 1)

    def commit_logits(self, batch: "ReceiverBatch", cand_weights: torch.Tensor, cand_greedy: torch.Tensor, cand_unresolved: torch.Tensor):
        dev = self.role.weight.device
        n_b = cand_weights.shape[0]
        role = self.role(torch.tensor(batch.receiver_role, device=dev))[None].expand(n_b, -1)
        g = batch.graph.to(dev).float()[None].expand(n_b, -1)
        w = batch.weight.to(dev).float()
        ent = -(w * w.clamp_min(1e-6).log()).sum()
        scal = torch.stack([cand_weights.to(dev).float(), cand_greedy.to(dev).float(), cand_unresolved.to(dev).float(),
                            torch.full((n_b,), float(len(w)), device=dev), w.max().expand(n_b), ent.expand(n_b)], -1)
        return self.commit(torch.cat([role, g, scal], -1)).squeeze(-1)

    def question_logits(self, batch: "ReceiverBatch", qfeat: "QuestionFeatures", head: str = "q_ref"):
        dev = self.role.weight.device
        rows = []
        w = batch.weight.to(dev).float()
        ent = -(w * w.clamp_min(1e-6).log()).sum()
        for q in qfeat.items:
            role = self.role(torch.tensor(batch.receiver_role, device=dev))
            scal = torch.tensor([q["wa"], q["wb"], abs(q["wa"] - q["wb"]), float(len(qfeat.items)), float(len(w)), float(w.max()), float(ent)], device=dev)
            rows.append(torch.cat([role, batch.graph.to(dev).float(), scal]))
        x = torch.stack(rows) if rows else torch.zeros(0, 8 + 8 + 7, device=dev)
        return getattr(self, head)(x).squeeze(-1)
