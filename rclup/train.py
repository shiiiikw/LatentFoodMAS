"""Phase A: clarification-supervised representation learning (Sec. 4.3, App. B.2, B.6).

Loss (Eq. 7): L_C (BCE over commit actions) + L_Q (BCE over admitted questions)
+ mu * L_Delta (squared difference between predicted and realized action-value
differences over all unordered admitted-action pairs), each prefix-root
weighted so every original question has equal total weight. Both question
terms backpropagate into the pooler, dependency reader and recursive carrier.

Usage:
  python -m rclup.train --collect collect/hoosier_repr_v1 --out models/hoosier_A --epochs 10
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .carrier import Carrier, M_SLOTS, ScalarControl
from .common import RUNS, write_json
from .features import build_features

D_BACKBONE = 4096


def load_prefixes(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        rec = torch.load(p, map_location="cpu", weights_only=False)
        # labels -> mean error per action
        commit = {int(b): statistics.mean(x["error"] for x in v) for b, v in rec["labels"]["commit"].items() if v}
        ask = {q: statistics.mean(x["error"] for x in v) for q, v in rec["labels"]["ask"].items() if v}
        if not commit:
            continue
        rec["y_commit"] = commit
        rec["y_ask"] = ask
        out.append(rec)
    return out


def _cand_scalars(rec: dict):
    cands = rec["data"]["receiver"]["candidates"]
    return (torch.tensor([c["weight"] for c in cands]), torch.tensor([float(c["greedy"]) for c in cands]),
            torch.tensor([float(c["is_unresolved"]) for c in cands]))


def forward_prefix(carrier, rt_pool, rec: dict, generic: bool = False, no_latent: bool = False, head: str = "q_ref"):
    data = rec["data"]
    if hasattr(carrier, "commit_logits"):  # ScalarControl
        feat = build_features(data, None, torch.zeros(M_SLOTS, 256))
        c_logit = carrier.commit_logits(feat.batch, *_cand_scalars(rec))
        q_logit = carrier.question_logits(feat.batch, feat.qfeat, head) if feat.qfeat.items else None
        return c_logit, q_logit, feat
    pooled = []
    for p in data["packets"]:
        if no_latent:
            pooled += [None] * len(p["candidates"])
        elif generic:
            states = [c["states"] for c in p["candidates"] if c.get("states") is not None]
            joint = carrier.pool_candidate(torch.cat(states, 0)) if states else None
            pooled += [joint] * len(p["candidates"])
        else:
            for c in p["candidates"]:
                pooled.append(carrier.pool_candidate(c["states"]) if c.get("states") is not None else None)
    feat = build_features(data, [None if x is None else x for x in pooled], carrier.pooler.null)
    # pooled slots carry gradients: rebuild latent tensor from the list (build_features detached via .float() on CPU copies)
    dev = carrier.state.weight.device
    lat = torch.stack([(x if x is not None else carrier.pooler.null).to(dev) for x in pooled])
    feat.batch.latent = lat
    read = carrier.read(feat.batch)
    q_logits = None
    if feat.qfeat.items:
        qx = carrier.question_features(read, feat.batch, feat.qfeat)
        q_logits = getattr(carrier, head)(qx).squeeze(-1)
    return read["commit_logit"], q_logits, feat


def loss_prefix(carrier, rec, mu: float, generic: bool = False, no_latent: bool = False, head: str = "q_ref", ask_key: str = "y_ask"):
    c_logit, q_logit, feat = forward_prefix(carrier, None, rec, generic, no_latent, head)
    rec = dict(rec, y_ask=rec.get(ask_key, rec["y_ask"]))
    dev = c_logit.device
    yc = rec["y_commit"]
    idx = [b for b in sorted(yc) if b < c_logit.shape[0]]
    lc = F.binary_cross_entropy_with_logits(c_logit[idx], torch.tensor([float(yc[b]) for b in idx], device=dev)) if idx else torch.zeros((), device=dev)
    lq = torch.zeros((), device=dev)
    qids = [it["qid"] for it in feat.qfeat.items]
    qsel = [i for i, q in enumerate(qids) if q in rec["y_ask"]]
    if q_logit is not None and qsel:
        lq = F.binary_cross_entropy_with_logits(q_logit[qsel], torch.tensor([float(rec["y_ask"][qids[i]]) for i in qsel], device=dev))
    # action-profile term over all unordered admitted-action pairs
    probs, ys = [], []
    for b in idx:
        probs.append(torch.sigmoid(c_logit[b])); ys.append(yc[b])
    for i in qsel:
        probs.append(torch.sigmoid(q_logit[i])); ys.append(rec["y_ask"][qids[i]])
    ld = torch.zeros((), device=dev)
    if len(probs) >= 2:
        P = torch.stack(probs); Y = torch.tensor([float(y) for y in ys], device=dev)
        diff_p = P[:, None] - P[None, :]
        diff_y = Y[:, None] - Y[None, :]
        iu = torch.triu_indices(len(probs), len(probs), offset=1)
        ld = ((diff_p[iu[0], iu[1]] - diff_y[iu[0], iu[1]]) ** 2).mean()
    return lc + lq + mu * ld, {"lc": float(lc), "lq": float(lq), "ld": float(ld)}


@torch.no_grad()
def evaluate(carrier, recs, generic=False, no_latent=False, head="q_ref", ask_key="y_ask") -> dict:
    carrier.eval()
    out = {"commit_nll": [], "ask_nll": [], "commit_brier": [], "ask_brier": [], "sel_regret": [], "pair_acc": []}
    for rec in recs:
        rec = dict(rec, y_ask=rec.get(ask_key, rec["y_ask"]))
        c_logit, q_logit, feat = forward_prefix(carrier, None, rec, generic, no_latent, head)
        pc = torch.sigmoid(c_logit).cpu()
        yc = rec["y_commit"]
        idx = [b for b in sorted(yc) if b < len(pc)]
        for b in idx:
            y = yc[b]; p = float(pc[b].clamp(1e-6, 1 - 1e-6))
            out["commit_nll"].append(-(y * torch.log(torch.tensor(p)) + (1 - y) * torch.log(torch.tensor(1 - p))).item())
            out["commit_brier"].append((p - y) ** 2)
        qids = [it["qid"] for it in feat.qfeat.items]
        acts = [(float(pc[b]), yc[b]) for b in idx]
        if q_logit is not None:
            pq = torch.sigmoid(q_logit).cpu()
            for i, q in enumerate(qids):
                if q in rec["y_ask"]:
                    y = rec["y_ask"][q]; p = float(pq[i].clamp(1e-6, 1 - 1e-6))
                    out["ask_nll"].append(-(y * torch.log(torch.tensor(p)) + (1 - y) * torch.log(torch.tensor(1 - p))).item())
                    out["ask_brier"].append((p - y) ** 2)
                    acts.append((p, y))
        if acts:
            best_pred = min(acts, key=lambda t: t[0])[1]
            best_true = min(a[1] for a in acts)
            out["sel_regret"].append(best_pred - best_true)
            pairs = [(acts[i], acts[j]) for i in range(len(acts)) for j in range(i + 1, len(acts)) if acts[i][1] != acts[j][1]]
            if pairs:
                out["pair_acc"].append(statistics.mean(((a[0] < b[0]) == (a[1] < b[1])) for a, b in pairs))
    return {k: (statistics.mean(v) if v else None) for k, v in out.items()} | {"n_prefixes": len(recs)}


def phase_bc(a):
    """Phase B: fit the deployment question head g_k (init from g_k0) on executor-matched labels (y_ask_star),
    carrier frozen. Phase C: fit the Eq. (9) calibrator on action-calibration prefixes."""
    out = Path(a.out) if Path(a.out).is_absolute() else RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in a.collect.split(","):
        d = Path(c) if Path(c).is_absolute() else RUNS / c
        paths += sorted(glob.glob(str(d / "prefixes" / "*.pt")))
    recs = load_prefixes(paths)
    carrier = (ScalarControl() if a.scalar else Carrier(D_BACKBONE)).to(a.device)
    carrier.load_state_dict(torch.load(a.init, map_location="cpu"))
    if a.phase == "B":
        for rec in recs:
            if "labels_star" in rec:
                rec["y_ask_star"] = {q: statistics.mean(x["error"] for x in v) for q, v in rec["labels_star"]["ask"].items() if v}
        recs = [r for r in recs if r.get("y_ask_star")]
        carrier.q_dep.load_state_dict(carrier.q_ref.state_dict())
        for n, p_ in carrier.named_parameters():
            p_.requires_grad_(n.startswith("q_dep."))
        opt = torch.optim.AdamW([p_ for p_ in carrier.parameters() if p_.requires_grad], lr=a.lr, weight_decay=a.wd)
        rng = random.Random(a.seed)
        history = []
        for epoch in range(a.epochs):
            carrier.train()
            rng.shuffle(recs)
            tot = 0.0
            for i in range(0, len(recs), a.batch):
                opt.zero_grad()
                for rec in recs[i: i + a.batch]:
                    l, _ = loss_prefix(carrier, rec, 0.0, a.generic_latent, a.no_latent, head="q_dep", ask_key="y_ask_star")
                    (l / len(recs[i: i + a.batch])).backward()
                    tot += float(l)
                opt.step()
            ev = evaluate(carrier, recs, a.generic_latent, a.no_latent, head="q_dep", ask_key="y_ask_star")
            history.append({"epoch": epoch, "loss": tot / max(1, len(recs)), "fit": ev})
            print(json.dumps(history[-1]))
        torch.save(carrier.state_dict(), out / "carrier.pt")
        write_json(out / "history.json", {"history": history, "args": vars(a)})
        return
    # Phase C: calibrator on (logit, group, label)
    from .carrier import Calibrator
    logits, groups, labels = [], [], []
    carrier.eval()
    with torch.no_grad():
        for rec in recs:
            role = rec["data"]["receiver"]["role"]
            c_logit, q_logit, feat = forward_prefix(carrier, None, rec, a.generic_latent, a.no_latent, head="q_dep")
            for b, y in rec["y_commit"].items():
                if b < len(c_logit):
                    logits.append(float(c_logit[b])); groups.append(f"{role}|commit"); labels.append(float(y))
            ask = rec.get("y_ask_star") or rec["y_ask"]
            if q_logit is not None:
                for i, it in enumerate(feat.qfeat.items):
                    if it["qid"] in ask:
                        logits.append(float(q_logit[i])); groups.append(f"{role}|ask"); labels.append(float(ask[it["qid"]]))
    cal = Calibrator(sorted(set(groups)), l2=1.0).to(a.device)
    cal.fit(torch.tensor(logits, device=a.device), groups, torch.tensor(labels, device=a.device))
    with torch.no_grad():
        p_ = cal(torch.tensor(logits, device=a.device), groups).cpu()
        y = torch.tensor(labels)
        brier = float(((p_ - y) ** 2).mean())
        raw = torch.sigmoid(torch.tensor(logits))
        brier_raw = float(((raw - y) ** 2).mean())
    torch.save({"groups": sorted(set(groups)), "l2": 1.0, "state_dict": cal.state_dict()}, out / "calibrator.pt")
    write_json(out / "calibration.json", {"n": len(labels), "brier_raw": brier_raw, "brier_calibrated": brier,
                                          "params": {k: v.tolist() for k, v in cal.state_dict().items()}, "args": vars(a)})
    print(json.dumps({"n": len(labels), "brier_raw": brier_raw, "brier_calibrated": brier}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collect", required=True, help="collection dir(s), comma separated")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--generic-latent", action="store_true", help="control: pool all candidate traces jointly")
    p.add_argument("--no-latent", action="store_true", help="control: rich text only (latent slots masked)")
    p.add_argument("--scalar", action="store_true", help="control: action-trained scalar controller")
    p.add_argument("--phase", default="A", choices=["A", "B", "C"])
    p.add_argument("--init", default=None, help="phase B/C: trained carrier checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.phase in ("B", "C"):
        return phase_bc(a)
    out = Path(a.out) if Path(a.out).is_absolute() else RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in a.collect.split(","):
        d = Path(c) if Path(c).is_absolute() else RUNS / c
        paths += sorted(glob.glob(str(d / "prefixes" / "*.pt")))
    recs = load_prefixes(paths)
    rng = random.Random(a.seed)
    roots = sorted({r["root_id"] for r in recs})
    rng.shuffle(roots)
    val_roots = set(roots[: max(1, int(len(roots) * a.val_frac))])
    train = [r for r in recs if r["root_id"] not in val_roots]
    val = [r for r in recs if r["root_id"] in val_roots]
    print(f"prefixes: train={len(train)} val={len(val)} roots={len(roots)}")
    torch.manual_seed(a.seed)
    carrier = (ScalarControl() if a.scalar else Carrier(D_BACKBONE)).to(a.device)
    opt = torch.optim.AdamW(carrier.parameters(), lr=a.lr, weight_decay=a.wd)
    best, history = None, []
    for epoch in range(a.epochs):
        carrier.train()
        rng.shuffle(train)
        tot, parts, t0 = 0.0, {"lc": 0.0, "lq": 0.0, "ld": 0.0}, time.monotonic()
        for i in range(0, len(train), a.batch):
            opt.zero_grad()
            batch_loss = 0.0
            for rec in train[i: i + a.batch]:
                l, pr = loss_prefix(carrier, rec, a.mu, a.generic_latent, a.no_latent)
                (l / len(train[i: i + a.batch])).backward()
                batch_loss += float(l)
                for k in parts:
                    parts[k] += pr[k]
            torch.nn.utils.clip_grad_norm_(carrier.parameters(), 1.0)
            opt.step()
            tot += batch_loss
        ev = evaluate(carrier, val, a.generic_latent, a.no_latent) if val else {}
        row = {"epoch": epoch, "train_loss": tot / max(1, len(train)), **{k: v / max(1, len(train)) for k, v in parts.items()},
               "val": ev, "seconds": time.monotonic() - t0}
        history.append(row)
        print(json.dumps(row))
        sel = (ev.get("commit_nll") or 0) + (ev.get("ask_nll") or 0)  # selection action-profile NLL
        if best is None or sel < best[0]:
            best = (sel, epoch)
            torch.save(carrier.state_dict(), out / "carrier.pt")
    write_json(out / "history.json", {"history": history, "best_epoch": best[1] if best else None, "args": vars(a)})


if __name__ == "__main__":
    main()
