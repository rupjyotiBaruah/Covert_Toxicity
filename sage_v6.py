#!/usr/bin/env python3
"""
SAGE v6 — Neighborhood-Adaptive Calibration (NAC) head-to-head
==============================================================
Thesis (ours + ACL'24 "Don't Go To Extremes"): implicit/covert toxicity is a
DECISION-BOUNDARY / calibration problem, not a representation problem — models
over-flag (recall >> precision) on locally ambiguous cases. NAC attacks that at
decision time.

Pipeline:
  * fine-tuned MuRIL + head, trained (a) CE-only and (b) CE + SUPERVISED CONTRASTIVE
    front-end (label-coherent neighborhoods).
Decision rules compared on the SAME embeddings/probs:
  CE@0.5 | CE+global-cal (v4) | HatePrototypes | SupCon+global-cal | NAC (ours)
NAC = ambiguity-conditioned LOCAL threshold: t(x)=t_g + lam*(amb(x)-amb_ref), where
amb(x) is the label-entropy of x's k nearest TRAIN neighbors; lam tuned on val.

Metrics: macro-F1, toxic-F1, toxic precision/recall, PR-AUC (threshold-free),
covert-slice toxic-F1, and model-level ECE (to show the miscalibration NAC fixes).

Usage:
  python sage_v6.py --model google/muril-base-cased --dataset local:./data_store/dravidian_tamil
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

from sage_v2_gonogo import load_data
from sage_v3 import set_seed, mine_lexicon

_PUNCT = ".,!?;:\"'()[]{}<>…-–—@#*/\\|~`^%$&+=_ \t\n"
def slang_density(text, s):
    t = [w.strip(_PUNCT).lower() for w in str(text).split()]; t = [w for w in t if w]
    return sum(1 for w in t if w in s) / len(t) if t else 0.0


# ---- model ---------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, model_name, finetune, hdim=256, pdim=128):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(model_name)
        if not finetune:
            for p in self.bb.parameters(): p.requires_grad = False
        d = self.bb.config.hidden_size
        self.proj = nn.Linear(d, pdim)                                  # SupCon projection
        self.head = nn.Sequential(nn.Linear(d, hdim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hdim, 2))

    def forward(self, ids, mask):
        H = self.bb(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (H * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(pooled), pooled, F.normalize(self.proj(pooled), dim=-1)


def supcon_loss(z, y, temp=0.1):
    """Supervised contrastive loss (Khosla et al.)."""
    B = z.size(0)
    sim = z @ z.T / temp
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    logits = sim - sim.max(1, keepdim=True).values.detach()
    log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True) + 1e-12)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~self_mask
    pos_cnt = pos.sum(1).clamp(min=1)
    return (-(log_prob * pos).sum(1) / pos_cnt).mean()


def train(model, tr, device, epochs, lr_head, lr_bb, finetune, class_w, beta, temp):
    model.to(device)
    non_bb = [p for n, p in model.named_parameters() if not n.startswith("bb.") and p.requires_grad]
    groups = [{"params": non_bb, "lr": lr_head}]
    if finetune: groups.append({"params": model.bb.parameters(), "lr": lr_bb})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    class_w = class_w.to(device)
    dl = DataLoader(TensorDataset(*tr), batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for ids, mask, y in dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            logits, _, z = model(ids, mask)
            loss = F.cross_entropy(logits, y, weight=class_w)
            if beta > 0:
                loss = loss + beta * supcon_loss(z, y, temp)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
    return model


@torch.no_grad()
def extract(model, ids, mask, device, bs=64):
    model.eval(); P, E = [], []
    for i in range(0, len(ids), bs):
        logits, pooled, _ = model(ids[i:i+bs].to(device), mask[i:i+bs].to(device))
        P.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        E.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(P).numpy(), torch.cat(E).numpy()


# ---- decision rules & metrics (all numpy) --------------------------------
def toxf1(y, pred):  return f1_score(y, pred, pos_label=1, average="binary", zero_division=0)
def macrof1(y, pred): return f1_score(y, pred, average="macro", zero_division=0)

def best_thr(yv, sv):
    cand = np.quantile(sv, np.linspace(0.05, 0.95, 19))
    bt, bf = float(cand[0]), -1.0
    for t in cand:
        f = toxf1(yv, (sv >= t).astype(int))
        if f > bf: bf, bt = f, float(t)
    return bt

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; N = len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i+1]) if i < bins-1 else (p >= edges[i]) & (p <= edges[i+1])
        if m.sum() == 0: continue
        e += abs(p[m].mean() - y[m].mean()) * m.sum() / N
    return e

def knn_ambiguity(zq, ztr, ytr, k):
    sims = zq @ ztr.T                                   # cosine (normalized)
    idx = np.argpartition(-sims, min(k, sims.shape[1]-1), axis=1)[:, :k]
    p_local = ytr[idx].mean(1)                          # local toxic rate
    return 1.0 - np.abs(2.0 * p_local - 1.0)           # ambiguity in [0,1]

def proto_scores(ztr, ytr, ze):
    p1 = ztr[ytr == 1].mean(0); p0 = ztr[ytr == 0].mean(0)
    p1 /= (np.linalg.norm(p1) + 1e-9); p0 /= (np.linalg.norm(p0) + 1e-9)
    return ze @ p1 - ze @ p0

def row(y, pred, score, cov):
    d = dict(macro=macrof1(y, pred), toxf1=toxf1(y, pred),
             prec=precision_score(y, pred, pos_label=1, zero_division=0),
             rec=recall_score(y, pred, pos_label=1, zero_division=0),
             prauc=average_precision_score(y, score))
    d["covert"] = toxf1(y[cov], pred[cov]) if cov.sum() > 5 and len(set(y[cov])) == 2 else float("nan")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--regime", choices=["frozen", "finetune"], default="finetune")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-bb", type=float, default=2e-5)
    ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-eval", type=int, default=1500)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=0.5, help="SupCon weight")
    ap.add_argument("--temp", type=float, default=0.1, help="SupCon temperature")
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="42,123,2024")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetune = a.regime == "finetune"
    print(f"[env] device={device} model={a.model} dataset={a.dataset} regime={a.regime}")

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model)
    train_all, evl = load_data(a.dataset, a.seed, a.max_train, a.max_eval)
    rng = np.random.default_rng(a.seed); perm = rng.permutation(len(train_all))
    nv = int(a.val_frac * len(train_all))
    val = [train_all[i] for i in perm[:nv]]; trn = [train_all[i] for i in perm[nv:]]
    ytr_list = [y for _, y in trn]
    print(f"[data] train={len(trn)} val={len(val)} test={len(evl)} toxic-frac={np.mean(ytr_list):.2f}")

    mined = mine_lexicon([t for t, _ in trn], ytr_list, top_k=200, min_freq=3)
    slang_set = {m.lower() for m in mined}

    def tok(texts):
        e = tk(texts, padding="max_length", truncation=True, max_length=a.maxlen, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]
    trI, trM = tok([t for t, _ in trn]); trY = torch.tensor(ytr_list)
    vlI, vlM = tok([t for t, _ in val]); vlY = np.array([y for _, y in val])
    teI, teM = tok([t for t, _ in evl]); teY = np.array([y for _, y in evl])
    ytr = np.array(ytr_list)
    cov = np.array([slang_density(t, slang_set) for t, _ in evl]) > 0

    cnt = np.bincount(trY.numpy(), minlength=2).astype(float)
    class_w = torch.tensor(cnt.sum() / (2 * np.clip(cnt, 1, None)), dtype=torch.float32)

    seeds = [int(s) for s in a.seeds.split(",")]
    methods = ["CE@0.5", "CE+global-cal", "HatePrototypes", "SupCon+global-cal", "NAC (ours)"]
    agg = {m: [] for m in methods}
    ece_log = {"CE": [], "SupCon": []}
    print(f"[run] regime={a.regime} epochs={a.epochs} beta={a.beta} k={a.knn_k} seeds={seeds}\n")

    for sd in seeds:
        t0 = time.time()
        outs = {}
        for tag, beta in [("CE", 0.0), ("SupCon", a.beta)]:
            set_seed(sd)
            m = Net(a.model, finetune)
            train(m, (trI, trM, trY), device, a.epochs, a.lr_head, a.lr_bb, finetune, class_w, beta, a.temp)
            p_tr, E_tr = extract(m, trI, trM, device)
            p_vl, E_vl = extract(m, vlI, vlM, device)
            p_te, E_te = extract(m, teI, teM, device)
            outs[tag] = dict(p_tr=p_tr, E_tr=E_tr, p_vl=p_vl, E_vl=E_vl, p_te=p_te, E_te=E_te)
            ece_log[tag].append(ece(teY, p_te))
            del m
            if device == "cuda": torch.cuda.empty_cache()

        ce, sc = outs["CE"], outs["SupCon"]
        # 1) CE @0.5
        agg["CE@0.5"].append(row(teY, (ce["p_te"] >= 0.5).astype(int), ce["p_te"], cov))
        # 2) CE + global calibration
        tg = best_thr(vlY, ce["p_vl"]);  agg["CE+global-cal"].append(row(teY, (ce["p_te"] >= tg).astype(int), ce["p_te"], cov))
        # 3) HatePrototypes (CE embeddings)
        s_vl = proto_scores(ce["E_tr"], ytr, ce["E_vl"]); s_te = proto_scores(ce["E_tr"], ytr, ce["E_te"])
        tp = best_thr(vlY, s_vl); agg["HatePrototypes"].append(row(teY, (s_te >= tp).astype(int), s_te, cov))
        # 4) SupCon + global calibration
        tgS = best_thr(vlY, sc["p_vl"]); agg["SupCon+global-cal"].append(row(teY, (sc["p_te"] >= tgS).astype(int), sc["p_te"], cov))
        # 5) NAC (ours) on SupCon embeddings: ambiguity-conditioned local threshold
        amb_vl = knn_ambiguity(sc["E_vl"], sc["E_tr"], ytr, a.knn_k)
        amb_te = knn_ambiguity(sc["E_te"], sc["E_tr"], ytr, a.knn_k)
        amb_ref = amb_vl.mean()
        best_lam, best_f = 0.0, -1.0
        for lam in np.linspace(-0.35, 0.35, 15):                       # data-driven direction
            t_vl = np.clip(tgS + lam * (amb_vl - amb_ref), 0.05, 0.95)
            f = toxf1(vlY, (sc["p_vl"] >= t_vl).astype(int))
            if f > best_f: best_f, best_lam = f, lam
        t_te = np.clip(tgS + best_lam * (amb_te - amb_ref), 0.05, 0.95)
        agg["NAC (ours)"].append(row(teY, (sc["p_te"] >= t_te).astype(int), sc["p_te"], cov))
        print(f"  seed {sd}: done in {time.time()-t0:.0f}s (NAC lam*={best_lam:+.2f})")

    def mu(m, k): return np.nanmean([r[k] for r in agg[m]])
    def sd_(m, k): return np.nanstd([r[k] for r in agg[m]])

    print("\n" + "=" * 96)
    print(f"REGIME {a.regime.upper()} — {len(seeds)} seeds mean±std   (test)")
    print(f"model calibration ECE (lower=better):  CE={np.mean(ece_log['CE']):.3f}   SupCon={np.mean(ece_log['SupCon']):.3f}")
    print(f"covert slice = {int(cov.sum())}/{len(teY)} test items contain a mined slang token")
    print("=" * 96)
    print(f"  {'method':20} {'macro-F1':>13} {'toxic-F1':>13} {'tox-P':>7} {'tox-R':>7} {'PR-AUC':>8} {'covert-F1':>10}")
    for m in methods:
        star = " *" if m == "NAC (ours)" else "  "
        print(f"{star}{m:20} {mu(m,'macro'):>7.4f}±{sd_(m,'macro'):.3f} {mu(m,'toxf1'):>7.4f}±{sd_(m,'toxf1'):.3f} "
              f"{mu(m,'prec'):>7.4f} {mu(m,'rec'):>7.4f} {mu(m,'prauc'):>8.4f} {mu(m,'covert'):>10.4f}")
    print("-" * 96)
    base = mu("CE+global-cal", "toxf1")
    print(f"  NAC vs CE+global-cal (v4):  toxic-F1 {mu('NAC (ours)','toxf1')-base:+.4f}  |  "
          f"covert-F1 {mu('NAC (ours)','covert')-mu('CE+global-cal','covert'):+.4f}  |  "
          f"tox-precision {mu('NAC (ours)','prec')-mu('CE+global-cal','prec'):+.4f}")
    print("=" * 96)
    print("GENUINE if NAC beats CE+global-cal AND SupCon+global-cal on toxic-F1/covert-F1 across seeds")
    print("(precision gain = it fixed the documented over-flagging). If it only ties -> honest")
    print("analysis paper: 'implicit code-mixed toxicity is a calibration problem'.")


if __name__ == "__main__":
    main()
