#!/usr/bin/env python3
"""
SAGE v4 — imbalance-aware, threshold-calibrated recipe for covert code-mixed toxicity
=====================================================================================
Motivation (grounded in our own v3 finding): a fine-tuned encoder is strong on macro-F1
but UNDER-DETECTS the minority toxic class — it trades recall for precision at the default
0.5 threshold. This script tests whether an imbalance-aware loss + per-language decision-
threshold calibration recovers toxic-class F1, the metric that actually matters here.

Configs compared (all: fine-tuned MuRIL + MLP head; identical except the loss):
    ce         : class-weighted cross-entropy               (the current baseline)
    focal      : class-balanced focal loss (gamma)          (down-weights easy negatives)
    logit_adj  : logit-adjusted softmax (tau * log prior)   (Menon et al., 2020)
Each is evaluated at the default 0.5 threshold AND at a threshold calibrated on a held-out
val split to maximize toxic-F1. Metrics: macro-F1, toxic-F1 (threshold-dependent) and
PR-AUC (threshold-free), averaged over seeds.

Usage:
    python sage_v4.py --model google/muril-base-cased --dataset local:./data_store/dravidian_tamil
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score

from sage_v2_gonogo import load_data
from sage_v3 import set_seed


class Net(nn.Module):
    def __init__(self, model_name, finetune=True):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(model_name)
        if not finetune:
            for p in self.bb.parameters(): p.requires_grad = False
        d = self.bb.config.hidden_size
        self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 2))

    def forward(self, ids, mask):
        H = self.bb(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (H * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(pooled)


# ---- losses --------------------------------------------------------------
def loss_ce(logits, y, class_w, prior, gamma, tau):
    return F.cross_entropy(logits, y, weight=class_w)

def loss_focal(logits, y, class_w, prior, gamma, tau):
    logp = F.log_softmax(logits, dim=-1)
    idx = torch.arange(len(y), device=y.device)
    logp_y = logp[idx, y]
    p_y = logp_y.exp()
    alpha = class_w[y]                                   # class-balanced focal
    return -(alpha * (1 - p_y).clamp(min=0) ** gamma * logp_y).mean()

def loss_logit_adj(logits, y, class_w, prior, gamma, tau):
    adj = tau * torch.log(prior.clamp(min=1e-12))        # (2,)
    return F.cross_entropy(logits + adj, y)              # inference uses plain logits

LOSSES = {"ce": loss_ce, "focal": loss_focal, "logit_adj": loss_logit_adj}


# ---- train / predict -----------------------------------------------------
def train(model, tr, device, epochs, lr_head, lr_bb, finetune, loss_fn, class_w, prior, gamma, tau):
    model.to(device)
    non_bb = [p for n, p in model.named_parameters() if not n.startswith("bb.") and p.requires_grad]
    groups = [{"params": non_bb, "lr": lr_head}]
    if finetune:
        groups.append({"params": model.bb.parameters(), "lr": lr_bb})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    class_w, prior = class_w.to(device), prior.to(device)
    dl = DataLoader(TensorDataset(*tr), batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for ids, mask, y in dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            loss = loss_fn(model(ids, mask), y, class_w, prior, gamma, tau)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
    return model


@torch.no_grad()
def predict(model, ids, mask, device, bs=64):
    model.eval(); out = []
    for i in range(0, len(ids), bs):
        logits = model(ids[i:i+bs].to(device), mask[i:i+bs].to(device))
        out.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return torch.cat(out).numpy()


def f1s_at(y, prob, thr):
    pred = (prob >= thr).astype(int)
    return (f1_score(y, pred, average="macro", zero_division=0),
            f1_score(y, pred, pos_label=1, average="binary", zero_division=0))

def best_threshold(yv, probv):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        _, f = f1s_at(yv, probv, t)
        if f > bf: bf, bt = f, t
    return bt


def tok(texts, tk, maxlen):
    e = tk(texts, padding="max_length", truncation=True, max_length=maxlen, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


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
    ap.add_argument("--val-frac", type=float, default=0.15, help="held-out for threshold calibration")
    ap.add_argument("--gamma", type=float, default=2.0, help="focal focusing")
    ap.add_argument("--tau", type=float, default=1.0, help="logit-adjustment strength")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="42,123,2024")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetune = a.regime == "finetune"
    print(f"[env] device={device} model={a.model} dataset={a.dataset} regime={a.regime}")

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model)
    train_all, evl = load_data(a.dataset, a.seed, a.max_train, a.max_eval)

    # carve a calibration val split from train (fixed by --seed, shared across model seeds)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(train_all))
    n_val = int(a.val_frac * len(train_all))
    val = [train_all[i] for i in perm[:n_val]]
    trn = [train_all[i] for i in perm[n_val:]]
    ytr = np.array([y for _, y in trn])
    print(f"[data] train={len(trn)} val={len(val)} test={len(evl)}  toxic-frac train={ytr.mean():.2f}")

    tr_ids, tr_mask = tok([t for t, _ in trn], tk, a.maxlen); tr_y = torch.tensor([y for _, y in trn])
    vl_ids, vl_mask = tok([t for t, _ in val], tk, a.maxlen); vl_y = np.array([y for _, y in val])
    te_ids, te_mask = tok([t for t, _ in evl], tk, a.maxlen); te_y = np.array([y for _, y in evl])
    tr = (tr_ids, tr_mask, tr_y)

    cnt = np.bincount(tr_y.numpy(), minlength=2).astype(float)
    class_w = torch.tensor(cnt.sum() / (2 * np.clip(cnt, 1, None)), dtype=torch.float32)
    prior = torch.tensor(cnt / cnt.sum(), dtype=torch.float32)

    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[run] regime={a.regime} epochs={a.epochs} seeds={seeds} gamma={a.gamma} tau={a.tau}\n")

    rows = {}  # loss -> list over seeds of dict(macro05,tox05,macroC,toxC,thr,prauc)
    for lname, lfn in LOSSES.items():
        rows[lname] = []
        t0 = time.time()
        for sd in seeds:
            set_seed(sd)
            m = Net(a.model, finetune=finetune)
            train(m, tr, device, a.epochs, a.lr_head, a.lr_bb, finetune, lfn, class_w, prior, a.gamma, a.tau)
            pv = predict(m, vl_ids, vl_mask, device)
            pt = predict(m, te_ids, te_mask, device)
            thr = best_threshold(vl_y, pv)                         # calibrate on val
            m05, t05 = f1s_at(te_y, pt, 0.5)
            mC, tC = f1s_at(te_y, pt, thr)
            rows[lname].append(dict(macro05=m05, tox05=t05, macroC=mC, toxC=tC,
                                    thr=thr, prauc=average_precision_score(te_y, pt)))
            del m
            if device == "cuda": torch.cuda.empty_cache()
        print(f"  [{lname:9}] {len(seeds)} seeds in {time.time()-t0:.0f}s")

    def mean(l, k): return np.mean([r[k] for r in rows[l]])
    def std(l, k):  return np.std([r[k] for r in rows[l]])

    print("\n" + "=" * 82)
    print(f"REGIME {a.regime.upper()} — {len(seeds)} seeds mean±std  (test macro-F1 / toxic-F1)")
    print("=" * 82)
    print(f"  {'loss':10} | {'thr=0.5 macro':>14} {'tox-F1':>9} | {'calib macro':>12} {'tox-F1':>9} {'thr*':>6} | {'PR-AUC':>8}")
    for l in LOSSES:
        print(f"  {l:10} | {mean(l,'macro05'):>8.4f}±{std(l,'macro05'):.3f} {mean(l,'tox05'):>8.4f} | "
              f"{mean(l,'macroC'):>7.4f}±{std(l,'macroC'):.3f} {mean(l,'toxC'):>8.4f} {mean(l,'thr'):>6.2f} | {mean(l,'prauc'):>8.4f}")
    print("-" * 82)
    base = mean("ce", "tox05")
    print(f"  baseline (ce, thr=0.5) toxic-F1 = {base:.4f}")
    best_l, best_v = max(((l, mean(l, "toxC")) for l in LOSSES), key=lambda x: x[1])
    print(f"  best config = {best_l} + calibration : toxic-F1 = {best_v:.4f}  (delta = {best_v-base:+.4f})")
    print("=" * 82)
    print("WORKING if best (loss+calibration) toxic-F1 clearly beats ce@0.5 across seeds.")
    print("Then scale to Malayalam/Kannada/Hinglish + 5 seeds + significance for the paper.")


if __name__ == "__main__":
    main()
