#!/usr/bin/env python3
"""
SAGE v7 — hard-negative supervised contrastive, head-to-head (PR-AUC ± std)
===========================================================================
v6 finding: a plain supervised-contrastive front-end lifted threshold-free PR-AUC
(0.621 -> 0.649) — the first representation gain in the project. NAC (local
calibration) was inert. This script tests whether HARD-NEGATIVE MINING amplifies
the contrastive gain, and confirms it is beyond seed noise.

Configs (each: fine-tuned MuRIL + head, evaluated at a val-calibrated threshold):
  CE               : cross-entropy only                       (baseline)
  SupCon           : + supervised contrastive (Khosla)        (the v6 lead)
  HardNeg-SupCon   : + hardness-reweighted negatives          (the amplifier)

Hardness reweighting (Robinson et al. style): negatives are reweighted by
softmax(beta_hn * sim) so the loss concentrates on the most confusable
(highest-similarity, different-label) pairs. beta_hn=0 recovers plain SupCon.

Reports every metric mean±std over seeds, prints per-seed PR-AUC, and a
win-count vs CE — so we can see if the gain is real, not noise.

Usage:
  python sage_v7.py --model google/muril-base-cased --dataset local:./data_store/dravidian_tamil
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, average_precision_score

from sage_v2_gonogo import load_data
from sage_v3 import set_seed, mine_lexicon
from sage_v6 import Net, extract, best_thr, ece, toxf1, macrof1, slang_density


def supcon_loss(z, y, temp=0.1, beta_hn=0.0):
    """Supervised contrastive loss with optional hard-negative reweighting.
       beta_hn=0 -> standard SupCon; beta_hn>0 -> concentrate on hard negatives."""
    B = z.size(0)
    sim = z @ z.T / temp
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~self_mask
    neg = y.unsqueeze(0) != y.unsqueeze(1)
    sim_m = sim.masked_fill(self_mask, -1e9)
    m = sim_m.max(1, keepdim=True).values.detach()
    exp = torch.exp(sim_m - m)                                   # self -> ~0
    if beta_hn > 0:
        with torch.no_grad():
            w = torch.softmax((beta_hn * sim).masked_fill(~neg, -1e9), dim=1)
            w = w * neg.sum(1, keepdim=True).clamp(min=1)        # mean-preserving reweight
        neg_term = (exp * neg * w).sum(1)
    else:
        neg_term = (exp * neg).sum(1)
    denom = (exp * pos).sum(1) + neg_term + 1e-12
    log_prob = (sim_m - m) - torch.log(denom).unsqueeze(1)       # (B,B)
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    if not valid.any():
        return z.new_zeros(())
    loss = -(log_prob * pos).sum(1)[valid] / pos_cnt[valid].clamp(min=1)
    return loss.mean()


def train(model, tr, device, epochs, lr_head, lr_bb, finetune, class_w, beta, temp, beta_hn):
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
                loss = loss + beta * supcon_loss(z, y, temp, beta_hn)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
    return model


def evaluate(p_vl, p_te, yv, yt, cov):
    t = best_thr(yv, p_vl); pred = (p_te >= t).astype(int)
    d = dict(macro=macrof1(yt, pred), toxf1=toxf1(yt, pred),
             prec=precision_score(yt, pred, pos_label=1, zero_division=0),
             rec=recall_score(yt, pred, pos_label=1, zero_division=0),
             prauc=average_precision_score(yt, p_te), ece=ece(yt, p_te))
    d["covert"] = toxf1(yt[cov], pred[cov]) if cov.sum() > 5 and len(set(yt[cov])) == 2 else float("nan")
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
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--beta-hn", type=float, default=1.0, help="hard-negative concentration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="42,123,2024,7,99")
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
    cov = np.array([slang_density(t, slang_set) for t, _ in evl]) > 0
    cnt = np.bincount(trY.numpy(), minlength=2).astype(float)
    class_w = torch.tensor(cnt.sum() / (2 * np.clip(cnt, 1, None)), dtype=torch.float32)

    seeds = [int(s) for s in a.seeds.split(",")]
    configs = [("CE", 0.0, 0.0), ("SupCon", a.beta, 0.0), ("HardNeg-SupCon", a.beta, a.beta_hn)]
    agg = {name: [] for name, _, _ in configs}
    print(f"[run] regime={a.regime} epochs={a.epochs} beta={a.beta} beta_hn={a.beta_hn} seeds={seeds}\n")

    for sd in seeds:
        t0 = time.time()
        for name, beta, bhn in configs:
            set_seed(sd)
            m = Net(a.model, finetune)
            train(m, (trI, trM, trY), device, a.epochs, a.lr_head, a.lr_bb, finetune, class_w, beta, a.temp, bhn)
            p_vl, _ = extract(m, vlI, vlM, device)
            p_te, _ = extract(m, teI, teM, device)
            agg[name].append(evaluate(p_vl, p_te, vlY, teY, cov))
            del m
            if device == "cuda": torch.cuda.empty_cache()
        print(f"  seed {sd}: done in {time.time()-t0:.0f}s")

    def mu(n, k): return np.nanmean([r[k] for r in agg[n]])
    def sd_(n, k): return np.nanstd([r[k] for r in agg[n]])

    print("\n" + "=" * 92)
    print(f"REGIME {a.regime.upper()} — {len(seeds)} seeds mean±std   (test, val-calibrated threshold)")
    print("=" * 92)
    print(f"  {'config':16} {'macro-F1':>13} {'toxic-F1':>13} {'PR-AUC':>14} {'tox-P':>7} {'tox-R':>7} {'ECE':>6}")
    for name, _, _ in configs:
        print(f"  {name:16} {mu(name,'macro'):>7.4f}±{sd_(name,'macro'):.3f} {mu(name,'toxf1'):>7.4f}±{sd_(name,'toxf1'):.3f} "
              f"{mu(name,'prauc'):>7.4f}±{sd_(name,'prauc'):.3f} {mu(name,'prec'):>7.4f} {mu(name,'rec'):>7.4f} {mu(name,'ece'):>6.3f}")
    print("-" * 92)
    # per-seed PR-AUC + win-count vs CE (is the gain real / consistent?)
    ce_pr = np.array([r["prauc"] for r in agg["CE"]])
    for name, _, _ in configs[1:]:
        pr = np.array([r["prauc"] for r in agg[name]])
        wins = int((pr > ce_pr).sum())
        print(f"  {name:16} PR-AUC per seed: {np.round(pr,4).tolist()}   vs CE: +{(pr-ce_pr).mean():.4f} "
              f"(wins {wins}/{len(seeds)})")
    print("=" * 92)
    best = max(configs[1:], key=lambda c: mu(c[0], "prauc"))[0]
    d = mu(best, "prauc") - mu("CE", "prauc")
    print(f"VERDICT cue: {best} PR-AUC gain over CE = {d:+.4f}.")
    print("  REAL if gain > CE's std AND wins on most seeds AND hard-neg >= plain SupCon.")
    print("  Then scale to Malayalam/Kannada/Hinglish + significance -> genuine contribution.")


if __name__ == "__main__":
    main()
