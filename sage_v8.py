#!/usr/bin/env python3
"""
SAGE v8 — cross-lingual transfer (Paper-2 foundation)
=====================================================
Tests whether the contrastive gain AMPLIFIES under cross-lingual transfer, and
whether language-adversarial alignment (DANN, Ganin et al.) helps transfer to
low-resource code-mixed languages.

Setup: train on SOURCE (e.g. Tamil, labeled) -> ZERO-SHOT eval on TARGET
languages (Malayalam/Kannada/Hinglish). Primary metric = PR-AUC (threshold-free,
the honest transfer metric); toxic-F1 reported at a SOURCE-val-calibrated
threshold (realistic zero-shot deployment).

Configs (all fine-tuned MuRIL):
  CE | SupCon | HardNeg-SupCon | HardNeg+Adv
  +Adv adds a gradient-reversal language discriminator over SOURCE (labeled) vs
  TARGET (unlabeled) text -> language-invariant toxicity representations.

Usage:
  python sage_v8.py --model google/muril-base-cased \\
    --source local:./data_store/dravidian_tamil \\
    --targets local:./data_store/dravidian_malayalam,local:./data_store/dravidian_kannada
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, average_precision_score

from sage_v2_gonogo import load_data
from sage_v3 import set_seed
from sage_v6 import best_thr, toxf1, macrof1
from sage_v7 import supcon_loss


# ---- gradient reversal (DANN) -------------------------------------------
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd); return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lambd, None
def grl(x, lambd): return _GRL.apply(x, lambd)

def cycle(loader):
    while True:
        for b in loader: yield b


class Net8(nn.Module):
    def __init__(self, model_name, finetune, hdim=256, pdim=128):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(model_name)
        if not finetune:
            for p in self.bb.parameters(): p.requires_grad = False
        d = self.bb.config.hidden_size
        self.proj = nn.Linear(d, pdim)
        self.tox = nn.Sequential(nn.Linear(d, hdim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hdim, 2))
        self.dom = nn.Sequential(nn.Linear(d, hdim), nn.ReLU(), nn.Linear(hdim, 2))   # language discriminator

    def pooled(self, ids, mask):
        H = self.bb(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return (H * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def forward(self, ids, mask):
        p = self.pooled(ids, mask)
        return self.tox(p), p, F.normalize(self.proj(p), dim=-1)

    def domain_logits(self, pooled, lambd):
        return self.dom(grl(pooled, lambd))


def train(model, src, tgt_unlab, device, epochs, lr_head, lr_bb, finetune,
          class_w, beta, temp, beta_hn, adv_w, adv_lambda):
    model.to(device)
    non_bb = [p for n, p in model.named_parameters() if not n.startswith("bb.") and p.requires_grad]
    groups = [{"params": non_bb, "lr": lr_head}]
    if finetune: groups.append({"params": model.bb.parameters(), "lr": lr_bb})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    class_w = class_w.to(device)
    src_loader = DataLoader(TensorDataset(*src), batch_size=32, shuffle=True)
    tgt_iter = cycle(DataLoader(TensorDataset(*tgt_unlab), batch_size=32, shuffle=True)) \
        if (adv_w > 0 and tgt_unlab is not None) else None
    for _ in range(epochs):
        model.train()
        for ids, mask, y in src_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            logits, pooled_s, z = model(ids, mask)
            loss = F.cross_entropy(logits, y, weight=class_w)
            if beta > 0:
                loss = loss + beta * supcon_loss(z, y, temp, beta_hn)
            if tgt_iter is not None:
                t_ids, t_mask = next(tgt_iter)
                pooled_t = model.pooled(t_ids.to(device), t_mask.to(device))
                d_s = model.domain_logits(pooled_s, adv_lambda)
                d_t = model.domain_logits(pooled_t, adv_lambda)
                ys = torch.zeros(d_s.size(0), dtype=torch.long, device=device)
                yt = torch.ones(d_t.size(0), dtype=torch.long, device=device)
                loss = loss + adv_w * (F.cross_entropy(d_s, ys) + F.cross_entropy(d_t, yt))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
    return model


@torch.no_grad()
def probs(model, ids, mask, device, bs=64):
    model.eval(); out = []
    for i in range(0, len(ids), bs):
        logits, _, _ = model(ids[i:i+bs].to(device), mask[i:i+bs].to(device))
        out.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return torch.cat(out).numpy()


def eval_dom(p_srcval, y_srcval, p_ev, y_ev):
    t = best_thr(y_srcval, p_srcval); pred = (p_ev >= t).astype(int)
    return dict(prauc=average_precision_score(y_ev, p_ev), toxf1=toxf1(y_ev, pred),
                prec=precision_score(y_ev, pred, pos_label=1, zero_division=0),
                rec=recall_score(y_ev, pred, pos_label=1, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--targets", default="", help="comma-sep local:<dir> targets for zero-shot")
    ap.add_argument("--regime", choices=["frozen", "finetune"], default="finetune")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-bb", type=float, default=2e-5)
    ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-eval", type=int, default=1500)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--beta-hn", type=float, default=1.0)
    ap.add_argument("--adv-w", type=float, default=0.3, help="domain-adversarial weight")
    ap.add_argument("--adv-lambda", type=float, default=1.0, help="gradient-reversal strength")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="42,123,2024")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetune = a.regime == "finetune"
    print(f"[env] device={device} model={a.model} regime={a.regime}")
    print(f"[transfer] SOURCE={a.source}  TARGETS={a.targets or '(none)'}")

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model)
    def tok(texts):
        e = tk(texts, padding="max_length", truncation=True, max_length=a.maxlen, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    # SOURCE (labeled)
    s_all, s_test = load_data(a.source, a.seed, a.max_train, a.max_eval)
    rng = np.random.default_rng(a.seed); perm = rng.permutation(len(s_all))
    nv = int(a.val_frac * len(s_all))
    s_val = [s_all[i] for i in perm[:nv]]; s_trn = [s_all[i] for i in perm[nv:]]
    syl = [y for _, y in s_trn]
    print(f"[source] train={len(s_trn)} val={len(s_val)} test={len(s_test)} toxic-frac={np.mean(syl):.2f}")
    sI, sM = tok([t for t, _ in s_trn]); sY = torch.tensor(syl)
    svI, svM = tok([t for t, _ in s_val]); svY = np.array([y for _, y in s_val])
    stI, stM = tok([t for t, _ in s_test]); stY = np.array([y for _, y in s_test])
    cnt = np.bincount(sY.numpy(), minlength=2).astype(float)
    class_w = torch.tensor(cnt.sum() / (2 * np.clip(cnt, 1, None)), dtype=torch.float32)

    # TARGETS (zero-shot eval + unlabeled pool for adversarial)
    domains = {"SOURCE (in-lang)": (stI, stM, stY)}
    tgt_unlab_texts = []
    for tp in [t for t in a.targets.split(",") if t.strip()]:
        try:
            t_all, t_test = load_data(tp, a.seed, a.max_train, a.max_eval)
        except Exception as e:
            print(f"[target] SKIP {tp}: {e}"); continue
        name = tp.split("/")[-1]
        tI, tM = tok([t for t, _ in t_test]); tY = np.array([y for _, y in t_test])
        domains[name] = (tI, tM, tY)
        tgt_unlab_texts += [t for t, _ in t_all][:a.max_train]
        print(f"[target] {name}: eval={len(t_test)} toxic-frac={tY.mean():.2f}")
    tgt_unlab = tok(tgt_unlab_texts) if tgt_unlab_texts else None

    configs = [("CE", 0.0, 0.0, 0.0), ("SupCon", a.beta, 0.0, 0.0),
               ("HardNeg-SupCon", a.beta, a.beta_hn, 0.0),
               ("HardNeg+Adv", a.beta, a.beta_hn, a.adv_w)]
    seeds = [int(s) for s in a.seeds.split(",")]
    agg = {c[0]: {d: [] for d in domains} for c in configs}
    print(f"[run] epochs={a.epochs} beta={a.beta} beta_hn={a.beta_hn} adv_w={a.adv_w} seeds={seeds}\n")

    for sd in seeds:
        t0 = time.time()
        for name, beta, bhn, advw in configs:
            set_seed(sd)
            m = Net8(a.model, finetune)
            train(m, (sI, sM, sY), tgt_unlab, device, a.epochs, a.lr_head, a.lr_bb, finetune,
                  class_w, beta, a.temp, bhn, advw, a.adv_lambda)
            p_sv = probs(m, svI, svM, device)
            for d, (dI, dM, dY) in domains.items():
                p_d = probs(m, dI, dM, device)
                agg[name][d].append(eval_dom(p_sv, svY, p_d, dY))
            del m
            if device == "cuda": torch.cuda.empty_cache()
        print(f"  seed {sd}: done in {time.time()-t0:.0f}s")

    def mu(c, d, k): return np.nanmean([r[k] for r in agg[c][d]])
    def sd_(c, d, k): return np.nanstd([r[k] for r in agg[c][d]])

    for d in domains:
        tag = "  (reference)" if d.startswith("SOURCE") else "  (ZERO-SHOT transfer)"
        print("\n" + "=" * 84)
        print(f"DOMAIN: {d}{tag}   — {len(seeds)} seeds mean±std")
        print("=" * 84)
        print(f"  {'config':16} {'PR-AUC':>15} {'toxic-F1@srcT':>15} {'tox-P':>7} {'tox-R':>7}")
        for c, _, _, _ in configs:
            print(f"  {c:16} {mu(c,d,'prauc'):>8.4f}±{sd_(c,d,'prauc'):.3f} "
                  f"{mu(c,d,'toxf1'):>8.4f}±{sd_(c,d,'toxf1'):.3f} {mu(c,d,'prec'):>7.4f} {mu(c,d,'rec'):>7.4f}")
        ce = mu("CE", d, "prauc")
        for c in ("HardNeg-SupCon", "HardNeg+Adv"):
            pr = np.array([r["prauc"] for r in agg[c][d]]); cepr = np.array([r["prauc"] for r in agg["CE"][d]])
            print(f"    {c:16} PR-AUC vs CE: {pr.mean()-cepr.mean():+.4f}  (wins {int((pr>cepr).sum())}/{len(seeds)})")
    print("\n" + "=" * 84)
    print("PAPER-2 EXISTS if contrastive/adv PR-AUC gain over CE is LARGER on ZERO-SHOT targets")
    print("than in-language (i.e., contrastive helps transfer more), consistent across seeds.")
    print("If transfer gain ~ in-language gain -> no distinct Paper 2; fold into Paper 1.")


if __name__ == "__main__":
    main()
