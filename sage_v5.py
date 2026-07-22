#!/usr/bin/env python3
"""
SAGE v5 — imbalance recipe + your 5 requested additions (verified & integrated)
===============================================================================
Baked into every run (the working recipe from v4 + your #3/#5):
  * fine-tuned MuRIL + configurable DEEPER head            (#3  --head-dims)
  * REDUCED backbone learning rate + 1 extra epoch         (#5  --lr-bb, --epochs)
  * class-weighted / focal loss + threshold CALIBRATION on a held-out val split
Opt-in experimental arms:
  * TARGET-IN-CONTEXT marking of mined slang tokens        (#4  --mark-targets)   <- most promising
  * KB similarity stream with E anchors + TEMPERATURE      (#1  --kb-stream --kb-experts 64
                                                             #2  --kb-temp 0.5)   <- falsified branch, tested honestly
Reports (per arm, mean±std over seeds): calibrated toxic-F1, macro-F1, PR-AUC, and
toxic-F1 on the COVERT slice (examples containing >=1 mined slang token) — the hook.

Example:
  python sage_v5.py --model google/muril-base-cased --dataset local:./data_store/dravidian_tamil \\
                    --mark-targets --kb-stream --kb-experts 64 --kb-temp 0.5
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score

from sage_v2_gonogo import load_data
from sage_v3 import set_seed, mine_lexicon, anchor_init, _PUNCT


# ---- #4 target-in-context marking ---------------------------------------
def mark_text(text, slang_set):
    """Wrap mined slang tokens with [TGT] ... [/TGT] so the encoder attends to them IN context."""
    out = []
    for w in str(text).split():
        core = w.strip(_PUNCT)
        if core and core.lower() in slang_set:
            out.append(w.replace(core, f"[TGT] {core} [/TGT]"))
        else:
            out.append(w)
    return " ".join(out)


def slang_density(text, slang_set):
    toks = [w.strip(_PUNCT).lower() for w in str(text).split()]
    toks = [t for t in toks if t]
    return sum(1 for t in toks if t in slang_set) / len(toks) if toks else 0.0


# ---- model ---------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, model_name, finetune, head_dims, vocab_size,
                 kb=False, anchors0=None, kb_temp=1.0):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(model_name)
        if vocab_size and vocab_size != self.bb.get_input_embeddings().weight.size(0):
            self.bb.resize_token_embeddings(vocab_size)            # for [TGT] markers (#4)
        if not finetune:
            for p in self.bb.parameters(): p.requires_grad = False
        d = self.bb.config.hidden_size
        self.kb, self.kb_temp = kb, kb_temp
        extra = 0
        if kb:
            self.anchors = nn.Parameter(anchors0.clone())          # #1 E anchors, trainable/in-space
            extra = 2 * anchors0.size(0)
        dims = [d + extra] + list(head_dims) + [2]                 # #3 configurable depth
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.ReLU(), nn.Dropout(0.1)]
        self.head = nn.Sequential(*layers)

    def forward(self, ids, mask):
        H = self.bb(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (H * m).sum(1) / m.sum(1).clamp(min=1e-6)
        if not self.kb:
            return self.head(pooled)
        An = F.normalize(self.anchors, dim=-1)
        sim = (F.normalize(H, dim=-1) @ An.T) / self.kb_temp       # #2 temperature
        kb_max = sim.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(1).values
        kb_mean = (sim * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(torch.cat([pooled, kb_max, kb_mean], dim=-1))


def compute_loss(logits, y, kind, class_w, gamma):
    if kind == "focal":
        logp = F.log_softmax(logits, dim=-1); idx = torch.arange(len(y), device=y.device)
        logp_y = logp[idx, y]; p_y = logp_y.exp()
        return -(class_w[y] * (1 - p_y).clamp(min=0) ** gamma * logp_y).mean()
    return F.cross_entropy(logits, y, weight=class_w)


def train(model, tr, device, epochs, lr_head, lr_bb, finetune, loss_kind, class_w, gamma):
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
            loss = compute_loss(model(ids, mask), y, loss_kind, class_w, gamma)
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


def toxf1(y, prob, thr):
    return f1_score(y, (prob >= thr).astype(int), pos_label=1, average="binary", zero_division=0)

def macrof1(y, prob, thr):
    return f1_score(y, (prob >= thr).astype(int), average="macro", zero_division=0)

def best_thr(yv, pv):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f = toxf1(yv, pv, t)
        if f > bf: bf, bt = f, t
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--regime", choices=["frozen", "finetune"], default="finetune")
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=4)                    # #5 +1 epoch
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-bb", type=float, default=1e-5)                # #5 reduced (was 2e-5)
    ap.add_argument("--head-dims", default="384,128")                  # #3 deeper head
    ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-eval", type=int, default=1500)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--mark-targets", action="store_true")            # #4
    ap.add_argument("--kb-stream", action="store_true")               # #1/#2
    ap.add_argument("--kb-experts", type=int, default=16)             # #1 E
    ap.add_argument("--kb-temp", type=float, default=1.0)             # #2 temperature
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="42,123,2024")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetune = a.regime == "finetune"
    head_dims = [int(x) for x in a.head_dims.split(",") if x]
    print(f"[env] device={device} model={a.model} dataset={a.dataset} regime={a.regime}")
    if a.mark_targets and not finetune:
        print("[warn] --mark-targets adds new token embeddings that only train under --regime finetune.")

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model)
    train_all, evl = load_data(a.dataset, a.seed, a.max_train, a.max_eval)

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(train_all)); nv = int(a.val_frac * len(train_all))
    val = [train_all[i] for i in perm[:nv]]; trn = [train_all[i] for i in perm[nv:]]
    ytr = np.array([y for _, y in trn])
    print(f"[data] train={len(trn)} val={len(val)} test={len(evl)} toxic-frac={ytr.mean():.2f}")

    # mine lexicon (auto-scale size with E so 64 anchors aren't degenerate) — #1 fix
    topk = max(120, 5 * a.kb_experts)
    mined = mine_lexicon([t for t, _ in trn], ytr.tolist(), top_k=topk, min_freq=3)
    slang_set = {m.lower() for m in mined}
    print(f"[lexicon] mined {len(mined)} tokens (topk={topk}). Top 15: {', '.join(mined[:15])}")

    anchors0 = None
    vocab_size = None
    if a.kb_stream:
        from transformers import AutoModel
        bbase = AutoModel.from_pretrained(a.model).to(device).eval()
        anchors0 = anchor_init(mined, tk, bbase, device, a.kb_experts).to(device)
        del bbase
        if device == "cuda": torch.cuda.empty_cache()
    if a.mark_targets:
        tk.add_special_tokens({"additional_special_tokens": ["[TGT]", "[/TGT]"]})
        vocab_size = len(tk)

    def tok(texts):
        e = tk(texts, padding="max_length", truncation=True, max_length=a.maxlen, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    # tokenize plain + (optionally) marked variants
    tr_txt = [t for t, _ in trn]; vl_txt = [t for t, _ in val]; te_txt = [t for t, _ in evl]
    plain = dict(tr=tok(tr_txt), vl=tok(vl_txt), te=tok(te_txt))
    marked = None
    if a.mark_targets:
        marked = dict(tr=tok([mark_text(t, slang_set) for t in tr_txt]),
                      vl=tok([mark_text(t, slang_set) for t in vl_txt]),
                      te=tok([mark_text(t, slang_set) for t in te_txt]))
    tr_y = torch.tensor([y for _, y in trn]); vl_y = np.array([y for _, y in val]); te_y = np.array([y for _, y in evl])
    cnt = np.bincount(tr_y.numpy(), minlength=2).astype(float)
    class_w = torch.tensor(cnt.sum() / (2 * np.clip(cnt, 1, None)), dtype=torch.float32)
    covert = np.array([slang_density(t, slang_set) for t in te_txt]) > 0     # covert slice mask

    # arms: base + (mark) + (kb)
    arms = [("base", plain, False)]
    if a.mark_targets: arms.append(("+mark(#4)", marked, False))
    if a.kb_stream:    arms.append((f"+kb E={a.kb_experts},T={a.kb_temp}(#1/2)", plain, True))

    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[run] loss={a.loss} head={head_dims} lr_bb={a.lr_bb} epochs={a.epochs} seeds={seeds}\n")
    results = {}
    for name, data, use_kb in arms:
        results[name] = []
        t0 = time.time()
        for sd in seeds:
            set_seed(sd)
            m = Net(a.model, finetune, head_dims, vocab_size,
                    kb=use_kb, anchors0=anchors0, kb_temp=a.kb_temp)
            train(m, (*data["tr"], tr_y), device, a.epochs, a.lr_head, a.lr_bb, finetune, a.loss, class_w, a.gamma)
            pv = predict(m, *data["vl"], device); pt = predict(m, *data["te"], device)
            thr = best_thr(vl_y, pv)
            r = dict(macro=macrof1(te_y, pt, thr), tox=toxf1(te_y, pt, thr),
                     prauc=average_precision_score(te_y, pt), thr=thr)
            # covert-slice toxic-F1 (hook)
            if covert.sum() > 5 and len(set(te_y[covert])) == 2:
                r["tox_covert"] = toxf1(te_y[covert], pt[covert], thr)
            else:
                r["tox_covert"] = float("nan")
            results[name].append(r); del m
            if device == "cuda": torch.cuda.empty_cache()
        print(f"  [{name:24}] {len(seeds)} seeds in {time.time()-t0:.0f}s")

    def mu(n, k): return np.nanmean([r[k] for r in results[n]])
    def sd_(n, k): return np.nanstd([r[k] for r in results[n]])

    print("\n" + "=" * 88)
    print(f"REGIME {a.regime.upper()} loss={a.loss} — {len(seeds)} seeds mean±std  (calibrated threshold)")
    print(f"covert slice = {int(covert.sum())}/{len(te_y)} test items contain a mined slang token")
    print("=" * 88)
    print(f"  {'arm':26} {'macro-F1':>13} {'toxic-F1':>13} {'PR-AUC':>11} {'tox-F1(covert)':>15}")
    for name, _, _ in arms:
        print(f"  {name:26} {mu(name,'macro'):>7.4f}±{sd_(name,'macro'):.3f} "
              f"{mu(name,'tox'):>7.4f}±{sd_(name,'tox'):.3f} {mu(name,'prauc'):>11.4f} {mu(name,'tox_covert'):>15.4f}")
    print("-" * 88)
    base_tox = mu("base", "tox")
    for name, _, _ in arms[1:]:
        print(f"  delta vs base — {name:24} toxic-F1 {mu(name,'tox')-base_tox:+.4f} "
              f" | PR-AUC {mu(name,'prauc')-mu('base','prauc'):+.4f}")
    print("=" * 88)
    print("Judge #4 (mark) and #1/#2 (kb) by delta toxic-F1 AND delta PR-AUC (threshold-free).")
    print("If PR-AUC doesn't move, the arm adds no real signal regardless of toxic-F1 wobble.")


if __name__ == "__main__":
    main()
