#!/usr/bin/env python3
"""
SAGE v3 (fixed) — data-mined, language-matched lexicon + in-space trainable anchors
===================================================================================
Fixes the two confounds found in the first v3 run:
  FIX 1  Language match: the lexicon is MINED FROM THE TRAINING DATA (tokens most
         associated with the toxic class via smoothed log-odds), so it is always in
         the dataset's own language(s) — no more Hindi anchors on Tamil data.
  FIX 2  In-space anchors: anchors are TRAINABLE parameters initialised from the mined
         tokens, so they live in (and co-adapt with) the encoder's representation
         space. This removes the base-vs-finetuned space mismatch that made +KB hurt.

Compares, per regime (frozen | finetune):
    baseline : backbone (mean-pool) -> MLP head
    +KB      : [backbone pool ; token-level lexical-prototype features] -> MLP head
Reports macro-F1 + toxic-class recall/F1.

Usage:
    python sage_v3.py --model google/muril-base-cased --dataset local:./data_store/dravidian_tamil --regime finetune
    python sage_v3.py --model google/muril-base-cased --dataset local:./data_store/dravidian_tamil --regime frozen
"""
from __future__ import annotations
import argparse, re, time
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score

from sage_v2_gonogo import load_data          # reuse the dataset loaders


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------------
# FIX 1 — mine a language-matched lexicon from the TRAINING split
# --------------------------------------------------------------------------
_PUNCT = ".,!?;:\"'()[]{}<>…-–—@#*/\\|~`^%$&+=_ \t\n"

def _word_tokens(t):
    """Whitespace tokens with edge punctuation stripped; keeps Tamil/Devanagari words
    whole (a letter-only regex would fragment them at combining vowel signs)."""
    out = []
    for w in str(t).lower().split():
        w = w.strip(_PUNCT)
        if len(w) >= 2 and any(ch.isalpha() for ch in w):
            out.append(w)
    return set(out)

def mine_lexicon(texts, labels, top_k=120, min_freq=3):
    """Top toxic-associated tokens by smoothed log-odds (train-only, no leakage)."""
    tox_df, cln_df = Counter(), Counter()
    for t, y in zip(texts, labels):
        (tox_df if y == 1 else cln_df).update(_word_tokens(t))
    n_t = int(np.sum(np.array(labels) == 1)); n_c = len(labels) - n_t
    scored = []
    for w in set(tox_df) | set(cln_df):
        if tox_df[w] + cln_df[w] < min_freq:
            continue
        p_t = (tox_df[w] + 1) / (n_t + 2); p_c = (cln_df[w] + 1) / (n_c + 2)
        scored.append((np.log(p_t / p_c), w))
    scored.sort(reverse=True)
    return [w for _, w in scored[:top_k]]


@torch.no_grad()
def embed_phrases(phrases, tok, bb, device):
    enc = tok(phrases, padding=True, truncation=True, max_length=8, return_tensors="pt").to(device)
    H = bb(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    return ((H * m).sum(1) / m.sum(1).clamp(min=1e-6)).cpu()      # (K,d)


def anchor_init(mined, tok, bb, device, E):
    """Initialise E anchors from clusters of the mined-token embeddings (base encoder)."""
    emb = embed_phrases(mined, tok, bb, device)                  # (K,d)
    if len(mined) >= E:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=E, n_init=4, random_state=42).fit(emb.numpy())
        init = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    else:
        init = emb.mean(0, keepdim=True).repeat(E, 1)
    return F.normalize(init, dim=-1)                             # (E,d)


# --------------------------------------------------------------------------
class KBDual(nn.Module):
    def __init__(self, model_name, anchors0, use_kb, finetune):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(model_name)
        if not finetune:
            for p in self.bb.parameters(): p.requires_grad = False
        d = self.bb.config.hidden_size
        self.use_kb = use_kb
        if use_kb:
            E = anchors0.size(0)
            self.anchors = nn.Parameter(anchors0.clone())       # FIX 2 — trainable, in-space
            in_dim = d + 2 * E
        else:
            in_dim = d
        self.head = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(256, 2))

    def forward(self, ids, mask):
        H = self.bb(input_ids=ids, attention_mask=mask).last_hidden_state    # (B,n,d)
        m = mask.unsqueeze(-1).float()
        pooled = (H * m).sum(1) / m.sum(1).clamp(min=1e-6)                    # (B,d)
        if not self.use_kb:
            return self.head(pooled)
        An = F.normalize(self.anchors, dim=-1)                               # (E,d) in-space
        Hn = F.normalize(H, dim=-1)
        sim = Hn @ An.T                                                       # (B,n,E)
        kb_max = sim.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(1).values # (B,E) strongest token/proto
        kb_mean = (sim * m).sum(1) / m.sum(1).clamp(min=1e-6)                 # (B,E) density
        return self.head(torch.cat([pooled, kb_max, kb_mean], dim=-1))


def tokenize(texts, tok, maxlen):
    enc = tok(texts, padding="max_length", truncation=True, max_length=maxlen, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def train_eval(model, tr, ev, device, epochs, lr_head, lr_bb, finetune, class_w):
    model.to(device)
    non_bb = [p for n, p in model.named_parameters() if not n.startswith("bb.") and p.requires_grad]
    groups = [{"params": non_bb, "lr": lr_head}]                 # head + trainable anchors
    if finetune:
        groups.append({"params": model.bb.parameters(), "lr": lr_bb})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    lossfn = nn.CrossEntropyLoss(weight=class_w.to(device))
    tl = DataLoader(TensorDataset(*tr), batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for ids, mask, y in tl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            loss = lossfn(model(ids, mask), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
    model.eval()
    probs = []
    el = DataLoader(TensorDataset(*ev[:2]), batch_size=64)
    with torch.no_grad():
        for ids, mask in el:
            logits = model(ids.to(device), mask.to(device))
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())     # P(toxic)
    prob = torch.cat(probs).numpy(); y = ev[2].numpy()
    pred = (prob >= 0.5).astype(int)
    try: roc = roc_auc_score(y, prob)
    except Exception: roc = float("nan")
    try: ap = average_precision_score(y, prob)
    except Exception: ap = float("nan")
    return dict(macro=f1_score(y, pred, average="macro", zero_division=0),
                tox_recall=recall_score(y, pred, pos_label=1, zero_division=0),
                tox_f1=f1_score(y, pred, pos_label=1, average="binary", zero_division=0),
                roc_auc=roc, pr_auc=ap)                                  # threshold-free signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--regime", choices=["frozen", "finetune"], default="finetune")
    ap.add_argument("--experts", type=int, default=4, help="number of lexical prototype anchors E")
    ap.add_argument("--lexicon-topk", type=int, default=120)
    ap.add_argument("--lexicon-minfreq", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-bb", type=float, default=2e-5)
    ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-eval", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42, help="data split/mining seed (held fixed)")
    ap.add_argument("--seeds", default="42,123,2024", help="model seeds to average over")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetune = a.regime == "finetune"
    print(f"[env] device={device}  model={a.model}  dataset={a.dataset}  regime={a.regime}")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(a.model)
    train, evl = load_data(a.dataset, a.seed, a.max_train, a.max_eval)
    tr_texts = [t for t, _ in train]; tr_y_list = [y for _, y in train]
    print(f"[data] train={len(train)} eval={len(evl)}  toxic-frac train={np.mean(tr_y_list):.2f}")

    # FIX 1 — mine the lexicon from the training split (language-matched)
    mined = mine_lexicon(tr_texts, tr_y_list, a.lexicon_topk, a.lexicon_minfreq)
    print(f"[lexicon] mined {len(mined)} toxic-associated tokens from TRAIN. Top 25:")
    print("          " + ", ".join(mined[:25]))

    base = AutoModel.from_pretrained(a.model).to(device).eval()
    a0 = anchor_init(mined, tok, base, device, a.experts)        # E anchors from mined tokens
    del base
    if device == "cuda": torch.cuda.empty_cache()

    tr_ids, tr_mask = tokenize(tr_texts, tok, a.maxlen)
    tr_y = torch.tensor(tr_y_list)
    ev_ids, ev_mask = tokenize([t for t, _ in evl], tok, a.maxlen)
    ev_y = torch.tensor([y for _, y in evl])
    tr, ev = (tr_ids, tr_mask, tr_y), (ev_ids, ev_mask, ev_y)

    cnt = np.bincount(tr_y.numpy(), minlength=2).astype(float)
    class_w = torch.tensor(cnt.sum() / (2 * np.clip(cnt, 1, None)), dtype=torch.float32)

    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[run] regime={a.regime} epochs={a.epochs} anchors E={a.experts} seeds={seeds}")
    agg = {"baseline": [], "+KB dual": []}
    for tag, use_kb in [("baseline", False), ("+KB dual", True)]:
        t0 = time.time()
        for sd in seeds:
            set_seed(sd)
            m = KBDual(a.model, a0.to(device), use_kb=use_kb, finetune=finetune)
            agg[tag].append(train_eval(m, tr, ev, device, a.epochs, a.lr_head, a.lr_bb, finetune, class_w))
            del m
            if device == "cuda": torch.cuda.empty_cache()
        print(f"  [{tag:9}] {len(seeds)} seeds done in {time.time()-t0:.0f}s")

    def ms(tag, key):
        v = np.array([r[key] for r in agg[tag]]); return v.mean(), v.std()

    print("\n" + "=" * 74)
    print(f"REGIME: {a.regime.upper()}   ({len(seeds)} seeds mean±std, toxic-frac {np.mean(tr_y_list):.2f})")
    print("=" * 74)
    print(f"  {'config':10} {'macro-F1':>14} {'toxic-F1':>14} {'ROC-AUC':>14} {'PR-AUC':>14}")
    for tag in ("baseline", "+KB dual"):
        cells = []
        for key in ("macro", "tox_f1", "roc_auc", "pr_auc"):
            mu, sd = ms(tag, key); cells.append(f"{mu:.4f}±{sd:.3f}")
        print(f"  {tag:10} " + " ".join(f"{c:>14}" for c in cells))
    print("-" * 74)
    for key, lbl in [("macro", "macro-F1"), ("tox_f1", "toxic-F1"),
                     ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
        d = ms("+KB dual", key)[0] - ms("baseline", key)[0]
        print(f"  delta {lbl:9} = {d:+.4f}")
    print("=" * 74)
    print("DECISIVE: if delta ROC-AUC AND delta PR-AUC are both > 0 and exceed their std,")
    print("  the KB features carry REAL signal -> the recall/toxic-F1 drop is a threshold")
    print("  (calibration) artifact, fixable -> a genuine (if modest) contribution.")
    print("  If the AUC deltas are ~0 or negative -> earlier macro gains were noise -> pivot.")


if __name__ == "__main__":
    main()
