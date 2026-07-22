#!/usr/bin/env python3
"""
SAGE v2 — go/no-go with HONEST ablation ladder + confound-free KN metric
========================================================================
Fixes after the first run (weak baseline + confounded KN):

  Ablation ladder (isolates WHERE any gain comes from):
    1. Linear head            — the (weak) frozen baseline
    2. MLP head               — CAPACITY-MATCHED baseline (same #params as experts)
    3. SAGE experts, RANDOM   — gated low-rank experts, NO KB grounding  (A5 control)
    4. SAGE experts, KB       — full method (category anchors + contrastive loss)

    full - MLP  = real gain over equal capacity
    full - RAND = gain attributable to SYMBOLIC KNOWLEDGE (the novelty claim)

  KN metric (no classifier/threshold confound):
    relative injection ‖add‖/‖h‖ on CLEAN vs TOXIC eval text.
    selectivity = inj_toxic / inj_clean   (>1 == gate injects more on toxic = good)
    plus clean-class precision: gated vs experts-zeroed (zeroed>gated => experts add noise).

Everything runs on cached frozen embeddings — minutes on a T4.

Usage:
    python sage_v2_gonogo.py --model google/muril-base-cased --dataset hatexplain
    python sage_v2_gonogo.py --model google/muril-base-cased --dataset local:./data_store/dravidian_tamil
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score

# --------------------------------------------------------------------------
# Category-labelled slang / covert-toxicity KB  (E = 4 linguistic categories)
# --------------------------------------------------------------------------
CATEGORIES = ["slang", "identity", "metaphor", "euphemism"]
KB = [
    ("chutiya","slang"),("bhenchod","slang"),("madarchod","slang"),("bsdk","slang"),
    ("gaandu","slang"),("harami","slang"),("bakchod","slang"),("saala","slang"),
    ("thevdiya","slang"),("punda","slang"),("otha","slang"),("myre","slang"),
    ("kutta","identity"),("chakka","identity"),("patti","identity"),
    ("you people","identity"),("these people","identity"),("go back","identity"),
    ("illegal alien","identity"),("thug","identity"),("welfare queen","identity"),
    ("snake in the grass","metaphor"),("parasite","metaphor"),("leech","metaphor"),
    ("cockroach","metaphor"),("vermin","metaphor"),("cancer","metaphor"),("disease","metaphor"),
    ("economic anxiety","euphemism"),("traditional values","euphemism"),
    ("law and order","euphemism"),("family values","euphemism"),
    ("real americans","euphemism"),("silent majority","euphemism"),
]

_SYN = [
    ("you are such an idiot",1),("this movie was great loved it",0),
    ("bloody hell what is wrong with you",1),("the weather today is lovely",0),
    ("these people are a disease on our country",1),("i enjoyed the book you recommended",0),
    ("go back to where you came from",1),("have a wonderful day my friend",0),
    ("what a parasite that politician is",1),("thanks for helping me move",0),
    ("shut up you worthless thug",1),("dinner was delicious thank you",0),
    ("you people are all the same",1),("i love walks in the park",0),
    ("real americans dont act like that",1),("congratulations on your promotion",0),
    ("bhenchod kya bakwaas hai",1),("aaj ka din bahut accha tha",0),
    ("saala harami kahin ka",1),("mujhe yeh gaana pasand aaya",0),
]
def synthetic_split(seed=42):
    rng = np.random.default_rng(seed); data = _SYN*60
    data = [data[i] for i in rng.permutation(len(data))]
    k = int(0.8*len(data)); return data[:k], data[k:]

def load_data(name, seed, max_train, max_eval):
    if name == "synthetic":
        return synthetic_split(seed)
    if name.startswith("local:"):
        import csv
        from pathlib import Path
        base = Path(name[len("local:"):])
        def _read(p):
            rows = []
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    r = csv.reader(f); next(r, None)
                    for row in r:
                        if len(row) >= 2 and row[0]:
                            try: rows.append((row[0], int(row[1])))
                            except ValueError: pass
            return rows
        train = _read(base / "train.csv")
        evl   = _read(base / "validation.csv") or _read(base / "test.csv")
        if not train:
            raise ValueError(f"no train.csv under {base} (run download_datasets.py first)")
        if not evl:
            k = int(0.8*len(train)); train, evl = train[:k], train[k:]
        if max_train: train = train[:max_train]
        if max_eval:  evl   = evl[:max_eval]
        return train, evl
    from datasets import load_dataset
    if name == "hatexplain":
        try: ds = load_dataset("Hate-speech-CNERG/hatexplain")
        except Exception: ds = load_dataset("Hate-speech-CNERG/hatexplain", trust_remote_code=True)
        def to_rows(split):
            rows = []
            for ex in split:
                text = " ".join(ex["post_tokens"])
                labs = ex["annotators"]["label"]; maj = max(set(labs), key=labs.count)
                rows.append((text, 0 if maj == 1 else 1))
            return rows
        train, evl = to_rows(ds["train"]), to_rows(ds["validation"])
    else:
        raise ValueError(f"unknown dataset {name}; use synthetic | hatexplain | local:<dir>")
    if max_train: train = train[:max_train]
    if max_eval:  evl   = evl[:max_eval]
    return train, evl

@torch.no_grad()
def embed(texts, tok, bb, device, bs=64, maxlen=64):
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=maxlen,
                  return_tensors="pt").to(device)
        h = bb(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        outs.append(((h*m).sum(1)/m.sum(1).clamp(min=1e-6)).cpu())
    return torch.cat(outs, 0)

# --------------------------------------------------------------------------
# Model: one class, four architectures
# --------------------------------------------------------------------------
class SAGEv2(nn.Module):
    def __init__(self, d, arch="experts", n_cat=4, rank=16, beta=5.0, kb_grounded=True):
        super().__init__()
        self.arch, self.beta, self.E, self.kb = arch, beta, n_cat, kb_grounded
        if arch == "linear":
            self.classifier = nn.Linear(d, 2)
        elif arch == "mlp":                              # capacity-matched to experts
            H = 2*n_cat*rank                             # == expert param budget (~d*H)
            self.mlp = nn.Sequential(nn.Linear(d, H), nn.ReLU(), nn.Dropout(0.1))
            self.classifier = nn.Linear(H, 2)
        elif arch == "experts":
            self.V = nn.Parameter(torch.empty(n_cat, d, rank)); nn.init.kaiming_uniform_(self.V, a=5**0.5)
            self.U = nn.Parameter(torch.empty(n_cat, rank, d)); nn.init.kaiming_uniform_(self.U, a=5**0.5)
            self.b = nn.Parameter(torch.zeros(n_cat, rank))
            self.classifier = nn.Linear(d, 2)
            self.register_buffer("anchors",  torch.zeros(n_cat, d))
            self.register_buffer("clean_mu", torch.zeros(d))
            self.register_buffer("clean_s2", torch.ones(d))
            self.register_buffer("tau",      torch.tensor(1.0))
        else:
            raise ValueError(arch)

    def set_stats(self, anchors, clean_mu, clean_s2, tau):
        self.anchors.copy_(anchors); self.clean_mu.copy_(clean_mu)
        self.clean_s2.copy_(clean_s2); self.tau.copy_(torch.tensor(float(tau)))

    def gate(self, h):
        d_clean = ((h - self.clean_mu)**2 / self.clean_s2).mean(-1)
        sel = torch.sigmoid(self.beta*(d_clean - self.tau))
        cat = torch.softmax(self.beta*(F.normalize(h,dim=-1) @ F.normalize(self.anchors,dim=-1).T), dim=-1)
        return sel.unsqueeze(-1) * cat

    def experts_out(self, h):
        down = torch.relu(torch.einsum("nd,edr->ner", h, self.V) + self.b)
        return torch.einsum("ner,erd->ned", down, self.U)

    def forward(self, h, force_on=False, zero_experts=False):
        if self.arch == "linear": return self.classifier(h), None
        if self.arch == "mlp":    return self.classifier(self.mlp(h)), None
        if zero_experts:
            add = torch.zeros_like(h)
        else:
            pi = torch.ones(h.size(0), self.E, device=h.device) if force_on else self.gate(h)
            add = (pi.unsqueeze(-1) * self.experts_out(h)).sum(1)
        return self.classifier(h + add), add

def loss_category(m, kb_emb, kb_lab):
    up = m.experts_out(kb_emb)
    up_true = up[torch.arange(len(kb_lab)), kb_lab]
    return (1 - F.cosine_similarity(up_true, m.anchors[kb_lab], dim=-1)).mean()

def loss_kn(add, labels):
    clean = labels == 0
    return add.new_zeros(()) if clean.sum() == 0 else (add[clean]**2).sum(-1).mean()

# --------------------------------------------------------------------------
def train_model(Xtr, Ytr, d, arch, kb_grounded, kb_emb, kb_lab,
                anchors, clean_mu, clean_s2, tau, epochs, lr, lcat, lkn, device):
    m = SAGEv2(d, arch=arch, kb_grounded=kb_grounded).to(device)
    if arch == "experts": m.set_stats(anchors, clean_mu, clean_s2, tau)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    Xtr, Ytr = Xtr.to(device), Ytr.to(device); N, bs = len(Xtr), 128
    for _ in range(epochs):
        m.train(); perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; xb, yb = Xtr[idx], Ytr[idx]
            logits, add = m(xb); loss = ce(logits, yb)
            if arch == "experts":
                loss = loss + lkn*loss_kn(add, yb)
                if kb_grounded: loss = loss + lcat*loss_category(m, kb_emb.to(device), kb_lab.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

@torch.no_grad()
def macro_f1(m, X, Y, device):
    pred = m(X.to(device))[0].argmax(-1).cpu().numpy()
    return f1_score(Y.numpy(), pred, average="macro", zero_division=0)

@torch.no_grad()
def kn_metrics(m, X, Y, device):
    Xd = X.to(device); _, add = m(Xd)
    rel = (add.norm(dim=-1) / Xd.norm(dim=-1).clamp(min=1e-6)).cpu().numpy()
    Yn = Y.numpy()
    inj_c, inj_t = rel[Yn==0].mean(), rel[Yn==1].mean()
    p_gate = precision_score(Yn, m(Xd)[0].argmax(-1).cpu().numpy(), pos_label=0, average="binary", zero_division=0)
    p_zero = precision_score(Yn, m(Xd, zero_experts=True)[0].argmax(-1).cpu().numpy(), pos_label=0, average="binary", zero_division=0)
    return dict(inj_c=inj_c, inj_t=inj_t, sel=inj_t/max(inj_c,1e-9), p_gate=p_gate, p_zero=p_zero)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-cat", type=float, default=0.1)
    ap.add_argument("--lambda-kn", type=float, default=0.1)
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-eval", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}  model={a.model}  dataset={a.dataset}")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(a.model)
    bb = AutoModel.from_pretrained(a.model).to(device).eval()
    for p in bb.parameters(): p.requires_grad = False
    d = bb.config.hidden_size

    train, evl = load_data(a.dataset, a.seed, a.max_train, a.max_eval)
    print(f"[data] train={len(train)}  eval={len(evl)}  "
          f"(toxic frac: train={np.mean([y for _,y in train]):.2f} eval={np.mean([y for _,y in evl]):.2f})")
    t0 = time.time()
    Xtr = embed([t for t,_ in train], tok, bb, device); Ytr = torch.tensor([y for _,y in train])
    Xev = embed([t for t,_ in evl],   tok, bb, device); Yev = torch.tensor([y for _,y in evl])
    kb_emb = embed([t for t,_ in KB], tok, bb, device)
    kb_lab = torch.tensor([CATEGORIES.index(c) for _,c in KB])
    print(f"[cache] frozen embeddings computed once in {time.time()-t0:.1f}s")

    anchors_kb = torch.stack([kb_emb[kb_lab==e].mean(0) for e in range(len(CATEGORIES))])
    g = torch.Generator().manual_seed(a.seed)
    anchors_rand = F.normalize(torch.randn(len(CATEGORIES), d, generator=g), dim=-1)  # A5: no KB
    clean = Xtr[Ytr==0]; clean_mu, clean_s2 = clean.mean(0), clean.var(0)+1e-6
    tau = torch.quantile(((clean-clean_mu)**2/clean_s2).mean(-1), 0.90).item()

    def T(arch, kb, anchors):
        return train_model(Xtr,Ytr,d,arch,kb,kb_emb,kb_lab,anchors,clean_mu,clean_s2,tau,
                           a.epochs,a.lr,a.lambda_cat,a.lambda_kn,device)
    m_lin = T("linear", False, None)
    m_mlp = T("mlp",    False, None)
    m_rnd = T("experts",False, anchors_rand)
    m_kb  = T("experts",True,  anchors_kb)
    f_lin,f_mlp = macro_f1(m_lin,Xev,Yev,device), macro_f1(m_mlp,Xev,Yev,device)
    f_rnd,f_kb  = macro_f1(m_rnd,Xev,Yev,device), macro_f1(m_kb,Xev,Yev,device)
    kn_rnd, kn_kb = kn_metrics(m_rnd,Xev,Yev,device), kn_metrics(m_kb,Xev,Yev,device)

    print("\n"+"="*66)
    print("ABLATION LADDER  (macro-F1, higher=better)")
    print("="*66)
    print(f"  1. Linear head        (weak baseline)          : {f_lin:.4f}")
    print(f"  2. MLP head           (CAPACITY-MATCHED base)   : {f_mlp:.4f}")
    print(f"  3. SAGE experts       (RANDOM, no KB / A5)      : {f_rnd:.4f}")
    print(f"  4. SAGE experts       (KB-anchored, FULL)       : {f_kb:.4f}")
    print("-"*66)
    print(f"  full - MLP   (real gain over equal capacity)    : {f_kb-f_mlp:+.4f}")
    print(f"  full - RAND  (gain from SYMBOLIC KNOWLEDGE)      : {f_kb-f_rnd:+.4f}   <- novelty test")
    print("="*66)
    print("SELECTIVE INJECTION  (relative ‖add‖/‖h‖ on eval; gate should inject less on clean)")
    print(f"  {'model':10} {'inj_clean':>10} {'inj_toxic':>10} {'selectivity':>12}")
    for nm,k in [("RAND",kn_rnd),("KB",kn_kb)]:
        print(f"  {nm:10} {k['inj_c']:>10.4f} {k['inj_t']:>10.4f} {k['sel']:>12.2f}")
    print("  (selectivity > 1  ==  more injection on toxic than clean = good)")
    print(f"\nCLEAN-class precision — gated vs experts-zeroed  (zeroed>gated => experts add noise on clean):")
    print(f"  KB   : gated={kn_kb['p_gate']:.4f}  zeroed={kn_kb['p_zero']:.4f}")
    print("="*66)
    print("GREEN if:  (full-MLP)>0  AND  (full-RAND)>0  AND  KB selectivity>1")

if __name__ == "__main__":
    main()
