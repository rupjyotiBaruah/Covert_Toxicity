### Hard-Negative Supervised Contrastive Learning with Calibrated Decisions for Covert Code-Mixed Toxicity

Reproducible experiment framework for the paper. Built on verified components
(`sage_v6.py`, `sage_v7.py`); the method is **not** the falsified symbolic/lexical
approach — it is a fine-tuned encoder + hard-negative supervised contrastive
front-end + calibrated decision threshold.

## Requested features — where each lives

| # | Feature | Where |
|---|---|---|
| 1 | Log file | `paper1_run.py` → `paper1_results/logs/run_<ts>.log` (file + console) |
| 2 | Checkpoints + resume | cell-level resume (skips completed `results/*.json`); `--save-models` dumps weights to `paper1_results/checkpoints/` |
| 3 | Different models | `--profile models`, or `models:[...]` in config (MuRIL, XLM-R, IndicBERT, mBERT) |
| 4 | Different datasets | `datasets:[...]` in config; Tamil/Malayalam/Kannada/HateXplain by default |
| 5 | Ablation study | 3 trained configs × 2 thresholds → ladder (−contrastive, −hard-neg, −calibration) in `paper1_report.py` |
| 6 | Supportive results | `paper1_report.py` → CSV, LaTeX tables, significance tests, figures |
| 7 | README | this file |
| 8 | Datasets + instructions | `PAPER1_DATASETS.md` |
| 9 | Time estimate | below + printed live with ETA |

## Quick start

```bash
pip install torch transformers datasets scikit-learn matplotlib tqdm

# 1. get data (see PAPER1_DATASETS.md)
python download_datasets.py --data-dir ./data_store \
    --datasets dravidian_tamil dravidian_malayalam dravidian_kannada hatexplain

# 2. smoke-test the pipeline (~10 min)
python paper1_run.py --profile minimal

# 3. full single-model matrix (~30-35 min)
python paper1_run.py --profile full

# 4. multi-model matrix (~2.5-3 h)
python paper1_run.py --profile models --save-models

# 5. build all paper artifacts
python paper1_report.py --results ./paper1_results/results --out ./paper1_results/report
```

Interrupt any time (Ctrl-C) and re-run the same command — completed cells are skipped.

## Method (what each config is)

Trained configs (retrained): `ce`, `supcon` (+SupCon), `hardneg` (+hard-negative SupCon).
Each evaluated at the default 0.5 threshold **and** a validation-calibrated threshold.
The paper's methods are assembled post-hoc:

- **CE@0.5** — naive baseline
- **CE+cal** — calibration only (isolates the calibration effect)
- **SupCon+cal** — contrastive front-end
- **Proposed = HardNeg-SupCon + cal** — full method
- Ablations: −calibration (`hardneg`@0.5), −hard-neg (`supcon`+cal), −contrastive (`ce`+cal)

## Outputs

```
paper1_results/
├── logs/run_<ts>.log            # feature 1
├── config_used.json
├── results/<model>__<dataset>__<config>__s<seed>.json   # metrics @0.5 & @cal + test probs
├── checkpoints/*.pt             # feature 2 (only with --save-models)
└── report/                      # feature 6 (from paper1_report.py)
    ├── results.csv
    ├── main_toxf1.tex, main_prauc.tex
    ├── ablation.tex
    ├── significance.txt         # paired bootstrap, Proposed vs CE+cal
    └── figures/ablation_bar.png, prauc_bar.png
```

## Tentative execution time (feature 9)

Single T4 GPU, base-size encoders, 3 epochs, 4000 train / 1500 test:

| Profile | Matrix | Cells | ~Time |
|---|---|---|---|
| `minimal` | 1 model × 2 ds × 3 cfg × 3 seeds | 18 | **~10 min** |
| `full` | 1 model × 4 ds × 3 cfg × 5 seeds | 60 | **~30–35 min** |
| `models` | 4 models × 4 ds × 3 cfg × 5 seeds | 240 | **~2.5–3 h** |

Notes: ~30 s/cell for base models on T4; large encoders (e.g. `xlm-roberta-large`)
are ~2.5–3× slower. Resume makes long runs safe to checkpoint across sessions.
CPU-only works for smoke tests but is far slower.

## Config file (optional)

```json
{
  "models": ["google/muril-base-cased", "xlm-roberta-base"],
  "datasets": ["dravidian_tamil", "dravidian_malayalam", "hatexplain"],
  "seeds": [42, 123, 2024, 7, 99],
  "epochs": 3, "max_train": 4000
}
```
Run with `python paper1_run.py --config my_config.json`.
