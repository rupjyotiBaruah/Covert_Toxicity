# SAGE-Fusion POC

Spectral-Anomaly Gated Expert Fusion for covert toxicity detection in code-mixed text.

The entire implementation lives in a single file, `main.py`. It has two modes:

- `--mode baseline` : a frozen backbone with a linear classification head.
- `--mode sage` : the full SAGE-Fusion model (frozen backbone, spectral-anomaly router, low-rank Hadamard fusion expert bank, classifier).

You run everything through `main.py`. There is nothing else to import or configure.

---

## Project layout

```
poc/
├── main.py            # the complete implementation and the only entry point
├── requirements.txt   # dependencies
├── README.md          # this file
└── CSL/               # the manuscript (LaTeX) - not needed to run the code
```

---

## Execution guide (single flow)

### Step 1 - Get the code

Open a terminal in the `poc/` folder (the folder that contains `main.py`).

### Step 2 - Create an environment and install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

A GPU is optional. The code uses CUDA automatically if it is available, otherwise it runs on CPU.

### Step 3 - Offline smoke test (verify the setup)

This uses a small built-in synthetic dataset and needs no dataset download.

```bash
python main.py --mode baseline --dataset smoke_test --backbone bert-base-uncased --epochs 2
python main.py --mode sage     --dataset smoke_test --backbone bert-base-uncased --epochs 2 --experts 4 --rank 16
```

If both commands train and print metrics, the setup is correct.

### Step 4 - Run on a real dataset

Datasets download automatically from the HuggingFace Hub on first use and are cached.

```bash
# baseline
python main.py --mode baseline --dataset hatexplain --backbone bert-base-uncased --epochs 5 --batch-size 16

# SAGE-Fusion
python main.py --mode sage --dataset hatexplain --backbone bert-base-uncased --epochs 5 --batch-size 16 --experts 4 --rank 16
```

For the production Indic backbone, set `--backbone AI4Bharat/IndicBERTv2-MLM-Supervised-only`.

### Step 5 - Read the results

Each run writes to `--output-dir` (default `./results`):

- `<mode>_best.pt` : the best checkpoint by macro-F1.
- `<mode>_results.json` : best metrics and the full per-epoch history.

Metrics are also printed live during training (loss, accuracy, macro-F1, binary-F1).

---

## Datasets

Pass one of these keys to `--dataset`:

| Key | Dataset | Source |
|-----|---------|--------|
| `smoke_test` | Synthetic (built-in, offline) | none |
| `hatexplain` | HateXplain | HuggingFace |
| `implicit_hate` | Implicit Hate Corpus | HuggingFace |
| `toxigen` | ToxiGen | HuggingFace |
| `dravidiancodemix_ta` | DravidianCodeMix Tamil-English | direct URL |
| `dravidiancodemix_ml` | DravidianCodeMix Malayalam-English | direct URL |
| `jigsaw_multilingual` | Jigsaw Multilingual | HuggingFace |

If a download fails, the loader falls back to the synthetic smoke-test data so a run never crashes.

---

## Command-line options

Run `python main.py --help` for the full list. The main ones:

```
--mode {baseline,sage}   Execution mode (default: baseline)
--backbone HF_ID         HuggingFace encoder (default: bert-base-uncased)
--dataset KEY            Dataset key (default: smoke_test)
--epochs N               Training epochs (default: 3)
--batch-size N           Batch size (default: 16)
--lr FLOAT               Learning rate (default: 1e-4)
--max-length N           Max sequence length (default: 128)
--pooling {cls,mean,max} Pooling strategy (default: cls)
--seed N                 Random seed (default: 42)
--output-dir PATH        Where to save checkpoints and results (default: ./results)

# SAGE-only options
--experts N              Number of symbolic experts (default: 4)
--rank N                 Low-rank bottleneck size (default: 16)
--beta FLOAT             Router sigmoid temperature (default: 1.0)
--tau-percentile FLOAT   Anomaly threshold percentile (default: 0.9)
--kb-path PATH           Optional custom slang lexicon file (token<TAB>category per line)
```

---

## Notes

- The backbone is frozen in both modes. Only the head (baseline) or the expert bank, Hadamard masks, and head (SAGE) are trained.
- In SAGE mode, `main.py` first calibrates the spectral router and initializes the experts from the built-in slang knowledge base, then trains.
- The `CSL/` folder holds the paper and is not used at runtime.
