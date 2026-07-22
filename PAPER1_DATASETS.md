# Paper-1 Datasets — download & storage (verified July 2026)

All datasets are normalized to a uniform `text,label` CSV (label: 0=benign, 1=toxic)
by `download_datasets.py`, and stored under a folder you choose (default `./data_store`).

## One-shot download

```bash
pip install datasets tqdm
# core code-mixed sets used in the paper (Tamil, Malayalam, Kannada) + English implicit
python download_datasets.py --data-dir ./data_store \
    --datasets dravidian_tamil dravidian_malayalam dravidian_kannada hatexplain
# optional: ToxiGen (needs a free HF token — gated)
python download_datasets.py --data-dir ./data_store --datasets toxigen --hf-token hf_xxx
python download_datasets.py --data-dir ./data_store --verify        # re-check counts
```

Files land as `./data_store/<key>/{train,validation,test}.csv` + `manifest.json`.
The experiment runner reads them via `local:./data_store/<key>` automatically.

## Verified sources (HuggingFace IDs, July 2026)

| Key (runner) | Dataset | HF ID / config | Languages | License | Auto? |
|---|---|---|---|---|---|
| `hatexplain` | HateXplain | `Hate-speech-CNERG/hatexplain` | English (implicit) | MIT | ✅ |
| `dravidian_tamil` | DravidianCodeMix | `community-datasets/offenseval_dravidian` · `tamil` | Tamil-English | CC BY 4.0 | ✅ |
| `dravidian_malayalam` | DravidianCodeMix | `community-datasets/offenseval_dravidian` · `malayalam` | Malayalam-English | CC BY 4.0 | ✅ |
| `dravidian_kannada` | DravidianCodeMix | `community-datasets/offenseval_dravidian` · `kannada` | Kannada-English | CC BY 4.0 | ✅ |
| `toxigen` | ToxiGen | `toxigen/toxigen-data` · `annotated` | English (implicit) | HF terms | ⚠️ token |

### Manual datasets (EULA / Kaggle / GitHub — not auto-downloadable)

Place a `raw.*` file under `./data_store/<key>/` then re-run `--verify`:

| Key | Source | File to place |
|---|---|---|
| `hinglish_hasoc` | https://hasocfire.github.io/ (accept EULA) | `raw.tsv` (`text<TAB>label`) |
| `implicit_hate` | https://github.com/SALT-NLP/implicit-hate (request access) | `raw.tsv` (`post<TAB>class`) |
| `jigsaw_multilingual` | https://kaggle.com/c/jigsaw-multilingual-toxic-comment-classification | `raw.csv` (`comment_text,toxic`) |
| `banglish` | https://data.mendeley.com/datasets/23dp3t88vk | `raw.csv` (`text,label`) |

## Notes
- Labels are binarized: DravidianCodeMix `Not_offensive`→0 else 1 (non-target-language rows dropped);
  HateXplain majority of {hatespeech,offensive}→1, `normal`→0; ToxiGen `toxicity_human`≥3→1.
- If a very new `datasets` version fails on script-based loaders, pin `pip install "datasets<3"`.
- Cache the `./data_store` folder (e.g., on Google Drive in Colab) to avoid re-downloading.
