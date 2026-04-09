# DITRL — Domain-Invariant Temporal Representation Learning

> **"Universal Time-Series Representations Do Not Transfer:
> A Cross-Domain Study and Framework"**

## Overview

DITRL is a domain-invariant representation learning framework for
cross-domain time-series generalization. It combines:

- **SpectralDAIN** — per-domain adaptive instance normalisation
  with learnable DFT frequency masking
- **Adversarial domain discriminator** — gradient reversal layer
  for distribution alignment
- **Prototype-guided contrastive loss** — cross-domain
  class-level semantic consistency

Evaluated on five real-world datasets under a leave-one-domain-out
(LODO) protocol against ERM, TS2Vec, and GPT4TS baselines.

---

## Datasets

| Domain | Dataset | Size | Source |
|--------|---------|------|--------|
| Healthcare | PTB-XL ECG | 21,837 records | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) |
| Finance | S&P 500 | 50 tickers, 14 years | Auto-downloaded (Stooq) |
| IoT | PAMAP2 | 9 subjects, 18 activities | [UCI](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) |
| Climate | ETTh1 | 17,420 hourly records | Auto-downloaded |
| Transportation | METR-LA | 207 sensors, 34,272 steps | [DCRNN repo](https://github.com/liyaguang/DCRNN) |

PTB-XL and PAMAP2 must be downloaded manually due to size/licensing.
All other datasets are downloaded automatically on first run.

---

## Installation

```bash
git clone https://github.com/shumaila0789/DITRL.git
cd DITRL
pip install -r requirements.txt
```

---

## Data Setup

Place downloaded datasets in the `data/` folder:

data/
├── ptbxl/
├── pamap2/
├── metrla/
└── ETTh1.csv

## Usage

```bash
# Full publication run (5 seeds, 50 epochs, all datasets)
python ditrl_pub_real.py

# Quick smoke-test (2 seeds, 15 epochs, 200 samples/class)
python ditrl_pub_real.py --quick

# Skip PTB-XL and PAMAP2 (uses ETTh1 as placeholder)
python ditrl_pub_real.py --skip_heavy

# Custom epoch count
python ditrl_pub_real.py --epochs 30
```

---

## Outputs

All outputs are saved in the same directory as the script:

| File | Description |
|------|-------------|
| `transfer_heatmap.png` | Source→target Macro-F1 matrix |
| `in_vs_cross_domain.png` | In-domain vs cross-domain bar chart |
| `embedding_tsne.png` | t-SNE embedding visualisation |
| `alignment_metrics.png` | MMD and Fréchet Distance comparison |
| `ablation_results.png` | Component ablation TS and DRI |
| `robustness_curves.png` | Perturbation robustness curves |
| `main_results.csv` | Macro-F1, Accuracy, AUC per method |
| `transfer_metrics.csv` | TS, DRI, PDR, CDG per method |
| `ablation_results.csv` | Ablation TS and DRI |
| `robustness_results.csv` | F1 under perturbation |

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Sequence length T | 64 |
| Patch size | 8 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 3 |
| Optimiser | Adam (lr=5×10⁻⁴) |
| Adversarial weight W_adv | 0.75 |
| Prototype weight W_proto | 0.70 |
| Prototype momentum | 0.90 |
| Input jitter std | 0.05 |
| Random seeds | 42, 137, 256, 512, 1024 |

---

## License

This project is licensed under the MIT License.