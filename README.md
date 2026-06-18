# MedTS-TTT

Official lightweight implementation of **MedTS-TTT: Test-Time Training for
Medical Time Series Classification**.

MedTS-TTT is a test-time training framework for medical time series. It targets
subject-level distribution shift in cross-subject evaluation, where samples from
the same subject should not appear in both training and test splits.

![MedTS-TTT overview](assets/model.png)

## Highlights

- **Closed-Loop Self-Alignment Test-Time Training (CLSA-TTT)** performs a
  single-step fast-weight update from each unlabeled test sample.
- **Gated Convolutional Backbone (GCB)** combines local temporal modeling,
  sample-wise adaptation, and gated token fusion.
- **Benchmark-friendly design** follows the subject-independent MedTS setup
  used by Medformer and MedTS_Evaluation.

## Quick Start

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

Run a forward-pass demo:

```bash
python demo.py
```

Expected output:

```text
input: torch.Size([8, 12, 1000])
logits: torch.Size([8, 5])
```

## Model Usage

```python
import torch
from MedTS_TTT import MedTSTTT

# Input shape: [batch, channels, time]
x = torch.randn(8, 12, 1000)

model = MedTSTTT(
    dim=128,
    max_channel=128,
    num_heads=8,
    num_layers=6,
    patch_size=8,
    num_classes=5,
)

logits = model(x)
```

The clean model implementation expects `[B, C, T]`.

## Benchmark Compatibility

This repository does not duplicate the full data preprocessing and training
framework from prior benchmark projects. Instead, it provides a small adapter
for the Medformer / Time-Series-Library style API.

See [benchmark/README.md](benchmark/README.md) for details.

Recommended benchmark references:

- [Medformer](https://github.com/DL4mHealth/Medformer)
- [MedTS_Evaluation](https://github.com/DL4mHealth/MedTS_Evaluation)

## Datasets

The paper evaluates on four public clinical datasets under subject-independent
splits, following the processed data and evaluation protocol of Medformer:

| Modality | Dataset | Task |
| --- | --- | --- |
| EEG | APAVA | Alzheimer's disease detection |
| EEG | ADFTD | HC / FTD / AD classification |
| ECG | PTB | myocardial infarction detection |
| ECG | PTB-XL | cardiac condition classification |

Processed data should follow the benchmark structure:

```text
dataset/DATA_NAME/
  Feature/
    feature_ID.npy
  Label/
    label.npy
```

## Main Results

Under subject-independent evaluation, MedTS-TTT achieves 11 top-1 rankings out
of 12 evaluations across 4 datasets, 9 baselines, and 3 metrics.

| Metric | Average Result |
| --- | ---: |
| Accuracy | 75.18 |
| Macro-F1 | 70.20 |
| Macro-AUROC | 86.86 |

Dataset-level results for MedTS-TTT:

| Dataset | Accuracy | Macro-F1 | Macro-AUROC |
| --- | ---: | ---: | ---: |
| APAVA | 83.03 | 81.52 | 89.96 |
| ADFTD | 58.10 | 55.18 | 75.41 |
| PTB | 85.04 | 80.71 | 91.31 |
| PTB-XL | 74.56 | 63.39 | 90.77 |

## Test-Time Alignment

![Feature alignment under test-time training](assets/ttt_alignment.png)

CLSA-TTT narrows the train-test feature gap under subject-level distribution
shift while preserving class discriminability.

## Citation

Citation information will be updated after the paper metadata is finalized.

## Acknowledgement

This project follows the medical time-series benchmark protocol and processed
data format introduced by Medformer and MedTS_Evaluation. We thank the authors
for establishing a useful evaluation ecosystem for subject-independent medical
time-series classification.
