# DEFT

Repository of paper "DEFT: Disentanglement-Enhanced Fine-Tuning for EEG Foundation Models".

This repository contains the training and evaluation package for DEFT across four EEG foundation backbones:

- `Codebrain/`: CodeBrain + Recursive Gaussian Disentangler
- `CBraMod/`: CBraMod + Recursive Gaussian Disentangler, including preprocessing scripts
- `CSBrain/`: CSBrain + Recursive Gaussian Disentangler
- `LaBraM/`: LaBraM + Recursive Gaussian Disentangler

The package keeps only code needed for downstream training and evaluation. Pretrained foundation checkpoints and EEG datasets are not committed; provide their paths through environment variables.

## Main Experiments

The main training target follows `TRAINING.md`: six subject-disjoint datasets, downstream fine-tuning only, and validation-based model selection.

| Dataset | Task | Input shape | # classes |
|---|---|---:|---:|
| BCIC-IV-2a | Motor imagery | `(22,4,200)` | 4 |
| FACED | Emotion recognition | `(32,10,200)` | 9 |
| ISRUC-S1 | Sleep staging | `(20,6,6000)` | 5 |
| MentalArith. | Workload assessment | `(20,5,200)` | 2 |
| BCIC-Speech | Speech decoding | `(60,3,200)` | 3 |
| Mumtaz2016 | MDD classification | `(19,5,200)` | 2 |

## Required Paths

Set the LMDB dataset root before running:

```bash
export DATA_ROOT=/path/to/lmdb_root
```

Set pretrained checkpoint paths before running:

```bash
export CODEBRAIN_CKPT=/path/to/CodeBrain.pth
export CBRAMOD_CKPT=/path/to/CBraMod.pth
export CSBRAIN_CKPT=/path/to/CSBrain.pth
export LABRAM_CKPT=/path/to/labram-base.pth
```

## One-Command Training

Run all four backbones:

```bash
bash run_all_main.sh
```

Useful overrides:

```bash
DATA_ROOT=/path/to/lmdb_root SEEDS="0" MODE="deft" GPU=0 bash run_all_main.sh
DATA_ROOT=/path/to/lmdb_root SEEDS="0 1 2" MODE="both" bash run_all_main.sh
```

`MODE` can be `baseline`, `deft`, or `both`. Each training run uses one GPU; launch separate shells with different `GPU` values if you want parallel execution across multiple GPUs.

## Per-Backbone Scripts

Each backbone has its own script:

```bash
bash Codebrain/scripts/train_main.sh
bash CBraMod/scripts/train_main.sh
bash CSBrain/scripts/train_main.sh
bash LaBraM/scripts/train_main.sh
```

Outputs are written to `outputs/main/<backbone>/...` by default.
