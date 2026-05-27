# CodeBrain (disentangle + downstream) — Training Guide

This repo contains two main training paths:

- **Downstream ("standard")** trainer: `Downstream/finetune_main.py`
- **Disentangle v7** trainer (API-style, produces `summary.json`): `finetune_disentangle.py` -> `disentangle_v7/train.py`

This README is written so another user can run training and evaluation on a fresh machine.

## 0) Repo Layout

- `Downstream/` : standard trainer entrypoint + training loop
- `disentangle_v7/` : disentangle v7 trainer + API
- `Datasets/` : dataset loaders (LMDB)
- `Models/` : model definitions per dataset
- `pretrain_weights/EEGSSM_weights.pth` : foundation weights (NOT tracked by git)
- `tools/` : automation helpers (`tick.sh`, `autopilot.py`) and utilities
- `docs/` : results tables + notes

## 1) Environment Setup (Recommended: isolated env)

Important: do **not** install other projects' deps into the same env while runs are active.

### Option A: Python venv (recommended for reproducibility)

```bash
# 1) clone
git clone git@github.com:wythedee/CodeBrain_disentangle.git
cd CodeBrain_disentangle

# 2) create venv
python3.11 -m venv .venv
source .venv/bin/activate

# 3) install PyTorch with CUDA (example: CUDA 12.1)
python -m pip install -U pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 4) install the rest
python -m pip install -r requirements.txt
```

### Option B: Conda (also fine)

```bash
conda create -n codebrain python=3.11 -y
conda activate codebrain

# pick one CUDA toolkit that matches your driver
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2) Data Preparation (LMDB)

All downstream datasets are expected to be **LMDB** directories containing:

- `data.mdb`, `lock.mdb`
- key `__keys__` -> a pickled dict with splits: `{"train": [...], "val": [...], "test": [...]}`
- each sample key -> a pickled dict with at least:
  - `sample`: numpy array
  - `label`: integer class id

If you already have our preprocessed LMDBs, you can point `--datasets_dir` to them.

### Quick sanity-check an LMDB

```bash
python tools/inspect_lmdb.py --lmdb_dir /path/to/processed_average --split train
```

## 3) Foundation Weights (required)

Many training commands expect:

- `pretrain_weights/EEGSSM_weights.pth`

This file is **not** tracked by git (large binary). You must copy it onto the new machine.
Expected path:

```text
<repo_root>/pretrain_weights/EEGSSM_weights.pth
```

## 4) Run: Standard (Downstream) Trainer

Entrypoint:

```bash
python Downstream/finetune_main.py ...
```

Example (BCIC-IV-2a):

```bash
CUDA_VISIBLE_DEVICES=0 \
python Downstream/finetune_main.py \
  --cuda 0 --seed 0 --epochs 50 --batch_size 64 --num_workers 16 \
  --lr 1e-4 --weight_decay 0.05 --optimizer AdamW --multi_lr \
  --downstream_dataset BCIC-IV-2a \
  --datasets_dir /path/to/MI_BCI_IV_2a/processed_average \
  --use_pretrained_weights --foundation_dir pretrain_weights/EEGSSM_weights.pth \
  --model_dir runs/train_bcic_std/model/ \
  --log_dir runs/train_bcic_std/log/
```

Notes:
- Logs are written under `--log_dir/<dataset>/...txt`
- Standard trainer prints lines like `Test Evaluation: acc: ...`.

## 5) Run: Disentangle v7 Trainer

Entrypoint:

```bash
python finetune_disentangle.py ...
```

Disentangle v7 writes:
- `model_dir/<run_name>/summary.json` with `best_test_acc`, `best_epoch`, and the config snapshot.

Example (SEED-V xsub555, codebook off):

```bash
CUDA_VISIBLE_DEVICES=0 \
python finetune_disentangle.py \
  --cuda 0 --seed 0 --epochs 50 --batch_size 64 --num_workers 16 \
  --lr 1e-4 --weight_decay 0.05 --optimizer AdamW --multi_lr \
  --select_best_by val_kappa \
  --foundation_dir pretrain_weights/EEGSSM_weights.pth \
  --temperature 0.07 --var_ratio_r 2.0 \
  --data_backend disentangle \
  --downstream_dataset SEED-V --dataset_name EMO_SEED_V \
  --datasets_dir /path/to/EMO_SEED_V/processed_average_cross \
  --subject_samples 0 \
  --no-bypass_mid_mlp --no-bypass_content_proj \
  --use_style_contrastive --use_style_invariance --use_var_ratio --use_film_reconstruction \
  --no-use_content_codebook \
  --lambda_task 1.0 --lambda_style_cl 0.02 --lambda_style_invar 0.001 --lambda_var_ratio 0.005 --lambda_rec 0.01 \
  --lambda_vq 0.0 --lambda_vq_usage 0.0 \
  --no-save_best_checkpoint \
  --model_dir runs/train_seedv/seedv_allloss_xsub555_nocodebook_s0
```

## 6) Automation (Optional)

For unattended monitoring + auto-launch (up to 4 GPUs), use:

```bash
./tools/tick.sh
```

- It prints alive runs (from `runs/.../pid_*.txt`) and last epoch lines.
- It appends any newly created `summary.json` to `docs/run_tables_20260218.md`-style audit markdown.
- It launches more runs via `tools/autopilot.py` until `MAX_RUNNING` is reached.

## 7) Results Snapshot

See:
- `docs/run_tables_20260218.md` (aggregated table of baseline + ablations)

## 8) Common Failure Modes

- **LMDB unpickle error** (`numpy._core...`): indicates an environment mismatch while unpickling.
  - Fix: keep a stable env; avoid changing numpy mid-run.
- **`subject_samples>0`** can be unstable on some datasets (observed early exits on SEED-V).
- **`latest_progress.tsv`** under `runs/` is considered human-curated; do not auto-edit it.
