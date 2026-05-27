# disentangle (LaBraM)

This folder adds a small, API-style wrapper layer around the upstream LaBraM
codebase.

Goals:
- Do **not** modify upstream training scripts
- Provide a clean programmatic / CLI interface for experiments
- Start with BCIC-IV-2a using the shared LMDB dataset format

## BCIC-IV-2a (LMDB)

We use the same LMDB record structure across downstream tasks:

- LMDB key `__keys__` -> dict with `train`/`val`/`test` splits
- each sample key -> dict with `sample` (22,4,200) and `label` (0..3)

### Fine-tune

```bash
conda activate labram
python -m disentangle.finetune_bcic2a \
  --lmdb_dir /path/to/lmdb_root/MI_BCI_IV_2a/processed_average \
  --finetune ./checkpoints/labram-base.pth \
  --output_dir ./checkpoints/finetune_bcic2a_seed0 \
  --log_dir ./log/finetune_bcic2a_seed0 \
  --eval_metric balanced_accuracy \
  --epochs 50 --batch_size 64 --lr 5e-4 --seed 0
```

Outputs:
- default: `summary.json` + `args.json`
- optional: add `--save_ckpt` to also write `best.pth` / `last.pth`

## SEED-IV (LMDB)

```bash
conda activate labram
python -m disentangle.finetune_seediv \
  --lmdb_dir /path/to/lmdb_root/EMO_SEED_IV/processed_average \
  --finetune ./checkpoints/labram-base.pth \
  --output_dir ./checkpoints/finetune_seediv_seed0 \
  --log_dir ./log/finetune_seediv_seed0 \
  --eval_metric balanced_accuracy \
  --epochs 30 --batch_size 64 --lr 5e-4 --seed 0
```

Note: checkpoint saving is disabled by default for both wrappers. Use
`--save_ckpt` only when you explicitly need model weights.

## Generic LMDB Fine-tuning (baseline + Gaussian disentangle)

This repo now includes a generic LMDB entrypoint under `disentangle/finetune_lmdb.py`.
It is designed for the shared LMDB format:

- `__keys__` stores `train` / `val` / `test`
- each sample record stores `sample`, `label`, and ideally `subject`

### Baseline sweep

```bash
python scripts/runners/run_labram_baseline_classifier_sweep.py \
  --python python \
  --dataset_name BCIC-IV-2a \
  --lmdb_dir /path/to/lmdb_root/MI_BCI_IV_2a/processed_average \
  --finetune ./checkpoints/labram-base.pth \
  --output_root ./checkpoints/lmdb_baseline_sweep
```

### Gaussian search

```bash
python scripts/runners/run_labram_gaussian_search.py \
  --python python \
  --dataset_name BCIC-IV-2a \
  --lmdb_dir /path/to/lmdb_root/MI_BCI_IV_2a/processed_average \
  --finetune ./checkpoints/labram-base.pth \
  --classifier all_patch_reps \
  --output_root ./checkpoints/lmdb_gaussian_search
```

### Notes

- Current preset channel maps cover: `BCIC-IV-2a`, `SEED-IV`, `SEED`, `HeBin2021-LR`, `HeBin2021-UD`, `MI-KoreaU`, `MI-Cho2017`, `PhysioNet-MI`.
- If a dataset is not covered, pass `--channel_names` explicitly.
- If a channel is not present in LaBraM's `standard_1020`, the script will stop and ask for a manual remap instead of silently guessing.
