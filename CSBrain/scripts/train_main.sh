#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${PYTHON:=python}"
: "${GPU:=0}"
: "${SEEDS:=0 1 2}"
: "${MODE:=both}"
: "${DATA_ROOT:?Set DATA_ROOT to the root directory containing the LMDB datasets}"
: "${OUTPUT_ROOT:=$REPO_DIR/../outputs/main}"
: "${CSBRAIN_CKPT:=$REPO_DIR/pretrained/CSBrain.pth}"

COMMON_ARGS=(
  --data_backend disentangle
  --cuda "$GPU"
  --epochs 20
  --batch_size 64
  --num_workers 8
  --lr 1e-4
  --weight_decay 0.05
  --optimizer AdamW
  --dropout 0.1
  --label_smoothing 0.1
  --multi_lr
  --no-frozen
  --use_pretrained_weights
  --foundation_dir "$CSBRAIN_CKPT"
  --select_best_by val
  --model CSBrain
  --n_layer 12
  --branch_impl gaussian
  --gaussian_num_transition_components 16
  --gaussian_remove_gate_init -2.0
  --lambda_task 1.0
  --temperature 0.07
  --var_ratio_r 2.0
  --no-use_content_codebook
  --no-task_use_quantized_content
  --no-recon_use_quantized_content
)

DEFT_ARGS=(
  --lambda_style_cl 0.01
  --lambda_style_invar 0.01
  --lambda_var_ratio 0.1
  --lambda_rec 0.02
  --use_style_contrastive
  --use_style_invariance
  --use_var_ratio
  --use_film_reconstruction
  --no-bypass_mid_mlp
  --no-bypass_content_proj
)

BASELINE_ARGS=(
  --lambda_style_cl 0.0
  --lambda_style_invar 0.0
  --lambda_var_ratio 0.0
  --lambda_rec 0.0
  --no-use_style_contrastive
  --no-use_style_invariance
  --no-use_var_ratio
  --no-use_film_reconstruction
  --bypass_mid_mlp
  --bypass_content_proj
)

run_one() {
  local variant="$1" seed="$2" dataset="$3" dataset_name="$4" rel_dir="$5" num_classes="$6" classifier="$7"
  local out_dir="$OUTPUT_ROOT/CSBrain/$variant/$dataset/seed${seed}"
  mkdir -p "$out_dir"
  local variant_args=("${DEFT_ARGS[@]}")
  if [[ "$variant" == "baseline" ]]; then
    variant_args=("${BASELINE_ARGS[@]}")
  fi
  "$PYTHON" "$REPO_DIR/finetune_v7.py" \
    "${COMMON_ARGS[@]}" "${variant_args[@]}" \
    --seed "$seed" \
    --downstream_dataset "$dataset" \
    --dataset_name "$dataset_name" \
    --datasets_dir "$DATA_ROOT/$rel_dir" \
    --num_of_classes "$num_classes" \
    --classifier "$classifier" \
    --model_dir "$out_dir"
}

while IFS='|' read -r dataset dataset_name rel_dir num_classes classifier; do
  [[ -z "$dataset" || "$dataset" =~ ^# ]] && continue
  for seed in $SEEDS; do
    if [[ "$MODE" == "baseline" || "$MODE" == "both" ]]; then
      run_one baseline "$seed" "$dataset" "$dataset_name" "$rel_dir" "$num_classes" "$classifier"
    fi
    if [[ "$MODE" == "deft" || "$MODE" == "both" ]]; then
      run_one deft "$seed" "$dataset" "$dataset_name" "$rel_dir" "$num_classes" "$classifier"
    fi
  done
done <<'DATASETS'
BCIC-IV-2a|MI_BCI_IV_2a|MI_BCI_IV_2a/processed_average|4|all_patch_reps
FACED|EMO_FACED|EMO_FACED/processed_average|9|all_patch_reps
ISRUC|SLEEP_05_isruc|SLEEP_05_isruc/processed_average|5|all_patch_reps
MentalArithmetic|STR_MentalArithmetic|STR_MentalArithmetic/processed_average|2|all_patch_reps
CS-BCIC-Track4|CS_04_BCIC_Track3|CS_04_BCIC_Track3/processed_subject_split_20260429|3|all_patch_reps
Mumtaz2016|MDD_Mumtaz|MDD_Mumtaz/processed_average|2|all_patch_reps
DATASETS

