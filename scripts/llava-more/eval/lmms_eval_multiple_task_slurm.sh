#!/bin/bash -l
#SBATCH --job-name=lmms_eval_aloe
#SBATCH --output=./log/%x-%A_%a.out
#SBATCH --error=./log/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=144000
#SBATCH --gres=gpu:h200:1
#SBATCH --time=6:00:00
#SBATCH --array=0-0

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/u/rmaser/LLaVA-MORE}"
MODEL_PATH="${MODEL_PATH:-aimagelab/LLaVA_MORE-gemma_2_9b-siglip2-finetuning}"
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-gemma_2}"
CONV_MODE="${CONV_MODE:-$MODEL_ARCHITECTURE}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/lmms_eval/logs}"
VISION_TOWER="${VISION_TOWER:-aloe://distilled/siglip2_so400m_432px}"
BATCH_SIZE="${BATCH_SIZE:-1}"
USE_ALOE="${USE_ALOE:-1}"

task_list=(
    pope
    mme
    gqa
    scienceqa_img
    mmmu_val
    seedbench
    ai2d
    textvqa_val
    vizwiz_vqa_val
)

task_count=${#task_list[@]}
task_index=${SLURM_ARRAY_TASK_ID:-0}
if (( task_index < 0 || task_index >= task_count )); then
    echo "SLURM_ARRAY_TASK_ID=$task_index is out of range for ${task_count} tasks."
    exit 1
fi

task_name=${task_list[$task_index]}

cd "$REPO_ROOT"
# Some activation scripts assume optional env vars may be unset, which conflicts with `set -u`.
set +u
eval "$(pixi shell-hook)"
set -u

export PYTHONPATH=.
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="pretrained=$MODEL_PATH,dtype=float16"
VISION_TOWER_ARGS=()
if [[ "$USE_ALOE" == "1" || "$USE_ALOE" == "true" || "$USE_ALOE" == "yes" ]]; then
    VISION_TOWER_ARGS=(--vision_tower "$VISION_TOWER")
fi

echo "Running task: $task_name"
echo "Model path: $MODEL_PATH"
echo "Vision tower override: $VISION_TOWER"
echo "Model architecture: $MODEL_ARCHITECTURE"
echo "Conversation mode: $CONV_MODE"
echo "ALOE_REPO_ROOT: $ALOE_REPO_ROOT"
echo "USE_ALOE: $USE_ALOE"
echo "Batch size: $BATCH_SIZE"
echo "Output dir: $OUTPUT_DIR"

pixi run python -u src/lmms_eval/__main__.py \
    --conv_mode "$CONV_MODE" \
    --model_architecture "$MODEL_ARCHITECTURE" \
    --task "$task_name" \
    --model llava \
    --model_args "$MODEL_ARGS" \
    "${VISION_TOWER_ARGS[@]}" \
    --device cuda:0 \
    --batch_size "$BATCH_SIZE" \
    --output "$OUTPUT_DIR" \
    --log_samples_suffix aloe-eval \
    --log_samples \
    --timezone Europe/Paris