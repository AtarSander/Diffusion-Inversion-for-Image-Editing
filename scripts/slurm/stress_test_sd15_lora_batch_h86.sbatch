#!/bin/bash
#SBATCH --job-name=sd15-lora-batch-stress
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com
set -euo pipefail
REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
RESULT_DIR=$REPO_DIR/logs/batch_size_stress
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
RESULT_LABEL=${RESULT_LABEL:-${MODEL_DTYPE}}
RESULT_LOG="$RESULT_DIR/sd15_lora_${RESULT_LABEL}_h86_${SLURM_JOB_ID}.log"
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sd15-lora-batch-stress-${SLURM_JOB_ID}
VENV_DIR="$JOB_TMP_DIR/venv"
module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=$REPO_DIR/cache/h86/batch_size_stress/huggingface
export UV_CACHE_DIR=$REPO_DIR/.uv-cache
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$RESULT_DIR" "$JOB_TMP_DIR" "$UV_CACHE_DIR" "$HF_HOME"
cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev
python3 -m diff_inversion.modeling.batch_size_stress_test --config-name train_sd15 \
  model.torch_dtype="$MODEL_DTYPE" model.variant=null \
  +stress_test.synthetic=true +stress_test.max_batch_size=512 >"$RESULT_LOG" 2>&1
grep "largest_micro_batch" "$RESULT_LOG"
echo "Full result log: $RESULT_LOG"
