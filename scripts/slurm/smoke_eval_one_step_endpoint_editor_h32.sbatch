#!/bin/bash
#SBATCH --job-name=endpoint-editor-smoke
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail
TASK_ID=${1:?Pass editor task id: 1=PnP, 2=MasaCtrl, 3=Pix2Pix-Zero}
REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
CHECKPOINT=${REPO_DIR}/checkpoints/sd15_inversion_lora_r16_uniform_e1/checkpoint_final.pt
MODULES=(unused diff_inversion.eval.pnp_lora diff_inversion.eval.masactrl_lora diff_inversion.eval.pix2pix_zero_lora)
CONFIGS=(unused eval/pnp_lora eval/masactrl_lora eval/pix2pix_zero_lora)
METHODS=(unused lora+pnp lora+masactrl lora+pix2pix-zero)
MODULE=${MODULES[${TASK_ID}]:-}
CONFIG=${CONFIGS[${TASK_ID}]:-}
METHOD=${METHODS[${TASK_ID}]:-}
[[ -n "${METHOD}" && "${TASK_ID}" != 0 ]] || { echo "Invalid task id: ${TASK_ID}" >&2; exit 2; }
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/endpoint-editor-smoke-${SLURM_JOB_ID}-${TASK_ID}
VENV_DIR=${JOB_TMP_DIR}/venv
SMOKE_DATA=${JOB_TMP_DIR}/pie_smoke
module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "${REPO_DIR}/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "${JOB_TMP_DIR}"; }
trap cleanup EXIT
mkdir -p "${SMOKE_DATA}"
ln -s "${REPO_DIR}/data/raw/PIE-Bench_v1/annotation_images" "${SMOKE_DATA}/annotation_images"
cd "${REPO_DIR}"
uv venv --python "$(command -v python3)" "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
uv sync --active --frozen --no-dev

python3 - "${REPO_DIR}/data/raw/PIE-Bench_v1/mapping_file.json" "${SMOKE_DATA}/mapping_file.json" <<'PY'
import json, sys
source, target = sys.argv[1:]
items = json.load(open(source))
key = next(iter(items))
json.dump({key: items[key]}, open(target, "w"))
PY

python3 -m "${MODULE}" --config-name "${CONFIG}" \
  "data_path=${SMOKE_DATA}" "output_path=${JOB_TMP_DIR}/output" \
  "edit_method_list=[${METHOD}]" lora=r16 lora.checkpoint_path="${CHECKPOINT}" \
  lora.mode=single +lora.inversion_mode=one_step_endpoint \
  cache_root="${JOB_CACHE_ROOT}/editing"
test -n "$(find "${JOB_TMP_DIR}/output" -type f -print -quit)"
echo "One-step endpoint ${METHOD} editor smoke passed."
