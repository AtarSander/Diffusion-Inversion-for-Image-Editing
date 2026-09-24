#!/bin/bash
#SBATCH --job-name=endpoint-p2p-smoke
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
REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
CHECKPOINT=${REPO_DIR}/checkpoints/sd15_inversion_lora_r16_uniform_e1/checkpoint_final.pt
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/endpoint-p2p-smoke-${SLURM_JOB_ID}
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

python3 -m diff_inversion.eval.p2p_lora --config-name eval/p2p_lora \
  "data_path=${SMOKE_DATA}" "output_path=${JOB_TMP_DIR}/output" \
  "edit_method_list=[lora+p2p]" lora=r16 lora.checkpoint_path="${CHECKPOINT}" \
  lora.mode=single +lora.inversion_mode=one_step_endpoint \
  cache_root="${JOB_CACHE_ROOT}/editing"
test -n "$(find "${JOB_TMP_DIR}/output" -type f -print -quit)"
echo "One-step endpoint P2P editor smoke passed."
