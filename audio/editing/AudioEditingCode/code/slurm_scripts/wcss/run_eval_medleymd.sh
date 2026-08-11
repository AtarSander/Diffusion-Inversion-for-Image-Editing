#!/bin/bash -l
# ABOUTME: PWR/WCSS array job scoring the six MedleyMD baseline runs with LPAPS, CLAP,
# ABOUTME: MuQ-MuLan and FAD + mel PSNR/SSIM. One array task per edit run.
#SBATCH --job-name=medleymd-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-5
#SBATCH --output=outputs/logs/slurm/eval-%A_%a.out
#SBATCH --error=outputs/logs/slurm/eval-%A_%a.err
#
# Submit from audio/ once the baseline array has finished:
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_eval.sh
#
# Prerequisites:
#   python -m venv .venv_eval && source .venv_eval/bin/activate
#   pip install -r requirements_eval.txt
#   pip install --no-deps ssr_eval==0.0.7 \
#     'audioldm_eval @ https://github.com/haoheliu/audioldm_eval/archive/8dc07ee7c42f9dc6e295460a1034175a0d49b436.tar.gz'
#   mkdir -p res/clap/pretrained && curl -L -o res/clap/pretrained/music_audioset_epoch_15_esc_90.14.pt \
#     https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
#   python -m editing.build_lower_bound --splits full

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv_eval/bin/activate

# evals/utils.py imports `evals` as a top-level package, and eval_medley imports `editing`/`src`.
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

EDITS_ROOT="${EDIT_OUTPUTS_DIR:-outputs/edits}/medleymd"

# The Stable Audio driver appends dataset_name to a path that already contains it, so its runs
# live one level deeper than the AudioLDM2 ones. Kept as-is to avoid changing the output layout.
RUNS=(
  "$EDITS_ROOT/audioldm2_ddpm/audioldm2_ddpm_cfgsrc3.0_cfgtar12.0_t100_s200/audios"
  "$EDITS_ROOT/audioldm2_ddim/audioldm2_ddim_cfgsrc3.0_cfgtar12.0_t200_s200/audios"
  "$EDITS_ROOT/audioldm2_sdedit/audioldm2_sdedit_cfgtar12.0_t100_s200/audios"
  "$EDITS_ROOT/medleymd/stable_audio/stableaudio_ddpm_cfgsrc1.0_cfgtar3.5_t50_s100/audios"
  "$EDITS_ROOT/medleymd/stable_audio/stableaudio_ddim_cfgsrc1.0_cfgtar3.5_t100_s100/audios"
  "$EDITS_ROOT/medleymd/stable_audio/stableaudio_sdedit_cfgtar3.5_t50_s100/audios"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array job}"
RUN_DIR="${RUNS[$TASK_ID]}"
echo "task=$TASK_ID node=$(hostname)"
echo "run_dir=$RUN_DIR"

n=$(find "$RUN_DIR" -name 'a*.wav' 2>/dev/null | wc -l)
echo "edits found: $n"
if [ "$n" -ne 696 ]; then
  echo "ERROR: expected 696 edits, found $n -- scoring an incomplete run would be misleading" >&2
  exit 1
fi

python -m editing.eval_medley --path_audio "$RUN_DIR"
echo "done: $RUN_DIR"
