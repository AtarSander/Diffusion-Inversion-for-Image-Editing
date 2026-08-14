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
#SBATCH --array=0-6
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

# sbatch exports the submitting shell by default, and env.py's load_dotenv uses override=False,
# so a value left over from an earlier `source .env` would beat the current .env in both the
# shell and Python. Clear them first so .env (or env.py's defaults) actually decides.
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv_eval/bin/activate

# evals/utils.py imports `evals` as a top-level package, and eval_medley imports `editing`/`src`.
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# Ask env.py rather than re-deriving the path here: it is the single source of truth that
# eval_medley itself uses, so the two cannot disagree.
EDITS_ROOT="$(python -c 'import sys; sys.path.insert(0, "editing/AudioEditingCode/code"); import env; print(env.PATH_EDIT_OUTPUTS)')/medleymd"
echo "resolved PATH_EDIT_OUTPUTS -> ${EDITS_ROOT%/medleymd}"

# The Stable Audio driver appends dataset_name to a path that already contains it, so its runs
# live one level deeper than the AudioLDM2 ones. Kept as-is to avoid changing the output layout.
RUNS=(
  "$EDITS_ROOT/audioldm2_ddpm/audioldm2_ddpm_cfgsrc3.0_cfgtar12.0_t100_s200/audios"
  "$EDITS_ROOT/audioldm2_ddim/audioldm2_ddim_cfgsrc3.0_cfgtar12.0_t200_s200/audios"
  "$EDITS_ROOT/audioldm2_sdedit/audioldm2_sdedit_cfgtar12.0_t100_s200/audios"
  "$EDITS_ROOT/medleymd/stable_audio/stableaudio_ddpm_cfgsrc1.0_cfgtar3.5_t50_s100/audios"
  "$EDITS_ROOT/medleymd/stable_audio/stableaudio_ddim_cfgsrc1.0_cfgtar3.5_t100_s100/audios"
  "$EDITS_ROOT/medleymd/stable_audio/stableaudio_sdedit_cfgtar3.5_t50_s100/audios"
  "$EDITS_ROOT/audioldm2_ddim/audioldm2_ddim_cfgsrc1.0_cfgtar12.0_t200_s200/audios"
)

# Index 7: a LoRA-inversion run. Its directory name comes from the checkpoint, so it is only
# resolvable with LORA_PATH set -- hence not in the default --array range.
if [ -n "${LORA_PATH:-}" ]; then
  source "editing/AudioEditingCode/code/slurm_scripts/wcss/lora_run_name.sh" || exit 1
  RUNS+=("$EDITS_ROOT/audioldm2_ddim/$(lora_run_name "$LORA_PATH")/audios")
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array job}"
if [ "$TASK_ID" -ge "${#RUNS[@]}" ]; then
  echo "ERROR: task $TASK_ID but only ${#RUNS[@]} runs are defined." >&2
  echo "  Scoring a LoRA run (index 7) needs LORA_PATH=<checkpoint.pt> in the environment." >&2
  exit 2
fi
RUN_DIR="${RUNS[$TASK_ID]}"
echo "task=$TASK_ID node=$(hostname)"
echo "run_dir=$RUN_DIR"

n=$(find "$RUN_DIR" -name 'a*.wav' 2>/dev/null | wc -l)
echo "edits found: $n"
if [ "$n" -ne 696 ]; then
  echo "ERROR: expected 696 edits, found $n -- scoring an incomplete run would be misleading" >&2
  exit 1
fi

# FAD and mel PSNR/SSIM need the paired reference. Check it up front: it is the last thing
# eval_medley touches, so a missing reference otherwise wastes the whole LPAPS/CLAP/MuLan pass.
REF_DIR="$(python -c 'import sys; sys.path.insert(0, "editing/AudioEditingCode/code"); import env; print(env.PATH_LOWER_BOUND_MEDLEY)')"
ref_n=$(find "$REF_DIR" -name 'a*.wav' 2>/dev/null | wc -l)
echo "reference files: $ref_n  ($REF_DIR)"
if [ "$ref_n" -ne "$n" ]; then
  echo "ERROR: reference has $ref_n files but the run has $n. Build it first:" >&2
  echo "  python -m editing.build_lower_bound --splits full" >&2
  exit 1
fi

python -m editing.eval_medley --path_audio "$RUN_DIR"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAILED rc=$rc: $RUN_DIR" >&2
  exit "$rc"
fi
echo "done: $RUN_DIR"
