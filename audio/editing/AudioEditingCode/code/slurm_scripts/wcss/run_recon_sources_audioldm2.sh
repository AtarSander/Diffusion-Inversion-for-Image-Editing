#!/bin/bash -l
# ABOUTME: PWR/WCSS job scoring AudioLDM2 real-audio reconstruction on MusicCaps vs MedleyDB for
# ABOUTME: no-LoRA / trajectory-LoRA / realfn-LoRA. Set TRAJ_CKPT and REALFN_CKPT before submit.
#SBATCH --job-name=lorainv-aldm2recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=outputs/logs/slurm/aldm2recon-%j.out
#SBATCH --error=outputs/logs/slurm/aldm2recon-%j.err
#
# Submit from audio/ once training is done (checkpoints live under LORAINV_CHECKPOINT_ROOT):
#   TRAJ_CKPT=$LORAINV_CHECKPOINT_ROOT/attn_r8_a4_lr5e-5/checkpoint_final.pt \
#   REALFN_CKPT=$LORAINV_CHECKPOINT_ROOT/aldm2realfn_r8_a4_lr5e-5/checkpoint_step_6000.pt \
#   sbatch --account=$HPC_PWR_ACCOUNT --partition=$HPC_PWR_PARTITION \
#     editing/AudioEditingCode/code/slurm_scripts/wcss/run_recon_sources_audioldm2.sh

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode/code:${PYTHONPATH:-}"

: "${TRAJ_CKPT:?set TRAJ_CKPT to a trajectory-trained AudioLDM2 adapter checkpoint}"
: "${REALFN_CKPT:?set REALFN_CKPT to the aldm2realfn checkpoint}"
for c in "$TRAJ_CKPT" "$REALFN_CKPT"; do
  [ -f "$c" ] && [ -f "${c%.pt}.json" ] || { echo "missing checkpoint or sidecar: $c" >&2; exit 1; }
done

python src/inversion_lora/recon_two_sources_audioldm2.py \
  --traj_ckpt "$TRAJ_CKPT" --realfn_ckpt "$REALFN_CKPT" --count "${COUNT:-35}"
echo "done"
