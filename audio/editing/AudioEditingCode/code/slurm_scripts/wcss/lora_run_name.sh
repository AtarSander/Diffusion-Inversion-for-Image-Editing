# ABOUTME: Derive the MedleyMD run directory name from an inversion-LoRA checkpoint path, shared
# ABOUTME: by the edit and eval jobs so they cannot disagree about where the run lives.

# Usage: lora_run_name /path/to/<config>/checkpoint_step_5000.pt
#     -> audioldm2_ddimlora_<config>_checkpoint_step_5000
lora_run_name() {
  local checkpoint="${1:?lora_run_name needs a checkpoint path}"
  echo "audioldm2_ddimlora_$(basename "$(dirname "$checkpoint")")_$(basename "$checkpoint" .pt)"
}
