# ABOUTME: Checks the plotter's run-name patterns against the names the sweep scripts actually
# ABOUTME: build, so a grid can never land on disk that the figure silently skips.

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "editing"))

from plot_lora_curves import MODELS  # noqa: E402

# (model, run directory name, expected fields). Names copied from the dry runs of
# sao_nfe_matched_configs.sh and sao_lora_sweep_configs.sh.
CASES = [
    ("stable_audio_nfe", "stableaudio_odeinv_nolora_hparam_nfe300_t99_s396_cfgtar3.5",
     {"mode": None, "tstart": "99", "steps": "396", "cfg_tar": "3.5", "nfe": "300"}),
    ("stable_audio_nfe", "stableaudio_ddpm_nolora_hparam_nfe300_t30_s120_cfgtar3.5",
     {"mode": "ddpm", "tstart": "30", "steps": "120", "cfg_tar": "3.5", "nfe": "300"}),
    ("stable_audio_nfe", "stableaudio_sdedit_nolora_hparam_nfe300_t150_s600_cfgtar3.5",
     {"mode": "sdedit", "tstart": "150", "steps": "600", "cfg_tar": "3.5", "nfe": "300"}),
    ("stable_audio", "stableaudio_odeinv_nolora_hparam_cfgtar3.5_t25_s100",
     {"mode": None, "tstart": "25", "steps": "100", "cfg_tar": "3.5"}),
    ("stable_audio", "stableaudio_ddpm_hparam_cfgtar7.0_t50_s100",
     {"mode": "ddpm", "tstart": "50", "steps": "100", "cfg_tar": "7.0"}),
]


@pytest.mark.parametrize("model,name,want", CASES)
def test_base_pattern_parses_sweep_names(model, name, want):
    """Every no-LoRA name the sweep scripts emit must parse, with the right fields."""
    match = MODELS[model]["base"].match(name)
    assert match is not None, f"{model} base pattern does not match {name}"
    for key, value in want.items():
        assert match.groupdict()[key] == value, (key, match.groupdict()[key], value)


@pytest.mark.parametrize("model,name,checkpoint,steps", [
    ("stable_audio_nfe",
     "stableaudio_odeinvlora_saocos_r8_a4_lr5e-5_checkpoint_step_4000_hparam"
     "_nfe300_t99_s396_cfgtar3.5",
     "saocos_r8_a4_lr5e-5_checkpoint_step_4000", "396"),
    ("stable_audio",
     "stableaudio_odeinvlora_hparam_saocos_r8_a4_lr5e-5_checkpoint_step_4000_ema"
     "_cfgtar3.5_t25_s100",
     "saocos_r8_a4_lr5e-5_checkpoint_step_4000_ema", "100"),
])
def test_lora_pattern_recovers_checkpoint(model, name, checkpoint, steps):
    """The LoRA pattern must recover the checkpoint label, which becomes the legend entry."""
    match = MODELS[model]["lora"].match(name)
    assert match is not None, f"{model} lora pattern does not match {name}"
    assert match.groupdict()["checkpoint"] == checkpoint
    assert match.groupdict()["steps"] == steps


def test_nfe_names_do_not_collide_with_the_hparam_model():
    """An nfe run must not be parsed by the plain hparam patterns, or it lands in the wrong figure."""
    name = "stableaudio_odeinv_nolora_hparam_nfe300_t99_s396_cfgtar3.5"
    spec = MODELS["stable_audio"]
    assert spec["base"].match(name) is None
    assert spec["lora"].match(name) is None
