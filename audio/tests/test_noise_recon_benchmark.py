# ABOUTME: Known-input checks for the noise-recon benchmark's sample table: deterministic captions
# ABOUTME: disjoint from LoRA training, and seed ranges that never let one arm reuse another's noise.

import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

AUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.noise_recon_benchmark import sample_table  # noqa: E402

CONFIG = OmegaConf.load(AUDIO_ROOT / "config/noise_recon_audioldm2.yaml")
TRAINING_CAPTIONS = AUDIO_ROOT / "editing/music_caps_dl/metadata/musiccaps-public.csv"


def test_table_has_n_samples_of_distinct_captions():
    table = sample_table(CONFIG)
    assert len(table) == CONFIG.num_prompts * CONFIG.seeds_per_prompt
    assert table["prompt"].nunique() == CONFIG.num_prompts
    assert (table.groupby("prompt").size() == CONFIG.seeds_per_prompt).all()


def test_table_is_deterministic_and_shared_by_both_models():
    other = OmegaConf.load(AUDIO_ROOT / "config/noise_recon_stable_audio.yaml")
    pd.testing.assert_frame_equal(sample_table(CONFIG), sample_table(other))


def test_captions_never_seen_in_lora_training():
    training = set(pd.read_csv(TRAINING_CAPTIONS)["caption"])
    assert not set(sample_table(CONFIG)["prompt"]) & training


def test_no_seed_is_shared_across_arms_or_samples():
    table = sample_table(CONFIG)
    seeds = pd.concat([table[c] for c in ("seed_gen", "seed_gaussian", "seed_ddpm")])
    assert seeds.is_unique
    # Training trajectories (42 + idx over MusicCaps), generated eval inputs, real-audio pairs.
    assert seeds.min() > 600000
