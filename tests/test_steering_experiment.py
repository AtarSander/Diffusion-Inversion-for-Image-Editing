"""Complete extract/fit/generate flow with real tiny UNet forwards and local pixels."""

import json

import pytest
import torch
from PIL import Image
from test_steering_runtime import make_example

from diff_inversion.steering import experiment
from diff_inversion.steering.config import load_config, read_sources
from diff_inversion.steering.runtime import SteeringRuntime


def test_end_to_end_uses_common_noise_for_every_variant_and_prompt(tmp_path, monkeypatch):
    pipe, cond, latent = make_example()
    rows = []
    for index, label in enumerate(("positive", "positive", "negative", "negative")):
        Image.new("RGB", (2, 2), color=(index * 30, 0, 0)).save(tmp_path / f"{index}.png")
        rows.append({"id": str(index), "path": f"{index}.png", "label": label})
    sources = tmp_path / "sources.jsonl"
    sources.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = {
        "source_manifest": str(sources),
        "model": {"id": "tiny"},
        "steps": 4,
        "size": 16,
        "null_seed": 123,
        "reader": "free",
        "inversion_prompt": "A painting",
        "reader_prompt": "A painting",
        "generation": {
            "prompts": ["A house", "A cat"],
            "seeds": [42, 43],
            "layers": "all",
            "start_call": 1,
            "stop_call": 4,
            "representations": ["SA"],
            "controls": ["true", "swap", "gaussian"],
            "strengths": [0.0, 1.0],
            "guidance_scale": 1.0,
        },
    }
    inputs = []

    class RecordingRuntime(SteeringRuntime):
        def generate(self, noise, *args, **kwargs):
            inputs.append(noise.clone())
            return super().generate(noise, *args, **kwargs)

    monkeypatch.setattr(experiment, "SteeringRuntime", RecordingRuntime)
    monkeypatch.setattr(experiment, "load_pipeline", lambda _config: pipe)
    monkeypatch.setattr(experiment, "conditioning", lambda *_args: cond)
    monkeypatch.setattr(
        experiment, "encode_image", lambda _pipe, path, _size: latent + int(path[-5]) * 0.03
    )
    monkeypatch.setattr(experiment, "decode_image", lambda *_args: Image.new("RGB", (2, 2)))
    experiment.extract(config, tmp_path / "features")
    experiment.fit(tmp_path / "features/features.pt", tmp_path / "vectors")
    experiment.generate(config, tmp_path / "vectors/directions.pt", tmp_path / "generations")
    manifest = json.loads((tmp_path / "generations/manifest.json").read_text())
    variants_per_unit = 1 + 2 * 3 * 2
    assert len(manifest["images"]) == 4 * variants_per_unit
    assert manifest["active_calls"] == [1, 2, 3]
    for unit in range(4):
        values = inputs[unit * variants_per_unit : (unit + 1) * variants_per_unit]
        assert all(torch.equal(values[0], value) for value in values)
    assert torch.equal(inputs[0], inputs[variants_per_unit])  # same seed across prompts
    assert not torch.equal(inputs[0], inputs[2 * variants_per_unit])
    for seed in (42, 43):
        rows = [row for row in manifest["images"] if row["seed"] == seed]
        assert len({row["starting_noise_sha256"] for row in rows}) == 1
        assert {row["prompt"] for row in rows} == {"A house", "A cat"}
        assert all((tmp_path / "generations" / row["path"]).is_file() for row in rows)
    with pytest.raises(FileExistsError):
        experiment.generate(config, tmp_path / "vectors/directions.pt", tmp_path / "generations")


def test_source_paths_and_heldout_separation(tmp_path):
    path = tmp_path / "sources.jsonl"
    path.write_text(json.dumps({"id": "a", "path": "a.png", "label": "positive", "split": "test"}))
    with pytest.raises(ValueError, match="Held-out"):
        read_sources(path)


def test_example_config_resolves_checkpoint_and_shared_seed():
    config = load_config("config/steering/van_gogh.yaml")
    assert config["generation"]["seeds"] == [42]
    assert len(config["generation"]["prompts"]) == 4
    assert config["source_manifest"].endswith("/data/steering/van_gogh_sources.jsonl")
    assert config["lora"]["r"] == 16
    assert config["lora"]["checkpoint_path"].startswith("/")
