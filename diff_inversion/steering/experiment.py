"""Three explicit stages: collect source features, fit directions, compare generations."""

import hashlib
import itertools
from pathlib import Path

import torch

from .config import read_sources, write_json
from .directions import fit_directions, scaled_offsets, select_layers
from .hooks import REPRESENTATIONS
from .images import conditioning, decode_image, encode_image, load_pipeline
from .runtime import SteeringRuntime

FEATURE_SOURCE = "inversion_states"


def new_directory(path):
    """Use a new destination so an experiment cannot silently overwrite old outputs."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def extract(config, output):
    sources = read_sources(config["source_manifest"])
    output = new_directory(output)
    write_json(output / "config.json", config)
    write_json(output / "sources.json", sources)
    pipe = load_pipeline(config)
    runtime = SteeringRuntime(pipe, steps=config["steps"])
    inverse_cond = conditioning(pipe, config["inversion_prompt"], config["size"])
    reader_cond = conditioning(pipe, config["reader_prompt"], config["size"])
    inversions = output / "inversions"
    inversions.mkdir()
    records = []
    for index, source in enumerate(sources):
        latent = encode_image(pipe, source["path"], config["size"])
        record = dict(source)
        for method in ("DDIM", "MW"):
            states = runtime.invert(latent, inverse_cond, method)
            torch.save(
                {
                    "source_id": source["id"],
                    "method": method,
                    "state_timesteps": runtime.state_timesteps,
                    "states": states,
                },
                inversions / f"source_{index:03d}_{method}.pt",
            )
            record[method] = runtime.read_inversion_states(states, reader_cond)
        records.append(record)
        print(f"Read {index + 1}/{len(sources)}: {source['id']}", flush=True)
    torch.save(
        {"config": config, "feature_source": FEATURE_SOURCE, "records": records},
        output / "features.pt",
    )


def fit(feature_path, output):
    features = torch.load(feature_path, map_location="cpu", weights_only=True)
    if features.get("feature_source") != FEATURE_SOURCE:
        raise ValueError("Features must come from saved inversion states; rerun extract")
    output = new_directory(output)
    config = features["config"]
    bundle = fit_directions(features["records"], null_seed=config["null_seed"])
    # Dict insertion order comes from observed CA execution, not named_modules().
    bundle["layer_order"] = list(features["records"][0]["DDIM"]["features"]["CA"])
    bundle["config"] = config
    bundle["feature_source"] = FEATURE_SOURCE
    bundle["sources"] = [
        {key: row[key] for key in ("id", "path", "label")} for row in features["records"]
    ]
    torch.save(bundle, output / "directions.pt")
    write_json(output / "config.json", config)
    write_json(output / "layers.json", bundle["layer_order"])


def generate(config, vector_path, output):
    bundle = torch.load(vector_path, map_location="cpu", weights_only=True)
    if bundle.get("feature_source") != FEATURE_SOURCE:
        raise ValueError("Directions must come from saved inversion states; rerun extract and fit")
    # Compare readable values, without hashing code/configs or binding to an HPC installation.
    for key in ("model", "lora", "steps", "size", "inversion_prompt", "reader_prompt"):
        if bundle["config"].get(key) != config.get(key):
            raise ValueError(f"Direction source and generation disagree at config.{key}")
    settings = config["generation"]
    layers = select_layers(bundle["layer_order"], settings["layers"])
    calls = list(range(settings["start_call"], settings["stop_call"]))
    if not calls or min(calls) < 0 or max(calls) >= config["steps"]:
        raise ValueError("Use 0-based denoising calls in [0, steps), with exclusive stop_call")
    if set(settings["representations"]) - set(REPRESENTATIONS):
        raise ValueError("Representations must be CA, SA or block")
    if set(settings["controls"]) - {"true", "swap", "gaussian"}:
        raise ValueError("Controls must be true, swap or gaussian")
    if len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("Generation repetitions must have distinct seeds")
    output = new_directory(output)
    write_json(output / "config.json", config)
    pipe = load_pipeline(config)
    runtime = SteeringRuntime(pipe, steps=config["steps"])
    comparisons = []
    manifest = {
        "config": config,
        "direction_config": bundle["config"],
        "feature_source": FEATURE_SOURCE,
        "direction_sources": bundle["sources"],
        "selected_layers": layers,
        "active_calls": calls,
        "images": comparisons,
    }
    for seed, (prompt_index, prompt) in itertools.product(
        settings["seeds"], enumerate(settings["prompts"])
    ):
        unit = output / f"seed_{seed}" / f"prompt_{prompt_index:02d}"
        unit.mkdir(parents=True)
        cond = conditioning(pipe, prompt, config["size"])
        noise = torch.randn(
            (1, 4, config["size"] // 8, config["size"] // 8),
            generator=torch.Generator().manual_seed(seed),
            dtype=pipe.unet.dtype,
        )
        torch.save(noise, unit / "starting_noise.pt")
        # The one hash requested for paired comparisons is the actual starting noise.
        noise_hash = hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest()

        def save(latent, filename, **variant):
            target = unit / filename
            decode_image(pipe, latent).save(target)
            comparisons.append(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "starting_noise_sha256": noise_hash,
                    "path": str(target.relative_to(output)),
                    **variant,
                }
            )
            write_json(output / "manifest.json", manifest)

        baseline = runtime.generate(noise, cond, guidance_scale=settings["guidance_scale"])
        save(baseline, "base.png", method="base", strength=0.0)
        variants = itertools.product(
            settings["representations"],
            ("DDIM", "MW"),
            settings["controls"],
            settings["strengths"],
        )
        for rep, method, control, strength in variants:
            offsets = scaled_offsets(bundle, rep, method, control, layers, calls, strength)
            latent = runtime.generate(
                noise,
                cond,
                guidance_scale=settings["guidance_scale"],
                representation=rep,
                offsets=offsets,
            )
            save(
                latent,
                f"{rep}_{method}_{control}_{strength:g}.png",
                representation=rep,
                method=method,
                control=control,
                strength=strength,
            )
        print(f"Generated prompt {prompt_index}, seed {seed}", flush=True)
