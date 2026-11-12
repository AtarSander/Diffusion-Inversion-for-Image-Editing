"""Single allocated-job entrypoint; requires a prepared steering Python environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    os.environ["SLURM_JOB_ID"]  # Never execute this workload on a login node.
    scratch = Path(os.environ["SLURM_TMPDIR"])
    repo = Path(__file__).resolve().parents[1]
    source = Path(os.environ["STEERING_SOURCE_MANIFEST"]).resolve()
    edits = Path(os.environ["STEERING_EDIT_MANIFEST"]).resolve()
    checkpoint = Path(os.environ["STEERING_CHECKPOINT"]).resolve()
    output = Path(os.environ["STEERING_OUTPUT_ROOT"]).resolve()
    # No downloads, environment installation, or overwriting existing runs.
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    preflight = json.loads((Path(os.environ["VISION_SD15_EVIDENCE_DIR"]) / "preflight.json").read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != preflight["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after preflight")
    source_rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    edit_rows = [json.loads(s) for s in edits.read_text().splitlines() if s.strip()]
    assert len(source_rows) == 128 and len({r["id"] for r in source_rows}) == 128
    assert all(sum(r["label"] == label for r in source_rows) == 64 for label in ("positive", "negative"))
    assert len(edit_rows) == 8 and len({r["id"] for r in edit_rows}) == 8
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    identity = {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "seed": 42, "null_seed": 123, "commands": commands,
                "model_id": preflight["model_id"], "model_revision": preflight["revision"],
                "preflight": preflight,
                "input_evidence": json.loads((Path(os.environ["VISION_SD15_EVIDENCE_DIR"]) / "input_evidence.json").read_text()),
                "config_sha256": {arm: hashlib.sha256((repo / f"config/steering/outline_reader_{arm}.json").read_bytes()).hexdigest() for arm in ("empty", "different")},
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "edit_sha256": hashlib.sha256(edits.read_bytes()).hexdigest(),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "checkpoint_assumption": "Complete SD1.5 attention-key/shape preflight passed; historical loss and steering identity unproven"}
    with tempfile.TemporaryDirectory(prefix="outline-reader-", dir=scratch) as temp:
        env = dict(os.environ)
        cache_vars = ["UV_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
                      "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "WANDB_DIR", "WANDB_CACHE_DIR",
                      "MPLCONFIGDIR", "XDG_CACHE_HOME", "TMPDIR", "PIP_CACHE_DIR"]
        for name in cache_vars:
            path = Path(temp) / name.lower()
            path.mkdir()
            env[name] = str(path)
        env["PYTHONPATH"] = str(repo)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        def run(module, *args):
            # Seed each stage identically; fit also uses its explicit null_seed.
            code = "import runpy,sys,random; import numpy as np,torch; random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42); m=sys.argv.pop(1); runpy.run_module(m,run_name='__main__')"
            cmd = [sys.executable, "-c", code, module, *map(str, args)]
            commands.append(cmd)
            (output / "provenance.json").write_text(json.dumps(identity, indent=2))
            subprocess.run(cmd, cwd=repo, env=env, check=True)
        for arm in ("empty", "different"):
            dest = output / arm
            dest.mkdir()
            config = json.loads((repo / f"config/steering/outline_reader_{arm}.json").read_text())
            config["source_manifest"] = str(source)
            config["lora"]["checkpoint_path"] = str(checkpoint)
            cfg = dest / "extract_config.json"
            cfg.write_text(json.dumps(config, indent=2))
            run("diff_inversion.steering", "extract", "--config", cfg, "--output", dest / "extract")
            run("diff_inversion.steering", "fit", "--features", dest / "extract/features.pt", "--output", dest / "fit")
            config["source_manifest"] = str(edits)
            config["edit"] = {"layers": "up_blocks.1,up_blocks.2,up_blocks.3", "start_call": 1, "stop_call": 50,
                "representations": ["SA"], "strengths": [-16., -8., -3., 3., 8., 16.],
                "controls": ["true"], "renorm": False}
            cfg = dest / "edit_config.json"
            cfg.write_text(json.dumps(config, indent=2))
            run("diff_inversion.steering", "edit", "--config", cfg, "--vectors", dest / "fit/directions.pt", "--output", dest / "edit")
            run("scripts.score_shape_size", "--run_dir", dest / "edit", "--threshold", 40.0)

    subprocess.run([sys.executable, str(repo / "scripts/diagnose_outline_readers.py"),
                    "--upstream", str(output.parent), "--output", str(output.parent / "comparison")], check=True)
    (output.parent / "SUCCESS.json").write_text(json.dumps({"status": "complete", "rows_per_arm":112,
        "paired_sources":8,"required_comparison":"comparison/diagnostics.json"},indent=2)+"\n")


if __name__ == "__main__":
    main()
