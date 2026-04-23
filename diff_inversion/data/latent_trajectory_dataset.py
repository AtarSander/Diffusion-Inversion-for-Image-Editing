from pathlib import Path
import json

from torch.utils.data import Dataset
from einops import reduce
import torch


class LatentTrajectoryDataset(Dataset):
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.items = []

        sample_dirs = sorted(self.root_dir.glob("sample_*"))

        for sample_dir in sample_dirs:
            latents_dir = sample_dir / "latents"
            pred_noises_dir = sample_dir / "pred_noises"
            if not latents_dir.exists():
                continue

            latent_paths = sorted(latents_dir.glob("x_*.pt"))
            pred_noises_paths = sorted(pred_noises_dir.glob("noise_*.pt"))
            if len(latent_paths) < 2:
                continue

            timesteps_path = sample_dir / "timesteps.json"
            prompt_path = sample_dir / "prompt.json"
            meta_path = sample_dir / "meta.json"

            timesteps = None
            if timesteps_path.exists():
                with timesteps_path.open("r", encoding="utf-8") as f:
                    timesteps = json.load(f)

            prompt_record = {}
            if prompt_path.exists():
                with prompt_path.open("r", encoding="utf-8") as f:
                    prompt_record = json.load(f)

            meta = {}
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)

            for step_idx in range(len(latent_paths) - 1):
                self.items.append(
                    {
                        "x_t_path": latent_paths[step_idx + 1],
                        "target_eps_path": pred_noises_paths[step_idx],
                        "timestep": timesteps[step_idx] if timesteps else None,
                        "prev_timestep": timesteps[step_idx + 1] if timesteps else None,
                        "prompt": prompt_record.get("prompt", ""),
                        "sample_idx": meta.get("sample_idx"),
                        "step_idx": step_idx,
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        x_t = torch.load(item["x_t_path"], map_location="cpu")
        target_eps = torch.load(item["target_eps_path"], map_location="cpu")

        return {
            "x_t": reduce(x_t, "1 c h w -> c h w", "mean"),
            "target_eps": reduce(target_eps, "1 c h w -> c h w", "mean"),
            "timestep": torch.tensor(
                -1 if item["timestep"] is None else item["timestep"],
                dtype=torch.long,
            ),
            "prev_timestep": torch.tensor(
                -1 if item["prev_timestep"] is None else item["prev_timestep"],
                dtype=torch.long,
            ),
            "prompt": item["prompt"],
            "sample_idx": item["sample_idx"],
            "step_idx": item["step_idx"],
        }
