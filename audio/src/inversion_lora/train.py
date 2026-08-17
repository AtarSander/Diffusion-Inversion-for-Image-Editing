# ABOUTME: Train a LoRA on AudioLDM2 so that DDIM inversion becomes near-exact, by distilling
# ABOUTME: the frozen teacher's epsilon at x_t into the student's prediction at the cleaner x_t-1.

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

AUDIO_ROOT = Path(__file__).resolve().parents[2]
for path in (AUDIO_ROOT, AUDIO_ROOT / "editing/AudioEditingCode/code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models import load_model  # noqa: E402

from src.inversion_lora.dataset import (  # noqa: E402
    AudioLDM2TrajectoryDataset,
    collate_trajectory_batch,
    split_sample_ids,
    transitions_below_timestep,
)
from src.inversion_lora.noise_metrics import noise_report  # noqa: E402
from src.inversion_lora.reconstruct import (  # noqa: E402
    generate_eval_latents,
    held_out_prompts,
    load_real_latents,
    real_audio_fixtures,
    reconstruction_metrics,
)

FROZEN_COMPONENTS = (
    "vae",
    "text_encoder",
    "text_encoder_2",
    "language_model",
    "projection_model",
    "vocoder",
)

def timestep_band(timesteps: torch.Tensor, band_top: int, num_bands: int) -> torch.Tensor:
    """Bucket timesteps into equal bands of `[0, band_top]`, noisiest first.

    Band 0 is `(band_top - width, band_top]` and the last band ends at t=0. Boundaries are
    half-open on the noisy side, so a timestep landing exactly on one belongs to the cleaner
    band. With a uniformly spaced DDIM grid this is the same as bucketing by step index, so
    "the first 25% of steps" reads the same either way.

    Args:
        timesteps: Integer timesteps `[B]`.
        band_top: Noisiest timestep the training set contains, i.e. the top of band 0.
        num_bands: Number of equal bands to split `[0, band_top]` into.

    Returns:
        Band index per element, in `[0, num_bands - 1]`.
    """
    assert timesteps.ndim == 1, timesteps.shape
    index = ((band_top - timesteps.double()) * num_bands / band_top).floor().long()
    assert int(index.min()) >= 0, (
        f"timestep {int(timesteps.max())} is above band_top={band_top}: the loss bands do not "
        "cover the training data, so train_max_timestep and num_loss_bands disagree"
    )
    return index.clamp(max=num_bands - 1)


def band_labels(band_top: int, num_bands: int, num_train_timesteps: int) -> list[str]:
    """Name each band by the share of the full schedule it covers, noisiest first.

    Four bands over the whole schedule give `q100_75`..`q25_00`; five over `t <= 250` give
    `q25_20`..`q05_00`.

    Args:
        band_top: Top of band 0, in timesteps.
        num_bands: Number of equal bands.
        num_train_timesteps: The scheduler's training horizon, e.g. 1000.

    Returns:
        One label per band.
    """
    width = band_top / num_bands
    labels = [
        f"q{100 * (band_top - i * width) / num_train_timesteps:02.0f}"
        f"_{100 * (band_top - (i + 1) * width) / num_train_timesteps:02.0f}"
        for i in range(num_bands)
    ]
    assert len(set(labels)) == num_bands, f"bands round to duplicate names: {labels}"
    return labels


class LoRAEMA:
    """Exponential moving average of the LoRA parameters.

    Only the adapter is averaged, which is 7.7M parameters at rank 8, so the shadow copy costs
    about 30 MB and the per-step update is negligible against a UNet forward.
    """

    def __init__(self, named_parameters: dict[str, torch.Tensor], decay: float):
        """Seed the shadow weights from the current parameters.

        Args:
            named_parameters: The trainable adapter parameters, by name.
            decay: Averaging decay; the shadow tracks roughly the last `1 / (1 - decay)` steps.
        """
        assert 0.0 <= decay < 1.0, f"decay must be in [0, 1), got {decay}"
        self.decay = decay
        self.shadow = {
            name: param.detach().clone().float() for name, param in named_parameters.items()
        }

    @torch.no_grad()
    def update(self, named_parameters: dict[str, torch.Tensor]) -> None:
        """Pull the shadow weights towards the live parameters by `1 - decay`."""
        for name, param in named_parameters.items():
            self.shadow[name].lerp_(param.detach().float(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        assert set(state) == set(self.shadow), "EMA state does not match the current adapter"
        self.shadow = {name: value.detach().clone().float() for name, value in state.items()}


def git_sha() -> str:
    """Return the current commit SHA so checkpoints are traceable to their code."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
    ).strip()


class AudioLDM2InversionTrainer:
    """Distils the frozen teacher's shifted epsilon into a LoRA on the AudioLDM2 UNet.

    The targets are precomputed, so the teacher is never run here; the LoRA stays enabled for
    the whole of training and the frozen weights receive no gradient.
    """

    def __init__(self, ldm, cfg: DictConfig, tracker: Any):
        self.ldm = ldm
        self.cfg = cfg
        self.tracker = tracker
        self.unet = ldm.model.unet
        self.device = ldm.device
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.global_step = 0
        self.baseline_reference = float("nan")
        self.num_train_timesteps = int(ldm.model.scheduler.config.num_train_timesteps)
        self.band_top = int(cfg.train_max_timestep or self.num_train_timesteps)
        self.band_labels = band_labels(
            self.band_top, int(cfg.num_loss_bands), self.num_train_timesteps
        )
        self._band_sums = torch.zeros(len(self.band_labels), dtype=torch.float64)
        self._band_counts = torch.zeros(len(self.band_labels), dtype=torch.float64)
        self.eval_fixtures: dict[str, Any] | None = None

        self._freeze_components()
        lora_cfg = OmegaConf.to_container(cfg.lora, resolve=True)
        preset = cfg.get("lora_preset")
        if preset is not None:
            presets = OmegaConf.to_container(cfg.lora_target_presets, resolve=True)
            if preset not in presets:
                raise KeyError(f"lora_preset={preset!r} not in {sorted(presets)}")
            lora_cfg["target_modules"] = presets[preset]
            logger.info("LoRA preset {}: {}", preset, lora_cfg["target_modules"])
        inject_adapter_in_model(
            LoraConfig(**lora_cfg),
            self.unet,
            adapter_name=str(cfg.adapter_name),
        )
        for name, param in self.unet.named_parameters():
            param.requires_grad_("lora" in name.lower())
        for param in self.unet.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

        self.trainable_parameters = [p for p in self.unet.parameters() if p.requires_grad]
        if not self.trainable_parameters:
            raise RuntimeError("No LoRA parameters were marked trainable.")
        logger.info(
            "Trainable LoRA parameters: {:,} across {} tensors",
            sum(p.numel() for p in self.trainable_parameters),
            len(self.trainable_parameters),
        )

        self.lora_config_used = lora_cfg
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )

        self.lora_named_parameters = {
            name: param for name, param in self.unet.named_parameters() if param.requires_grad
        }
        self.ema: LoRAEMA | None = None
        if cfg.get("ema_decay") is not None:
            self.ema = LoRAEMA(self.lora_named_parameters, float(cfg.ema_decay))
            logger.info(
                "EMA enabled: decay={} (averages roughly the last {:.0f} steps)",
                float(cfg.ema_decay),
                1.0 / (1.0 - float(cfg.ema_decay)),
            )

    def _freeze_components(self) -> None:
        for name in FROZEN_COMPONENTS:
            component = getattr(self.ldm.model, name, None)
            if component is not None and hasattr(component, "requires_grad_"):
                component.requires_grad_(False)
                component.eval()

    def predict_noise(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run the student UNet on the cleaner latent with the cached conditioning."""
        x_clean = batch["x_clean"].to(device=self.device, dtype=self.unet.dtype)
        timestep = batch["timestep"].to(device=self.device)
        hidden = batch["generated_prompt_embeds"].to(device=self.device, dtype=self.unet.dtype)
        t5_embeds = batch["t5_prompt_embeds"].to(device=self.device, dtype=self.unet.dtype)
        t5_mask = batch["t5_attention_mask"].to(device=self.device)

        assert x_clean.shape[0] == timestep.shape[0] == hidden.shape[0], (
            x_clean.shape,
            timestep.shape,
            hidden.shape,
        )

        model_input = self.ldm.model.scheduler.scale_model_input(x_clean, timestep)
        return self.ldm.unet_forward(
            model_input,
            timestep=timestep,
            encoder_hidden_states=hidden,
            class_labels=t5_embeds,
            encoder_attention_mask=t5_mask,
        )[0].sample

    def forward_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """No-CFG loss: MSE between the student's epsilon and the cached teacher epsilon."""
        return self.per_example_loss(batch).mean()

    def per_example_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Same loss as `forward_loss` but kept per batch element, for the quartile breakdown."""
        student_eps = self.predict_noise(batch)
        target_eps = batch["target_eps"].to(device=self.device, dtype=self.unet.dtype)
        assert student_eps.shape == target_eps.shape, (student_eps.shape, target_eps.shape)
        squared = (student_eps.float() - target_eps.float()) ** 2
        return squared.flatten(1).mean(dim=1)

    @torch.no_grad()
    def baseline_loss(self, val_loader: DataLoader, max_batches: int | None) -> float:
        """Loss with the LoRA disabled: the shift gap the adapter has to close.

        Measured over the deterministic validation loader rather than one training batch. The
        per-transition loss is heavy-tailed -- a handful of steps near t=0 carry most of the mass
        -- so a single batch of 32 has a standard deviation about half its own mean, and the
        number is not even comparable between runs: injecting a rank-r adapter consumes RNG in
        proportion to r, which shifts the shuffled loader and hands each rank a different first
        batch. The teacher is identical in every run, so this must be one number.

        Args:
            val_loader: Validation loader, unshuffled.
            max_batches: Cap on batches, or None for all of them.

        Returns:
            Mean LoRA-disabled loss.
        """
        self.set_lora_enabled(False)
        try:
            return self.validate(val_loader, max_batches)["val/loss"]
        finally:
            self.set_lora_enabled(True)

    def record_bands(self, losses: torch.Tensor, timesteps: torch.Tensor) -> None:
        """Accumulate per-example losses into their noise-schedule bands."""
        assert losses.shape == timesteps.shape, (losses.shape, timesteps.shape)
        bands = timestep_band(timesteps.cpu(), self.band_top, len(self.band_labels))
        values = losses.detach().double().cpu()
        self._band_sums.scatter_add_(0, bands, values)
        self._band_counts.scatter_add_(0, bands, torch.ones_like(values))

    def pop_band_losses(self) -> dict[str, float]:
        """Mean loss per band since the last call, then reset the accumulator.

        Bands with no examples in the window are omitted rather than reported as zero.
        """
        out = {}
        for index, label in enumerate(self.band_labels):
            count = float(self._band_counts[index])
            if count > 0:
                out[f"train/loss_{label}"] = float(self._band_sums[index]) / count
        self._band_sums.zero_()
        self._band_counts.zero_()
        return out

    @contextmanager
    def scheduler_steps(self, num_inference_steps: int):
        """Temporarily re-grid the DDIM scheduler, then restore the training grid.

        Training never reads `scheduler.timesteps` -- it uses the timestep cached with each
        transition, and `scale_model_input` is a no-op for DDIM -- so this only affects the
        reconstruction eval. It is restored regardless, so the two cannot drift apart.
        """
        original = int(self.cfg.num_inference_steps)
        if num_inference_steps == original:
            yield
            return
        self.ldm.model.scheduler.set_timesteps(num_inference_steps, device=self.device)
        try:
            yield
        finally:
            self.ldm.model.scheduler.set_timesteps(original, device=self.device)

    @contextmanager
    def ema_weights(self):
        """Temporarily install the EMA weights into the adapter, then restore the live ones.

        Swapping in place keeps everything downstream -- checkpoint format, eval, the adapter
        toggle -- identical between the raw and averaged weights, with no second copy of the
        model. A no-op when EMA is disabled.
        """
        if self.ema is None:
            yield
            return
        backup = {
            name: param.detach().clone() for name, param in self.lora_named_parameters.items()
        }
        try:
            with torch.no_grad():
                for name, param in self.lora_named_parameters.items():
                    param.copy_(self.ema.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, param in self.lora_named_parameters.items():
                    param.copy_(backup[name])

    def set_lora_enabled(self, enabled: bool) -> None:
        """Toggle every injected adapter; disabling restores the frozen teacher exactly."""
        toggled = 0
        for module in self.unet.modules():
            if module is not self.unet and hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled)
                toggled += 1
        if not toggled:
            raise AttributeError("No injected LoRA layers exposed enable_adapters().")

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, max_batches: int | None) -> dict[str, float]:
        """Average validation loss over up to `max_batches` batches."""
        self.unet.eval()
        losses: list[float] = []
        for idx, batch in enumerate(tqdm(val_loader, desc="validation", leave=False)):
            if max_batches is not None and idx >= max_batches:
                break
            loss = self.forward_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss at step {self.global_step}")
            losses.append(float(loss))
        self.unet.train()
        return {"val/loss": sum(losses) / len(losses)} if losses else {}

    def prepare_reconstruction_fixtures(
        self, prompts: list[str], audio_paths: list[Path], audio_prompts: list[str]
    ) -> None:
        """Build the fixed eval set once, and measure plain DDIM inversion on it as the baseline.

        The fixtures are frozen for the whole run so the reconstruction curves are comparable
        across steps. The no-LoRA baseline is measured once here rather than at every eval: the
        teacher never changes, so re-measuring it would burn the same compute for the same number.
        """
        self.set_lora_enabled(False)
        try:
            with self.scheduler_steps(int(self.cfg.recon_num_inference_steps)):
                generated, initial_noise = generate_eval_latents(
                    self.ldm,
                    prompts,
                    seed=int(self.cfg.seed),
                    batch_size=int(self.cfg.recon_batch_size),
                    duration_s=float(self.cfg.recon_duration_s),
                )
                real, real_names = load_real_latents(
                    self.ldm,
                    audio_paths,
                    seed=int(self.cfg.seed),
                    duration_s=float(self.cfg.recon_duration_s),
                )
        finally:
            self.set_lora_enabled(True)

        assert generated.shape[1:] == real.shape[1:], (generated.shape, real.shape)
        logger.info(
            "Reconstruction fixtures: {} generated {} real, latent {}; first real crop {}",
            generated.shape[0],
            real.shape[0],
            tuple(generated.shape[1:]),
            real_names[0],
        )

        self.eval_fixtures = {
            "generated": generated,
            "generated_noise": initial_noise,
            "generated_prompts": prompts,
            "real": real,
            "real_prompts": audio_prompts,
            "real_names": real_names,
        }

        # Logged under the same `eval/` names at step 0, so W&B shows one continuous curve whose
        # first point is plain DDIM. A separate eval_no_lora/* namespace made the reference
        # invisible on the panel that mattered.
        baseline = self._reconstruction_pass(set_lora_enabled=None, prefix="eval")
        logger.info("Plain DDIM inversion baseline (logged as step 0): {}", baseline)
        self.tracker.log(baseline, step=0)
        self.baseline_reconstruction = baseline

    def _reconstruction_pass(
        self, set_lora_enabled: Any, prefix: str
    ) -> dict[str, float]:
        """Score reconstruction and noise statistics on both fixture sets."""
        assert self.eval_fixtures is not None, "call prepare_reconstruction_fixtures first"
        fixtures = self.eval_fixtures
        metrics: dict[str, float] = {}

        with self.scheduler_steps(int(self.cfg.recon_num_inference_steps)):
            for kind in ("generated", "real"):
                scores, inverted = reconstruction_metrics(
                    self.ldm,
                    fixtures[kind],
                    fixtures[f"{kind}_prompts"],
                    batch_size=int(self.cfg.recon_batch_size),
                    set_lora_enabled=set_lora_enabled,
                    progress=lambda message: logger.debug(message),
                )
                metrics.update(
                    {f"{prefix}/{kind}/{key}": value for key, value in scores.items()}
                )
                # Generated samples have a true initial noise to compare against; real audio has
                # none, so a fresh standard normal draw is the only available reference.
                gaussian = torch.randn(
                    inverted.shape, generator=torch.Generator().manual_seed(int(self.cfg.seed))
                )
                reference = fixtures["generated_noise"] if kind == "generated" else gaussian
                metrics.update(
                    noise_report(
                        inverted,
                        reference,
                        prefix=f"{prefix}/{kind}/noise",
                        reference_is_ground_truth=kind == "generated",
                        seed=int(self.cfg.seed),
                    )
                )
                # The same statistics on the clean latent being inverted. It is structured
                # audio, so it should look markedly non-Gaussian: the control that shows these
                # metrics discriminate, rather than reading at the floor for everything.
                metrics.update(
                    noise_report(
                        fixtures[kind],
                        gaussian,
                        prefix=f"{prefix}/{kind}/latent",
                        reference_is_ground_truth=False,
                        seed=int(self.cfg.seed),
                    )
                )
        return metrics

    def reconstruction_eval(self) -> dict[str, float]:
        """Reconstruction and noise metrics for the current adapter."""
        self.unet.eval()
        try:
            return self._reconstruction_pass(
                set_lora_enabled=self.set_lora_enabled, prefix="eval"
            )
        finally:
            self.unet.train()

    def save_checkpoint(self, filename: str, save_training_state: bool = False) -> Path:
        """Save the LoRA adapter, plus optimizer state when resuming matters."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_state = get_peft_model_state_dict(
            self.unet, adapter_name=str(self.cfg.adapter_name)
        )
        path = self.checkpoint_dir / filename
        torch.save(adapter_state, path)

        meta = {
            "global_step": self.global_step,
            "git_sha": git_sha(),
            "model_id": str(self.cfg.model_id),
            "num_inference_steps": int(self.cfg.num_inference_steps),
            "lora": self.lora_config_used,
            "adapter_name": str(self.cfg.adapter_name),
        }
        with path.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self.ema is not None:
            # Written through the same code path as the raw weights, so the EMA checkpoint is
            # loadable by anything that loads a normal one.
            with self.ema_weights():
                # get_peft_model_state_dict returns views onto the live parameters, so this must
                # be cloned before the context restores them, or the file holds the raw weights.
                ema_state = {
                    key: value.detach().clone()
                    for key, value in get_peft_model_state_dict(
                        self.unet, adapter_name=str(self.cfg.adapter_name)
                    ).items()
                }
            ema_path = path.with_name(f"{path.stem}_ema{path.suffix}")
            torch.save(ema_state, ema_path)
            with ema_path.with_suffix(".json").open("w", encoding="utf-8") as f:
                json.dump({**meta, "ema_decay": float(self.cfg.ema_decay)}, f, indent=2)

        if save_training_state:
            state = {
                "global_step": self.global_step,
                "lora_state_dict": adapter_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            if self.ema is not None:
                state["ema_state_dict"] = self.ema.state_dict()
            torch.save(state, self.checkpoint_dir / f"training_state_{filename}")
        logger.info("Saved checkpoint {}", path)
        return path

    def load_training_state(self, path: str | Path) -> int:
        """Restore adapter and optimizer state, returning the step to resume from."""
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        set_peft_model_state_dict(
            self.unet, state["lora_state_dict"], adapter_name=str(self.cfg.adapter_name)
        )
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.ema is not None:
            if "ema_state_dict" not in state:
                raise KeyError(
                    f"{path} has no EMA state but ema_decay is set. Resuming would restart the "
                    "average from the current weights and silently misreport eval/ema/*."
                )
            self.ema.load_state_dict(state["ema_state_dict"])
        return int(state["global_step"])

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        max_train_steps: int,
        initial_global_step: int = 0,
    ) -> None:
        """Run the training loop with gradient accumulation and a hard non-finite guard."""
        self.global_step = initial_global_step
        accum = max(1, int(self.cfg.gradient_accumulation_steps))
        micro_step = 0
        recent: list[float] = []
        self.unet.train()
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(total=max_train_steps, initial=self.global_step, desc="train steps")

        if val_loader is not None:
            self.baseline_reference = self.baseline_loss(val_loader, self.cfg.max_val_batches)
            logger.info(
                "LoRA-disabled loss over the validation split: {:.3e}", self.baseline_reference
            )
            self.tracker.log(
                {"train/loss_lora_disabled": self.baseline_reference}, step=self.global_step
            )

        logged_first = False
        while self.global_step < max_train_steps:
            for batch in train_loader:
                if not logged_first:
                    logger.info(
                        "First batch: x_clean={} t={} t5={}",
                        tuple(batch["x_clean"].shape),
                        batch["timestep"].tolist(),
                        tuple(batch["t5_prompt_embeds"].shape),
                    )
                    logged_first = True

                per_example = self.per_example_loss(batch)
                loss = per_example.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at step {self.global_step}: {float(loss)}"
                    )
                (loss / accum).backward()
                self.record_bands(per_example, batch["timestep"])
                recent.append(float(loss))
                micro_step += 1
                if micro_step % accum:
                    continue

                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_parameters,
                        float(self.cfg.max_grad_norm),
                        error_if_nonfinite=True,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.lora_named_parameters)
                self.global_step += 1
                progress.update(1)

                if self._due(self.cfg.log_every_steps):
                    train_loss = sum(recent) / len(recent)
                    bands = self.pop_band_losses()
                    # Also to the logger: with the tracker disabled the curve would otherwise be
                    # invisible, which makes a smoke run unreadable.
                    logger.info(
                        "step {} train/loss={:.6f} (LoRA-disabled baseline {:.6f}) {}",
                        self.global_step,
                        train_loss,
                        self.baseline_reference,
                        " ".join(f"{k.split('/')[-1]}={v:.6f}" for k, v in bands.items()),
                    )
                    self.tracker.log({"train/loss": train_loss, **bands}, step=self.global_step)
                    recent = []
                if val_loader is not None and self._due(self.cfg.eval_every_steps):
                    metrics = self.validate(val_loader, self.cfg.max_val_batches)
                    if self.ema is not None:
                        with self.ema_weights():
                            ema_metrics = self.validate(val_loader, self.cfg.max_val_batches)
                        metrics.update({"val/loss_ema": ema_metrics["val/loss"]})
                    if metrics:
                        logger.info("step {} {}", self.global_step, metrics)
                        self.tracker.log(metrics, step=self.global_step)
                if self.eval_fixtures is not None and self._due(self.cfg.recon_every_steps):
                    metrics = self.reconstruction_eval()
                    logger.info("step {} reconstruction {}", self.global_step, metrics)
                    self.tracker.log(metrics, step=self.global_step)
                if self._due(self.cfg.save_every_steps):
                    self.save_checkpoint(
                        f"checkpoint_step_{self.global_step}.pt",
                        save_training_state=bool(self.cfg.save_training_state),
                    )
                if self.global_step >= max_train_steps:
                    break

        progress.close()
        self.save_checkpoint("checkpoint_final.pt", save_training_state=bool(self.cfg.save_training_state))

    def _due(self, interval: Any) -> bool:
        interval = int(interval)
        return interval > 0 and self.global_step > 0 and self.global_step % interval == 0


class NullTracker:
    """Stand-in tracker so CPU smoke runs need no W&B."""

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        pass

    def finish(self) -> None:
        pass


@hydra.main(config_path="../../config", config_name="train_inversion_lora", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the AudioLDM2 inversion LoRA on cached trajectories."""
    # override=True matches env.py: .env is the only place configuration is edited, so a stale
    # exported variable in the submitting shell must not win.
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    device = torch.device(str(cfg.device))
    if device.type != "cpu":
        torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.seed))

    train_ids, val_ids = split_sample_ids(cfg.data_root, float(cfg.val_fraction), int(cfg.seed))
    train_dataset = AudioLDM2TrajectoryDataset(cfg.data_root, sample_ids=train_ids)
    logger.info(
        "train: {:,} transitions from {} trajectories", len(train_dataset), len(train_ids)
    )
    if cfg.train_max_timestep:
        keep = transitions_below_timestep(train_dataset, int(cfg.train_max_timestep))
        logger.info(
            "train: restricted to t <= {}: {:,} of {:,} transitions",
            int(cfg.train_max_timestep),
            len(keep),
            len(train_dataset),
        )
        train_dataset = Subset(train_dataset, keep)
    val_loader = None
    if val_ids:
        val_dataset = AudioLDM2TrajectoryDataset(cfg.data_root, sample_ids=val_ids)
        logger.info("val:   {:,} transitions from {} trajectories", len(val_dataset), len(val_ids))

        # Flat indices run trajectory-major, so truncating an unshuffled loader to
        # max_val_batches would only ever score the earliest timesteps of the first few
        # trajectories -- the noisy end, where the shift gap is ~300x smaller. Draw a fixed
        # random subset instead: unbiased across the schedule, and identical in every run.
        val_size = len(val_dataset)
        if cfg.max_val_batches:
            capped = int(cfg.max_val_batches) * int(cfg.batch_size)
            if capped < val_size:
                order = torch.randperm(
                    val_size, generator=torch.Generator().manual_seed(int(cfg.seed))
                )
                val_dataset = Subset(val_dataset, order[:capped].tolist())
                logger.info("val:   scoring a fixed random subset of {:,} transitions", capped)

        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            collate_fn=collate_trajectory_batch,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_trajectory_batch,
        drop_last=False,
    )

    ldm = load_model(str(cfg.model_id), device, int(cfg.num_inference_steps), edit_method="ddim")

    tracker: Any = NullTracker()
    if str(cfg.wandb_mode) != "disabled":
        import wandb

        tracker = wandb.init(
            project=str(cfg.wandb_project),
            name=cfg.run_name,
            mode=str(cfg.wandb_mode),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    trainer = AudioLDM2InversionTrainer(ldm, cfg, tracker)
    initial_step = 0
    if cfg.resume_from:
        initial_step = trainer.load_training_state(cfg.resume_from)
        logger.info("Resumed at step {}", initial_step)

    if int(cfg.recon_every_steps) > 0:
        audio_paths, audio_prompts = real_audio_fixtures(
            cfg.recon_prompts_csv,
            cfg.recon_audio_root,
            count=int(cfg.recon_num_real),
            seed=int(cfg.seed),
        )
        trainer.prepare_reconstruction_fixtures(
            prompts=held_out_prompts(cfg.data_root, val_ids, int(cfg.recon_num_generated)),
            audio_paths=audio_paths,
            audio_prompts=audio_prompts,
        )

    trainer.train(
        train_loader,
        val_loader,
        max_train_steps=int(cfg.max_train_steps),
        initial_global_step=initial_step,
    )
    tracker.finish()
    logger.success("Training finished at step {}", trainer.global_step)


if __name__ == "__main__":
    main()
