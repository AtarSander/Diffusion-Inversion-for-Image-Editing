# ABOUTME: Train a LoRA on Stable Audio Open so that DPMSolver inversion becomes near-exact, by
# ABOUTME: distilling the frozen teacher's prediction at x_t into the student's at the cleaner x_t-1.

import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.dataset import (  # noqa: E402
    AudioLDM2TrajectoryDataset,
    collate_stable_audio_batch,
    split_sample_ids,
    transitions_below_timestep,
    transitions_with_room_below,
)
from src.inversion_lora.stable_audio import ExactDPMSolver, load_teacher  # noqa: E402
from src.inversion_lora.train import (  # noqa: E402
    AudioLDM2InversionTrainer,
    NullTracker,
    band_labels,
)

STABLE_AUDIO_CONDITIONING_KEYS = ("text_audio",)
# What the cached dataset must be, or the loss is not the inversion error. See
# output/sao_schedules/REPORT.md for why each of the three is what it is.
REQUIRED_META = {
    "schedule": "cosine",
    "solver": "first_order_ode",
    "pairing": "matched_timestep",
    "target_space": "data_prediction",
}


class StableAudioInversionTrainer(AudioLDM2InversionTrainer):
    """Distils the frozen teacher's shifted prediction into a LoRA on the Stable Audio DiT.

    Everything the base trainer does is model-independent -- adapter injection, EMA, the loss
    bands, checkpointing, the loop -- so only the forward differs: Stable Audio conditions on one
    cross-attention tensor plus duration and rotary embeddings held by the teacher, and its
    denoiser takes per-example timesteps directly.
    """

    def __init__(self, ldm, cfg: DictConfig, tracker: Any):
        super().__init__(ldm, cfg, tracker)
        self.solver = ExactDPMSolver(ldm.model.scheduler)
        # The base class bands the schedule against num_train_timesteps=1000, which is meaningless
        # on a grid whose timesteps run 0.99..0.19: every example would land in one band.
        self.band_top = max(self.solver.timesteps)
        self.band_labels = band_labels(
            self.band_top, int(cfg.num_loss_bands), self.band_top
        )
        self._band_sums = torch.zeros(len(self.band_labels), dtype=torch.float64)
        self._band_counts = torch.zeros(len(self.band_labels), dtype=torch.float64)
        logger.info(
            "solver grid: {} steps, t {:.4g}..{:.4g}, sigma {:.4g}..{:.4g}",
            len(self.solver.timesteps), self.solver.timesteps[0], self.solver.timesteps[-1],
            float(self.solver.sigmas[0]), float(self.solver.sigmas[-2]),
        )
        self.cycle_enabled = bool(cfg.get("cycle", {}).get("enabled", False))
        self.cycle_steps = int(cfg.get("cycle", {}).get("steps", 1))
        if self.cycle_enabled:
            # The balance is read off the deepest adapter tensor, so the two probe backwards stop
            # near the end of the network instead of traversing all 24 blocks.
            anchors = [n for n in self.lora_named_parameters if "lora_B" in n]
            assert anchors, f"no lora_B tensor among {list(self.lora_named_parameters)[:4]}"
            self.balance_anchor = self.lora_named_parameters[anchors[-1]]
            a, b = self.solver.coefficients(0)
            # B^2 is the ratio of loss *values*. The gradient ratio the balancer equalises also
            # carries the factor (I + (B/A) J) from differentiating the teacher at z_hat, so the
            # reported lambda sits below 1/B^2 by however much that Jacobian amplifies.
            logger.info(
                "cycle loss ON: steps={} target_ratio={} anchor={} | grid A={:.5f} B={:.5f} "
                "B/A={:.4f}, so the value ratio is ~B^2={:.3e} (1/B^2={:.0f}) while lambda "
                "balances gradients and will read lower",
                self.cycle_steps, float(cfg.cycle.target_ratio), anchors[-1],
                float(a), float(b), float(b) / float(a), float(b) ** 2, 1.0 / float(b) ** 2,
            )

    def log_first_batch(self, batch: dict[str, Any]) -> None:
        """Print the first training batch's shapes, Stable Audio's single conditioning included."""
        logger.info(
            "First batch: x_clean={} t={} text_audio={}",
            tuple(batch["x_clean"].shape),
            batch["timestep"].tolist(),
            tuple(batch["text_audio"].shape),
        )

    def predict_noise(self, batch: dict[str, Any]) -> torch.Tensor:
        """The student's data prediction at the cleaner latent, at that latent's own timestep.

        The target is the teacher's data prediction at the noisier latent, which is exactly what the
        reverse step consumed, so closing this difference makes the exact inverse exact. Everything
        is in data-prediction space rather than raw network output: that is the quantity the solver
        consumes, and on an EDM grid the two differ by a sigma-dependent scale.
        """
        x_clean = batch["x_clean"].to(device=self.device, dtype=self.unet.dtype)
        timestep = batch["timestep"].to(device=self.device, dtype=torch.float32)
        text_audio = batch["text_audio"].to(device=self.device, dtype=self.unet.dtype)
        assert x_clean.shape[0] == timestep.shape[0] == text_audio.shape[0], (
            x_clean.shape,
            timestep.shape,
            text_audio.shape,
        )
        sigmas = self.solver.sigma_for(timestep).to(dtype=x_clean.dtype)
        model_input = self.solver.model_input_batch(x_clean, sigmas)
        conditional = self.solver.data_prediction_batch(
            x_clean, self.ldm.forward(model_input, timestep, text_audio), sigmas
        )
        if self.uncond is None:
            return conditional
        # The step this is distilled from was driven by the guided combination, so the student
        # must be guided too -- a merged adapter perturbs both branches at deployment.
        unconditional = self.solver.data_prediction_batch(
            x_clean,
            self.ldm.forward(model_input, timestep, self.uncond.expand(x_clean.shape[0], -1, -1)),
            sigmas,
        )
        return unconditional + self.guidance_scale * (conditional - unconditional)

    def build_uncond(self, ldm) -> torch.Tensor:
        """Stable Audio's unconditional cross-attention states `[1, S, D]`.

        The base class calls `encode_text([""], negative=True)`, which is AudioLDM2's three-tensor
        signature; Stable Audio carries a single tensor from `encode_prompt`.
        """
        return ldm.encode_prompt("").detach()

    def data_prediction_at(
        self, x: torch.Tensor, index: torch.Tensor, text_audio: torch.Tensor
    ) -> torch.Tensor:
        """The network's data prediction at latent `x`, read at grid step `index`.

        Args:
            x: Latents `[B, C, L]`.
            index: Grid step indices `[B]`, on the CPU.
            text_audio: Cross-attention states `[B, S, D]`.

        Returns:
            Data predictions `[B, C, L]`.
        """
        assert x.shape[0] == index.shape[0] == text_audio.shape[0], (x.shape, index.shape)
        sigmas = self.solver.sigmas[index].to(device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        timestep = torch.tensor(
            [self.solver.timesteps[int(i)] for i in index], device=x.device, dtype=torch.float32
        )
        raw = self.ldm.forward(self.solver.model_input_batch(x, sigmas), timestep, text_audio)
        return self.solver.data_prediction_batch(x, raw, sigmas)

    def cycle_per_example(self, batch: dict[str, Any], student: torch.Tensor) -> torch.Tensor:
        """Round-trip error of a `cycle_steps`-step inversion followed by the same many gen steps.

        One step reduces to `B^2 ||D_phi(x_{k+1}) - D_theta(z_hat_k)||^2`: the inverse cancels the
        A of the generation step, so only the prediction difference survives, scaled by B. For
        `k > 1` the chain is walked with the adapter and walked back with the frozen teacher, which
        penalises how per-step residuals *compound* -- the thing a single step cannot see.

        Only the final inversion step carries gradient; the earlier ones run under `no_grad`, which
        keeps k adapter graphs from being alive at once.

        Args:
            batch: One training batch.
            student: The adapter's data prediction at the stored (cleaner) latent, with graph.

        Returns:
            Per-example squared round-trip error `[B]`.
        """
        x_clean = batch["x_clean"].to(device=self.device, dtype=self.unet.dtype)
        timestep = batch["timestep"].to(device=self.device, dtype=torch.float32)
        text_audio = batch["text_audio"].to(device=self.device, dtype=self.unet.dtype)
        k = int(self.cycle_steps)

        # The stored timestep is the cleaner latent's own, at grid index `top`; the reverse step
        # that produced it runs from `top - 1`, so that is the first step to undo.
        top = self.solver.index_for(timestep)
        assert int(top.min()) >= k, (
            f"a {k}-step cycle needs {k} steps below the stored latent, but one example sits at "
            f"grid index {int(top.min())}; restrict the dataset or lower cycle_steps"
        )

        # Walk up (noisier). The first step uses the prediction we already have; later ones need
        # fresh adapter calls, which run detached.
        x, indices = x_clean, []
        for step in range(k):
            index = top - 1 - step
            indices.append(index)
            if step == 0:
                prediction = student
            else:
                with torch.no_grad():
                    prediction = self.data_prediction_at(x, index + 1, text_audio)
            a, b = (c.to(device=x.device, dtype=x.dtype) for c in
                    self.solver.coefficients_batch(index))
            x = (x - b * prediction) / a

        # Walk back down with the frozen teacher. Gradient flows through x the whole way.
        # This cannot be combined with gradient checkpointing: the recompute happens during the
        # backward, after this context has exited, so the blocks would be re-run with the adapter
        # ON and the saved activations would not match. Memory has to come from the batch size.
        with self.lora_disabled():
            for index in reversed(indices):
                prediction = self.data_prediction_at(x, index, text_audio)
                a, b = (c.to(device=x.device, dtype=x.dtype) for c in
                        self.solver.coefficients_batch(index))
                x = a * x + b * prediction

        return ((x.float() - x_clean.float()) ** 2).flatten(1).mean(dim=1)

    def adaptive_weight(self, inversion: torch.Tensor, cycle: torch.Tensor) -> torch.Tensor:
        """Gradient-norm balance between the two terms, measured at the deepest adapter tensor.

        The VQGAN adaptive weight: scale the second loss so its gradient norm matches the first's,
        times a target ratio. Detached, so it is a weight and not something to differentiate
        through. Computed on one late LoRA tensor rather than the whole adapter, which keeps the
        two extra backward passes short.

        Args:
            inversion: The inversion loss scalar.
            cycle: The cycle loss scalar.

        Returns:
            A detached scalar weight.
        """
        g_inv = torch.autograd.grad(inversion, self.balance_anchor, retain_graph=True)[0]
        g_cyc = torch.autograd.grad(cycle, self.balance_anchor, retain_graph=True)[0]
        weight = g_inv.norm() / (g_cyc.norm() + float(self.cfg.cycle.eps))
        return weight.clamp(max=float(self.cfg.cycle.lambda_max)).detach() * float(
            self.cfg.cycle.target_ratio
        )

    def training_loss(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Inversion loss, plus the adaptively weighted cycle loss when it is enabled."""
        student = self.predict_noise(batch)
        target = self.target_noise(batch)
        assert student.shape == target.shape, (student.shape, target.shape)
        per_example = ((student.float() - target.float()) ** 2).flatten(1).mean(dim=1)
        inversion = per_example.mean()
        if not self.cycle_enabled:
            return inversion, per_example, {}

        cycle = self.cycle_per_example(batch, student).mean()
        weight = self.adaptive_weight(inversion, cycle)
        return (
            inversion + weight * cycle,
            per_example,
            {
                "train/loss_inv": float(inversion),
                "train/loss_cycle": float(cycle),
                "train/lambda_cycle": float(weight),
                "train/cycle_value_share": float(weight * cycle) / max(float(inversion), 1e-12),
            },
        )


def check_dataset_convention(data_root: str, guidance_scale: float | None = None) -> None:
    """Refuse a cached dataset that was not generated the way the loss assumes.

    A dataset from the beta grid, or with the shifted pairing, or holding raw network outputs, would
    train silently against the wrong quantity -- which is exactly what happened once already.

    Args:
        data_root: Dataset directory holding `sample_*`.
        guidance_scale: Guidance the loss will be formed at; None skips the check.
    """
    samples = sorted(Path(data_root).glob("sample_*/meta.json"))
    if not samples:
        raise FileNotFoundError(f"no sample_*/meta.json under {data_root}")
    meta = json.loads(samples[0].read_text())
    if guidance_scale is not None and float(meta.get("guidance_scale", 1.0)) != guidance_scale:
        raise ValueError(
            f"{samples[0].parent} was generated at guidance "
            f"{meta.get('guidance_scale', 1.0)}, but training asks for {guidance_scale}. The "
            "target is the prediction the reverse step consumed, so these must agree."
        )
    wrong = {k: meta.get(k) for k, v in REQUIRED_META.items() if meta.get(k) != v}
    if wrong:
        raise ValueError(
            f"{samples[0].parent} was generated under a different convention: {wrong}, expected "
            f"{REQUIRED_META}. Regenerate with generate_trajectories_stable_audio.py."
        )
    logger.info("dataset convention: {}", {k: meta[k] for k in REQUIRED_META})


def build_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader | None, set[int]]:
    """Build the train and validation loaders over a cached Stable Audio trajectory set.

    Splits by trajectory, optionally restricts training to the cleanest tail of the schedule, and
    scores validation on a fixed random subset so the reported loss covers the whole schedule
    rather than the first few trajectories' noisiest steps.

    Args:
        cfg: The resolved training config.

    Returns:
        `(train_loader, val_loader, val_ids)`; `val_loader` is None when nothing is held out.
    """
    check_dataset_convention(cfg.data_root, float(cfg.get("guidance_scale", 1.0)))
    train_ids, val_ids = split_sample_ids(cfg.data_root, float(cfg.val_fraction), int(cfg.seed))
    load_uncond = float(cfg.get("guidance_scale", 1.0)) != 1.0
    train_dataset = AudioLDM2TrajectoryDataset(
        cfg.data_root,
        sample_ids=train_ids,
        conditioning_keys=STABLE_AUDIO_CONDITIONING_KEYS,
        timestep_dtype=torch.float32,
        load_uncond=load_uncond,
    )
    logger.info("train: {:,} transitions from {} trajectories", len(train_dataset), len(train_ids))
    if cfg.train_max_timestep:
        keep = transitions_below_timestep(train_dataset, int(cfg.train_max_timestep))
        logger.info(
            "train: restricted to t <= {}: {:,} of {:,} transitions",
            int(cfg.train_max_timestep),
            len(keep),
            len(train_dataset),
        )
        train_dataset = Subset(train_dataset, keep)

    cycle_steps = int(cfg.get("cycle", {}).get("steps", 1))
    if cfg.get("cycle", {}).get("enabled", False) and cycle_steps > 1:
        # Validation is deliberately left unfiltered: it scores the inversion loss alone, so
        # keeping the same transitions across every arm makes val/loss comparable.
        base = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
        room = set(transitions_with_room_below(base, cycle_steps))
        if isinstance(train_dataset, Subset):
            keep = [i for i in train_dataset.indices if i in room]
            train_dataset = Subset(base, keep)
        else:
            train_dataset = Subset(base, sorted(room))
        logger.info(
            "train: a {}-step cycle needs {} grid points below each latent: {:,} transitions kept",
            cycle_steps, cycle_steps, len(train_dataset),
        )

    val_loader = None
    if val_ids:
        val_dataset = AudioLDM2TrajectoryDataset(
            cfg.data_root,
            sample_ids=val_ids,
            conditioning_keys=STABLE_AUDIO_CONDITIONING_KEYS,
            timestep_dtype=torch.float32,
            load_uncond=load_uncond,
        )
        logger.info("val:   {:,} transitions from {} trajectories", len(val_dataset), len(val_ids))
        capped = int(cfg.max_val_batches) * int(cfg.batch_size)
        if cfg.max_val_batches and capped < len(val_dataset):
            order = torch.randperm(
                len(val_dataset), generator=torch.Generator().manual_seed(int(cfg.seed))
            )
            val_dataset = Subset(val_dataset, order[:capped].tolist())
            logger.info("val:   scoring a fixed random subset of {:,} transitions", capped)
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            collate_fn=collate_stable_audio_batch,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_stable_audio_batch,
        drop_last=False,
    )
    return train_loader, val_loader, val_ids


@hydra.main(
    config_path="../../config", config_name="train_inversion_lora_stable_audio", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Train the Stable Audio Open inversion LoRA on cached trajectories."""
    # override=True matches env.py: .env is the only place configuration is edited, so a stale
    # exported variable in the submitting shell must not win.
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    device = torch.device(str(cfg.device))
    if device.type != "cpu":
        torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.seed))

    train_loader, val_loader, _ = build_loaders(cfg)

    teacher = load_teacher(
        str(cfg.model_id),
        device,
        int(cfg.num_inference_steps),
        duration_s=cfg.duration_s,
        schedule=str(cfg.schedule),
    )

    tracker: Any = NullTracker()
    if str(cfg.wandb_mode) != "disabled":
        import wandb

        tracker = wandb.init(
            project=str(cfg.wandb_project),
            name=cfg.run_name,
            mode=str(cfg.wandb_mode),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    trainer = StableAudioInversionTrainer(teacher, cfg, tracker)
    initial_step = 0
    if cfg.resume_from:
        initial_step = trainer.load_training_state(cfg.resume_from)
        logger.info("Resumed at step {}", initial_step)

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
