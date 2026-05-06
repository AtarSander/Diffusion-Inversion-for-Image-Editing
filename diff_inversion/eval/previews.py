from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import torch

from diff_inversion.eval.sample_metrics import load_rgb_tensor, plain_area_mask


def normalize_to_uint8(tensor: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    if vmax <= vmin:
        return torch.zeros_like(tensor, dtype=torch.uint8)
    normalized = (tensor - vmin) / (vmax - vmin)
    return normalized.clamp(0, 1).mul(255).round().to(torch.uint8)


def channel_grid_image(tensor: torch.Tensor, vmin: float, vmax: float) -> Image.Image:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"Expected [C,H,W] tensor for preview, got {tuple(tensor.shape)}")

    channels = min(int(tensor.shape[0]), 4)
    height = int(tensor.shape[1])
    width = int(tensor.shape[2])
    canvas = Image.new("L", (width * 2, height * 2), color=0)

    for channel_idx in range(channels):
        channel = normalize_to_uint8(tensor[channel_idx], vmin, vmax)
        image = Image.fromarray(channel.numpy(), mode="L")
        canvas.paste(image, ((channel_idx % 2) * width, (channel_idx // 2) * height))

    return canvas.convert("RGB")


def pca_rgb_image(
    tensor: torch.Tensor, components: torch.Tensor | None = None
) -> tuple[Image.Image, torch.Tensor]:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"Expected [C,H,W] tensor for PCA preview, got {tuple(tensor.shape)}")

    channels, height, width = tensor.shape
    flat = tensor.permute(1, 2, 0).reshape(-1, channels)
    centered = flat - flat.mean(dim=0, keepdim=True)

    if components is None:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        components = vh[: min(3, vh.shape[0])]

    projected = centered @ components.T
    if projected.shape[1] < 3:
        projected = torch.nn.functional.pad(projected, (0, 3 - projected.shape[1]))
    projected = projected[:, :3].reshape(height, width, 3)

    channels_u8 = []
    for channel_idx in range(3):
        channel = projected[:, :, channel_idx]
        vmin = float(channel.quantile(0.01).item())
        vmax = float(channel.quantile(0.99).item())
        channels_u8.append(normalize_to_uint8(channel, vmin, vmax))

    image = torch.stack(channels_u8, dim=-1).numpy()
    return Image.fromarray(image, mode="RGB"), components


def load_final_image_preview(final_image_path: Path, target_size: tuple[int, int]) -> Image.Image:
    with Image.open(final_image_path) as image:
        image = ImageOps.contain(image.convert("RGB"), target_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", target_size, color="white")
    x = (target_size[0] - image.width) // 2
    y = (target_size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def fit_panel(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    image = ImageOps.contain(image.convert("RGB"), target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, color="white")
    x = (target_size[0] - image.width) // 2
    y = (target_size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def write_image_comparison(
    output_path: Path,
    final_image_path: Path,
    reconstructed_image_path: Path,
    plain_threshold: float,
) -> dict[str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(final_image_path) as image:
        reference_size = image.size
    reference = load_rgb_tensor(final_image_path)
    reconstructed = load_rgb_tensor(reconstructed_image_path, size=reference_size)
    error = (reconstructed - reference).abs()
    error_vmax = float(error.quantile(0.99).item())
    error_u8 = normalize_to_uint8(error, 0.0, max(error_vmax, 1e-8))
    error_image = Image.fromarray(error_u8.permute(1, 2, 0).numpy(), mode="RGB")
    plain_mask = plain_area_mask(reference, threshold=plain_threshold)
    plain_mask_image = Image.fromarray(
        plain_mask.to(torch.uint8).mul(255).numpy(),
        mode="L",
    ).convert("RGB")

    error_path = output_path.with_name(f"{output_path.stem}_abs_image_error.png")
    plain_mask_path = output_path.with_name(f"{output_path.stem}_plain_mask.png")
    error_image.save(error_path)
    plain_mask_image.save(plain_mask_path)

    panel_size = (320, 320)
    panels = [
        ("final.png", load_final_image_preview(final_image_path, panel_size)),
        ("reconstructed.png", load_final_image_preview(reconstructed_image_path, panel_size)),
        ("abs image error", fit_panel(error_image, panel_size)),
        ("plain-area mask", fit_panel(plain_mask_image, panel_size)),
    ]
    label_height = 18
    canvas = Image.new(
        "RGB",
        (panel_size[0] * len(panels), panel_size[1] + label_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(panels):
        x = idx * panel_size[0]
        draw.text((x + 4, 3), label, fill="black")
        canvas.paste(image, (x, label_height))
    canvas.save(output_path)

    return {
        "image_comparison_preview_path": output_path.as_posix(),
        "abs_image_error_path": error_path.as_posix(),
        "plain_mask_path": plain_mask_path.as_posix(),
    }


def write_noise_images(
    output_path: Path,
    initial_noise: torch.Tensor,
    inverted_noise: torch.Tensor,
    final_image_path: Path | None = None,
) -> dict[str, str]:
    combined = torch.cat([initial_noise.flatten(), inverted_noise.flatten()])
    vmin = float(combined.quantile(0.01).item())
    vmax = float(combined.quantile(0.99).item())
    error = (inverted_noise - initial_noise).abs()
    error_vmax = float(error.quantile(0.99).item())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_path = output_path.with_name(f"{output_path.stem}_input_noise.png")
    inverted_path = output_path.with_name(f"{output_path.stem}_inverted_noise.png")
    error_path = output_path.with_name(f"{output_path.stem}_abs_error.png")

    input_image = channel_grid_image(initial_noise, vmin, vmax)
    inverted_image = channel_grid_image(inverted_noise, vmin, vmax)
    error_image = channel_grid_image(error, 0.0, max(error_vmax, 1e-8))

    input_image.save(input_path)
    inverted_image.save(inverted_path)
    error_image.save(error_path)

    pca_input_path = output_path.with_name(f"{output_path.stem}_pca_input_noise.png")
    pca_inverted_path = output_path.with_name(f"{output_path.stem}_pca_inverted_noise.png")
    pca_error_path = output_path.with_name(f"{output_path.stem}_pca_abs_error.png")
    pca_input_image, components = pca_rgb_image(initial_noise)
    pca_inverted_image, _ = pca_rgb_image(inverted_noise, components=components)
    pca_error_image, _ = pca_rgb_image(error)
    pca_input_image.save(pca_input_path)
    pca_inverted_image.save(pca_inverted_path)
    pca_error_image.save(pca_error_path)

    output_paths = {
        "preview_path": output_path.as_posix(),
        "input_noise_image_path": input_path.as_posix(),
        "inverted_noise_image_path": inverted_path.as_posix(),
        "abs_error_image_path": error_path.as_posix(),
        "pca_input_noise_image_path": pca_input_path.as_posix(),
        "pca_inverted_noise_image_path": pca_inverted_path.as_posix(),
        "pca_abs_error_image_path": pca_error_path.as_posix(),
    }
    panels = [
        ("input: latents/x_000.pt", input_image),
        ("inverted: inverted_noise.pt", inverted_image),
        ("abs error", error_image),
    ]

    label_height = 18
    panel_width, panel_height = panels[0][1].size
    if final_image_path is not None and final_image_path.exists():
        final_preview_path = output_path.with_name(f"{output_path.stem}_final_image.png")
        final_image = load_final_image_preview(final_image_path, (panel_width, panel_height))
        final_image.save(final_preview_path)
        panels.insert(0, ("final: final.png", final_image))
        output_paths["final_image_path"] = final_image_path.as_posix()
        output_paths["final_image_preview_path"] = final_preview_path.as_posix()

    canvas = Image.new(
        "RGB", (panel_width * len(panels), panel_height + label_height), color="white"
    )
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(panels):
        x = idx * panel_width
        draw.text((x + 4, 3), label, fill="black")
        canvas.paste(image, (x, label_height))

    canvas.save(output_path)
    return output_paths
