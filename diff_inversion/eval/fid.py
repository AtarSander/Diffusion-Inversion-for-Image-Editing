import json
import os
from pathlib import Path
import random
import shutil

from cleanfid import fid as cleanfid_fid
import hydra
from hydra.utils import to_absolute_path
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import PIL
import torch
from torchvision import transforms
from tqdm import tqdm


def tensors_to_pil_single(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return transforms.ToPILImage()(tensor)


def seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def tens_to_imagedir(tens_path: Path, ds_idx: int):
    dir_name = tens_path.name[:-3]
    save_dir = (tens_path.parent / dir_name).resolve()
    if save_dir.exists():
        logger.info("Directory {} already exists", save_dir)
    else:
        os.makedirs(save_dir, exist_ok=True)
        tensor = torch.load(tens_path, weights_only=False, map_location="cpu")
        if "celeba_ldm_256" in str(tens_path):
            images = ldm_tensor_to_images(tensor)
        else:
            images = [tensors_to_pil_single(tensor[idx]) for idx in range(tensor.shape[0])]

        for idx, sample in tqdm(
            enumerate(images), total=len(images), desc=f"Saving images for dataset {ds_idx}"
        ):
            sample.save(save_dir / f"{idx}.png")

    return save_dir


def check_is_dir_with_images(path: Path):
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"Directory/file {path} does not exist")
    if any(path.glob("*.png")) or any(path.glob("*.jpg")) or any(path.glob("*.jpeg")):
        logger.info("Directory {} exists with images", path)
        return True
    elif path.name.endswith(".pt"):
        return False
    else:
        raise ValueError(f"Directory/file {path} does not exist")


def ldm_tensor_to_images(ldm_out):
    image_processed = ldm_out.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
    return [PIL.Image.fromarray(img_processed) for img_processed in image_processed]


def run_fid_paths(
    path1: str,
    path2: str,
    seed: int = 42,
    mode: str = "legacy_tensorflow",
    num_workers: int = 16,
    batch_size: int = 64,
    device: str | torch.device = "cuda",
):
    seed_everywhere(seed)
    device = torch.device(device)
    logger.info("Using device: {}", device)

    feat_model = cleanfid_fid.build_feature_extractor(mode, device)

    ppath1, ppath2 = Path(path1).resolve(), Path(path2).resolve()
    ppath1_dir_created = False
    ppath2_dir_created = False
    if not check_is_dir_with_images(ppath1):
        ppath1: Path = tens_to_imagedir(ppath1, ds_idx=1)
        ppath1_dir_created = True
    if not check_is_dir_with_images(ppath2):
        ppath2: Path = tens_to_imagedir(ppath2, ds_idx=2)
        ppath2_dir_created = True

    fid_score = cleanfid_fid.compare_folders(
        ppath1.as_posix(),
        ppath2.as_posix(),
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
    )
    if ppath1_dir_created:
        shutil.rmtree(ppath1.as_posix(), ignore_errors=True)
    if ppath2_dir_created:
        shutil.rmtree(ppath2.as_posix(), ignore_errors=True)

    return float(fid_score)


def get_reference_statistics_ours(
    name, res, mode="clean", model_name="inception_v3", seed=0, split="test", metric="FID"
):
    if name == "imagenet":
        if res in [256, 64]:
            np_array = np.load(f"res/eval/VIRTUAL_imagenet{res}_labeled.npz")
            mu = np_array["mu"]
            sigma = np_array["sigma"]
            return mu, sigma
    elif name == "celeba":
        if res == 64:
            np_array = np.load("res/eval/celeba_64.npz")
            mu = np_array["mu"]
            sigma = np_array["sigma"]
            return mu, sigma
        else:
            raise ValueError(f"Invalid resolution for CelebA: {res}")
    else:
        return cleanfid_fid.get_reference_statistics(
            name=name,
            res=res,
            mode=mode,
            model_name=model_name,
            seed=seed,
            split=split,
            metric=metric,
        )


def fid_folder_ours(
    fdir,
    dataset_name,
    dataset_res,
    dataset_split,
    model=None,
    mode="clean",
    model_name="inception_v3",
    num_workers=12,
    batch_size=128,
    device=torch.device("cuda"),
    verbose=True,
    custom_image_tranform=None,
    custom_fn_resize=None,
):
    ref_mu, ref_sigma = get_reference_statistics_ours(
        dataset_name, dataset_res, mode=mode, model_name=model_name, seed=0, split=dataset_split
    )
    np_feats = cleanfid_fid.get_folder_features(
        fdir,
        model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description="",
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
        custom_fn_resize=custom_fn_resize,
    )
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = cleanfid_fid.frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


def run_fid_path_ds(
    path: str,
    dataset_name: str,
    dataset_res: int,
    dataset_split: str,
    seed: int = 42,
    mode: str = "legacy_tensorflow",
    model_name: str = "inception_v3",
    num_workers: int = 16,
    batch_size: int = 64,
    device: str | torch.device = "cuda",
):
    seed_everywhere(seed)
    device = torch.device(device)
    logger.info("Using device: {}", device)

    feat_model = cleanfid_fid.build_feature_extractor(mode, device)

    ppath = Path(path).resolve()
    ppath_dir_created = False
    if not check_is_dir_with_images(ppath):
        ppath: Path = tens_to_imagedir(ppath, ds_idx=1)
        ppath_dir_created = True

    fid_score = fid_folder_ours(
        fdir=ppath.as_posix(),
        dataset_name=dataset_name,
        dataset_res=dataset_res,
        dataset_split=dataset_split,
        model=feat_model,
        model_name=model_name,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
    )

    if ppath_dir_created:
        shutil.rmtree(ppath.as_posix(), ignore_errors=True)

    return float(fid_score)


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _validate_fid_config(cfg: DictConfig) -> None:
    has_path_ref = cfg.path_ref is not None
    has_ds_name = cfg.ds_name is not None

    if has_path_ref == has_ds_name:
        raise ValueError("Exactly one of path_ref or ds_name must be provided")

    if has_ds_name and (cfg.dataset_res is None or cfg.dataset_split is None):
        raise ValueError("dataset_res and dataset_split must be provided if ds_name is provided")


@hydra.main(config_path="../../config/eval", config_name="fid", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("FID config:\n{}", OmegaConf.to_yaml(cfg))

    # pass path like experiments/noise_latent_interpolate/latent/celeba_ldm_256/T_100/alpha_0.0/samples_decoded.pt
    # script will go through all dirs in T_100, find all alphas, and run fid for each alpha, alphas should be sorted to avoid confusion
    # path_ref stays unchanged
    _validate_fid_config(cfg)

    output_dir = _resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path_outputs = _resolve_path(cfg.path_outputs)
    path_ref = _resolve_path(cfg.path_ref) if cfg.path_ref is not None else None
    filename = path_outputs.name
    parent_T = path_outputs.parent.parent
    alphas = sorted(list(parent_T.glob("alpha_*")))
    results = {
        "path_ref": path_ref.as_posix() if path_ref is not None else None,
        "path_outputs": path_outputs.as_posix(),
        "ds_name": cfg.ds_name,
        "dataset_res": cfg.dataset_res,
        "dataset_split": cfg.dataset_split,
        "seed": cfg.seed,
        "mode": cfg.mode,
        "model_name": cfg.model_name,
        "num_workers": cfg.num_workers,
        "batch_size": cfg.batch_size,
        "device": cfg.device,
    }
    outputs = {}
    logger.info("Running FID sweep over {} alpha directories", len(alphas))
    for alpha in tqdm(alphas, desc="Running FID"):
        alpha_str = str(alpha).split("_")[-1]
        path_outputs_alpha = alpha / filename
        logger.info("Running FID for {} and {}", path_outputs_alpha, path_ref)
        if path_ref is not None:
            fid_score = run_fid_paths(
                path1=path_outputs_alpha,
                path2=path_ref,
                seed=cfg.seed,
                mode=cfg.mode,
                num_workers=cfg.num_workers,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
        else:
            fid_score = run_fid_path_ds(
                path=path_outputs_alpha,
                dataset_name=cfg.ds_name,
                dataset_res=cfg.dataset_res,
                dataset_split=cfg.dataset_split,
                seed=cfg.seed,
                mode=cfg.mode,
                model_name=cfg.model_name,
                num_workers=cfg.num_workers,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
        outputs[alpha_str] = fid_score
    results["outputs"] = outputs

    output_path = output_dir / cfg.output_filename
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.success("Saved FID scores to {}", output_path)


if __name__ == "__main__":
    main()
