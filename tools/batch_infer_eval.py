import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modelv2 import MultiFocusFusionModel


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NORMALIZE = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch inference for multi-focus fusion using a trained checkpoint, "
            "followed by quantitative evaluation against ground-truth images."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints_mfiwh_full/best_model.pth",
        help="Path to the trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--source1",
        type=str,
        default="data/MFI-WHU/source_1",
        help="Directory with the first focus stack (e.g. near-focus images).",
    )
    parser.add_argument(
        "--source2",
        type=str,
        default="data/MFI-WHU/source_2",
        help="Directory with the second focus stack (e.g. far-focus images).",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="data/MFI-WHU/full_clear",
        help="Directory holding the all-in-focus ground-truth images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/batch_infer",
        help="Directory to store fused outputs.",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default=None,
        help="Optional path to save per-image metrics as CSV.",
    )
    parser.add_argument(
        "--clip_weight",
        type=float,
        default=0.2,
        help="CLIP branch weight (keep in sync with training if changed).",
    )
    parser.add_argument(
        "--force_size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional spatial size (H W) to resize both sources and GT before fusion.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Set to 'cuda' or 'cpu'. Defaults to CUDA when available.",
    )
    return parser.parse_args()


def list_common_filenames(dirs: Iterable[str]) -> List[str]:
    sets = []
    for folder in dirs:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")
        files = {f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))}
        if not files:
            raise RuntimeError(f"No image files found in {folder}")
        sets.append(files)
    common = set.intersection(*sets)
    if not common:
        raise RuntimeError("No common filenames across the provided directories.")
    return sorted(common)


def load_pil_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_image(img: Image.Image, size: Optional[Tuple[int, int]]) -> torch.Tensor:
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    return NORMALIZE(img)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return image * std + mean


def _fspecial_gauss(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    grid_x = coords.view(1, -1).repeat(size, 1)
    grid_y = coords.view(-1, 1).repeat(1, size)
    kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def ssim_value(img1: torch.Tensor, img2: torch.Tensor, size: int = 11, sigma: float = 1.5) -> float:
    if img1.size() != img2.size():
        raise ValueError("Inputs to SSIM must share the same shape.")

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    img1 = img1.to(dtype=torch.float32)
    img2 = img2.to(dtype=torch.float32)
    device = img1.device

    window = _fspecial_gauss(size, sigma, device, img1.dtype)
    window = window.view(1, 1, size, size).repeat(img1.size(1), 1, 1, 1)

    pad = size // 2
    img1_pad = F.pad(img1, (pad, pad, pad, pad), mode="reflect")
    img2_pad = F.pad(img2, (pad, pad, pad, pad), mode="reflect")

    mu1 = F.conv2d(img1_pad, window, stride=1, groups=img1.size(1))
    mu2 = F.conv2d(img2_pad, window, stride=1, groups=img2.size(1))
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1_pad * img1_pad, window, stride=1, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2_pad * img2_pad, window, stride=1, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1_pad * img2_pad, window, stride=1, groups=img1.size(1)) - mu1_mu2

    k1, k2, data_range = 0.01, 0.03, 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def psnr_value(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def mae_value(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def prepare_model(ckpt_path: str, device: torch.device, clip_weight: float) -> MultiFocusFusionModel:
    model = MultiFocusFusionModel(
        block_size=32,
        overlap=4,
        clip_patch_size=32,
        clip_stride=16,
        clip_weight=clip_weight,
        train_backbone=True,
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")

    model.eval()
    return model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    ensure_dir(args.output_dir)

    filenames = list_common_filenames([args.source1, args.source2, args.gt])
    print(f"[INFO] Found {len(filenames)} common images.")

    model = prepare_model(args.ckpt, device, args.clip_weight)

    per_image_metrics = []
    progress = tqdm(filenames, desc="Batch inference", unit="img")

    for name in progress:
        path1 = os.path.join(args.source1, name)
        path2 = os.path.join(args.source2, name)
        gt_path = os.path.join(args.gt, name)

        raw1 = load_pil_image(path1)
        raw2 = load_pil_image(path2)
        gt_raw = load_pil_image(gt_path)

        if args.force_size is None and (raw1.size != raw2.size or raw1.size != gt_raw.size):
            # Default to source1 size when mismatched dimensions are detected.
            target_size = (raw1.height, raw1.width)
        else:
            target_size = tuple(args.force_size) if args.force_size is not None else None

        img1_tensor = preprocess_image(raw1, target_size).unsqueeze(0).to(device)
        img2_tensor = preprocess_image(raw2, target_size).unsqueeze(0).to(device)

        with torch.no_grad():
            fused_norm, _ = model(img1_tensor, img2_tensor)

        fused_denorm = denormalize(fused_norm.squeeze(0)).clamp(0.0, 1.0).cpu()
        fused_pil = transforms.ToPILImage()(fused_denorm)
        fused_pil.save(os.path.join(args.output_dir, os.path.splitext(name)[0] + "_fused.png"))

        if target_size is not None:
            gt_raw = gt_raw.resize((fused_pil.width, fused_pil.height), Image.BILINEAR)

        gt_tensor = TF.to_tensor(gt_raw)

        ssim = ssim_value(fused_denorm.unsqueeze(0), gt_tensor.unsqueeze(0))
        psnr = psnr_value(fused_denorm, gt_tensor)
        mae = mae_value(fused_denorm, gt_tensor)

        per_image_metrics.append(
            {
                "filename": name,
                "psnr": psnr,
                "ssim": ssim,
                "mae": mae,
            }
        )

        progress.set_postfix({"PSNR": f"{psnr:.2f}", "SSIM": f"{ssim:.3f}", "MAE": f"{mae:.4f}"})

    mean_psnr = sum(m["psnr"] for m in per_image_metrics) / len(per_image_metrics)
    mean_ssim = sum(m["ssim"] for m in per_image_metrics) / len(per_image_metrics)
    mean_mae = sum(m["mae"] for m in per_image_metrics) / len(per_image_metrics)

    print("\n[RESULTS] Aggregated metrics across fused images:")
    print(f"  • Mean PSNR: {mean_psnr:.2f} dB")
    print(f"  • Mean SSIM: {mean_ssim:.4f}")
    print(f"  • Mean MAE : {mean_mae:.4f}")

    if args.metrics_csv:
        ensure_dir(os.path.dirname(args.metrics_csv) or ".")
        with open(args.metrics_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["filename", "psnr", "ssim", "mae"])
            writer.writeheader()
            writer.writerows(per_image_metrics)
        print(f"[INFO] Saved per-image metrics to {args.metrics_csv}")


if __name__ == "__main__":
    main()
