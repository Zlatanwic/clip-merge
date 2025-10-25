import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
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
        description="Batch inference for multi-focus fusion without evaluation (robust save, optional reflect padding & border crop)."
    )
    parser.add_argument("--ckpt", type=str, default="checkpoints_mfiwh_full/best_model.pth",
                        help="Path to the trained checkpoint (.pth).")
    parser.add_argument("--source1", type=str, default="data/MFI-WHU/source_1",
                        help="Directory with the first focus stack (e.g. near-focus images).")
    parser.add_argument("--source2", type=str, default="data/MFI-WHU/source_2",
                        help="Directory with the second focus stack (e.g. far-focus images).")
    parser.add_argument("--output_dir", type=str, default="results/infer_only",
                        help="Directory to store fused outputs.")
    parser.add_argument("--clip_weight", type=float, default=0.2,
                        help="CLIP branch weight (keep consistent with training unless intentionally changing).")
    parser.add_argument("--force_size", type=int, nargs=2, metavar=("H", "W"), default=None,
                        help="Optional spatial size (H W) to resize both inputs before fusion.")
    parser.add_argument("--device", type=str, default=None,
                        help="Set to 'cuda' or 'cpu'. Defaults to CUDA when available.")
    parser.add_argument("--save_focus_maps", action="store_true",
                        help="If set, saves the two focus maps alongside fused images for inspection.")
    parser.add_argument("--match_strategy", type=str, default="name", choices=["name", "suffix"],
                        help=("How to match files between the two directories. "
                              "'name' requires identical filenames; 'suffix' strips everything before the first delimiter."))
    parser.add_argument("--match_delimiter", type=str, default="_",
                        help="Delimiter used when --match_strategy=suffix to locate the shared suffix.")
    # --- 新增：边界处理相关 ---
    parser.add_argument("--pad_reflect", type=int, default=16,
                        help="Reflect padding (pixels) applied before model; removed after model. 0 to disable. Helps remove black border.")
    parser.add_argument("--crop_border", type=int, default=0,
                        help="Optionally crop this many pixels from each side before saving (after removing reflect pad). 0 to disable.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_key(filename: str, strategy: str, delimiter: str) -> str:
    if strategy == "name":
        return filename
    base, _ = os.path.splitext(filename)
    if delimiter and delimiter in base:
        suffix = base.split(delimiter, 1)[1]
    else:
        suffix = base
    return suffix


def collect_common_keys(
    dirs: Iterable[str], strategy: str, delimiter: str
) -> Tuple[List[str], List[Dict[str, str]]]:
    file_maps: List[Dict[str, str]] = []
    for folder in dirs:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")
        mapping: Dict[str, str] = {}
        entries = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
        if not entries:
            raise RuntimeError(f"No image files found in {folder}")
        for name in entries:
            key = normalize_key(name, strategy, delimiter)
            if key in mapping:
                raise RuntimeError(
                    f"Duplicate match key '{key}' detected in directory {folder}. "
                    "Adjust naming or choose a different --match_strategy/--match_delimiter."
                )
            mapping[key] = name
        file_maps.append(mapping)

    common_keys = set.intersection(*(set(m.keys()) for m in file_maps))
    if not common_keys:
        raise RuntimeError("No common filenames across the provided directories.")
    return sorted(common_keys), file_maps


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


# ---------- 安全保存工具（消除 NaN/Inf & 溢出警告） ----------
def safe_to_uint8_chw(img_chw: torch.Tensor) -> Image.Image:
    """
    img_chw: torch.Tensor [C,H,W], expected roughly in [0,1] but robust to NaN/Inf / slight overflow.
    returns: PIL.Image RGB
    """
    x = img_chw.detach().cpu().float()
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = torch.clamp(x, 0.0, 1.0)
    x = (x * 255.0).round().byte()          # [C,H,W], uint8
    x = x.permute(1, 2, 0).numpy()          # [H,W,C]
    return Image.fromarray(x, mode="RGB")


def safe_save_gray(arr: torch.Tensor, path: str) -> None:
    """
    保存单通道灰度图（例如 focus map）。
    arr: Tensor [H,W] or [1,H,W] or [C,H,W] (取第一通道)，值域预期 0..1，内部会做安全处理。
    """
    a = arr.detach().cpu().float()
    if a.dim() == 3:
        a = a[0]
    a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = torch.clamp(a, 0.0, 1.0)
    a = (a * 255.0).round().byte().numpy()
    Image.fromarray(a, mode="L").save(path)


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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    ensure_dir(args.output_dir)

    keys, file_maps = collect_common_keys([args.source1, args.source2], args.match_strategy, args.match_delimiter)
    print(f"[INFO] Found {len(keys)} common images.")

    model = prepare_model(args.ckpt, device, args.clip_weight)

    progress = tqdm(keys, desc="Batch inference", unit="img")

    PAD = max(0, int(args.pad_reflect))
    CROP = max(0, int(args.crop_border))

    for key in progress:
        name1 = file_maps[0][key]
        name2 = file_maps[1][key]
        path1 = os.path.join(args.source1, name1)
        path2 = os.path.join(args.source2, name2)

        raw1 = load_pil_image(path1)
        raw2 = load_pil_image(path2)

        if args.force_size is None and raw1.size != raw2.size:
            target_size = (raw1.height, raw1.width)
        else:
            target_size = tuple(args.force_size) if args.force_size is not None else None

        img1 = preprocess_image(raw1, target_size).unsqueeze(0).to(device)  # [1,C,H,W]
        img2 = preprocess_image(raw2, target_size).unsqueeze(0).to(device)

        # ---- 反射 padding（去黑边的关键） ----
        if PAD > 0:
            img1 = F.pad(img1, (PAD, PAD, PAD, PAD), mode="reflect")
            img2 = F.pad(img2, (PAD, PAD, PAD, PAD), mode="reflect")

        with torch.no_grad():
            fused_norm, focus_maps = model(img1, img2)  # [1,C,H+2PAD,W+2PAD] if padded

        # ---- 去掉我们加的 padding，恢复到原始 HxW ----
        if PAD > 0:
            fused_norm = fused_norm[..., PAD:-PAD, PAD:-PAD]  # [1,C,H,W]
            if isinstance(focus_maps, torch.Tensor) and focus_maps.dim() >= 3:
                focus_maps = focus_maps[..., PAD:-PAD, PAD:-PAD]

        # ---- 反标准化到 [0,1]，安全保存 ----
        fused_denorm = denormalize(fused_norm.squeeze(0))  # [C,H,W], ~ [0,1] after clamp
        # 可选再裁一个 very small 边框（与 GT 对齐时也裁），仅为保险
        if CROP > 0:
            fused_denorm = fused_denorm[:, CROP:-CROP, CROP:-CROP]
            if isinstance(focus_maps, torch.Tensor) and focus_maps.dim() >= 3:
                focus_maps = focus_maps[:, :, CROP:-CROP, CROP:-CROP]

        fused_img = safe_to_uint8_chw(torch.clamp(fused_denorm, 0.0, 1.0))
        fused_base = (os.path.splitext(key)[0] if args.match_strategy == "name" else key)
        fused_path = os.path.join(args.output_dir, f"{fused_base}_fused.png")
        fused_img.save(fused_path)

        if args.save_focus_maps and isinstance(focus_maps, torch.Tensor):
            focus_dir = os.path.join(args.output_dir, "focus_maps")
            ensure_dir(focus_dir)
            # 假设 focus_maps shape 为 [1, 2, H, W] 或 [B, C, H, W]
            # 取前两个通道保存
            w1 = focus_maps[:, 0:1].squeeze(0)  # [1,H,W]
            w2 = focus_maps[:, 1:2].squeeze(0)
            save_focus_gray_1 = os.path.join(focus_dir, f"{fused_base}_source1.png")
            save_focus_gray_2 = os.path.join(focus_dir, f"{fused_base}_source2.png")
            safe_save_gray(w1, save_focus_gray_1)
            safe_save_gray(w2, save_focus_gray_2)

    print(f"[DONE] Fused images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
