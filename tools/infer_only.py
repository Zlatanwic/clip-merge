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

# 导入模型
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
        description="Batch inference for multi-focus fusion (Adapted for Gated Model)."
    )
    parser.add_argument("--ckpt", type=str, default="checkpoints_mfiwh_gated/best_model.pth",
                        help="Path to the trained checkpoint (.pth).")
    parser.add_argument("--source1", type=str, default="Lytro/A",
                        help="Directory with the first focus stack.")
    parser.add_argument("--source2", type=str, default="Lytro/B",
                        help="Directory with the second focus stack.")
    parser.add_argument("--output_dir", type=str, default="results/Lytro_gated_infer",
                        help="Directory to store fused outputs.")
    # 注意：clip_weight 参数已不再使用，但保留以防脚本报错，只是不会传入模型
    parser.add_argument("--clip_weight", type=float, default=0.3,
                        help="[UNUSED in Gated Model] Kept for compatibility.")
    parser.add_argument("--force_size", type=int, nargs=2, metavar=("H", "W"), default=None,
                        help="Optional spatial size (H W) to resize both inputs before fusion.")
    parser.add_argument("--device", type=str, default=None,
                        help="Set to 'cuda' or 'cpu'. Defaults to CUDA when available.")
    parser.add_argument("--save_focus_maps", action="store_true",
                        help="If set, saves focus maps AND gate maps.")
    parser.add_argument("--match_strategy", type=str, default="name", choices=["name", "suffix"],
                        help="Match by 'name' (identical) or 'suffix'.")
    parser.add_argument("--match_delimiter", type=str, default="_",
                        help="Delimiter used when --match_strategy=suffix.")
    # --- 边界处理 ---
    parser.add_argument("--pad_reflect", type=int, default=16,
                        help="Reflect padding pixels. Helps remove black border.")
    parser.add_argument("--crop_border", type=int, default=0,
                        help="Optionally crop this many pixels from output.")
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


# ---------- 安全保存工具 ----------
def safe_to_uint8_chw(img_chw: torch.Tensor) -> Image.Image:
    x = img_chw.detach().cpu().float()
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = torch.clamp(x, 0.0, 1.0)
    x = (x * 255.0).round().byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x, mode="RGB")


def safe_save_gray(arr: torch.Tensor, path: str) -> None:
    """保存单通道灰度图"""
    a = arr.detach().cpu().float()
    if a.dim() == 3:
        a = a[0]
    a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = torch.clamp(a, 0.0, 1.0)
    a = (a * 255.0).round().byte().numpy()
    Image.fromarray(a, mode="L").save(path)


def prepare_model(ckpt_path: str, device: torch.device) -> MultiFocusFusionModel:
    # 🔥 修正 1: 这里不再传入 clip_weight
    model = MultiFocusFusionModel(
        block_size=32,
        overlap=4,
        clip_patch_size=32,
        clip_stride=16,
        # clip_weight=clip_weight,  <-- 已删除
        train_backbone=True,
    ).to(device)

    print(f"[INFO] Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    ensure_dir(args.output_dir)

    keys, file_maps = collect_common_keys([args.source1, args.source2], args.match_strategy, args.match_delimiter)
    print(f"[INFO] Found {len(keys)} common images.")

    # 🔥 修正 2: prepare_model 调用不传 clip_weight
    model = prepare_model(args.ckpt, device)

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

        img1 = preprocess_image(raw1, target_size).unsqueeze(0).to(device)
        img2 = preprocess_image(raw2, target_size).unsqueeze(0).to(device)

        # ---- 反射 padding ----
        if PAD > 0:
            img1 = F.pad(img1, (PAD, PAD, PAD, PAD), mode="reflect")
            img2 = F.pad(img2, (PAD, PAD, PAD, PAD), mode="reflect")

        with torch.no_grad():
            # 🔥 修正 3: 接收 3 个返回值 (fused, focus, gate)
            fused_norm, focus_maps, gate_map = model(img1, img2)

        # ---- 去除 Padding ----
        if PAD > 0:
            fused_norm = fused_norm[..., PAD:-PAD, PAD:-PAD]
            if isinstance(focus_maps, torch.Tensor):
                focus_maps = focus_maps[..., PAD:-PAD, PAD:-PAD]
            if isinstance(gate_map, torch.Tensor):
                gate_map = gate_map[..., PAD:-PAD, PAD:-PAD]

        # ---- 反标准化与保存 ----
        fused_denorm = denormalize(fused_norm.squeeze(0))
        
        # 可选 Crop
        if CROP > 0:
            fused_denorm = fused_denorm[:, CROP:-CROP, CROP:-CROP]
            if isinstance(focus_maps, torch.Tensor):
                focus_maps = focus_maps[:, :, CROP:-CROP, CROP:-CROP]
            if isinstance(gate_map, torch.Tensor):
                gate_map = gate_map[:, :, CROP:-CROP, CROP:-CROP]

        fused_img = safe_to_uint8_chw(torch.clamp(fused_denorm, 0.0, 1.0))
        fused_base = (os.path.splitext(key)[0] if args.match_strategy == "name" else key)
        fused_path = os.path.join(args.output_dir, f"{fused_base}_fused.png")
        fused_img.save(fused_path)

        # ---- 保存 Focus Map 和 Gate Map ----
        if args.save_focus_maps:
            focus_dir = os.path.join(args.output_dir, "focus_maps")
            ensure_dir(focus_dir)
            
            # 保存 Focus Map (Source 1 & 2)
            if isinstance(focus_maps, torch.Tensor):
                w1 = focus_maps[:, 0:1].squeeze(0)
                w2 = focus_maps[:, 1:2].squeeze(0)
                safe_save_gray(w1, os.path.join(focus_dir, f"{fused_base}_source1_map.png"))
                safe_save_gray(w2, os.path.join(focus_dir, f"{fused_base}_source2_map.png"))
            
            # 🔥 新增：保存 Gate Map
            # Gate 越亮(接近1)，表示该区域越倾向于使用 CLIP 结果
            if isinstance(gate_map, torch.Tensor):
                g_map = gate_map.squeeze(0) # [1, H, W]
                safe_save_gray(g_map, os.path.join(focus_dir, f"{fused_base}_gate_map.png"))

    print(f"[DONE] Fused images saved to {args.output_dir}")


if __name__ == "__main__":
    main()